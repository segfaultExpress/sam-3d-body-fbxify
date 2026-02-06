from typing import Dict, Any, List, Tuple, Optional
from fbxify.tracking.tracker import TrackletTracker
from fbxify.tracking.tracking_config import TrackingConfig
from fbxify.tracking.tracklet import Detection, Tracklet
from fbxify.tracking.metrics import (
    mean_abs_diff,
    l2_distance,
    bbox_iou_xywh,
    similarity_from_distance,
)


class MHRTracker(TrackletTracker):
    """
    Simple temporal association tracker using MHR parameter consistency.
    """

    def build_tracklets(
        self,
        estimation_results: Dict[str, Dict[str, Any]],
        config: TrackingConfig
    ) -> List[Tracklet]:
        frames = self._build_detections(estimation_results)
        active: List[Tracklet] = []
        finished: List[Tracklet] = []
        next_track_id = 0
        print("Running Tracking Inference...")

        for frame_index in sorted(frames.keys()):
            detections = frames[frame_index]

            # Finalize inactive tracklets
            still_active = []
            expired_count = 0
            for tracklet in active:
                if frame_index - tracklet.end_frame > config.max_gap_frames:
                    finished.append(tracklet)
                    expired_count += 1
                else:
                    still_active.append(tracklet)
            active = still_active

            if not detections:
                print("--------------------------------")
                print("Frame Index: ", frame_index)
                print(f"  [{frame_index}] People actively being tracked {len(active)}")
                print(f"  [{frame_index}] Distinct people found this frame {len(detections)}")
                print(f"  [{frame_index}] Re-IDs - 0 already exist, [0] new ids assigned")
                print(f"  [{frame_index}] Tracklets expired this frame {expired_count}")
                continue

            matches = self._match_detections(active, detections, config)
            matched_detections = set()
            matched_tracklets = set()

            for track_idx, det_idx, score in matches:
                if score < config.min_similarity:
                    continue
                if track_idx in matched_tracklets or det_idx in matched_detections:
                    continue
                active[track_idx].add_detection(detections[det_idx])
                matched_tracklets.add(track_idx)
                matched_detections.add(det_idx)

            # Create new tracklets for unmatched detections
            for det_idx, detection in enumerate(detections):
                if det_idx in matched_detections:
                    continue
                tracklet = Tracklet(track_id=next_track_id)
                next_track_id += 1
                tracklet.add_detection(detection)
                active.append(tracklet)

            reid_count = len(matched_detections)
            new_id_count = len(detections) - reid_count
            print("--------------------------------")
            print("Frame Index: ", frame_index)
            print(f"  [{frame_index}] People actively being tracked {len(active)}")
            print(f"  [{frame_index}] Distinct people found this frame {len(detections)}")
            print(f"  [{frame_index}] Re-IDs - {reid_count} already exist, [{new_id_count}] new ids assigned")
            print(f"  [{frame_index}] Tracklets expired this frame {expired_count}")

        finished.extend(active)
        finished = [t for t in finished if len(t.detections) >= config.min_tracklet_length]

        if config.merge_max_gap_frames > 0:
            finished = self._merge_tracklets(finished, config)

        return finished

    def _build_detections(self, estimation_results: Dict[str, Dict[str, Any]]) -> Dict[int, List[Detection]]:
        frames: Dict[int, List[Detection]] = {}
        for frame_index_str, frame_data in estimation_results.items():
            try:
                frame_index = int(frame_index_str)
            except Exception:
                continue
            detections: List[Detection] = []
            if isinstance(frame_data, dict):
                for person_id, data in frame_data.items():
                    if not isinstance(data, dict):
                        continue
                    detection = Detection(
                        frame_index=frame_index,
                        person_id=str(person_id),
                        shape_params=data.get("shape_params"),
                        pred_cam_t=data.get("pred_cam_t"),
                        pred_global_rots=data.get("pred_global_rots"),
                        pred_joint_coords=data.get("pred_joint_coords"),
                        pred_keypoints_3d=data.get("pred_keypoints_3d"),
                        bbox_xywh=data.get("bbox_xywh"),
                    )
                    detections.append(detection)
            frames[frame_index] = detections
        return frames

    def _match_detections(
        self,
        tracklets: List[Tracklet],
        detections: List[Detection],
        config: TrackingConfig,
    ) -> List[Tuple[int, int, float]]:
        pairs: List[Tuple[int, int, float]] = []
        for t_idx, tracklet in enumerate(tracklets):
            last_det = tracklet.last_detection
            if last_det is None:
                continue
            for d_idx, detection in enumerate(detections):
                score = self._similarity(last_det, detection, config)
                pairs.append((t_idx, d_idx, score))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def _similarity(self, a: Detection, b: Detection, config: TrackingConfig) -> float:
        total_weight = 0.0
        score = 0.0

        if config.use_shape_params:
            dist = mean_abs_diff(a.shape_params, b.shape_params)
            sim = similarity_from_distance(dist, config.shape_distance_threshold)
            score += config.shape_weight * sim
            total_weight += config.shape_weight

        if config.use_pred_cam_t:
            dist = l2_distance(a.pred_cam_t, b.pred_cam_t)
            sim = similarity_from_distance(dist, config.cam_distance_threshold)
            score += config.cam_weight * sim
            total_weight += config.cam_weight

        if config.use_pose_aux:
            dist = mean_abs_diff(a.pred_global_rots, b.pred_global_rots)
            sim = similarity_from_distance(dist, config.pose_distance_threshold)
            score += config.pose_weight * sim
            total_weight += config.pose_weight

        if config.use_bbox_iou:
            iou = bbox_iou_xywh(a.bbox_xywh, b.bbox_xywh)
            sim = iou if iou is not None else 0.0
            score += config.iou_weight * sim
            total_weight += config.iou_weight

        if total_weight <= 0:
            return 0.0
        return score / total_weight

    def _merge_tracklets(self, tracklets: List[Tracklet], config: TrackingConfig) -> List[Tracklet]:
        if not tracklets:
            return tracklets
        tracklets = sorted(tracklets, key=lambda t: t.start_frame)
        merged: List[Tracklet] = []
        i = 0
        while i < len(tracklets):
            base = tracklets[i]
            j = i + 1
            best_idx = None
            best_score = config.min_similarity
            while j < len(tracklets):
                candidate = tracklets[j]
                gap = candidate.start_frame - base.end_frame
                if gap < 0:
                    j += 1
                    continue
                if gap > config.merge_max_gap_frames:
                    break
                score = self._similarity(base.last_detection, candidate.detections[0], config)
                if score >= best_score:
                    best_score = score
                    best_idx = j
                j += 1
            if best_idx is not None:
                candidate = tracklets[best_idx]
                merged_tracklet = Tracklet(track_id=base.track_id)
                for det in base.detections + candidate.detections:
                    merged_tracklet.add_detection(det)
                tracklets.pop(best_idx)
                tracklets[i] = merged_tracklet
            else:
                merged.append(base)
                i += 1
        return merged
