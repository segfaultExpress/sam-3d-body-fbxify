from typing import Dict, Any, List, Tuple, Optional
import numpy as np
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

    Cam similarity is linear: sim = max(0, 1 - L2(pred_cam_t_a, pred_cam_t_b) / cam_distance_threshold).
    So cam=0.0 means L2 distance >= threshold (e.g. person moved a lot, or was occluded many frames
    and reappeared elsewhere). It does not drop off exponentially. Use min_cam_similarity to reject
    these "teleport" matches. Use min_pose_similarity to reject weak pose agreement (e.g. pose ~30%
    same is often a different person).
    """

    def build_tracklets(
        self,
        estimation_results: Dict[str, Dict[str, Any]],
        config: TrackingConfig,
        debug_per_frame: bool = False,
        step_through: bool = False,
        debug_start_frame: int = 0,
    ) -> List[Tracklet]:
        frames = self._build_detections(estimation_results)
        active: List[Tracklet] = []
        finished: List[Tracklet] = []
        next_track_id = 0
        print("Running Tracking Inference...")

        for frame_index in sorted(frames.keys()):
            # Intense debug (matrix, assignments, Enter pause) only from this frame onward
            do_intense = (debug_per_frame or step_through) and (frame_index >= debug_start_frame)

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
                if do_intense and debug_per_frame:
                    self._debug_frame_header(frame_index, active, detections, expired_count, 0, 0)
                elif not do_intense:
                    if frame_index == debug_start_frame - 1 and (debug_per_frame or step_through):
                        print(f"  ... frame {frame_index} (debug starts at frame {debug_start_frame})")
                    elif not (debug_per_frame or step_through) or debug_start_frame <= 0:
                        print("--------------------------------")
                        print("Frame Index: ", frame_index)
                        print(f"  [{frame_index}] People actively being tracked {len(active)}")
                        print(f"  [{frame_index}] Distinct people found this frame {len(detections)}")
                        print(f"  [{frame_index}] Re-IDs - 0 already exist, [0] new ids assigned")
                        print(f"  [{frame_index}] Tracklets expired this frame {expired_count}")
                if do_intense and step_through:
                    input("Press Enter for next frame...")
                continue

            # Build (track_idx, det_idx, score) and optionally (score, breakdown) for debug
            matches = self._match_detections(active, detections, config)
            if do_intense and debug_per_frame:
                self._debug_frame_header(frame_index, active, detections, expired_count, None, None)
                self._debug_similarity_matrix(active, detections, config, matches)
            matched_detections = set()
            matched_tracklets = set()
            accepted_pairs: List[Tuple[int, int]] = []  # (track_idx, det_idx) when debug

            for track_idx, det_idx, score in matches:
                if score < config.min_similarity:
                    continue
                if track_idx in matched_tracklets or det_idx in matched_detections:
                    continue
                active[track_idx].add_detection(detections[det_idx])
                matched_tracklets.add(track_idx)
                matched_detections.add(det_idx)
                if do_intense and debug_per_frame:
                    accepted_pairs.append((track_idx, det_idx))

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
            if do_intense and debug_per_frame:
                self._debug_assignments(active, detections, accepted_pairs, matched_detections, reid_count, new_id_count)
            elif not do_intense:
                if frame_index == debug_start_frame - 1 and (debug_per_frame or step_through):
                    print(f"  ... frame {frame_index} (debug starts at frame {debug_start_frame})")
                elif not (debug_per_frame or step_through) or debug_start_frame <= 0:
                    print("--------------------------------")
                    print("Frame Index: ", frame_index)
                    print(f"  [{frame_index}] People actively being tracked {len(active)}")
                    print(f"  [{frame_index}] Distinct people found this frame {len(detections)}")
                    print(f"  [{frame_index}] Re-IDs - {reid_count} already exist, [{new_id_count}] new ids assigned")
                    print(f"  [{frame_index}] Tracklets expired this frame {expired_count}")
            if do_intense and step_through:
                input("Press Enter for next frame...")

        finished.extend(active)
        finished = [t for t in finished if len(t.detections) >= config.min_tracklet_length]

        if config.merge_max_gap_frames > 0:
            finished = self._merge_tracklets(finished, config)

        return finished

    def _debug_frame_header(
        self,
        frame_index: int,
        active: List[Tracklet],
        detections: List[Detection],
        expired_count: int,
        reid_count: Optional[int],
        new_id_count: Optional[int],
    ) -> None:
        reid_s = str(reid_count) if reid_count is not None else "0"
        new_s = str(new_id_count) if new_id_count is not None else "0"
        print("--------------------------------")
        print(f"[DEBUG] Frame Index: {frame_index}")
        print(f"  Active tracklets: {len(active)}")
        for t_idx, t in enumerate(active):
            last = t.last_detection
            bbox = last.bbox_xywh if last else None
            bbox_s = f" bbox={bbox}" if bbox else ""
            print(f"    track_idx={t_idx} track_id={t.track_id} last_frame={t.end_frame} last_person_id={last.person_id if last else None}{bbox_s}")
        print(f"  Detections this frame: {len(detections)}")
        for d_idx, d in enumerate(detections):
            bbox_s = f" bbox={d.bbox_xywh}" if d.bbox_xywh else ""
            print(f"    det_idx={d_idx} person_id={d.person_id}{bbox_s}")
        print(f"  Tracklets expired this frame: {expired_count}")
        if reid_count is not None and new_id_count is not None:
            print(f"  Re-IDs - {reid_s} matched to existing, {new_s} new track(s) assigned")

    def _debug_similarity_matrix(
        self,
        active: List[Tracklet],
        detections: List[Detection],
        config: TrackingConfig,
        matches: List[Tuple[int, int, float]],
    ) -> None:
        min_cam = getattr(config, "min_cam_similarity", 0.0)
        min_pose = getattr(config, "min_pose_similarity", 0.0)
        print("  Similarity (track_idx, det_idx) -> total [breakdown]:")
        for (t_idx, d_idx, score) in matches:
            last = active[t_idx].last_detection
            det = detections[d_idx]
            track_id = active[t_idx].track_id
            tracklet_age = len(active[t_idx].detections)
            _, breakdown = self._similarity_breakdown(
                last, det, config,
                tracklet_age=tracklet_age,
                tracklet=active[t_idx],
            )
            above = " ACCEPT" if score >= config.min_similarity else ""
            if score == 0.0 and above == "":
                reason = []
                if config.use_pred_cam_t and min_cam > 0 and breakdown.get("cam", 1.0) < min_cam:
                    reason.append("cam < min")
                if config.use_pose_aux and min_pose > 0 and breakdown.get("pose", 1.0) < min_pose:
                    reason.append("pose < min")
                if reason:
                    above = " REJECT (" + ", ".join(reason) + ")"
            print(f"    (idx={t_idx} track_id={track_id}, det_idx={d_idx}) -> {score:.4f} {breakdown}{above}")

    def _debug_assignments(
        self,
        active: List[Tracklet],
        detections: List[Detection],
        accepted_pairs: List[Tuple[int, int]],
        matched_detections: set,
        reid_count: int,
        new_id_count: int,
    ) -> None:
        print("  Accepted (track continued): det_idx person_id -> track_id")
        for (t_idx, d_idx) in accepted_pairs:
            if t_idx < len(active) and d_idx < len(detections):
                print(f"    det_idx={d_idx} person_id={detections[d_idx].person_id} -> track_id={active[t_idx].track_id}")
        print(f"  New track(s) assigned: {new_id_count} detection(s)")
        for d_idx, det in enumerate(detections):
            if d_idx in matched_detections:
                continue
            print(f"    det_idx={d_idx} person_id={det.person_id} -> new track (added to active)")

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
                tracklet_age = len(tracklet.detections)
                score = self._similarity(
                    last_det, detection, config,
                    tracklet_age=tracklet_age,
                    tracklet=tracklet,
                )
                pairs.append((t_idx, d_idx, score))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def _track_shape_average(
        self,
        tracklet: Tracklet,
        config: TrackingConfig,
    ) -> Optional[List[float]]:
        """
        Spike-removed weighted average of shape_params over the track's recent detections.
        Used for matching only when track is mature (>= shape_maturity_frames).
        Returns None if not enough data or shape params missing.
        """
        maturity = getattr(config, "shape_maturity_frames", 20)
        max_frames = getattr(config, "shape_avg_max_frames", 50)
        mad_factor = getattr(config, "shape_spike_mad_factor", 3.0)
        decay = getattr(config, "shape_avg_weight_decay", 0.98)

        # Collect shape_params from most recent detections
        shapes: List[np.ndarray] = []
        for det in reversed(tracklet.detections):
            if len(shapes) >= max_frames:
                break
            if det.shape_params is not None and len(det.shape_params) > 0:
                shapes.append(np.array(det.shape_params, dtype=np.float32))
        shapes.reverse()  # oldest first

        if len(shapes) < maturity:
            return None
        stack = np.stack(shapes, axis=0)
        # Per-component median
        median_shape = np.median(stack, axis=0)
        # L1 distance of each frame to median (mean abs diff per row)
        dists = np.mean(np.abs(stack - median_shape), axis=1)
        median_d = np.median(dists)
        mad_d = np.median(np.abs(dists - median_d))
        if mad_d <= 1e-9:
            mad_d = 1e-9
        # Keep frames within median_d + mad_factor * MAD
        threshold = median_d + mad_factor * mad_d
        keep = dists <= threshold
        if not np.any(keep):
            keep = np.ones(len(dists), dtype=bool)
        stack = stack[keep]
        n = len(stack)
        # Weights: newest = 1.0, older = decay^1, decay^2, ...
        weights = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=np.float32)
        weights /= weights.sum()
        avg_shape = np.average(stack, axis=0, weights=weights)
        return avg_shape.tolist()

    def _similarity(
        self,
        a: Detection,
        b: Detection,
        config: TrackingConfig,
        tracklet_age: Optional[int] = None,
        tracklet: Optional[Tracklet] = None,
    ) -> float:
        total, _ = self._similarity_breakdown(
            a, b, config,
            tracklet_age=tracklet_age,
            tracklet=tracklet,
        )
        return total

    def _similarity_breakdown(
        self,
        a: Detection,
        b: Detection,
        config: TrackingConfig,
        tracklet_age: Optional[int] = None,
        tracklet: Optional[Tracklet] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Return (total_weighted_score, breakdown_dict) for debug logging.
        When tracklet_age < shape_maturity_frames, shape weight is 0 (pose+cam only).
        When mature, shape comparison uses spike-removed weighted average of track shape if available.
        When mature and shape_sim >= high_shape_threshold, min_cam gate can be skipped.
        """
        total_weight = 0.0
        score = 0.0
        breakdown: Dict[str, float] = {}

        maturity = getattr(config, "shape_maturity_frames", 20)
        use_shape = config.use_shape_params and (
            tracklet_age is None or tracklet_age >= maturity
        )
        if tracklet_age is not None and tracklet_age < maturity and config.use_shape_params:
            breakdown["_shape_gated"] = 1.0  # hint for debug: shape not used

        if config.use_shape_params:
            # For mature tracklets, use spike-removed weighted average shape when available
            track_shape = None
            if tracklet is not None and tracklet_age is not None and tracklet_age >= maturity:
                track_shape = self._track_shape_average(tracklet, config)
            shape_a = track_shape if track_shape is not None else a.shape_params
            dist = mean_abs_diff(shape_a, b.shape_params)
            sim = similarity_from_distance(dist, config.shape_distance_threshold)
            breakdown["shape"] = round(sim, 4)
            if use_shape:
                score += config.shape_weight * sim
                total_weight += config.shape_weight

        if config.use_pred_cam_t:
            dist = l2_distance(a.pred_cam_t, b.pred_cam_t)
            sim = similarity_from_distance(dist, config.cam_distance_threshold)
            score += config.cam_weight * sim
            total_weight += config.cam_weight
            breakdown["cam"] = round(sim, 4)

        if config.use_pose_aux:
            dist = mean_abs_diff(a.pred_global_rots, b.pred_global_rots)
            sim = similarity_from_distance(dist, config.pose_distance_threshold)
            score += config.pose_weight * sim
            total_weight += config.pose_weight
            breakdown["pose"] = round(sim, 4)

        if config.use_bbox_iou:
            iou = bbox_iou_xywh(a.bbox_xywh, b.bbox_xywh)
            sim = iou if iou is not None else 0.0
            score += config.iou_weight * sim
            total_weight += config.iou_weight
            breakdown["iou"] = round(sim, 4)

        if total_weight <= 0:
            return 0.0, breakdown
        total = score / total_weight

        # Reject matches where cam disagrees strongly (person cannot teleport between frames)
        min_cam = getattr(config, "min_cam_similarity", 0.0)
        high_override = getattr(config, "high_shape_override_cam", False)
        high_threshold = getattr(config, "high_shape_threshold", 0.95)
        shape_sim = breakdown.get("shape", 0.0)
        skip_cam_gate = (
            use_shape
            and high_override
            and shape_sim >= high_threshold
        )
        if not skip_cam_gate and config.use_pred_cam_t and min_cam > 0:
            cam_sim = breakdown.get("cam", 0.0)
            if cam_sim < min_cam:
                return 0.0, breakdown
        # Reject matches where pose agrees too weakly
        min_pose = getattr(config, "min_pose_similarity", 0.0)
        if config.use_pose_aux and min_pose > 0:
            pose_sim = breakdown.get("pose", 0.0)
            if pose_sim < min_pose:
                return 0.0, breakdown
        return total, breakdown

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
                tracklet_age = len(base.detections)
                score = self._similarity(
                    base.last_detection,
                    candidate.detections[0],
                    config,
                    tracklet_age=tracklet_age,
                    tracklet=base,
                )
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
