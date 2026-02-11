from typing import Dict, Any, List, Optional
import numpy as np
from fbxify.tracking.tracklet import Tracklet
from fbxify.tracking.mhr_tracker import MHRTracker
from fbxify.tracking.tracking_config import TrackingConfig


def apply_frame_assignments(
    estimation_results: Dict[str, Dict[str, Any]],
    frame_assignments: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, Any]]:
    """
    Remap estimation results so person_id keys become track_id (stable across frames).

    frame_assignments: { frame_key: { original_person_id_str: track_id_int } }
    Returns a new estimation_results dict with the same structure but keys remapped to track_id.
    """
    remapped: Dict[str, Dict[str, Any]] = {}
    for frame_key, frame_data in estimation_results.items():
        if not isinstance(frame_data, dict):
            remapped[frame_key] = frame_data if frame_data is not None else {}
            continue
        frame_map = frame_assignments.get(frame_key, {})
        remapped[frame_key] = {}
        for orig_id_str, track_id in frame_map.items():
            if orig_id_str in frame_data:
                remapped[frame_key][str(track_id)] = frame_data[orig_id_str]
    return remapped


class TrackingManager:
    """
    Orchestrates inference tracking and prepares metadata for JSON export.
    """

    def __init__(self, tracker: Optional[MHRTracker] = None):
        self.tracker = tracker or MHRTracker()

    def run(
        self,
        estimation_results: Dict[str, Dict[str, Any]],
        config: TrackingConfig,
        mode: str = "Inference Tracking",
        debug_per_frame: bool = False,
        step_through: bool = False,
        debug_start_frame: int = 0,
    ) -> Dict[str, Any]:
        if not config.enabled:
            return {
                "mode": mode,
                "enabled": False,
                "config": config.to_dict(),
                "tracklets": [],
            }

        tracklets = self.tracker.build_tracklets(
            estimation_results,
            config,
            debug_per_frame=debug_per_frame,
            step_through=step_through,
            debug_start_frame=debug_start_frame,
        )
        tracking_metadata = {
            "mode": mode,
            "enabled": True,
            "config": config.to_dict(),
            "tracklets": self._tracklets_to_dict(tracklets, config),
        }
        bg = getattr(self.tracker, "last_background_filtering", None)
        if bg is not None:
            tracking_metadata["background_filtering"] = bg

        if config.export_frame_assignments:
            tracking_metadata["frame_assignments"] = self._build_frame_assignments(tracklets)

        return tracking_metadata

    def _tracklets_to_dict(self, tracklets: List[Tracklet], config: TrackingConfig) -> List[Dict[str, Any]]:
        data = []
        for tracklet in tracklets:
            tracklet_data = tracklet.to_dict(include_detections=config.export_tracklet_detections)
            tracklet_data["summary"] = self._summarize_tracklet(tracklet)
            data.append(tracklet_data)
        return data

    def _summarize_tracklet(self, tracklet: Tracklet) -> Dict[str, Any]:
        shape_params = []
        cam_t = []
        for det in tracklet.detections:
            if det.shape_params is not None:
                shape_params.append(np.array(det.shape_params, dtype=np.float32))
            if det.pred_cam_t is not None:
                cam_t.append(np.array(det.pred_cam_t, dtype=np.float32))

        summary: Dict[str, Any] = {}
        if shape_params:
            summary["shape_params_mean"] = np.mean(np.stack(shape_params, axis=0), axis=0).tolist()
        if cam_t:
            summary["pred_cam_t_mean"] = np.mean(np.stack(cam_t, axis=0), axis=0).tolist()
        return summary

    def _build_frame_assignments(self, tracklets: List[Tracklet]) -> Dict[str, Dict[str, int]]:
        assignments: Dict[str, Dict[str, int]] = {}
        for tracklet in tracklets:
            for det in tracklet.detections:
                frame_key = str(det.frame_index)
                if frame_key not in assignments:
                    assignments[frame_key] = {}
                assignments[frame_key][str(det.person_id)] = tracklet.track_id
        return assignments
