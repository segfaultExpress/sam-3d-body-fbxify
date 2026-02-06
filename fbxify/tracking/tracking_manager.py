from typing import Dict, Any, List, Optional
import numpy as np
from fbxify.tracking.tracklet import Tracklet
from fbxify.tracking.mhr_tracker import MHRTracker
from fbxify.tracking.tracking_config import TrackingConfig


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
    ) -> Dict[str, Any]:
        if not config.enabled:
            return {
                "mode": mode,
                "enabled": False,
                "config": config.to_dict(),
                "tracklets": [],
            }

        tracklets = self.tracker.build_tracklets(estimation_results, config)
        tracking_metadata = {
            "mode": mode,
            "enabled": True,
            "config": config.to_dict(),
            "tracklets": self._tracklets_to_dict(tracklets, config),
        }

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
