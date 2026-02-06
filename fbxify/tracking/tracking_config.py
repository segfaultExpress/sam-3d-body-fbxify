from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import json


@dataclass
class TrackingConfig:
    """
    Configuration for inference tracking based on parameter consistency.
    """
    enabled: bool = True

    # Tracklet creation/merge thresholds
    max_gap_frames: int = 5
    merge_max_gap_frames: int = 12
    min_tracklet_length: int = 2
    min_similarity: float = 0.4

    # Distance thresholds (lower distance => higher similarity)
    shape_distance_threshold: float = 1.2
    cam_distance_threshold: float = 0.8
    pose_distance_threshold: float = 0.9
    iou_distance_threshold: float = 0.5

    # Weights for similarity score
    shape_weight: float = 0.5
    cam_weight: float = 0.3
    pose_weight: float = 0.2
    iou_weight: float = 0.0

    # Feature toggles
    use_shape_params: bool = True
    use_pred_cam_t: bool = True
    use_pose_aux: bool = True
    use_bbox_iou: bool = False

    # Export controls
    export_frame_assignments: bool = True
    export_tracklet_detections: bool = True
    export_mot_bboxes: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TrackingConfig":
        cfg = TrackingConfig()
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    @staticmethod
    def load_json(path: str) -> "TrackingConfig":
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Tracking config JSON must be an object")
        return TrackingConfig.from_dict(data)

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
