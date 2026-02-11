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

    # Minimum cam similarity when use_pred_cam_t is True (rejects "teleport" matches)
    # Pairs with cam similarity below this get total similarity 0. Default 0.01 rejects cam=0.
    min_cam_similarity: float = 0.01

    # Minimum pose similarity when use_pose_aux is True (rejects weak pose agreement)
    # Pairs with pose similarity below this get total similarity 0. Default 0.0 = no gate.
    # Set e.g. 0.3–0.4 to reject matches where pose is only ~30% the same.
    min_pose_similarity: float = 0.0

    # Age-aware (moving gate): trust shape only after this many detections in the tracklet
    shape_maturity_frames: int = 20
    # When mature and shape_sim >= high_shape_threshold, allow match even if cam < min_cam (re-emerged person)
    high_shape_override_cam: bool = True
    high_shape_threshold: float = 0.95

    # Mature-track shape: spike-removed weighted average over recent detections
    # Max number of detections to use for track shape average (most recent)
    shape_avg_max_frames: int = 50
    # Exclude frames whose L1 distance to median shape > median_dist + this * MAD (outlier removal)
    shape_spike_mad_factor: float = 3.0
    # Exponential weight decay per frame going back in time (1.0 = most recent, decay^1 = one frame back)
    shape_avg_weight_decay: float = 0.98

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

    # -----------------------------
    # Background Filtering (optional)
    # -----------------------------
    # Master toggle (default off)
    bg_filter_enabled: bool = False

    # Detection-level gates (applied per frame before association)
    # 0 disables a gate.
    bg_min_bbox_height_px: float = 0.0
    bg_min_bbox_area_px2: float = 0.0
    # Keep detections with pred_cam_t[2] <= this (0 disables). Only applies when pred_cam_t exists.
    bg_depth_max_z: float = 0.0
    # Keep nearest z-quantile per frame (0 disables). Example: 0.7 keeps nearest 70% by z.
    bg_keep_nearest_z_quantile: float = 0.0

    # Automatic size cutoffs (applied per frame; uses bbox height/area)
    # - none: disabled
    # - percentile: drop smallest X fraction (bg_size_percentile)
    # - otsu: Otsu threshold on log(size)
    # - gmm2: 2-component 1D GMM on log(size)
    bg_size_auto_method: str = "none"
    # For percentile method: fraction to drop from the small end (0 disables).
    bg_size_percentile: float = 0.0
    bg_size_feature: str = "bbox_height"  # "bbox_height" | "bbox_area"

    # Tracklet-level scoring (stable-subject filter)
    bg_tracklet_score_enabled: bool = True
    # Keep tracklets with foreground_score >= threshold. 0 disables.
    bg_tracklet_score_threshold: float = 0.0
    bg_min_tracklet_frames_for_scoring: int = 10

    # Foreground score weights (sum not required; normalized internally)
    bg_w_length: float = 0.45
    bg_w_size: float = 0.25
    bg_w_size_stability: float = 0.15
    bg_w_centering: float = 0.15

    # Auto ROI (computed, not user-provided)
    bg_auto_roi_enabled: bool = False
    bg_auto_roi_window_frames: int = 30
    bg_auto_roi_point: str = "bottom_center"  # "bottom_center" | "center"
    bg_auto_roi_mad_k: float = 3.0
    bg_auto_roi_min_radius_px: float = 60.0
    bg_auto_roi_smoothing_alpha: float = 0.2

    # Optional second-pass refinement (rerun association with auto ROI gate)
    bg_refine_second_pass: bool = False

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
