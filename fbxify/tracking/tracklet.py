from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Detection:
    frame_index: int
    person_id: str
    shape_params: Optional[List[float]] = None
    pred_cam_t: Optional[List[float]] = None
    pred_global_rots: Optional[List[Any]] = None
    pred_joint_coords: Optional[List[Any]] = None
    pred_keypoints_3d: Optional[List[Any]] = None
    bbox_xywh: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "person_id": self.person_id,
            "shape_params": self.shape_params,
            "pred_cam_t": self.pred_cam_t,
            "pred_global_rots": self.pred_global_rots,
            "pred_joint_coords": self.pred_joint_coords,
            "pred_keypoints_3d": self.pred_keypoints_3d,
            "bbox_xywh": self.bbox_xywh,
        }


@dataclass
class Tracklet:
    track_id: int
    detections: List[Detection] = field(default_factory=list)

    def add_detection(self, detection: Detection) -> None:
        self.detections.append(detection)

    @property
    def start_frame(self) -> int:
        return self.detections[0].frame_index if self.detections else -1

    @property
    def end_frame(self) -> int:
        return self.detections[-1].frame_index if self.detections else -1

    @property
    def last_detection(self) -> Optional[Detection]:
        return self.detections[-1] if self.detections else None

    def to_dict(self, include_detections: bool = True) -> Dict[str, Any]:
        data = {
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "length": len(self.detections),
        }
        if include_detections:
            data["detections"] = [d.to_dict() for d in self.detections]
        return data
