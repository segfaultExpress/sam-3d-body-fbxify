from typing import Any, Optional
import numpy as np


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.array(value)
    except Exception:
        return None


def mean_abs_diff(a: Any, b: Any) -> Optional[float]:
    arr_a = _to_numpy(a)
    arr_b = _to_numpy(b)
    if arr_a is None or arr_b is None:
        return None
    if arr_a.size == 0 or arr_b.size == 0:
        return None
    try:
        diff = np.abs(arr_a.astype(np.float32) - arr_b.astype(np.float32))
        return float(np.mean(diff))
    except Exception:
        return None


def l2_distance(a: Any, b: Any) -> Optional[float]:
    arr_a = _to_numpy(a)
    arr_b = _to_numpy(b)
    if arr_a is None or arr_b is None:
        return None
    if arr_a.size == 0 or arr_b.size == 0:
        return None
    try:
        diff = arr_a.astype(np.float32) - arr_b.astype(np.float32)
        return float(np.linalg.norm(diff))
    except Exception:
        return None


def bbox_iou_xywh(a: Any, b: Any) -> Optional[float]:
    arr_a = _to_numpy(a)
    arr_b = _to_numpy(b)
    if arr_a is None or arr_b is None:
        return None
    if arr_a.size < 4 or arr_b.size < 4:
        return None
    ax, ay, aw, ah = arr_a[:4]
    bx, by, bw, bh = arr_b[:4]
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = (aw * ah) + (bw * bh) - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def similarity_from_distance(distance: Optional[float], threshold: float) -> float:
    """
    Linear similarity from distance: sim = max(0, 1 - distance / threshold).
    """
    if distance is None:
        return 0.0
    if threshold <= 0:
        return 0.0
    return max(0.0, 1.0 - (distance / threshold))
