from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from fbxify.tracking.tracklet import Detection, Tracklet
from fbxify.tracking.tracking_config import TrackingConfig


@dataclass
class BackgroundFilterStats:
    """Lightweight counters for debugging/metadata."""

    frames_seen: int = 0
    detections_seen: int = 0
    detections_kept: int = 0

    dropped_missing_bbox: int = 0
    dropped_min_height: int = 0
    dropped_min_area: int = 0
    dropped_depth_max_z: int = 0
    dropped_keep_nearest_z_quantile: int = 0
    dropped_auto_size: int = 0
    dropped_auto_roi: int = 0

    tracklets_seen: int = 0
    tracklets_kept: int = 0
    tracklets_dropped_by_score: int = 0
    tracklets_guardrail_kept: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames_seen": self.frames_seen,
            "detections_seen": self.detections_seen,
            "detections_kept": self.detections_kept,
            "dropped_missing_bbox": self.dropped_missing_bbox,
            "dropped_min_height": self.dropped_min_height,
            "dropped_min_area": self.dropped_min_area,
            "dropped_depth_max_z": self.dropped_depth_max_z,
            "dropped_keep_nearest_z_quantile": self.dropped_keep_nearest_z_quantile,
            "dropped_auto_size": self.dropped_auto_size,
            "dropped_auto_roi": self.dropped_auto_roi,
            "tracklets_seen": self.tracklets_seen,
            "tracklets_kept": self.tracklets_kept,
            "tracklets_dropped_by_score": self.tracklets_dropped_by_score,
            "tracklets_guardrail_kept": self.tracklets_guardrail_kept,
        }


def _bbox_xywh(det: Detection) -> Optional[Tuple[float, float, float, float]]:
    b = det.bbox_xywh
    if b is None or len(b) < 4:
        return None
    try:
        x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    except Exception:
        return None
    if not np.isfinite([x, y, w, h]).all():
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _bbox_size(det: Detection, feature: str) -> Optional[float]:
    bb = _bbox_xywh(det)
    if bb is None:
        return None
    _, _, w, h = bb
    if feature == "bbox_area":
        return float(w * h)
    # default: bbox_height
    return float(h)


def _bbox_point(det: Detection, point_kind: str) -> Optional[Tuple[float, float]]:
    bb = _bbox_xywh(det)
    if bb is None:
        return None
    x, y, w, h = bb
    if point_kind == "center":
        return float(x + 0.5 * w), float(y + 0.5 * h)
    # default: bottom_center
    return float(x + 0.5 * w), float(y + 1.0 * h)


def _pred_cam_z(det: Detection) -> Optional[float]:
    t = det.pred_cam_t
    if t is None or len(t) < 3:
        return None
    try:
        z = float(t[2])
    except Exception:
        return None
    if not np.isfinite(z):
        return None
    return z


def _sigmoid(x: float) -> float:
    x = float(x)
    if x >= 60:
        return 1.0
    if x <= -60:
        return 0.0
    return float(1.0 / (1.0 + np.exp(-x)))


def _otsu_threshold(values: np.ndarray, bins: int = 128) -> Optional[float]:
    """Otsu threshold for 1D float array. Returns threshold value (same scale)."""
    if values.size < 8:
        return None
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not np.isfinite([vmin, vmax]).all():
        return None
    if vmax <= vmin + 1e-9:
        return None
    hist, edges = np.histogram(values, bins=bins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    if hist.sum() <= 0:
        return None
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) * 0.5)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom <= 1e-12] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    k = int(np.nanargmax(sigma_b2))
    thr = float((edges[k] + edges[k + 1]) * 0.5)
    return thr


def _gmm2_threshold(values: np.ndarray, iters: int = 30) -> Optional[float]:
    """Fit a 2-component 1D GMM with EM; return decision boundary between means."""
    if values.size < 12:
        return None
    x = values.astype(np.float64)
    x = x[np.isfinite(x)]
    if x.size < 12:
        return None

    # Initialize means by quantiles
    m1 = float(np.quantile(x, 0.3))
    m2 = float(np.quantile(x, 0.7))
    if m2 <= m1 + 1e-6:
        return None
    v1 = float(np.var(x) + 1e-3)
    v2 = float(np.var(x) + 1e-3)
    w1, w2 = 0.5, 0.5

    for _ in range(iters):
        # E-step
        p1 = w1 * (1.0 / np.sqrt(2.0 * np.pi * v1)) * np.exp(-0.5 * (x - m1) ** 2 / v1)
        p2 = w2 * (1.0 / np.sqrt(2.0 * np.pi * v2)) * np.exp(-0.5 * (x - m2) ** 2 / v2)
        s = p1 + p2 + 1e-12
        r1 = p1 / s
        r2 = p2 / s
        # M-step
        n1 = float(np.sum(r1))
        n2 = float(np.sum(r2))
        if n1 < 1e-6 or n2 < 1e-6:
            break
        w1 = n1 / (n1 + n2)
        w2 = 1.0 - w1
        m1 = float(np.sum(r1 * x) / n1)
        m2 = float(np.sum(r2 * x) / n2)
        v1 = float(np.sum(r1 * (x - m1) ** 2) / n1 + 1e-6)
        v2 = float(np.sum(r2 * (x - m2) ** 2) / n2 + 1e-6)

    # Ensure ordering
    if m2 < m1:
        m1, m2 = m2, m1
        v1, v2 = v2, v1
        w1, w2 = w2, w1

    # Compute intersection of two weighted Gaussians.
    # Solve: w1*N(x|m1,v1) = w2*N(x|m2,v2)
    # => ax^2 + bx + c = 0
    a = (1.0 / v2) - (1.0 / v1)
    b = (-2.0 * m2 / v2) + (2.0 * m1 / v1)
    c = (m2 * m2 / v2) - (m1 * m1 / v1) + 2.0 * np.log((w2 * np.sqrt(v1) + 1e-12) / (w1 * np.sqrt(v2) + 1e-12))

    if abs(a) < 1e-9:
        # Similar variances; midpoint is fine.
        return float(0.5 * (m1 + m2))

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return float(0.5 * (m1 + m2))
    sqrt_disc = float(np.sqrt(disc))
    rA = (-b + sqrt_disc) / (2.0 * a)
    rB = (-b - sqrt_disc) / (2.0 * a)
    lo, hi = float(m1), float(m2)
    # Pick intersection between means when possible
    candidates = [r for r in (rA, rB) if lo <= r <= hi]
    if candidates:
        return float(candidates[0])
    # Otherwise pick closest to midpoint
    mid = 0.5 * (lo + hi)
    return float(rA if abs(rA - mid) < abs(rB - mid) else rB)


def filter_detections_for_frame(
    detections: List[Detection],
    config: TrackingConfig,
    stats: Optional[BackgroundFilterStats] = None,
    auto_roi: Optional[Tuple[float, float, float]] = None,  # (cx, cy, r)
) -> Tuple[List[Detection], List[Tuple[Detection, str]]]:
    """Filter detections using bg_* config.

    Returns:
        (kept, dismissed) where *dismissed* is a list of
        ``(detection, reason_string)`` tuples for debug logging.
    """
    dismissed: List[Tuple[Detection, str]] = []

    if not getattr(config, "bg_filter_enabled", False):
        return detections, dismissed

    if stats is not None:
        stats.frames_seen += 1
        stats.detections_seen += len(detections)

    kept = list(detections)

    # Detection-level bbox gates
    min_h = float(getattr(config, "bg_min_bbox_height_px", 0.0) or 0.0)
    min_a = float(getattr(config, "bg_min_bbox_area_px2", 0.0) or 0.0)

    if min_h > 0 or min_a > 0 or auto_roi is not None or getattr(config, "bg_size_auto_method", "none") != "none":
        out = []
        for d in kept:
            bb = _bbox_xywh(d)
            if bb is None:
                if auto_roi is not None:
                    if stats is not None:
                        stats.dropped_missing_bbox += 1
                    dismissed.append((d, "Missing BBox"))
                    continue
                # If bbox is missing but ROI/size filter isn't required, keep.
                out.append(d)
                continue
            _, _, w, h = bb
            if min_h > 0 and h < min_h:
                if stats is not None:
                    stats.dropped_min_height += 1
                dismissed.append((d, f"BBox Too Small (h={h:.1f} < min_h={min_h:.1f})"))
                continue
            if min_a > 0 and (w * h) < min_a:
                if stats is not None:
                    stats.dropped_min_area += 1
                dismissed.append((d, f"BBox Too Small (area={w * h:.0f} < min_area={min_a:.0f})"))
                continue
            out.append(d)
        kept = out

    # Depth gates
    depth_max_z = float(getattr(config, "bg_depth_max_z", 0.0) or 0.0)
    if depth_max_z > 0:
        out = []
        for d in kept:
            z = _pred_cam_z(d)
            if z is not None and z > depth_max_z:
                if stats is not None:
                    stats.dropped_depth_max_z += 1
                dismissed.append((d, f"Too Far (z={z:.2f} > max_z={depth_max_z:.2f})"))
                continue
            out.append(d)
        kept = out

    keep_q = float(getattr(config, "bg_keep_nearest_z_quantile", 0.0) or 0.0)
    if keep_q > 0 and keep_q < 1 and len(kept) > 1:
        zs = []
        idxs = []
        for i, d in enumerate(kept):
            z = _pred_cam_z(d)
            if z is None:
                continue
            zs.append(z)
            idxs.append(i)
        if len(zs) >= 2:
            zs_arr = np.asarray(zs, dtype=np.float64)
            order = np.argsort(zs_arr)  # nearest = smaller z
            k = int(np.ceil(keep_q * len(order)))
            k = max(1, min(k, len(order)))
            keep_indices = set(idxs[int(order[j])] for j in range(k))
            out = []
            for i, d in enumerate(kept):
                if i in keep_indices or _pred_cam_z(d) is None:
                    out.append(d)
                else:
                    if stats is not None:
                        stats.dropped_keep_nearest_z_quantile += 1
                    z = _pred_cam_z(d)
                    dismissed.append((d, f"Depth Quantile Cutoff (z={z:.2f}, keep_q={keep_q:.2f})"))
            kept = out

    # Auto size cutoff
    method = str(getattr(config, "bg_size_auto_method", "none") or "none").lower()
    feature = str(getattr(config, "bg_size_feature", "bbox_height") or "bbox_height")
    if method != "none" and len(kept) > 2:
        sizes = []
        dets = []
        for d in kept:
            s = _bbox_size(d, feature)
            if s is None or s <= 0:
                continue
            sizes.append(float(s))
            dets.append(d)
        if len(sizes) >= 3:
            sizes_arr = np.asarray(sizes, dtype=np.float64)
            log_sizes = np.log(sizes_arr + 1e-6)
            thr_log: Optional[float] = None
            if method == "percentile":
                drop_p = float(getattr(config, "bg_size_percentile", 0.0) or 0.0)
                drop_p = max(0.0, min(drop_p, 1.0))
                if drop_p > 0 and drop_p < 1:
                    thr_log = float(np.quantile(log_sizes, drop_p))
            elif method == "otsu":
                thr_log = _otsu_threshold(log_sizes)
            elif method == "gmm2":
                thr_log = _gmm2_threshold(log_sizes)

            if thr_log is not None and np.isfinite(thr_log):
                thr_raw = float(np.exp(thr_log))
                out = []
                for d in kept:
                    s = _bbox_size(d, feature)
                    if s is None or s <= 0:
                        out.append(d)
                        continue
                    if np.log(float(s) + 1e-6) < thr_log:
                        if stats is not None:
                            stats.dropped_auto_size += 1
                        dismissed.append((d, f"Auto Size Cutoff ({feature}={s:.1f} < thr={thr_raw:.1f}, method={method})"))
                        continue
                    out.append(d)
                kept = out

    # Auto ROI gate (when provided)
    if auto_roi is not None and len(kept) > 0:
        cx, cy, r = auto_roi
        r2 = float(r) * float(r)
        point_kind = str(getattr(config, "bg_auto_roi_point", "bottom_center") or "bottom_center")
        out = []
        for d in kept:
            pt = _bbox_point(d, point_kind)
            if pt is None:
                if stats is not None:
                    stats.dropped_missing_bbox += 1
                dismissed.append((d, "Missing BBox (ROI gate)"))
                continue
            dx = float(pt[0] - cx)
            dy = float(pt[1] - cy)
            dist = float(np.sqrt(dx * dx + dy * dy))
            if dist * dist <= r2:
                out.append(d)
            else:
                if stats is not None:
                    stats.dropped_auto_roi += 1
                dismissed.append(
                    (
                        d,
                        f"Outside ROI (pt=({pt[0]:.1f}, {pt[1]:.1f}), roi=({cx:.1f}, {cy:.1f}), dist={dist:.1f} > r={r:.1f})",
                    )
                )
        kept = out

    if stats is not None:
        stats.detections_kept += len(kept)
    return kept, dismissed


def filter_frames(
    frames: Dict[int, List[Detection]],
    config: TrackingConfig,
    stats: Optional[BackgroundFilterStats] = None,
    auto_roi_by_frame: Optional[Dict[int, Tuple[float, float, float]]] = None,
) -> Tuple[Dict[int, List[Detection]], Dict[int, List[Tuple[Detection, str]]]]:
    """Apply detection-level filtering across all frames.

    Returns:
        (filtered_frames, dismissed_by_frame) where *dismissed_by_frame* maps
        frame_index -> list of (detection, reason_string) tuples.
    """
    dismissed_by_frame: Dict[int, List[Tuple[Detection, str]]] = {}
    if not getattr(config, "bg_filter_enabled", False):
        return frames, dismissed_by_frame
    out: Dict[int, List[Detection]] = {}
    for fi, dets in frames.items():
        roi = auto_roi_by_frame.get(fi) if auto_roi_by_frame is not None else None
        kept, dismissed = filter_detections_for_frame(dets, config, stats=stats, auto_roi=roi)
        out[fi] = kept
        if dismissed:
            dismissed_by_frame[fi] = dismissed
    return out, dismissed_by_frame


def score_tracklets(
    tracklets: List[Tracklet],
    config: TrackingConfig,
) -> List[Tuple[Tracklet, float, Dict[str, float]]]:
    """Return list of (tracklet, score, feature_breakdown)."""
    if not tracklets:
        return []

    feature = str(getattr(config, "bg_size_feature", "bbox_height") or "bbox_height")
    point_kind = str(getattr(config, "bg_auto_roi_point", "bottom_center") or "bottom_center")
    min_frames = int(getattr(config, "bg_min_tracklet_frames_for_scoring", 10) or 0)

    # Precompute per-tracklet size stats and per-frame centers
    all_log_sizes = []
    track_stats: Dict[int, Dict[str, float]] = {}

    frame_points: Dict[int, List[Tuple[float, float]]] = {}
    for t in tracklets:
        sizes = []
        points = []
        for det in t.detections:
            s = _bbox_size(det, feature)
            if s is not None and s > 0:
                sizes.append(float(s))
            pt = _bbox_point(det, point_kind)
            if pt is not None:
                points.append((det.frame_index, float(pt[0]), float(pt[1])))
        if sizes:
            log_m = float(np.log(np.median(sizes) + 1e-6))
            all_log_sizes.append(log_m)
            q75, q25 = np.percentile(sizes, [75, 25])
            iqr = float(q75 - q25)
            med = float(np.median(sizes))
            rel_iqr = float(iqr / (med + 1e-6))
        else:
            log_m = float("nan")
            rel_iqr = float("nan")
            med = float("nan")

        for fi, px, py in points:
            frame_points.setdefault(int(fi), []).append((px, py))

        track_stats[t.track_id] = {
            "log_size_med": log_m,
            "rel_size_iqr": rel_iqr,
            "size_med": med,
            "length": float(len(t.detections)),
        }

    log_sizes_arr = np.asarray([v for v in all_log_sizes if np.isfinite(v)], dtype=np.float64)
    if log_sizes_arr.size >= 2:
        mu = float(np.mean(log_sizes_arr))
        sigma = float(np.std(log_sizes_arr) + 1e-6)
    else:
        mu, sigma = 0.0, 1.0

    # Per-frame robust center + spread (MAD of distances)
    frame_center: Dict[int, Tuple[float, float, float]] = {}
    for fi, pts in frame_points.items():
        if not pts:
            continue
        arr = np.asarray(pts, dtype=np.float64)
        cx = float(np.median(arr[:, 0]))
        cy = float(np.median(arr[:, 1]))
        d = np.sqrt((arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2)
        med_d = float(np.median(d))
        mad_d = float(np.median(np.abs(d - med_d)) + 1e-6)
        spread = float(med_d + (3.0 * mad_d))
        frame_center[fi] = (cx, cy, spread)

    # Weights
    w_len = float(getattr(config, "bg_w_length", 0.45) or 0.0)
    w_size = float(getattr(config, "bg_w_size", 0.25) or 0.0)
    w_stab = float(getattr(config, "bg_w_size_stability", 0.15) or 0.0)
    w_cent = float(getattr(config, "bg_w_centering", 0.15) or 0.0)
    w_sum = w_len + w_size + w_stab + w_cent
    if w_sum <= 1e-9:
        w_len, w_size, w_stab, w_cent = 1.0, 0.0, 0.0, 0.0
        w_sum = 1.0
    w_len /= w_sum
    w_size /= w_sum
    w_stab /= w_sum
    w_cent /= w_sum

    # Length saturation constant
    tau = float(max(30, min_frames * 3, 60))

    scored: List[Tuple[Tracklet, float, Dict[str, float]]] = []
    for t in tracklets:
        st = track_stats.get(t.track_id, {})
        length = float(st.get("length", len(t.detections)))

        # Length term
        len_term = float(1.0 - np.exp(-length / max(tau, 1.0)))

        # Size term (relative within this video)
        log_s = float(st.get("log_size_med", float("nan")))
        if np.isfinite(log_s):
            size_term = _sigmoid((log_s - mu) / sigma)
        else:
            size_term = 0.5

        # Stability term (lower IQR => higher)
        rel_iqr = float(st.get("rel_size_iqr", float("nan")))
        if np.isfinite(rel_iqr):
            stab_term = float(np.exp(-2.0 * max(0.0, rel_iqr)))
        else:
            stab_term = 0.5

        # Centering term
        d_norms = []
        for det in t.detections:
            pt = _bbox_point(det, point_kind)
            if pt is None:
                continue
            fc = frame_center.get(int(det.frame_index))
            if fc is None:
                continue
            cx, cy, spread = fc
            dx = float(pt[0] - cx)
            dy = float(pt[1] - cy)
            d = float(np.sqrt(dx * dx + dy * dy))
            d_norms.append(d / max(spread, 1.0))
        if d_norms:
            d_med = float(np.median(np.asarray(d_norms, dtype=np.float64)))
            cent_term = float(1.0 / (1.0 + d_med))
        else:
            cent_term = 0.5

        raw = (w_len * len_term) + (w_size * size_term) + (w_stab * stab_term) + (w_cent * cent_term)

        # Penalize very short tracklets
        if min_frames > 0 and length < float(min_frames):
            raw *= float(max(0.0, length / float(min_frames)))

        breakdown = {
            "length": round(len_term, 4),
            "size": round(size_term, 4),
            "size_stability": round(stab_term, 4),
            "centering": round(cent_term, 4),
            "score": round(float(raw), 4),
        }
        scored.append((t, float(raw), breakdown))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def filter_tracklets_by_score(
    tracklets: List[Tracklet],
    config: TrackingConfig,
    stats: Optional[BackgroundFilterStats] = None,
) -> Tuple[List[Tracklet], List[Dict[str, Any]]]:
    """
    Filter tracklets by background scoring config.
    Returns (kept_tracklets, per_tracklet_debug_rows).
    """
    if not getattr(config, "bg_filter_enabled", False):
        return tracklets, []
    if not getattr(config, "bg_tracklet_score_enabled", True):
        return tracklets, []

    thr = float(getattr(config, "bg_tracklet_score_threshold", 0.0) or 0.0)
    scored = score_tracklets(tracklets, config)
    rows = [
        {
            "track_id": t.track_id,
            "length": len(t.detections),
            **breakdown,
        }
        for (t, _s, breakdown) in scored
    ]

    if stats is not None:
        stats.tracklets_seen += len(tracklets)

    if thr <= 0.0:
        # Threshold disabled: do not drop, but still return debug rows for metadata.
        if stats is not None:
            stats.tracklets_kept += len(tracklets)
        return tracklets, rows

    kept = [t for (t, s, _b) in scored if s >= thr]

    if not kept and scored:
        # Guardrail: keep best tracklet so output isn't empty.
        kept = [scored[0][0]]
        if stats is not None:
            stats.tracklets_guardrail_kept += 1

    if stats is not None:
        stats.tracklets_kept += len(kept)
        stats.tracklets_dropped_by_score += max(0, len(tracklets) - len(kept))

    return kept, rows


def compute_auto_roi_by_frame(
    tracklets: List[Tracklet],
    config: TrackingConfig,
) -> Dict[int, Tuple[float, float, float]]:
    """
    Build a moving circular ROI from tracklet detections.
    Returns {frame_index: (cx, cy, r)}.
    """
    if not getattr(config, "bg_auto_roi_enabled", False):
        return {}

    point_kind = str(getattr(config, "bg_auto_roi_point", "bottom_center") or "bottom_center")
    window = int(getattr(config, "bg_auto_roi_window_frames", 30) or 30)
    window = max(1, window)
    mad_k = float(getattr(config, "bg_auto_roi_mad_k", 3.0) or 3.0)
    min_r = float(getattr(config, "bg_auto_roi_min_radius_px", 60.0) or 0.0)
    alpha = float(getattr(config, "bg_auto_roi_smoothing_alpha", 0.2) or 0.0)
    alpha = max(0.0, min(alpha, 1.0))

    # Collect points by frame from tracklets
    pts_by_frame: Dict[int, List[Tuple[float, float]]] = {}
    for t in tracklets:
        for det in t.detections:
            pt = _bbox_point(det, point_kind)
            if pt is None:
                continue
            fi = int(det.frame_index)
            pts_by_frame.setdefault(fi, []).append((float(pt[0]), float(pt[1])))

    if not pts_by_frame:
        return {}

    frames_sorted = sorted(pts_by_frame.keys())
    out: Dict[int, Tuple[float, float, float]] = {}

    ema_cx = None
    ema_cy = None
    ema_r = None

    # Sliding window across frame indices (not time in seconds; good enough here)
    for fi in frames_sorted:
        # Gather points in [fi-window+1, fi]
        w_pts = []
        start = fi - window + 1
        for fj in range(start, fi + 1):
            if fj in pts_by_frame:
                w_pts.extend(pts_by_frame[fj])
        if len(w_pts) < 3:
            continue
        arr = np.asarray(w_pts, dtype=np.float64)
        cx = float(np.median(arr[:, 0]))
        cy = float(np.median(arr[:, 1]))
        d = np.sqrt((arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2)
        med_d = float(np.median(d))
        mad_d = float(np.median(np.abs(d - med_d)) + 1e-6)
        r = float(max(min_r, med_d + mad_k * mad_d))

        if ema_cx is None:
            ema_cx, ema_cy, ema_r = cx, cy, r
        else:
            ema_cx = (1.0 - alpha) * float(ema_cx) + alpha * cx
            ema_cy = (1.0 - alpha) * float(ema_cy) + alpha * cy
            ema_r = (1.0 - alpha) * float(ema_r) + alpha * r

        out[fi] = (float(ema_cx), float(ema_cy), float(ema_r))

    return out

