"""Compare cli_bench_quantized outputs against a baseline and render visual diffs.

For every configuration folder under --bench_dir (each with results.json + poses_{a,b}.npz)
this computes, per pass and per frame, the deviation from the baseline configuration:

  * mesh:      per-vertex displacement (mm) mean / p95 / max
  * keypoints: 3D (mm) and 2D (px) error grouped by arms / legs / hands / head+torso
  * joints:    geodesic angle (deg) between predicted global joint rotations
  * camera:    translation delta (mm)

and renders panels for a few frames: input + 2D skeleton overlay (baseline green, candidate
red, arms/legs thick), baseline mesh, candidate mesh, and a per-vertex displacement heatmap
(front + side) so limb differences are easy to spot.

  python -m fbxify.cli_compare_quantized --bench_dir workspace/quant_bench --baseline fp32-normal
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _mhr_names() -> List[str]:
    """Keypoint names without importing the sam_3d_body package __init__ (needs torchvision)."""
    try:
        from sam_3d_body.metadata.mhr70 import mhr_names

        return list(mhr_names)
    except Exception:
        import importlib.util

        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sam_3d_body", "metadata", "mhr70.py")
        spec = importlib.util.spec_from_file_location("mhr70_names", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return list(mod.mhr_names)


mhr_names = _mhr_names()

GROUP_RULES = {
    "arms": ("shoulder", "elbow", "wrist", "olecranon", "cubital", "acromion"),
    "legs": ("hip", "knee", "ankle", "heel", "toe"),
    "hands": ("thumb", "index", "middle", "ring", "pinky"),
    "head_torso": ("nose", "eye", "ear", "neck"),
}

# (name_a, name_b, limb) bones drawn on the 2D overlay; limb bones are drawn thicker.
BONES = [
    ("left-shoulder", "right-shoulder", False),
    ("left-hip", "right-hip", False),
    ("left-shoulder", "left-hip", False),
    ("right-shoulder", "right-hip", False),
    ("neck", "nose", False),
    ("left-shoulder", "left-elbow", True),
    ("left-elbow", "left-wrist", True),
    ("right-shoulder", "right-elbow", True),
    ("right-elbow", "right-wrist", True),
    ("left-hip", "left-knee", True),
    ("left-knee", "left-ankle", True),
    ("left-ankle", "left-heel", True),
    ("left-ankle", "left-big-toe-tip", True),
    ("right-hip", "right-knee", True),
    ("right-knee", "right-ankle", True),
    ("right-ankle", "right-heel", True),
    ("right-ankle", "right-big-toe-tip", True),
]

BASELINE_COLOR = (0.65, 0.74, 0.86)   # light blue (RGB 0..1)
CANDIDATE_COLOR = (0.98, 0.62, 0.25)  # orange
HEAT_MAX_MM = 40.0


def parse_args():
    p = argparse.ArgumentParser(description="Compare quantized/fast SAM 3D Body runs against a baseline")
    p.add_argument("--bench_dir", default="workspace/quant_bench")
    p.add_argument("--baseline", default="fp32-normal")
    p.add_argument("--configs", nargs="*", default=None, help="Config folder names (default: all with results.json)")
    p.add_argument("--render_frames", type=int, default=4, help="Frames per pass to render")
    p.add_argument("--tile_height", type=int, default=420)
    p.add_argument("--heat_max_mm", type=float, default=HEAT_MAX_MM)
    p.add_argument("--faces", default="", help="Optional faces .npy (default: checkpoint MHR faces via LOD cache or embedl sam3dbody_faces.npy)")
    p.add_argument("--no_render", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------------- metrics
def _groups() -> Dict[str, List[int]]:
    out = {k: [] for k in GROUP_RULES}
    for i, name in enumerate(mhr_names):
        for g, keys in GROUP_RULES.items():
            if any(k in name for k in keys):
                out[g].append(i)
                break
    return out


GROUPS = _groups()
NAME_TO_IDX = {n: i for i, n in enumerate(mhr_names)}


def _load_poses(cfg_dir: str, pas: str) -> Optional[Dict[str, np.ndarray]]:
    path = os.path.join(cfg_dir, f"poses_{pas}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _align(base: Dict[str, np.ndarray], cand: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    bi = {int(s): j for j, s in enumerate(base["source_indices"])}
    ci = {int(s): j for j, s in enumerate(cand["source_indices"])}
    common = sorted(set(bi) & set(ci))
    return np.array([bi[s] for s in common], dtype=int), np.array([ci[s] for s in common], dtype=int)


def _rot_angle_deg(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Geodesic angle between batches of rotation matrices (..., 3, 3)."""
    rel = np.matmul(r1, np.swapaxes(r2, -1, -2))
    tr = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _summ(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return {}
    return {
        "mean": round(float(x.mean()), 3),
        "median": round(float(np.median(x)), 3),
        "p95": round(float(np.percentile(x, 95)), 3),
        "max": round(float(x.max()), 3),
    }


# A frame counts as a "target switch" (the candidate reconstructed a different person /
# a grossly different placement) when the camera translation or the mean vertex
# displacement jumps far beyond anything quantization noise can produce.
SWITCH_CAM_T_MM = 300.0
SWITCH_VERTEX_MM = 150.0


def _metrics(base, cand, b_idx, c_idx) -> Dict:
    out: Dict = {"n_frames": int(len(b_idx)), "per_frame": {}}
    if len(b_idx) == 0:
        return out
    dv = np.linalg.norm(base["pred_vertices"][b_idx] - cand["pred_vertices"][c_idx], axis=-1) * 1000.0  # (N, V)
    out["vertex_mm"] = _summ(dv)
    out["per_frame"]["vertex_mean_mm"] = [round(float(v), 2) for v in dv.mean(axis=1)]
    out["per_frame"]["vertex_max_mm"] = [round(float(v), 2) for v in dv.max(axis=1)]

    d3 = np.linalg.norm(base["pred_keypoints_3d"][b_idx] - cand["pred_keypoints_3d"][c_idx], axis=-1) * 1000.0
    d2 = np.linalg.norm(base["pred_keypoints_2d"][b_idx] - cand["pred_keypoints_2d"][c_idx], axis=-1)
    out["keypoints_3d_mm"] = _summ(d3)
    out["keypoints_2d_px"] = _summ(d2)
    out["groups"] = {}
    for g, idx in GROUPS.items():
        if not idx:
            continue
        out["groups"][g] = {"kp3d_mm": _summ(d3[:, idx]), "kp2d_px": _summ(d2[:, idx])}
        out["per_frame"][f"{g}_mm"] = [round(float(v), 2) for v in d3[:, idx].mean(axis=1)]

    if base["pred_global_rots"].ndim == 4 and base["pred_global_rots"].shape[-2:] == (3, 3):
        ang = _rot_angle_deg(base["pred_global_rots"][b_idx], cand["pred_global_rots"][c_idx])  # (N, J)
        out["joint_rotation_deg"] = _summ(ang)
        out["per_frame"]["joint_rot_mean_deg"] = [round(float(v), 3) for v in ang.mean(axis=1)]
    if base["global_rot"].shape[-2:] == (3, 3):
        out["global_rot_deg"] = _summ(_rot_angle_deg(base["global_rot"][b_idx], cand["global_rot"][c_idx]))
    cam = np.linalg.norm(base["pred_cam_t"][b_idx] - cand["pred_cam_t"][c_idx], axis=-1) * 1000.0
    out["cam_t_mm"] = _summ(cam)
    out["per_frame"]["cam_t_mm"] = [round(float(v), 1) for v in cam]
    return out


def compare_pass(base: Dict[str, np.ndarray], cand: Dict[str, np.ndarray]) -> Dict:
    b_idx, c_idx = _align(base, cand)
    if len(b_idx) == 0:
        return {"n_frames": 0}
    out = _metrics(base, cand, b_idx, c_idx)
    cam = np.asarray(out["per_frame"]["cam_t_mm"])
    vm = np.asarray(out["per_frame"]["vertex_mean_mm"])
    switched = (cam > SWITCH_CAM_T_MM) | (vm > SWITCH_VERTEX_MM)
    out["target_switch_frames"] = [int(base["source_indices"][b_idx[i]]) for i in np.where(switched)[0]]
    out["n_target_switch"] = int(switched.sum())
    # Pose precision on the frames where both runs reconstructed the same person.
    keep = ~switched
    stable = _metrics(base, cand, b_idx[keep], c_idx[keep])
    stable.pop("per_frame", None)
    out["stable"] = stable
    return out


# ----------------------------------------------------------------------------- rendering
def _faces_from(args, bench_dir: str) -> Optional[np.ndarray]:
    candidates = []
    if args.faces:
        candidates.append(args.faces)
    cache = os.environ.get("CACHE_DIR", "/fbxify/cache").rstrip("/")
    ckpt = os.environ.get("CHECKPOINTS_DIR", "/fbxify/checkpoints").rstrip("/")
    candidates += [
        os.path.join(bench_dir, "faces.npy"),
        os.path.join(os.environ.get("FBXIFY_QUANTIZED_DIR", os.path.join(ckpt, "embedl-sam-3d-body")), "sam3dbody_faces.npy"),
        os.path.join(cache, "lod_faces", "lod1_faces.npy"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            faces = np.load(c)
            print(f"faces: {c} {faces.shape}", flush=True)
            return faces.astype(np.int64)
    print("faces: none found; mesh renders disabled (pass --faces path/to/faces.npy)", flush=True)
    return None


def _colormap(values01: np.ndarray) -> np.ndarray:
    """(N,) in 0..1 -> (N,4) uint8 RGBA, blue -> green -> yellow -> red."""
    try:
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap("turbo")
        rgba = cmap(np.clip(values01, 0, 1))
        return (rgba * 255).astype(np.uint8)
    except Exception:
        v = np.clip(values01, 0, 1)
        r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
        return (np.stack([r, g, b, np.ones_like(v)], -1) * 255).astype(np.uint8)


def render_mesh(image_bgr: np.ndarray, vertices: np.ndarray, faces: np.ndarray, cam_t: np.ndarray,
                focal: float, color=None, vertex_colors=None, side_view=False) -> np.ndarray:
    import pyrender
    import trimesh

    h, w = image_bgr.shape[:2]
    img = image_bgr.astype(np.float32) / 255.0
    mesh = trimesh.Trimesh(vertices.copy(), faces.copy(), process=False,
                           vertex_colors=vertex_colors if vertex_colors is not None else None)
    if side_view:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]))
    if vertex_colors is None:
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode="OPAQUE",
                                                      baseColorFactor=(*color, 1.0))
        pmesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    else:
        pmesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=(0.35, 0.35, 0.35))
    scene.add(pmesh, "mesh")
    cam_pose = np.eye(4)
    t = cam_t.copy()
    t[0] *= -1.0
    cam_pose[:3, 3] = t
    scene.add(pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=w / 2.0, cy=h / 2.0, zfar=1e12), pose=cam_pose)
    from sam_3d_body.visualization.renderer import create_raymond_lights

    for node in create_raymond_lights():
        scene.add_node(node)
    r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    rgba, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()
    rgba = rgba.astype(np.float32) / 255.0
    rgb = rgba[:, :, :3][:, :, ::-1]  # pyrender returns RGB; convert to BGR
    alpha = rgba[:, :, 3:4]
    out = rgb * alpha + img * (1 - alpha)
    return (out * 255).astype(np.uint8)


def _crop_box(bbox: np.ndarray, w: int, h: int, pad: float = 0.25) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    size = max(x2 - x1, y2 - y1) * (1 + pad)
    x1, x2 = int(max(0, cx - size / 2)), int(min(w, cx + size / 2))
    y1, y2 = int(max(0, cy - size / 2)), int(min(h, cy + size / 2))
    return x1, y1, x2, y2


def _draw_skeleton(img: np.ndarray, kp2d: np.ndarray, color_bgr, limb_thickness=5, torso_thickness=2):
    for a, b, limb in BONES:
        ia, ib = NAME_TO_IDX.get(a), NAME_TO_IDX.get(b)
        if ia is None or ib is None:
            continue
        pa, pb = kp2d[ia], kp2d[ib]
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color_bgr,
                 limb_thickness if limb else torso_thickness, cv2.LINE_AA)
    for name in ("left-wrist", "right-wrist", "left-elbow", "right-elbow", "left-knee", "right-knee",
                 "left-ankle", "right-ankle"):
        p = kp2d[NAME_TO_IDX[name]]
        cv2.circle(img, (int(p[0]), int(p[1])), 5, color_bgr, -1, cv2.LINE_AA)
    return img


def _tile(img: np.ndarray, crop, height: int, label: str, sub: str = "") -> np.ndarray:
    x1, y1, x2, y2 = crop
    t = img[y1:y2, x1:x2]
    scale = height / max(1, t.shape[0])
    t = cv2.resize(t, (max(1, int(t.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (t.shape[1], 44 if sub else 26), (20, 20, 20), -1)
    cv2.putText(t, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(t, sub, (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1, cv2.LINE_AA)
    return t


def _legend_bar(height: int, vmax: float) -> np.ndarray:
    bar = np.zeros((height, 56, 3), dtype=np.uint8) + 255
    grad = _colormap(np.linspace(1, 0, height - 40))[:, :3][:, ::-1]
    bar[20:-20, 8:28] = grad[:, None, :]
    cv2.putText(bar, f"{vmax:.0f}", (30, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, "0", (30, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, "mm", (28, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return bar


def render_frame_panel(frame_bgr: np.ndarray, base: Dict[str, np.ndarray], cand: Dict[str, np.ndarray],
                       bj: int, cj: int, faces: Optional[np.ndarray], cand_name: str, base_name: str,
                       tile_h: int, heat_max_mm: float) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    crop = _crop_box(base["bbox"][bj], w, h)
    bv, cv_ = base["pred_vertices"][bj], cand["pred_vertices"][cj]
    dv_mm = np.linalg.norm(bv - cv_, axis=-1) * 1000.0
    arms = np.linalg.norm(base["pred_keypoints_3d"][bj][GROUPS["arms"]] - cand["pred_keypoints_3d"][cj][GROUPS["arms"]], axis=-1).mean() * 1000
    legs = np.linalg.norm(base["pred_keypoints_3d"][bj][GROUPS["legs"]] - cand["pred_keypoints_3d"][cj][GROUPS["legs"]], axis=-1).mean() * 1000

    tiles = []
    sk = frame_bgr.copy()
    _draw_skeleton(sk, base["pred_keypoints_2d"][bj], (0, 200, 0))
    _draw_skeleton(sk, cand["pred_keypoints_2d"][cj], (0, 0, 255))
    tiles.append(_tile(sk, crop, tile_h, "2D skeleton", f"green={base_name}  red={cand_name}"))

    if faces is not None:
        focal_b, focal_c = float(base["focal_length"][bj]), float(cand["focal_length"][cj])
        tiles.append(_tile(render_mesh(frame_bgr, bv, faces, base["pred_cam_t"][bj], focal_b, color=BASELINE_COLOR),
                           crop, tile_h, base_name, "baseline mesh"))
        tiles.append(_tile(render_mesh(frame_bgr, cv_, faces, cand["pred_cam_t"][cj], focal_c, color=CANDIDATE_COLOR),
                           crop, tile_h, cand_name, f"arms {arms:.1f}mm  legs {legs:.1f}mm"))
        colors = _colormap(dv_mm / heat_max_mm)
        white = np.full_like(frame_bgr, 255)
        heat_front = render_mesh(white, cv_, faces, cand["pred_cam_t"][cj], focal_c, vertex_colors=colors)
        tiles.append(_tile(heat_front, crop, tile_h, "vertex displacement",
                           f"mean {dv_mm.mean():.1f}  max {dv_mm.max():.1f} mm"))
        # Side view: re-centre so the rotated mesh stays in frame.
        side = render_mesh(white, cv_ - cv_.mean(0, keepdims=True), faces,
                           np.array([0.0, 0.0, float(cand["pred_cam_t"][cj][2])]), focal_c,
                           vertex_colors=colors, side_view=True)
        cx, cy = w // 2, h // 2
        half = (crop[3] - crop[1]) // 2
        side_crop = (max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half))
        tiles.append(_tile(side, side_crop, tile_h, "side view", "same colour scale"))
        tiles.append(_legend_bar(tile_h, heat_max_mm))
    return np.concatenate(tiles, axis=1)


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    bench = args.bench_dir
    names = args.configs or sorted(d for d in os.listdir(bench) if os.path.exists(os.path.join(bench, d, "results.json")))
    if args.baseline not in names:
        print(f"baseline {args.baseline!r} not found in {bench}; have {names}")
        return 1
    out_dir = os.path.join(bench, "compare")
    os.makedirs(out_dir, exist_ok=True)
    faces = None if args.no_render else _faces_from(args, bench)

    summary = {"baseline": args.baseline, "groups": {g: [mhr_names[i] for i in idx] for g, idx in GROUPS.items()},
               "heat_max_mm": args.heat_max_mm, "configs": {}, "images": {}}
    base_dir = os.path.join(bench, args.baseline)
    base_poses = {p: _load_poses(base_dir, p) for p in ("a", "b")}

    for name in names:
        cfg_dir = os.path.join(bench, name)
        with open(os.path.join(cfg_dir, "results.json"), encoding="utf-8") as f:
            results = json.load(f)
        entry = {"results": results, "quality": {}}
        cand_poses = {p: _load_poses(cfg_dir, p) for p in ("a", "b")}
        for pas in ("a", "b"):
            if base_poses[pas] is None or cand_poses[pas] is None:
                continue
            entry["quality"][pas] = compare_pass(base_poses[pas], cand_poses[pas])
        # Pooled over both passes: "all" includes target switches, "stable" excludes them.
        pooled = {"all": {}, "stable": {}, "n_target_switch": 0, "n_frames": 0}
        for p in ("a", "b"):
            q = entry["quality"].get(p, {})
            pooled["n_target_switch"] += int(q.get("n_target_switch", 0))
            pooled["n_frames"] += int(q.get("n_frames", 0))
        for scope in ("all", "stable"):
            def _q(p):
                q = entry["quality"].get(p, {})
                return q if scope == "all" else q.get("stable", {})
            for key in ("vertex_mm", "keypoints_3d_mm", "keypoints_2d_px", "joint_rotation_deg", "cam_t_mm"):
                vals = [_q(p)[key]["mean"] for p in ("a", "b") if key in _q(p)]
                if vals:
                    pooled[scope][key + "_mean"] = round(float(np.mean(vals)), 3)
            for g in GROUPS:
                vals = [_q(p)["groups"][g]["kp3d_mm"]["mean"] for p in ("a", "b")
                        if "groups" in _q(p) and g in _q(p)["groups"]]
                if vals:
                    pooled[scope][f"{g}_kp3d_mm_mean"] = round(float(np.mean(vals)), 3)
        entry["quality"]["pooled"] = pooled
        summary["configs"][name] = entry
        print(f"{name:14s} vs {args.baseline}: {pooled}", flush=True)

        if faces is None or name == args.baseline:
            continue
        summary["images"][name] = {}
        for pas in ("a", "b"):
            b, c = base_poses[pas], cand_poses[pas]
            if b is None or c is None:
                continue
            b_idx, c_idx = _align(b, c)
            if len(b_idx) == 0:
                continue
            picks = sorted(set(int(round(x)) for x in np.linspace(0, len(b_idx) - 1, min(args.render_frames, len(b_idx)))))
            cfg_out = os.path.join(out_dir, name)
            os.makedirs(cfg_out, exist_ok=True)
            paths = []
            for k in picks:
                bj, cj = int(b_idx[k]), int(c_idx[k])
                frame = cv2.imread(os.path.join(base_dir, str(b["frame_files"][bj])))
                if frame is None:
                    continue
                panel = render_frame_panel(frame, b, c, bj, cj, faces, name, args.baseline, args.tile_height, args.heat_max_mm)
                p = os.path.join(cfg_out, f"pass_{pas}_frame_{k:02d}_src{int(b['source_indices'][bj]):04d}.png")
                cv2.imwrite(p, panel)
                paths.append(os.path.relpath(p, bench))
            summary["images"][name][pas] = paths
            print(f"  rendered {len(paths)} panels for pass {pas}", flush=True)

    # Cross-config grids: same frame, one row per candidate.
    if faces is not None:
        for pas in ("a", "b"):
            rows_by_frame: Dict[str, List[np.ndarray]] = {}
            for name in names:
                for p in summary["images"].get(name, {}).get(pas, []):
                    key = p.split("_frame_")[1]
                    img = cv2.imread(os.path.join(bench, p))
                    if img is not None:
                        rows_by_frame.setdefault(key, []).append(img)
            for key, rows in rows_by_frame.items():
                wmax = max(r.shape[1] for r in rows)
                rows = [cv2.copyMakeBorder(r, 0, 4, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
                grid_path = os.path.join(out_dir, f"grid_pass_{pas}_frame_{key}")
                cv2.imwrite(grid_path, np.concatenate(rows, axis=0))
                summary["images"].setdefault("_grids", []).append(os.path.relpath(grid_path, bench))

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {os.path.join(out_dir, 'summary.json')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
