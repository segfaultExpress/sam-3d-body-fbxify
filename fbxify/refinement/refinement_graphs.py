import os
import math
import tempfile
from typing import Dict, Any, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - runtime optional
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None


def _ensure_output_dir(output_dir: Optional[str]) -> str:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    return tempfile.mkdtemp(prefix="refinement_graphs_")


def _plot_joint_grid(
    frames: np.ndarray,
    data: np.ndarray,
    title: str,
    output_path: str,
    row_label: str = "Joint",
) -> None:
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Expected [T][J][3] data for {title}, got {data.shape}")

    num_joints = data.shape[1]
    rows = max(1, num_joints)
    cols = 3

    # Increase per-row height so joint/rotation grids are readable.
    fig_height = max(3.0, rows * 0.6)
    fig_width = cols * 3.2
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True)
    if rows == 1:
        axes = np.array([axes])

    axis_labels = ["X", "Y", "Z"]
    for joint_idx in range(rows):
        for axis_idx in range(cols):
            ax = axes[joint_idx, axis_idx]
            series = data[:, joint_idx, axis_idx]
            ax.plot(frames, series, linewidth=0.6)
            if joint_idx == 0:
                ax.set_title(axis_labels[axis_idx], fontsize=8)
            if axis_idx == 0:
                ax.set_ylabel(f"{row_label} {joint_idx}", fontsize=6)
            ax.tick_params(labelsize=6)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def convert_refinement_logs(refinement_logs: Dict[str, Any], output_dir: Optional[str] = None) -> List[str]:
    """
    Convert refinement logs into graph PNGs.
    Returns a list of generated file paths.
    """
    if plt is None:
        raise ImportError(f"matplotlib is required for Graph Refinement: {_MATPLOTLIB_IMPORT_ERROR}")

    if not refinement_logs or "persons" not in refinement_logs:
        return []

    output_dir = _ensure_output_dir(output_dir)
    generated_files: List[str] = []

    persons = refinement_logs.get("persons", {})
    for person_id, person_logs in persons.items():
        frames = np.array(person_logs.get("frames", []), dtype=float)
        if frames.size == 0:
            continue

        # pred_joint_coords
        pred_joint_coords = person_logs.get("pred_joint_coords")
        if pred_joint_coords is not None:
            coords_arr = np.array(pred_joint_coords, dtype=float)
            out_path = os.path.join(output_dir, f"pred_joint_coords_refinement_person_{person_id}.png")
            _plot_joint_grid(frames, coords_arr, f"pred_joint_coords delta (person {person_id})", out_path)
            generated_files.append(out_path)

        # pred_global_rots
        pred_global_rots = person_logs.get("pred_global_rots")
        if pred_global_rots is not None:
            rots_arr = np.array(pred_global_rots, dtype=float)
            out_path = os.path.join(output_dir, f"pred_global_rots_refinement_person_{person_id}.png")
            _plot_joint_grid(frames, rots_arr, f"pred_global_rots delta (person {person_id})", out_path)
            generated_files.append(out_path)

        # pred_cam_t
        pred_cam_t = person_logs.get("pred_cam_t")
        if pred_cam_t is not None:
            cam_arr = np.array(pred_cam_t, dtype=float)
            cam_arr = cam_arr.reshape((-1, 1, 3))
            out_path = os.path.join(output_dir, f"pred_cam_t_refinement_person_{person_id}.png")
            _plot_joint_grid(frames, cam_arr, f"pred_cam_t delta (person {person_id})", out_path, row_label="Axis")
            generated_files.append(out_path)

        # global_rot
        global_rot = person_logs.get("global_rot")
        if global_rot is not None:
            global_arr = np.array(global_rot, dtype=float)
            global_arr = global_arr.reshape((-1, 1, 3))
            out_path = os.path.join(output_dir, f"global_rot_refinement_person_{person_id}.png")
            _plot_joint_grid(frames, global_arr, f"global_rot delta (person {person_id})", out_path, row_label="Axis")
            generated_files.append(out_path)

    return generated_files
