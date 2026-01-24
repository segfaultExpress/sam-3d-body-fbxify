from __future__ import annotations
import numpy as np
import tempfile
import os
import json
import subprocess
import time
import shutil
import re
from tqdm import tqdm

from dataclasses import dataclass
from typing import Dict, List, Optional

from fbxify.metadata import PROFILES, MHR_KEYPOINT_INDEX
from fbxify.i18n import Translator, DEFAULT_LANGUAGE

MHR_EXTENDED_KEYPOINT_INDEX = {
    **MHR_KEYPOINT_INDEX,
    "hips": 70,
    "spine": 71,
    "upper_spine": 72,
    "chest": 73,
    "occipital": 74,
    "head": 75,
    "left_mid_foot": 76,
    "right_mid_foot": 77,
    "left_shoulder_root": 78,
    "right_shoulder_root": 79,
}

def get_profile(profile_name):
    return PROFILES[profile_name]

def to_serializable(obj, _seen=None):
    """Recursively convert numpy types to plain Python for JSON dumping."""
    if _seen is None:
        _seen = set()
    
    # Check for circular references
    obj_id = id(obj)
    if obj_id in _seen:
        return f"<circular reference to {type(obj).__name__}>"
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        _seen.add(obj_id)
        result = {k: to_serializable(v, _seen) for k, v in obj.items()}
        _seen.remove(obj_id)
        return result
    if isinstance(obj, (list, tuple)):
        _seen.add(obj_id)
        result = [to_serializable(v, _seen) for v in obj]
        _seen.remove(obj_id)
        return result
    return obj


def export_to_fbx(metadata, joint_mapping, root_motion, rest_pose, faces, mesh_obj_path=None, lod_fbx_path=None, 
                  progress_callback=None, lang=DEFAULT_LANGUAGE):
    tmp_dir = tempfile.mkdtemp(prefix="sam3d_fbx_")
    
    try:
        metadata_path = os.path.join(tmp_dir, "metadata.json")
        joint_mapping_path = os.path.join(tmp_dir, "armature_joint_mapping.json")
        root_motion_path = os.path.join(tmp_dir, "root_motion.json")
        rest_pose_path = os.path.join(tmp_dir, "armature_rest_pose.json")
        faces_path = os.path.join(tmp_dir, "faces.json")
        script_path = os.path.join(tmp_dir, "blender_script.py")
        fbx_path = os.path.join(tmp_dir, "output.fbx")
        
        with open(metadata_path, "w") as f:
            json.dump({"metadata": metadata}, f)
        with open(joint_mapping_path, "w") as f:
            json.dump({"joint_mapping": joint_mapping}, f)
        with open(root_motion_path, "w") as f:
            json.dump({"root_motion": to_serializable(root_motion)}, f)
        with open(rest_pose_path, "w") as f:
            json.dump({"rest_pose": rest_pose}, f)
        with open(faces_path, "w") as f:
            json.dump({"faces": faces.tolist()}, f)

        # Copy mesh files if provided
        lod_fbx_path_tmp = None
        mesh_obj_path_tmp = None
        
        # Copy LOD FBX if provided (this contains the mesh for MHR profile)
        if lod_fbx_path and os.path.exists(lod_fbx_path) and os.path.getsize(lod_fbx_path) > 0:
            lod_fbx_path_tmp = os.path.join(tmp_dir, "lod.fbx")
            shutil.copyfile(lod_fbx_path, lod_fbx_path_tmp)
        
        # Copy mesh OBJ if provided (optional, used for reskinning)
        if mesh_obj_path and os.path.exists(mesh_obj_path) and os.path.getsize(mesh_obj_path) > 0:
            mesh_obj_path_tmp = os.path.join(tmp_dir, "mesh.obj")
            shutil.copyfile(mesh_obj_path, mesh_obj_path_tmp)

        # Copy the contents of blender_script.py to the temporary directory
        # This allows editing the file directly with IDE support
        script_source = os.path.join(os.path.dirname(__file__), "blender_utils", "build_armature_and_pose.py")
        shutil.copyfile(script_source, script_path)
        
        # Build command line arguments
        cmd_args = [
            "blender", "-b",
            "--python", script_path,
            "--",
            metadata_path, joint_mapping_path, root_motion_path, rest_pose_path, faces_path, fbx_path
        ]
        
        # Add mesh paths if provided (at least one must be provided for mesh inclusion)
        if lod_fbx_path_tmp or mesh_obj_path_tmp:
            cmd_args.extend([lod_fbx_path_tmp or "", mesh_obj_path_tmp or ""])
        else:
            cmd_args.extend(["", ""])  # Empty strings to maintain argument count
        
        # Get num_keyframes for progress bar
        num_keyframes = metadata.get("num_keyframes", 0)
        translator = Translator(lang)
        
        # Run subprocess with stdout capture to parse progress
        process = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=tmp_dir
        )
        
        # Create progress bar
        progress_bar = tqdm(total=num_keyframes, desc=translator.t("progress.applying_poses", frame_num=0, total_frames=num_keyframes), unit="keyframe") if num_keyframes > 0 else None
        
        # Parse output line by line
        progress_pattern = re.compile(r'PROGRESS: (\d+)/(\d+)')
        
        try:
            for line in process.stdout:
                # Parse progress messages (don't print these)
                match = progress_pattern.search(line)
                if match:
                    frame_num = int(match.group(1))
                    total_frames = int(match.group(2))
                    if progress_bar:
                        progress_bar.n = frame_num
                        progress_bar.total = total_frames
                        # Update description with current frame number
                        progress_bar.set_description(translator.t("progress.applying_poses", frame_num=frame_num, total_frames=total_frames))
                        progress_bar.refresh()
                        # Close progress bar immediately when we reach 100%
                        if frame_num >= total_frames:
                            progress_bar.close()
                            progress_bar = None
                    
                    # Update progress if callback is provided
                    # Pass normalized progress (0.0 to 1.0) - caller will handle weighing
                    if progress_callback and total_frames > 0:
                        tqdm_progress = frame_num / total_frames
                        progress_callback(tqdm_progress, translator.t("progress.applying_poses", frame_num=frame_num, total_frames=total_frames))
                    
                    # Skip printing the PROGRESS line
                    continue
                
                # Print all other output (warnings, errors, etc.)
                # Use tqdm.write() to avoid interfering with progress bar if it exists
                if progress_bar:
                    tqdm.write(line.rstrip())
                else:
                    print(line, end='', flush=True)
            
            # Wait for process to complete (should already be done since we read all output)
            return_code = process.wait()
            
            # Ensure progress bar is closed (in case we didn't reach 100% via PROGRESS messages)
            if progress_bar:
                progress_bar.close()
                progress_bar = None
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd_args)
        except Exception as e:
            if progress_bar:
                progress_bar.close()
            raise
        
        # Use profile_name and id from metadata for filename
        profile_name = metadata.get("profile_name", "unknown")
        person_id = metadata.get("id", "unknown")
        timestamp = int(time.time())
        final_path = f"/tmp/{profile_name}_{person_id}_{timestamp:010d}.fbx"
        shutil.copyfile(fbx_path, final_path)
        
        return final_path
    
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except:
            pass


def convert_to_blender_coords(vec):
    """SAM3D座標系 → Blender座標系"""
    x, y, z = vec
    # To match mixamo, I rotate the armature x=90 later
    return np.array([x, -y, -z])


def get_keypoint(joints, name):
    """キーポイント名からジョイント位置を取得"""
    idx = MHR_EXTENDED_KEYPOINT_INDEX.get(name)
    if idx is None:
        return None
    return joints[idx]


def safe_get_keypoint(joints, name):
    """安全にキーポイントを取得"""
    pt = get_keypoint(joints, name)
    if pt is None:
        return None
    if np.any(np.isnan(pt)) or np.any(np.abs(pt) > 100):
        return None
    return pt

def add_helper_keypoints(joints_3d):
    """Helper keypoints that are used for json mapping to refer to points not already included by default in the joints_3d array"""
    # Helper keypoints
    kp = lambda name: get_keypoint(joints_3d, name)
    safe_kp = lambda name: safe_get_keypoint(joints_3d, name)
    
    # Hips - require both hips to be valid
    left_hip = safe_kp("left_hip")
    right_hip = safe_kp("right_hip")
    if left_hip is None or right_hip is None:
        raise ValueError("Missing required keypoints: left_hip or right_hip is None. Frame may have incomplete pose estimation data.")
    hips = (left_hip + right_hip) * 0.5
    
    # Neck - required for body calculations
    neck = safe_kp("neck")
    if neck is None:
        raise ValueError("Missing required keypoint: neck is None. Frame may have incomplete pose estimation data.")
    
    # Body up
    body_up = neck - hips
    torso_len = np.linalg.norm(body_up)
    if torso_len < 1e-6:
        body_up = np.array([0, 1, 0], float)
        torso_len = 1.0
    body_up = body_up / torso_len
    
    # Spine
    spine = hips + body_up * (torso_len * 0.30)
    
    # Upper spine (between spine and chest, used by mixamo)
    left_shoulder = safe_kp("left_shoulder")
    right_shoulder = safe_kp("right_shoulder")
    if left_shoulder is None or right_shoulder is None:
        raise ValueError("Missing required keypoints: left_shoulder or right_shoulder is None. Frame may have incomplete pose estimation data.")
    shoulder_center = (left_shoulder + right_shoulder) * 0.5
    chest_hint = (shoulder_center * 2 + neck) / 3
    upper_spine = _interpolate_curve_smooth(spine, chest_hint, neck, 0.5)
    
    # Chest
    chest = _interpolate_curve_smooth(spine, chest_hint, neck, 2/3)

    # Head
    head = _compute_head_position(joints_3d, body_up, neck)
    
    # Lower Head
    occipital = _compute_occipital(neck, head, kp("nose"), kp("right_ear"), ratio=0.8, down_offset_ratio=0.08)
    
    # Shoulders (already computed above, but keeping for consistency)
    left_shoulder_root = neck * 0.4 + left_shoulder * 0.6
    right_shoulder_root = neck * 0.4 + right_shoulder * 0.6
        
    # The mid point between the heel, big toe and little toe for each foot
    left_heel = safe_kp("left_heel")
    left_big_toe = safe_kp("left_big_toe")
    left_small_toe = safe_kp("left_small_toe")
    if left_heel is None or left_big_toe is None or left_small_toe is None:
        raise ValueError("Missing required keypoints for left foot. Frame may have incomplete pose estimation data.")
    left_mid_foot = (left_heel + left_big_toe + left_small_toe) * 0.33
    
    right_heel = safe_kp("right_heel")
    right_big_toe = safe_kp("right_big_toe")
    right_small_toe = safe_kp("right_small_toe")
    if right_heel is None or right_big_toe is None or right_small_toe is None:
        raise ValueError("Missing required keypoints for right foot. Frame may have incomplete pose estimation data.")
    right_mid_foot = (right_heel + right_big_toe + right_small_toe) * 0.33

    helper_keypoints = {
        "hips": hips,
        "spine": spine,
        "upper_spine": upper_spine,
        "chest": chest,
        "occipital": occipital,
        "head": head,
        "left_mid_foot": left_mid_foot,
        "right_mid_foot": right_mid_foot,
        "left_shoulder_root": left_shoulder_root,
        "right_shoulder_root": right_shoulder_root,
    }

    # Convert helper keypoints to numpy array and concatenate with joints_3d
    helper_array = np.array(list(helper_keypoints.values()))
    joints_3d = np.concatenate([joints_3d, helper_array], axis=0)

    return joints_3d



def _to_vec3(x):
    """Accept list/tuple/np.array; return float64 (3,)."""
    v = np.asarray(x, dtype=np.float64).reshape(3)
    return v

def _normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros(3, dtype=np.float64), 0.0
    return v / n, n

def _compute_occipital(
    neck,
    head,
    nose,
    right_ear,
    ratio=0.80,
    down_offset_ratio=0.08,
):
    """
    Compute a 'skull_base' point (neck↔head connection center) using:
      - a point at t% along the neck→head_center segment, then
      - a downward normal derived from the plane defined by (neck→head_center, nose-right_ear).

    Inputs are 3D points (iterables of length 3).
    Returns:
      skull_base (np.ndarray shape (3,))
      down_dir   (np.ndarray shape (3,))  # unit vector (or zeros if degenerate)

    Notes:
      - 'down_offset_ratio' is relative to head scale: ||head - neck||.
      - If the plane is degenerate (colinear), it falls back to the un-offset point.
    """
    neck = _to_vec3(neck)
    head = _to_vec3(head)
    nose = _to_vec3(nose)
    right_ear = _to_vec3(right_ear)

    # Base point at 80% (or t) along neck→head.
    nh = head - neck
    nh_dir, nh_len = _normalize(nh)
    if nh_len < 1e-6:
        # Head and neck coincide; can't do anything meaningful.
        return neck.copy()

    ratio = float(np.clip(ratio, 0.0, 1.0))
    base = neck + ratio * nh

    # Define plane using:
    #   a = neck→head direction
    #   b = (nose - right_ear) as a face-ish axis
    # Plane normal: n = a × b
    b = nose - right_ear
    n = np.cross(nh_dir, b)
    down_dir, n_len = _normalize(n)

    if n_len < 1e-6:
        # Degenerate: can't compute a stable normal, so return the base point.
        return base

    # Choose sign so "down" points toward the neck (prevents flipping).
    # Use the full neck→head direction as a stable reference (more stable than base→neck).
    # We want down_dir to be generally aligned with the direction toward the neck.
    # Project down_dir onto the direction from head to neck (-nh_dir)
    proj_toward_neck = np.dot(down_dir, -nh_dir)
    
    # Use a threshold to avoid unstable flipping when projection is near zero
    # This prevents rapid sign changes when the cross product direction is nearly
    # perpendicular to the neck→head axis
    threshold = 0.05  # Only flip if clearly misaligned
    if proj_toward_neck < -threshold:
        down_dir = -down_dir
    # If near zero, maintain previous sign (hysteresis) or use a default
    # For simplicity, if very close to zero, don't flip (maintain current orientation)

    # Offset amount (scaled by neck→head length)
    down_offset_ratio = float(max(0.0, down_offset_ratio))
    offset = down_offset_ratio * nh_len

    skull_base = base + down_dir * offset
    return skull_base

def _interpolate_curve(p0, p1, p2, t):
    """3点曲線補間"""
    seg0 = np.linalg.norm(p1 - p0)
    seg1 = np.linalg.norm(p2 - p1)
    total = max(seg0 + seg1, 1e-6)
    s = t * total
    if s <= seg0:
        return (1 - s/seg0) * p0 + (s/seg0) * p1 if seg0 > 1e-6 else p0.copy()
    else:
        s2 = s - seg0
        return (1 - s2/seg1) * p1 + (s2/seg1) * p2 if seg1 > 1e-6 else p1.copy()


def _interpolate_curve_smooth(p0, p1, p2, t):
    """Smooth quadratic Bezier curve interpolation"""
    # Clamp t to [0, 1]
    t = np.clip(t, 0.0, 1.0)
    # Quadratic Bezier: (1-t)^2 * P0 + 2*(1-t)*t * P1 + t^2 * P2
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def _compute_head_position(joints_3d, body_up, neck):
    """頭の位置を計算"""
    safe_kp = lambda name: safe_get_keypoint(joints_3d, name)
    kp = lambda name: get_keypoint(joints_3d, name)
    
    ls = kp("left_shoulder")
    rs = kp("right_shoulder")
    torso_right = rs - ls
    if np.linalg.norm(torso_right) > 1e-6:
        torso_right = torso_right / np.linalg.norm(torso_right)
    else:
        torso_right = np.array([1, 0, 0], float)
    
    torso_forward = np.cross(body_up, torso_right)
    if np.linalg.norm(torso_forward) > 1e-6:
        torso_forward = torso_forward / np.linalg.norm(torso_forward)
    else:
        torso_forward = np.array([0, 0, 1], float)
    
    left_ear = safe_kp("left_ear")
    right_ear = safe_kp("right_ear")
    nose = safe_kp("nose")
    
    if left_ear is not None and right_ear is not None:
        return (left_ear + right_ear) * 0.5 + body_up * 0.06
    elif left_ear is not None:
        return left_ear + torso_right * 0.07 + body_up * 0.06
    elif right_ear is not None:
        return right_ear - torso_right * 0.07 + body_up * 0.06
    elif nose is not None:
        return nose - torso_forward * 0.08 + body_up * 0.06
    else:
        return neck + body_up * 0.15

def _get_finger_positions(joints_3d, side):
    """指のジョイント位置を取得"""
    kp = lambda name: get_keypoint(joints_3d, name)
    prefixes = [f"{side}_thumb", f"{side}_forefinger", f"{side}_middle_finger",
                f"{side}_ring_finger", f"{side}_pinky_finger"]
    fingers = []
    for prefix in prefixes:
        fingers.extend([kp(f"{prefix}_third_joint"), kp(f"{prefix}2"), kp(f"{prefix}3")])
    return fingers


##################################### Extrinsics Utils #####################################

@dataclass
class ExtrinsicsEntry:
    frame_index: Optional[int]
    qvec: np.ndarray  # [qw, qx, qy, qz] (world->camera)
    tvec: np.ndarray  # [tx, ty, tz] (world->camera)


def parse_extrinsics_file(file_path: str) -> List[ExtrinsicsEntry]:
    """
    Parse a COLMAP images.txt-style extrinsics file.
    
    Expected lines:
    IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME
    Followed by a POINTS2D line (which we skip).
    
    Note: IMAGE_ID is treated as a sample index, not an absolute frame index.
    If values are 1-based, they are normalized to 0-based for interpolation.
    """
    entries: List[ExtrinsicsEntry] = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split()
        # Skip POINTS2D lines (usually 2D coordinates in the second line)
        if len(parts) < 10:
            continue

        try:
            frame_index = int(parts[0])
            qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64)
            tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float64)
        except (ValueError, IndexError):
            continue

        entries.append(ExtrinsicsEntry(frame_index=frame_index, qvec=qvec, tvec=tvec))

    return entries


def build_frame_extrinsics(
    frame_count: int,
    sample_rate: int,
    entries: List[ExtrinsicsEntry],
) -> List[Dict[str, np.ndarray]]:
    """
    Build per-frame extrinsics with interpolation.
    
    Returns a list of dicts with:
      - R_cw, T_cw (camera -> world)
      - R_wc, T_wc (world -> camera)
    """
    if frame_count <= 0:
        return []

    if not entries:
        return [_identity_extrinsics() for _ in range(frame_count)]

    if sample_rate is None:
        sample_rate = 0
    # Leave sample_rate at 0 so _build_sample_frames can infer
    # an endpoint-inclusive rate based on entries and frame_count.

    inferred_rate = None
    if sample_rate == 0:
        denom = max(len(entries) - 1, 1)
        inferred_rate = frame_count / denom if frame_count > 0 else 0
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("sample_rate: ", inferred_rate if inferred_rate is not None else sample_rate)
    print("frame_count: ", frame_count)
    print("entries: ", len(entries))
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    sample_frames = _build_sample_frames(entries, sample_rate, frame_count)
    sample_frames, entries = _sort_samples(sample_frames, entries)

    q_cw_list = []
    t_cw_list = []
    for entry in entries:
        q_wc = _normalize_quat(entry.qvec)
        q_cw = _invert_quat(q_wc)
        R_wc = _qvec_to_rotmat(q_wc)
        R_cw = R_wc.T
        t_wc = entry.tvec.reshape(3)
        t_cw = -R_cw @ t_wc
        q_cw_list.append(q_cw)
        t_cw_list.append(t_cw)

    per_frame: List[Dict[str, np.ndarray]] = []
    for frame_idx in range(frame_count):
        if frame_idx <= sample_frames[0]:
            q_cw = q_cw_list[0]
            t_cw = t_cw_list[0]
        elif frame_idx >= sample_frames[-1]:
            q_cw = q_cw_list[-1]
            t_cw = t_cw_list[-1]
        else:
            right = int(np.searchsorted(sample_frames, frame_idx, side="right"))
            left = max(0, right - 1)
            right = min(right, len(sample_frames) - 1)
            left_frame = sample_frames[left]
            right_frame = sample_frames[right]
            if right_frame == left_frame:
                alpha = 0.0
            else:
                alpha = (frame_idx - left_frame) / (right_frame - left_frame)
            q_cw = _quat_slerp(q_cw_list[left], q_cw_list[right], alpha)
            t_cw = _lerp_vec(t_cw_list[left], t_cw_list[right], alpha)

        R_cw = _qvec_to_rotmat(q_cw)
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw

        T_cw = np.eye(4, dtype=np.float64)
        T_cw[:3, :3] = R_cw
        T_cw[:3, 3] = t_cw

        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = R_wc
        T_wc[:3, 3] = t_wc

        per_frame.append(
            {
                "R_cw": R_cw,
                "t_cw": t_cw,
                "R_wc": R_wc,
                "t_wc": t_wc,
                "T_cw": T_cw,
                "T_wc": T_wc,
            }
        )

    # output every T_wc after interpolation as pretty JSON
    output_path = os.path.join(os.getcwd(), "T_wc_fbxify.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([frame["T_wc"].tolist() for frame in per_frame], f, indent=2)

    return per_frame


def _build_sample_frames(entries: List[ExtrinsicsEntry], sample_rate: int, frame_count: int) -> List[int]:
    sample_indices = [e.frame_index for e in entries if e.frame_index is not None]
    if len(sample_indices) != len(entries):
        # Fallback to sequential samples if any index is missing.
        sample_indices = list(range(len(entries)))

    # Normalize 1-based indices to 0-based so entry 1 maps to frame 0.
    if sample_indices and min(sample_indices) >= 1:
        sample_indices = [idx - 1 for idx in sample_indices]

    if sample_rate is None or sample_rate <= 0:
        # Use frame_count/(N-1) so a downsample rate like 20 is preserved.
        denom = max(len(entries) - 1, 1)
        sample_rate = frame_count / denom if frame_count > 0 else 0

    sample_frames = []
    for idx in sample_indices:
        frame = int(round(idx * sample_rate))
        if frame_count > 0:
            frame = max(0, min(frame, frame_count - 1))
        sample_frames.append(frame)
    return sample_frames


def _sort_samples(sample_frames: List[int], entries: List[ExtrinsicsEntry]):
    pairs = sorted(zip(sample_frames, entries), key=lambda x: x[0])
    frames_sorted = [p[0] for p in pairs]
    entries_sorted = [p[1] for p in pairs]
    return frames_sorted, entries_sorted


def _identity_extrinsics() -> Dict[str, np.ndarray]:
    return {
        "R_cw": np.eye(3, dtype=np.float64),
        "T_cw": np.eye(4, dtype=np.float64),
        "R_wc": np.eye(3, dtype=np.float64),
        "T_wc": np.eye(4, dtype=np.float64),
    }


def _is_monotonic(values: List[int]) -> bool:
    return all(a <= b for a, b in zip(values, values[1:]))


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _invert_quat(q: np.ndarray) -> np.ndarray:
    q = _normalize_quat(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _lerp_vec(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        return _normalize_quat(_lerp_vec(q0, q1, t))

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * q0) + (s1 * q1)


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = _normalize_quat(qvec)
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
