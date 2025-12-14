import numpy as np
import tempfile
import os
import json
import subprocess
import time
import shutil

from fbxify.metadata import PROFILES, MHR_KEYPOINT_INDEX

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

def export_to_fbx(metadata, joint_mapping, rest_pose, faces):
    tmp_dir = tempfile.mkdtemp(prefix="sam3d_fbx_")
    
    try:
        joint_mapping_path = os.path.join(tmp_dir, "armature_joint_mapping.json")
        rest_pose_path = os.path.join(tmp_dir, "armature_rest_pose.json")
        faces_path = os.path.join(tmp_dir, "faces.json")
        script_path = os.path.join(tmp_dir, "blender_script.py")
        fbx_path = os.path.join(tmp_dir, "output.fbx")
        metadata_path = os.path.join(tmp_dir, "metadata.json")
        
        with open(metadata_path, "w") as f:
            json.dump({"metadata": metadata}, f)
        with open(joint_mapping_path, "w") as f:
            json.dump({"joint_mapping": joint_mapping}, f)
        with open(rest_pose_path, "w") as f:
            json.dump({"rest_pose": rest_pose}, f)
        with open(faces_path, "w") as f:
            json.dump({"faces": faces.tolist()}, f)

        # Copy the contents of blender_script.py to the temporary directory
        # This allows editing the file directly with IDE support
        script_source = os.path.join(os.path.dirname(__file__), "blender_utils", "build_armature_and_pose.py")
        shutil.copyfile(script_source, script_path)
        
        subprocess.run([
            "blender", "-b",
            "--python", script_path,
            "--",
            metadata_path, joint_mapping_path, rest_pose_path, faces_path, fbx_path
        ], check=True, cwd=tmp_dir)
        
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
    
    # Hips
    left_hip = kp("left_hip")
    right_hip = kp("right_hip")
    hips = (left_hip + right_hip) * 0.5
    neck = kp("neck")
    
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
    left_shoulder = kp("left_shoulder")
    right_shoulder = kp("right_shoulder")
    shoulder_center = (left_shoulder + right_shoulder) * 0.5
    chest_hint = (shoulder_center * 2 + neck) / 3
    upper_spine = _interpolate_curve_smooth(spine, chest_hint, neck, 0.5)
    
    # Chest
    chest = _interpolate_curve_smooth(spine, chest_hint, neck, 2/3)
    
    # Lower Head
    left_acromion = kp("left_acromion")
    right_acromion = kp("right_acromion")
    occipital = (left_acromion + right_acromion) * 0.5

    # Head
    head = _compute_head_position(joints_3d, body_up, neck)
    
    # Shoulders
    left_shoulder = kp("left_shoulder")
    right_shoulder = kp("right_shoulder")
    left_shoulder_root = neck * 0.4 + left_shoulder * 0.6
    right_shoulder_root = neck * 0.4 + right_shoulder * 0.6
        
    # The mid point between the heel, big toe and little toe for each foot
    left_mid_foot = (kp("left_heel") + kp("left_big_toe") + kp("left_small_toe")) * 0.33
    right_mid_foot = (kp("right_heel") + kp("right_big_toe") + kp("right_small_toe")) * 0.33

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
