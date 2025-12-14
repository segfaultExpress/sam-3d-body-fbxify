import torch
import numpy as np
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from fbxify.utils import convert_to_blender_coords, get_keypoint, get_profile, add_helper_keypoints
from fbxify.metadata import JOINT_NAMES_TO_INDEX
import json
import os

class PoseEstimator:    
    def __init__(self, checkpoint_path, mhr_path, device=None, fov_name="moge2", fov_path=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SAM 3D Body on {self.device}...")
        
        model, cfg = load_sam_3d_body(
            checkpoint_path=checkpoint_path,
            device=self.device,
            mhr_path=mhr_path
        )
        
        # Initialize FOV estimator if requested
        fov_estimator = None
        if fov_name:
            fov_path = fov_path or os.environ.get("SAM3D_FOV_PATH", "")
            try:
                from tools.build_fov_estimator import FOVEstimator
                fov_estimator = FOVEstimator(name=fov_name, device=self.device, path=fov_path)
                print(f"FOV estimator '{fov_name}' loaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to load FOV estimator '{fov_name}': {e}")
                print("Continuing without FOV estimator (will use default FOV).")
        
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=fov_estimator,
        )
        self.faces = self.estimator.faces

    def populate_joint_mapping(self, joint_to_bone_mapping, joint_rotations, joint_coords, keypoints_3d, keyframe_index):
        # joint_to_bone_mapping is a nested dictionary that contains the bone name, the mapping type, some reference material, and its children
        # populate the joint rotation and coords for each bone

        self._populate_joint_and_children_rec(joint_to_bone_mapping, joint_rotations, joint_coords, keypoints_3d, keyframe_index) # pass in root bone

        return joint_to_bone_mapping # side effect programming but it's okay

    def _populate_joint_and_children_rec(self, bone, joint_rotations, joint_coords, keypoints_3d, keyframe_index):
        # Check if bone is a dictionary first to avoid AttributeError
        if not isinstance(bone, dict) or bone.get("name") is None:
            return # not a valid bone structure

        self._populate_data(bone, joint_rotations, joint_coords, keypoints_3d, keyframe_index)

        for child in bone.get("children", []):
            self._populate_joint_and_children_rec(child, joint_rotations, joint_coords, keypoints_3d, keyframe_index)

    def _populate_data(self, bone, joint_rotations, joint_coords, keypoints_3d, keyframe_index):
        if bone.get("method") is None:
            return # not a valid mapping method
        
        # Initialize data dictionary if it doesn't exist (preserve existing data for multi-frame processing)
        if "data" not in bone or not isinstance(bone["data"], dict):
            bone["data"] = {}

        if bone.get("method") == "direct_rotation":
            self._populate_direct_rotation(bone, joint_rotations, joint_coords, keyframe_index)
        elif bone.get("method") == "keypoint_with_global_rot_roll":
            self._populate_keypoint_with_global_rot_roll(bone, joint_rotations, keypoints_3d, keyframe_index)

    def _populate_direct_rotation(self, bone, joint_rotations, joint_coords, keyframe_index):
        # For direct referencing, populate the rotation and coords from the reference joint

        reference = bone.get("reference")
        if reference is None:
            return # no reference, so no data to populate

        joint_index = JOINT_NAMES_TO_INDEX[reference]

        data = bone["data"]

        if "rotation" not in data or not isinstance(data["rotation"], list):
            data["rotation"] = []
        
        # Ensure list has exactly keyframe_index elements before we add the new one
        # This handles sequential processing (0, 1, 2...) correctly
        while len(data["rotation"]) < keyframe_index:
            data["rotation"].append(None)
        
        # Now append the new value (will be at index keyframe_index)
        data["rotation"].append(joint_rotations[joint_index].tolist())

        if "coords" not in data or not isinstance(data["coords"], list):
            data["coords"] = []
        
        # Ensure list has exactly keyframe_index elements before we add the new one
        while len(data["coords"]) < keyframe_index:
            data["coords"].append(None)
        
        # Now append the new value (will be at index keyframe_index)
        data["coords"].append(joint_coords[joint_index].tolist())

    def _populate_keypoint_with_global_rot_roll(self, bone, joint_rotations, keypoints_3d, keyframe_index):
        # for keypoint and roll referencing, populate dir_vector and roll_vector 

        point_reference = bone.get("point_reference")
        if point_reference is None:
            print(f"  WARNING: [{bone['name']}] has 'keypoint with global rot roll' but no point reference found")
            return # no point reference, so no data to populate

        roll_reference = bone.get("roll_reference")
        if roll_reference is None:
            print(f"  WARNING: [{bone['name']}] has 'keypoint with global rot roll' but no roll reference found")
            return # no roll reference, so no data to populate

        roll_joint_index = JOINT_NAMES_TO_INDEX[roll_reference]

        from_keypoint_data = get_keypoint(keypoints_3d, point_reference[0])

        if from_keypoint_data is None:
            print(f"  WARNING: [{bone['name']}] has 'keypoint with global rot roll' but {point_reference[0]} not found in keypoints_3d")
            return # no from keypoint data, so no data to populate

        to_keypoint_data = get_keypoint(keypoints_3d, point_reference[1])
        if to_keypoint_data is None:
            print(f"  WARNING: [{bone['name']}] has 'keypoint with global rot roll' but {point_reference[1]} not found in keypoints_3d")
            return # no to keypoint data, so no data to populate

        dir_vector = to_keypoint_data - from_keypoint_data
        dir_vector = convert_to_blender_coords(dir_vector)

        data = bone["data"]

        if "dir_vector" not in data or not isinstance(data["dir_vector"], list):
            data["dir_vector"] = []
        
        # Ensure list has exactly keyframe_index elements before we add the new one
        while len(data["dir_vector"]) < keyframe_index:
            data["dir_vector"].append(None)
        
        # Now append the new value (will be at index keyframe_index)
        data["dir_vector"].append(dir_vector.tolist())

        if "roll_vector" not in data or not isinstance(data["roll_vector"], list):
            data["roll_vector"] = []
        
        # Ensure list has exactly keyframe_index elements before we add the new one
        while len(data["roll_vector"]) < keyframe_index:
            data["roll_vector"].append(None)
        
        # Now append the new value (will be at index keyframe_index)
        data["roll_vector"].append(joint_rotations[roll_joint_index].tolist())

    def process_single_frame(self, profile_name, image_path, keyframe_index, joint_to_bone_mapping=None):
        """
        Process a single frame and populate joint_to_bone_mapping for the given keyframe_index.
        
        Args:
            profile_name: Profile name (e.g., "mixamo", "unity", "mhr")
            image_path: Path to the image file
            keyframe_index: Index of this keyframe (0-based)
            joint_to_bone_mapping: Optional existing joint_to_bone_mapping to append to.
                                   If None, loads a fresh one from the profile.
        
        Returns:
            dict with keys: joint_to_bone_mapping, rest_pose, vertices, keypoints_3d
        """
        outputs_raw = self.estimator.process_one_image(image_path)
        outputs = self._pick_largest_person(outputs_raw)
        
        joint_coords = outputs["pred_joint_coords"]
        joint_rotations = outputs["pred_global_rots"]
        vertices = outputs["pred_vertices"]
        keypoints_3d = outputs["pred_keypoints_3d"]

        # Add the helper keypoints to the keypoints_3d array before calculating the joint mapping
        keypoints_3d = add_helper_keypoints(keypoints_3d)

        profile = get_profile(profile_name)

        # Load or use existing joint_to_bone_mapping
        if joint_to_bone_mapping is None:
            joint_mapping_rel_path = profile["joint_mapping_path"]
            joint_mapping_path = os.path.join(os.path.dirname(__file__), joint_mapping_rel_path)
            with open(joint_mapping_path, "r", encoding="utf-8") as f:
                joint_to_bone_mapping = json.load(f)

        armature_rest_pose_rel_path = profile["armature_rest_pose_path"]
        armature_rest_pose_path = os.path.join(os.path.dirname(__file__), armature_rest_pose_rel_path)
        with open(armature_rest_pose_path, "r", encoding="utf-8") as f:
            armature_rest_pose = json.load(f)

        # Populate joint mapping for this keyframe (modifies joint_to_bone_mapping in-place)
        self.populate_joint_mapping(joint_to_bone_mapping, joint_rotations, joint_coords, keypoints_3d, keyframe_index)
        
        vertices = np.array([convert_to_blender_coords(v) for v in vertices])
        
        return {
            "joint_to_bone_mapping": joint_to_bone_mapping,
            "rest_pose": armature_rest_pose,
            "vertices": vertices.tolist(),
            "keypoints_3d": keypoints_3d.tolist(),
        }

    def process_image(self, profile_name, image_path):
        """Process a single image (backward compatibility wrapper)."""
        result = self.process_single_frame(profile_name, image_path, 0, joint_to_bone_mapping=None)
        result["metadata"] = self.create_metadata(profile_name)
        return result

    def create_metadata(self, profile_name, num_keyframes=1):
        return {
            "num_keyframes": num_keyframes,
            "profile_name": profile_name
        }
    
    def _pick_largest_person(self, outputs):
        if isinstance(outputs, dict):
            return outputs
        sizes = []
        for o in outputs:
            verts = o["pred_vertices"]
            sizes.append((np.max(verts[:,0]) - np.min(verts[:,0])) +
                        (np.max(verts[:,1]) - np.min(verts[:,1])))
        return outputs[int(np.argmax(sizes))]