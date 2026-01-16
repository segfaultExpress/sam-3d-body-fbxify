"""
FBX Data Preparation Manager - Transforms estimation results into FBX-ready data.

This module is responsible for:
- Loading estimation results from JSON
- Building joint-to-bone mappings
- Building root motion data
- Managing rest poses and metadata
"""
import json
import os
import numpy as np
from typing import Dict, List, Optional, Any
from fbxify.utils import convert_to_blender_coords, get_keypoint, get_profile, add_helper_keypoints
from fbxify.metadata import JOINT_NAMES_TO_INDEX
from fbxify import VERSION


class FbxDataPrepManager:
    """
    Manages data preparation for FBX export (the transformation part).
    Transforms estimation results into joint mappings, root motions, etc.
    """
    
    armature_rest_poses_cached = {}  # armature rest pose cache
    joint_to_bone_mappings_cached = {}  # joint to bone mapping cache
    
    def __init__(self):
        """Initialize the FBX data preparation manager."""
        pass
    
    @staticmethod
    def validate_estimation_results(estimation_results: Dict[str, Dict[str, Any]], 
                                     use_root_motion: bool = True) -> None:
        """
        Validate that estimation results contain all required keys.
        
        This is a common validation function that can be used before processing
        estimation results to ensure all required data is present.
                
        Args:
            estimation_results: Dictionary in format {frame_X: {person_id: estimation_data}}
            use_root_motion: Whether root motion keys are required
            
        Raises:
            ValueError: If required keys are missing from estimation data
        """
        # Required keys for FBX data prep
        required_keys = ["pred_joint_coords", "pred_global_rots", "pred_keypoints_3d"]
        root_motion_keys = ["global_rot", "pred_cam_t"] if use_root_motion else []
        
        # Find first frame with data (optimized: O(1) instead of O(n*m))
        if not estimation_results:
            raise ValueError("Error: Estimation results is empty. May need to re-run the estimator on this file.")
        
        # Get first frame
        first_frame_key = next(iter(estimation_results))
        frame_data = estimation_results[first_frame_key]
        
        if not frame_data:
            raise ValueError(f"Error: Frame {first_frame_key} has no person data. May need to re-run the estimator on this file.")
        
        # Get first person in first frame
        first_person_key = next(iter(frame_data))
        estimation_data = frame_data[first_person_key]
        
        # Check required keys
        missing_keys = []
        for key in required_keys:
            if key not in estimation_data:
                missing_keys.append(key)
        
        # Check root motion keys if needed
        if use_root_motion:
            for key in root_motion_keys:
                if key not in estimation_data:
                    missing_keys.append(key)
        
        if missing_keys:
            frame_index = int(first_frame_key)
            person_id = str(first_person_key)
            error_msg = (
                f"Error: Missing required key(s) {missing_keys} in frame {frame_index}, person {person_id}. "
                f"This may indicate a version compatibility issue. "
                f"Please check that the estimation JSON file was created with the same version of the software "
                f"(current version: {VERSION}). May need to re-run the estimator on this file."
            )
            raise ValueError(error_msg)
    
    def prepare_from_estimation(self, estimation_results: Dict[str, Dict[str, Any]], 
                                profile_name: str, use_root_motion: bool = True,
                                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare FBX data from estimation results.
        
        Args:
            estimation_results: Dictionary in format {frame_X: {person_id: estimation_data}}
            profile_name: Profile name (e.g., "mixamo", "unity", "mhr")
            use_root_motion: Whether to track root motion
            metadata: Optional metadata dict containing num_people for validation
            
        Returns:
            Dictionary containing:
            - joint_to_bone_mappings: Dict mapping person_id to joint_to_bone_mapping
            - root_motions: Dict mapping person_id to list of root motion data (if use_root_motion)
            
        Raises:
            ValueError: If required keys are missing from estimation data
        """
        # Validate estimation results before processing
        self.validate_estimation_results(estimation_results, use_root_motion)
        
        joint_to_bone_mappings = {}
        root_motions = {} if use_root_motion else None
        
        # Get all unique person IDs across all frames
        all_person_ids = set()
        for frame_data in estimation_results.values():
            for person_id in frame_data.keys():
                all_person_ids.add(person_id)
        
        # Validate num_people from metadata if available
        if metadata is not None and metadata.get("num_people") is not None:
            metadata_num_people = metadata["num_people"]
            actual_num_people = len(all_person_ids)
            if metadata_num_people != actual_num_people:
                print(f"  [WARNING] Metadata num_people ({metadata_num_people}) does not match actual person IDs found ({actual_num_people})")
                print(f"  [INFO] This may indicate frames were skipped or person IDs changed during processing")
            else:
                print(f"  [INFO] Verified num_people from metadata: {metadata_num_people} person(s)")
        
        # Initialize joint mappings and root motions for each person
        for person_id in all_person_ids:
            joint_to_bone_mappings[person_id] = self.get_joint_to_bone_mapping(profile_name, use_cache=False)
            if use_root_motion:
                root_motions[person_id] = []
        
        # Process each frame
        for frame_index_str, frame_data in estimation_results.items():
            frame_index = int(frame_index_str)
            
            # Skip empty frames
            if not frame_data:
                continue
            
            # Process each person in this frame
            for person_id_str, estimation_data in frame_data.items():
                person_id = str(person_id_str)
                
                # Skip if estimation_data is None or empty
                if not estimation_data or not isinstance(estimation_data, dict):
                    print(f"  [WARNING] Frame {frame_index}, person {person_id}: empty or invalid data, skipping")
                    continue
                
                # Skip if required data is None (missing frame)
                if (estimation_data.get("pred_joint_coords") is None or 
                    estimation_data.get("pred_global_rots") is None or 
                    estimation_data.get("pred_keypoints_3d") is None):
                    print(f"  [WARNING] Frame {frame_index}, person {person_id}: missing required keys, skipping")
                    continue
                
                # Convert estimation data back to numpy arrays for processing
                try:
                    joint_coords = np.array(estimation_data["pred_joint_coords"])
                    joint_rotations = np.array(estimation_data["pred_global_rots"])
                    keypoints_3d = np.array(estimation_data["pred_keypoints_3d"])
                except (ValueError, TypeError) as e:
                    print(f"  [WARNING] Frame {frame_index}, person {person_id}: failed to convert to numpy arrays: {e}, skipping")
                    continue
                
                # Validate that arrays are valid (not empty, correct shape, no None values)
                def is_valid_array(arr, name):
                    """Check if numpy array is valid for processing."""
                    if arr is None or arr.size == 0:
                        return False, f"{name} is empty or None"
                    if arr.dtype == object:
                        # Object arrays might contain None - check for them
                        if any(x is None for x in arr.flatten()):
                            return False, f"{name} contains None values"
                    # Check for NaN in numeric arrays
                    if np.issubdtype(arr.dtype, np.number) and np.any(np.isnan(arr)):
                        return False, f"{name} contains NaN values"
                    return True, None
                
                # Validate all arrays
                validation_failed = False
                for arr, name in [(joint_coords, "joint_coords"), 
                                  (joint_rotations, "joint_rotations"), 
                                  (keypoints_3d, "keypoints_3d")]:
                    is_valid, error_msg = is_valid_array(arr, name)
                    if not is_valid:
                        print(f"  [WARNING] Frame {frame_index}, person {person_id}: {error_msg}, skipping")
                        validation_failed = True
                        break
                
                if validation_failed:
                    continue
                
                # Add the helper keypoints to the keypoints_3d array before calculating the joint mapping
                keypoints_3d = add_helper_keypoints(keypoints_3d)
                
                # Get joint mapping for this person
                joint_to_bone_mapping = joint_to_bone_mappings[person_id]
                
                # Populate joint mapping for this keyframe (modifies joint_to_bone_mapping in-place)
                self.populate_joint_mapping(joint_to_bone_mapping, joint_rotations, joint_coords, keypoints_3d, frame_index)
                
                # Add root motion if enabled
                if use_root_motion:
                    # Skip if root motion data is None
                    if (estimation_data.get("global_rot") is not None and 
                        estimation_data.get("pred_cam_t") is not None):
                        root_motion = root_motions[person_id]
                        root_motion.append({
                            "frame_index": frame_index,  # 0-based frame index to match joint_mapping
                            "global_rot": estimation_data["global_rot"],
                            "pred_cam_t": estimation_data["pred_cam_t"],
                        })
        
        return {
            "joint_to_bone_mappings": joint_to_bone_mappings,
            "root_motions": root_motions
        }
    
    def populate_joint_mapping(self, joint_to_bone_mapping, joint_rotations, joint_coords, keypoints_3d, keyframe_index):
        """
        Populate joint rotation and coords for each bone in the joint_to_bone_mapping.
        
        Args:
            joint_to_bone_mapping: Nested dictionary containing bone name, mapping type, reference material, and children
            joint_rotations: Array of joint rotations
            joint_coords: Array of joint coordinates
            keypoints_3d: Array of 3D keypoints
            keyframe_index: Index of this keyframe (0-based)
        """
        self._populate_joint_and_children_rec(joint_to_bone_mapping, joint_rotations, joint_coords, keypoints_3d, keyframe_index)
        return joint_to_bone_mapping  # side effect programming but it's okay

    def _populate_joint_and_children_rec(self, bone, joint_rotations, joint_coords, keypoints_3d, keyframe_index):
        """Recursively populate joint data for bone and its children."""
        # Check if bone is a dictionary first to avoid AttributeError
        if not isinstance(bone, dict) or bone.get("name") is None:
            return  # not a valid bone structure

        self._populate_data(bone, joint_rotations, joint_coords, keypoints_3d, keyframe_index)

        for child in bone.get("children", []):
            self._populate_joint_and_children_rec(child, joint_rotations, joint_coords, keypoints_3d, keyframe_index)

    def _populate_data(self, bone, joint_rotations, joint_coords, keypoints_3d, keyframe_index):
        """Populate data for a single bone based on its method."""
        if bone.get("method") is None:
            return  # not a valid mapping method
        
        # Initialize data dictionary if it doesn't exist (preserve existing data for multi-frame processing)
        if "data" not in bone or not isinstance(bone["data"], dict):
            bone["data"] = {}

        if bone.get("method") == "direct_rotation":
            self._populate_direct_rotation(bone, joint_rotations, joint_coords, keyframe_index)
        elif bone.get("method") == "keypoint_with_global_rot_roll":
            self._populate_keypoint_with_global_rot_roll(bone, joint_rotations, keypoints_3d, keyframe_index)

    def _populate_direct_rotation(self, bone, joint_rotations, joint_coords, keyframe_index):
        """For direct referencing, populate the rotation and coords from the reference joint."""
        reference = bone.get("reference")
        if reference is None:
            return  # no reference, so no data to populate

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
        """For keypoint and roll referencing, populate dir_vector and roll_vector."""
        point_reference = bone.get("point_reference")
        if point_reference is None:
            print(f"  WARNING: [{bone['name']}] has 'keypoint with global rot roll' but no point reference found")
            return  # no point reference, so no data to populate

        roll_reference = bone.get("roll_reference")
        if roll_reference is None:
            print(f"  WARNING: [{bone['name']}] has 'keypoint with global rot roll' but no roll reference found")
            return  # no roll reference, so no data to populate

        roll_joint_index = JOINT_NAMES_TO_INDEX[roll_reference]

        from_keypoint_data = get_keypoint(keypoints_3d, point_reference[0])

        if from_keypoint_data is None:
            print(f"  WARNING: [{bone['name']}] has 'keypoint with global rot roll' but {point_reference[0]} not found in keypoints_3d")
            return  # no from keypoint data, so no data to populate

        to_keypoint_data = get_keypoint(keypoints_3d, point_reference[1])
        if to_keypoint_data is None:
            print(f"  WARNING: [{bone['name']}] has 'keypoint with global rot roll' but {point_reference[1]} not found in keypoints_3d")
            return  # no to keypoint data, so no data to populate

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

    def get_joint_to_bone_mapping(self, profile_name, use_cache=True):
        """
        Get joint-to-bone mapping structure for a profile.
        
        Args:
            profile_name: Profile name (e.g., "mixamo", "unity", "mhr")
            use_cache: Whether to use cached mapping
            
        Returns:
            Joint-to-bone mapping dictionary
        """
        if use_cache and profile_name in self.joint_to_bone_mappings_cached:
            return self.joint_to_bone_mappings_cached[profile_name]

        profile = get_profile(profile_name)
        joint_mapping_rel_path = profile["joint_mapping_path"]
        joint_mapping_path = os.path.join(os.path.dirname(__file__), joint_mapping_rel_path)
        with open(joint_mapping_path, "r", encoding="utf-8") as f:
            joint_to_bone_mapping = json.load(f)

        if self.joint_to_bone_mappings_cached is None or self.joint_to_bone_mappings_cached == {}:
            self.joint_to_bone_mappings_cached = {}
            self.joint_to_bone_mappings_cached[profile_name] = joint_to_bone_mapping

        return joint_to_bone_mapping

    def get_armature_rest_pose(self, profile_name, use_cache=True):
        """
        Get armature rest pose for a profile.
        
        Args:
            profile_name: Profile name (e.g., "mixamo", "unity", "mhr")
            use_cache: Whether to use cached rest pose
            
        Returns:
            Armature rest pose dictionary
        """
        if use_cache and profile_name in self.armature_rest_poses_cached:
            return self.armature_rest_poses_cached[profile_name]

        profile = get_profile(profile_name)
        armature_rest_pose_rel_path = profile["armature_rest_pose_path"]
        armature_rest_pose_path = os.path.join(os.path.dirname(__file__), armature_rest_pose_rel_path)
        with open(armature_rest_pose_path, "r", encoding="utf-8") as f:
            armature_rest_pose = json.load(f)

        if self.armature_rest_poses_cached is None or self.armature_rest_poses_cached == {}:
            self.armature_rest_poses_cached = {}
            self.armature_rest_poses_cached[profile_name] = armature_rest_pose

        return armature_rest_pose

    def create_metadata(self, profile_name, id, num_keyframes=1, fps=30.0):
        """
        Create metadata for FBX export.
        
        Args:
            profile_name: Profile name
            id: Person ID
            num_keyframes: Number of keyframes
            fps: Frames per second
            
        Returns:
            Metadata dictionary
        """
        return {
            "num_keyframes": num_keyframes,
            "id": id,
            "profile_name": profile_name,
            "fps": fps
        }

