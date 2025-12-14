import torch
import numpy as np
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from fbxify.utils import convert_to_blender_coords, get_keypoint, get_profile, add_helper_keypoints
from fbxify.metadata import JOINT_NAMES_TO_INDEX
import json
import os

class PoseEstimator:

    armature_rest_poses_cached = {} # armature rest pose cache to avoid loading from json per frame or per person
    joint_to_bone_mappings_cached = {} # joint to bone mapping cache to avoid loading from json per frame or per person

    def __init__(self, checkpoint_path, mhr_path, device=None, detector_name=None, detector_path=None, fov_name="moge2", fov_path=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SAM 3D Body on {self.device}...")
        
        model, cfg = load_sam_3d_body(
            checkpoint_path=checkpoint_path,
            device=self.device,
            mhr_path=mhr_path
        )
        
        # Initialize human detector if requested
        human_detector = None
        if detector_name:
            detector_path = detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
            try:
                from tools.build_detector import HumanDetector
                human_detector = HumanDetector(name=detector_name, device=self.device, path=detector_path)
                print(f"Human detector '{detector_name}' loaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to load human detector '{detector_name}': {e}")
                print("Continuing without human detector.")
        
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
            human_detector=human_detector,
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

    def process_single_frame(self, 
        profile_name,
        image_path,
        keyframe_index,
        joint_to_bone_mappings=None,
        num_people=1,
        bboxes=None) -> list[dict]:
        """
        Process a single frame and populate joint_to_bone_mapping for the given keyframe_index.
        
        Args:
            profile_name: Profile name (e.g., "mixamo", "unity", "mhr")
            image_path: Path to the image file
            keyframe_index: Index of this keyframe (0-based)
            joint_to_bone_mappings: Optional existing joint_to_bone_mappings for each person.
        
        Returns:
            dict for each person: joint_to_bone_mapping, rest_pose, vertices, keypoints_3d
        """

        ids = []
        if num_people > 0 and bboxes is None:
            ids = [i for i in range(num_people)] # without bboxes, name people 0, 1, 2, ...
        else:
            ids = self._get_person_ids(bboxes)

        ## IMPORTANT: process_one_image returns a list of dicts, in ORDER of bboxes passed in!
        outputs_raw = self.estimator.process_one_image(image_path, bboxes=bboxes)

        all_results = {} # dict for each person: joint_to_bone_mapping, rest_pose, vertices, keypoints_3d
        for i, id in enumerate(ids):
            if bboxes is not None:
                outputs = outputs_raw[i]
            else:
                # without bboxes, pick the ith largest person in the frame
                # "but wait, won't the largest person change in some situations?" 
                # yes - pass in your own re-id (gta-link is good) bbox data if you want consistency
                outputs = self._pick_largest_person(outputs_raw, i)

            if outputs is None:
                continue # no output for this person, skip it
        
            joint_coords = outputs["pred_joint_coords"]
            joint_rotations = outputs["pred_global_rots"]
            # vertices = outputs["pred_vertices"] # TODO: Not sure what to do with this for now - we're armature only
            keypoints_3d = outputs["pred_keypoints_3d"]

            # Add the helper keypoints to the keypoints_3d array before calculating the joint mapping
            keypoints_3d = add_helper_keypoints(keypoints_3d)

            # Load or use existing joint_to_bone_mapping
            if joint_to_bone_mappings is None:
                joint_to_bone_mappings = {}

            if i not in joint_to_bone_mappings:
                # force load from json to prevent weird dict mutation issues
                joint_to_bone_mapping = self.get_joint_to_bone_mapping(profile_name, use_cache=False)
            else:
                joint_to_bone_mapping = joint_to_bone_mappings[i]

            # Populate joint mapping for this keyframe (modifies joint_to_bone_mapping in-place)
            self.populate_joint_mapping(joint_to_bone_mapping, joint_rotations, joint_coords, keypoints_3d, keyframe_index)
            
            # vertices = np.array([convert_to_blender_coords(v) for v in vertices]) # TODO: Not sure what to do with this for now - we're armature only
        
            all_results[id] = {
                "joint_to_bone_mapping": joint_to_bone_mapping,
                # "vertices": vertices.tolist(), # TODO: Not sure what to do with this for now - we're armature only
                # "keypoints_3d": keypoints_3d.tolist(), # Not used downstream, maybe eventually?
            }

        return all_results


    def get_joint_to_bone_mapping(self, profile_name, use_cache=True):
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

    def create_metadata(self, profile_name, id, num_keyframes=1):
        return {
            "num_keyframes": num_keyframes,
            "id": id,
            "profile_name": profile_name
        }

    def _get_person_ids(self, bboxes):
        """
        Assumes MOT format bboxes, aka
        [frame_index, person_id, x1, y1, w, h, conf:optional, x_world:optional, y_world:optional, z_world:optional]
        We just need the first six columns overall for bbox support

        Returns:
            list of person ids
        """
        seen = set()
        out = []
        for b in bboxes:
            if len(b) >= 2 and b[1] is not None and b[1] not in seen:
                seen.add(b[1])
                out.append(b[1])
        return out

    def _pick_largest_person(self, outputs, person_index=0):
        """
        Pick the index-th largest person from detections.
        
        Args:
            outputs: Either a dict (single person) or list of dicts (multiple people)
            person_index: Which person to pick (0 = largest, 1 = second largest, etc.)
        
        Returns:
            The selected person dict, or None if no people detected or person_index is out of bounds
        """
        if isinstance(outputs, dict):
            if person_index == 0:
                return outputs
            else:
                return None  # Only 1 person detected, but person_index > 0 requested
        
        if len(outputs) == 0:
            return None  # No people detected
        
        # Calculate size for each person
        sizes = []
        for o in outputs:
            verts = o["pred_vertices"]
            size = (np.max(verts[:,0]) - np.min(verts[:,0])) + (np.max(verts[:,1]) - np.min(verts[:,1]))
            sizes.append(size)
        
        # Check if we have enough people
        if person_index >= len(outputs):
            return None  # Not enough people detected
        
        # Sort by size (largest first) and return the index-th largest
        sorted_indices = np.argsort(sizes)[::-1]  # Descending order
        selected_index = sorted_indices[person_index]
        return outputs[selected_index]