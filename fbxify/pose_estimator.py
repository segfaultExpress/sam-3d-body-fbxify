import torch
import numpy as np
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.data.utils.io import load_image
from fbxify.utils import convert_to_blender_coords, get_keypoint, get_profile, add_helper_keypoints
from fbxify.metadata import JOINT_NAMES_TO_INDEX
import json
import os
import random
from fbxify.utils import to_serializable

class PoseEstimator:

    cached_cam_int = None # cache for camera intrinsics to avoid re-running FOV estimator per frame

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

    def cache_cam_int_from_images(self, img_paths, average_of=1):
        if self.estimator.fov_estimator is None:
            print("Camera Intrinsics caching turned on, but there was no FOV estimator loaded. Please load an FOV estimator.")
            return

        # Randomly sample average_of images from img_paths

        if average_of > len(img_paths):
            print(f"Warning: Requested to average over {average_of} images, but only {len(img_paths)} images are available. Using all images.")
            average_of = len(img_paths)

        img_paths = random.sample(img_paths, average_of)

        cam_ints = []
        for img_path in img_paths:
            img = load_image(img_path, backend="cv2", image_format="bgr")
            cam_int = self.estimator.fov_estimator.get_cam_intrinsics(img)
            cam_ints.append(cam_int)

        # Average the camera intrinsics (they should all be tensors with shape (1, 3, 3))
        # Stack them, compute mean, and ensure it's a tensor
        cam_ints_stacked = torch.stack(cam_ints, dim=0)  # (N, 1, 3, 3)
        self.cached_cam_int = torch.mean(cam_ints_stacked, dim=0)  # (1, 3, 3)
        print(f"Cached camera intrinsics (averaged of {average_of} images): {self.cached_cam_int}")

    def cache_cam_int_from_file(self, cam_int):
        """
        Cache camera intrinsics from file or array.
        
        Accepts either:
        1. MoGe format: 3x3 numpy array or torch tensor with intrinsics matrix
           [[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]]
        
        2. COLMAP format: path to cameras.txt file or dict with COLMAP parameters
           - File format: "CAMERA_ID MODEL WIDTH HEIGHT [params...]"
           - For PINHOLE: "1 PINHOLE W H FX FY CX CY"
           - For SIMPLE_PINHOLE: "1 SIMPLE_PINHOLE W H F CX CY"
           - Dict format: {"model": "PINHOLE", "width": W, "height": H, "fx": FX, "fy": FY, "cx": CX, "cy": CY}
        
        Args:
            cam_int: Either a 3x3 numpy array/tensor (MoGe) or file path/dict (COLMAP)
        """
        # Check if it's a numpy array or torch tensor (MoGe format)
        if isinstance(cam_int, (np.ndarray, torch.Tensor)):
            # Convert to tensor if numpy array
            if isinstance(cam_int, np.ndarray):
                cam_int = torch.from_numpy(cam_int).float()
            
            # Validate shape and convert to (1, 3, 3) format
            if cam_int.shape == (3, 3):
                # Add batch dimension: (3, 3) -> (1, 3, 3)
                self.cached_cam_int = cam_int.unsqueeze(0)
                print(f"Cached camera intrinsics from MoGe format (3x3 matrix):\n{self.cached_cam_int}")
                return
            elif cam_int.shape == (1, 3, 3):
                # Already in correct format
                self.cached_cam_int = cam_int.float()
                print(f"Cached camera intrinsics from MoGe format (batched 1x3x3 matrix):\n{self.cached_cam_int}")
                return
            else:
                raise ValueError(f"MoGe format must be 3x3 or 1x3x3 array, got shape {cam_int.shape}")
        
        # Check if it's a file path (COLMAP format or MoGe text format)
        elif isinstance(cam_int, str) and os.path.isfile(cam_int):
            # Try to parse as COLMAP format first
            try:
                with open(cam_int, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        # Skip comments and empty lines
                        if not line or line.startswith('#'):
                            continue
                        
                        parts = line.split()
                        if len(parts) < 4:
                            continue
                        
                        # Check if it looks like COLMAP format (has MODEL keyword)
                        if parts[1] in ["PINHOLE", "SIMPLE_PINHOLE"]:
                            camera_id = parts[0]
                            model = parts[1]
                            width = float(parts[2])
                            height = float(parts[3])
                            
                            if model == "PINHOLE":
                                if len(parts) != 8:
                                    raise ValueError(f"PINHOLE model requires 8 parameters, got {len(parts)}")
                                fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                            elif model == "SIMPLE_PINHOLE":
                                if len(parts) != 7:
                                    raise ValueError(f"SIMPLE_PINHOLE model requires 7 parameters, got {len(parts)}")
                                f = float(parts[4])
                                fx = fy = f
                                cx, cy = float(parts[5]), float(parts[6])
                            else:
                                raise ValueError(f"Unsupported COLMAP camera model: {model}. Only PINHOLE and SIMPLE_PINHOLE are supported.")
                            
                            # Build 3x3 intrinsics matrix as tensor with batch dimension (1, 3, 3)
                            self.cached_cam_int = torch.tensor([
                                [[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]]
                            ], dtype=torch.float32)
                            
                            print(f"Cached camera intrinsics from COLMAP file (camera_id={camera_id}, model={model}):\n{self.cached_cam_int}")
                            return
            except (ValueError, IndexError) as e:
                # If COLMAP parsing fails, try MoGe format (3x3 matrix)
                pass
            
            # Try to parse as MoGe format (3x3 matrix text file)
            try:
                # Read all lines and try to parse as 3x3 matrix
                with open(cam_int, 'r') as f:
                    lines = f.readlines()
                    matrix_data = []
                    for line in lines:
                        line = line.strip()
                        # Skip comments and empty lines
                        if not line or line.startswith('#'):
                            continue
                        # Try to parse as space or comma separated numbers
                        if ',' in line:
                            row = [float(x.strip()) for x in line.split(',')]
                        else:
                            row = [float(x) for x in line.split()]
                        if len(row) == 3:
                            matrix_data.append(row)
                    
                    if len(matrix_data) == 3:
                        cam_int_matrix = np.array(matrix_data, dtype=np.float32)
                        # Validate it's a valid intrinsics matrix
                        if cam_int_matrix.shape == (3, 3) and cam_int_matrix[2, 0] == 0 and cam_int_matrix[2, 1] == 0 and cam_int_matrix[2, 2] == 1:
                            # Convert to tensor and add batch dimension: (3, 3) -> (1, 3, 3)
                            self.cached_cam_int = torch.from_numpy(cam_int_matrix).float().unsqueeze(0)
                            print(f"Cached camera intrinsics from MoGe text file (3x3 matrix):\n{self.cached_cam_int}")
                            return
            except (ValueError, IndexError) as e:
                pass
            
            raise ValueError(f"Could not parse camera intrinsics file: {cam_int}. Expected COLMAP format (cameras.txt) or MoGe format (3x3 matrix).")
        
        # Check if it's a dict (COLMAP format as dict)
        elif isinstance(cam_int, dict):
            model = cam_int.get("model", "PINHOLE").upper()
            width = float(cam_int.get("width", 1920))
            height = float(cam_int.get("height", 1080))
            
            if model == "PINHOLE":
                fx = float(cam_int.get("fx", cam_int.get("f", width)))
                fy = float(cam_int.get("fy", cam_int.get("f", height)))
                cx = float(cam_int.get("cx", width / 2))
                cy = float(cam_int.get("cy", height / 2))
            elif model == "SIMPLE_PINHOLE":
                f = float(cam_int.get("f", (width + height) / 2))
                fx = fy = f
                cx = float(cam_int.get("cx", width / 2))
                cy = float(cam_int.get("cy", height / 2))
            else:
                raise ValueError(f"Unsupported COLMAP camera model: {model}. Only PINHOLE and SIMPLE_PINHOLE are supported.")
            
            # Build 3x3 intrinsics matrix as tensor with batch dimension (1, 3, 3)
            self.cached_cam_int = torch.tensor([
                [[fx, 0, cx],
                 [0, fy, cy],
                 [0, 0, 1]]
            ], dtype=torch.float32)
            
            print(f"Cached camera intrinsics from COLMAP dict (model={model}):\n{self.cached_cam_int}")
            return
        
        else:
            raise ValueError(f"Unsupported camera intrinsics format. Expected:\n"
                           f"  - 3x3 numpy array/tensor (MoGe format)\n"
                           f"  - Path to COLMAP cameras.txt file (string)\n"
                           f"  - Dict with COLMAP parameters\n"
                           f"Got: {type(cam_int)}")

    def process_single_frame(self, 
        profile_name,
        image_path,
        keyframe_index,
        joint_to_bone_mappings=None,
        root_motions=None,
        num_people=1,
        bboxes=None):
        """
        Process a single frame and populate joint_to_bone_mapping for the given keyframe_index.
        
        Args:
            profile_name: Profile name (e.g., "mixamo", "unity", "mhr")
            image_path: Path to the image file
            keyframe_index: Index of this keyframe (0-based)
            joint_to_bone_mappings: Optional existing joint_to_bone_mappings for each person.
            bboxes: Either a list of tuples (person_id, x1, y1, w, h, ...) or numpy array in xyxy format
        
        Returns:
            dict for each person: joint_to_bone_mapping, rest_pose, vertices, keypoints_3d
        """

        ids = []
        bboxes_numpy = None
        
        if num_people > 0 and bboxes is None:
            ids = [i for i in range(num_people)] # without bboxes, name people 0, 1, 2, ...
        else:
            # Check if bboxes is a list (original format) or numpy array
            if isinstance(bboxes, list):
                # Original format: list of tuples (person_id, x1, y1, w, h, ...)
                ids = self._get_person_ids(bboxes)
                # Convert to numpy array in xyxy format for process_one_image
                bboxes_numpy = self._convert_bboxes_to_numpy(bboxes)
            elif isinstance(bboxes, np.ndarray):
                # Already in numpy format - extract person IDs if possible
                # If numpy array, we can't extract person IDs, so use indices
                ids = [i for i in range(len(bboxes))]
                bboxes_numpy = bboxes
            else:
                raise ValueError(f"bboxes must be either a list or numpy array, got {type(bboxes)}")

        ## IMPORTANT: process_one_image returns a list of dicts, in ORDER of bboxes passed in!
        outputs_raw = self.estimator.process_one_image(image_path, bboxes=bboxes_numpy, cam_int=self.cached_cam_int)

        with open("outputs_raw.json", "w") as f:
            json.dump(to_serializable(outputs_raw), f)

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

            if id not in joint_to_bone_mappings.keys():
                # force load from json to prevent weird dict mutation issues
                joint_to_bone_mapping = self.get_joint_to_bone_mapping(profile_name, use_cache=False)
            else:
                joint_to_bone_mapping = joint_to_bone_mappings[id]

            if root_motions is None:
                root_motions = {}

            if id not in root_motions.keys():
                root_motions[id] = []

            root_motion = root_motions[id]

            # Populate joint mapping for this keyframe (modifies joint_to_bone_mapping in-place)
            self.populate_joint_mapping(joint_to_bone_mapping, joint_rotations, joint_coords, keypoints_3d, keyframe_index)

            root_motion.append({
                "global_rot": outputs["global_rot"],
                "pred_cam_t": outputs["pred_cam_t"], # colmap can and should be used to reverse this, assuming both camera and person are moving. But we assume static camera as far as outputs are concerned
            })

            # Extract body parameters for mesh generation
            scale_params = outputs.get("scale_params")
            shape_params = outputs.get("shape_params")
            
            # Log what we got from outputs
            if scale_params is None or shape_params is None:
                print(f"  WARNING: Missing body params for person {id}")
                print(f"    Available keys in outputs: {list(outputs.keys())}")
                if scale_params is None:
                    print(f"    scale_params is None")
                if shape_params is None:
                    print(f"    shape_params is None")
            else:
                # Log parameter info
                if hasattr(scale_params, 'shape'):
                    print(f"  Person {id} body params: scale={scale_params.shape}, shape={shape_params.shape}")
                elif isinstance(scale_params, (list, np.ndarray)):
                    print(f"  Person {id} body params: scale={len(scale_params) if isinstance(scale_params, list) else scale_params.shape}, shape={len(shape_params) if isinstance(shape_params, list) else shape_params.shape}")
            
            # vertices = np.array([convert_to_blender_coords(v) for v in vertices]) # TODO: Not sure what to do with this for now - we're armature only
        
            all_results[id] = {
                "joint_to_bone_mapping": joint_to_bone_mapping,
                "root_motion": root_motion,
                "scale_params": scale_params.tolist() if scale_params is not None and hasattr(scale_params, 'tolist') else (scale_params if scale_params is not None else None),
                "shape_params": shape_params.tolist() if shape_params is not None and hasattr(shape_params, 'tolist') else (shape_params if shape_params is not None else None),
                # "vertices": vertices.tolist(), # TODO: Not sure what to do with this for now - we're armature only
                # "keypoints_3d": keypoints_3d.tolist(), # Not used downstream, maybe eventually?
            }

        return all_results, outputs_raw


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

    def create_metadata(self, profile_name, id, num_keyframes=1, fps=30.0):
        return {
            "num_keyframes": num_keyframes,
            "id": id,
            "profile_name": profile_name,
            "fps": fps
        }

    def _get_person_ids(self, bboxes):
        """
        Extract person IDs from bbox list.
        
        Input format: list of tuples (person_id, x1, y1, w, h, conf, x_world, y_world, z_world)
        where person_id is at index 0.

        Returns:
            list of person ids
        """
        seen = set()
        out = []
        for b in bboxes:
            # person_id is at index 0 in the tuple
            if len(b) >= 1 and b[0] is not None and b[0] not in seen:
                seen.add(b[0])
                out.append(b[0])
        return out

    def _convert_bboxes_to_numpy(self, bbox_list):
        """
        Convert list of bbox tuples to numpy array in xyxy format.
        
        Input format: list of (person_id, x1, y1, w, h, conf, x_world, y_world, z_world)
        where person_id is at index 0, x1 at index 1, y1 at index 2, w at index 3, h at index 4.
        Output format: numpy array of shape (N, 4) with [x1, y1, w, h] format 
        """
        if bbox_list is None or len(bbox_list) == 0:
            return None
        
        bboxes_xyxy = []
        for bbox_tuple in bbox_list:
            # Extract x1, y1, w, h (indices 1, 2, 3, 4 in the tuple, since index 0 is person_id)
            x1 = float(bbox_tuple[1])
            y1 = float(bbox_tuple[2])
            w = float(bbox_tuple[3])
            h = float(bbox_tuple[4])
            # Convert to xyxy format: [x1, y1, w, h]
            # bboxes_xyxy.append([x1, y1, w, h]) 
            bboxes_xyxy.append([x1, y1, x1+w, y1+h])
        
        return np.array(bboxes_xyxy, dtype=np.float32)

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
    
    def test_regenerate_vertices_from_outputs_raw(self, outputs_raw_path="outputs_raw.json", person_index=0, output_obj_path="test_regenerated.obj"):
        """
        Test method to regenerate vertices from outputs_raw.json and save as OBJ file.
        
        This verifies that outputs_raw contains enough information to regenerate
        the mesh by calling mhr_head.mhr_forward() with the saved parameters.
        
        Args:
            outputs_raw_path: Path to outputs_raw.json file
            person_index: Index of person to regenerate (default: 0)
            output_obj_path: Path to save the OBJ file
            
        Returns:
            tuple: (vertices_np, output_obj_path)
        """
        import trimesh
        
        # Load outputs_raw.json
        print(f"Loading {outputs_raw_path}...")
        with open(outputs_raw_path, "r") as f:
            outputs_raw = json.load(f)
        
        if person_index >= len(outputs_raw):
            raise ValueError(f"person_index {person_index} out of range. Only {len(outputs_raw)} people found.")
        
        output = outputs_raw[person_index]
        print(f"Processing person {person_index}...")
        
        # Get MHR head from estimator
        mhr_head = self.estimator.model.head_pose
        faces = self.estimator.faces
        
        # Convert numpy arrays to torch tensors
        device = self.device
        
        # Extract parameters from outputs_raw
        global_rot = torch.from_numpy(np.array(output["global_rot"])).float().to(device)
        body_pose_params = torch.from_numpy(np.array(output["body_pose_params"])).float().to(device)
        hand_pose_params = torch.from_numpy(np.array(output["hand_pose_params"])).float().to(device)
        scale_params = torch.from_numpy(np.array(output["scale_params"])).float().to(device)
        shape_params = torch.from_numpy(np.array(output["shape_params"])).float().to(device)
        expr_params = torch.from_numpy(np.array(output["expr_params"])).float().to(device) if output.get("expr_params") is not None else None
        
        # global_trans is typically zeros (as set in mhr_head.forward)
        global_trans = torch.zeros_like(global_rot).to(device)
        
        # Add batch dimension if needed
        if len(global_rot.shape) == 1:
            global_rot = global_rot.unsqueeze(0)
        if len(body_pose_params.shape) == 1:
            body_pose_params = body_pose_params.unsqueeze(0)
        if len(hand_pose_params.shape) == 1:
            hand_pose_params = hand_pose_params.unsqueeze(0)
        if len(scale_params.shape) == 1:
            scale_params = scale_params.unsqueeze(0)
        if len(shape_params.shape) == 1:
            shape_params = shape_params.unsqueeze(0)
        if expr_params is not None and len(expr_params.shape) == 1:
            expr_params = expr_params.unsqueeze(0)
        if len(global_trans.shape) == 1:
            global_trans = global_trans.unsqueeze(0)
        
        print("Calling mhr_forward...")
        # Call mhr_forward to regenerate vertices
        with torch.no_grad():
            vertices = mhr_head.mhr_forward(
                global_trans=global_trans,
                global_rot=global_rot,
                body_pose_params=body_pose_params,
                hand_pose_params=hand_pose_params,
                scale_params=scale_params,
                shape_params=shape_params,
                expr_params=expr_params,
                return_keypoints=False,
                do_pcblend=True,
                return_joint_coords=False,
                return_model_params=False,
                return_joint_rotations=False,
            )
        
        # Convert to numpy and apply camera system correction (same as in mhr_head.forward)
        vertices_np = vertices.cpu().numpy()[0]  # Remove batch dimension
        vertices_np[..., [1, 2]] *= -1  # Camera system difference
        
        print(f"Generated {len(vertices_np)} vertices")
        if "pred_vertices" in output:
            orig_shape = output["pred_vertices"]
            if isinstance(orig_shape, list):
                orig_shape = np.array(orig_shape).shape
            print(f"Original vertices shape: {orig_shape}")
        
        # Create trimesh and save as OBJ
        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces)
        mesh.export(output_obj_path)
        print(f"Saved regenerated mesh to {output_obj_path}")
        
        # Compare with original vertices if available
        if "pred_vertices" in output:
            orig_vertices = np.array(output["pred_vertices"])
            if orig_vertices.shape == vertices_np.shape:
                diff = np.abs(orig_vertices - vertices_np)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f"Comparison with original vertices:")
                print(f"  Max difference: {max_diff:.6f}")
                print(f"  Mean difference: {mean_diff:.6f}")
                if max_diff < 1e-5:
                    print("  ✓ Vertices match exactly!")
                elif max_diff < 1e-3:
                    print("  ✓ Vertices are very close (likely numerical precision)")
                else:
                    print("  ⚠ Vertices differ significantly")
        
        return vertices_np, output_obj_path
    
    def generate_tpose_from_outputs_raw(self, outputs_raw_path="outputs_raw.json", person_index=0, output_obj_path="tpose.obj"):
        """
        Generate a T-pose/A-pose mesh with the same body shape and scale from outputs_raw.json,
        but with all pose parameters set to zero (no pose).
        
        This keeps the same body shape (shape_params) and scale (scale_params) but removes
        all pose information (global_rot, body_pose_params, hand_pose_params, expr_params).
        
        Args:
            outputs_raw_path: Path to outputs_raw.json file
            person_index: Index of person to use (default: 0)
            output_obj_path: Path to save the OBJ file
            
        Returns:
            tuple: (vertices_np, output_obj_path)
        """
        import trimesh
        
        # Load outputs_raw.json
        print(f"Loading {outputs_raw_path}...")
        with open(outputs_raw_path, "r") as f:
            outputs_raw = json.load(f)
        
        if person_index >= len(outputs_raw):
            raise ValueError(f"person_index {person_index} out of range. Only {len(outputs_raw)} people found.")
        
        output = outputs_raw[person_index]
        print(f"Processing person {person_index} for T-pose generation...")
        
        # Get MHR head from estimator
        mhr_head = self.estimator.model.head_pose
        faces = self.estimator.faces
        
        # Convert numpy arrays to torch tensors
        device = self.device
        
        # Extract shape and scale parameters (keep these)
        scale_params = torch.from_numpy(np.array(output["scale_params"])).float().to(device)
        shape_params = torch.from_numpy(np.array(output["shape_params"])).float().to(device)
        
        # Set all pose parameters to zero (T-pose/A-pose)
        # Global rotation: zeros (no rotation)
        global_rot = torch.zeros(3).float().to(device)
        global_trans = torch.zeros(3).float().to(device)
        
        # Body pose: zeros (T-pose)
        body_pose_params = torch.zeros(133).float().to(device)
        
        # Hand pose: None (use body_pose_params zeros for hands, which should give straight fingers)
        # If we pass zeros here, it adds hand_pose_mean which gives relaxed/bent fingers
        # By passing None, we skip replace_hands_in_pose and use the zeros from body_pose_params
        hand_pose_params = None
        
        # Expression: zeros (no facial expression)
        expr_params = torch.zeros(72).float().to(device) if output.get("expr_params") is not None else None
        
        # Add batch dimension
        global_rot = global_rot.unsqueeze(0)
        global_trans = global_trans.unsqueeze(0)
        body_pose_params = body_pose_params.unsqueeze(0)
        scale_params = scale_params.unsqueeze(0) if len(scale_params.shape) == 1 else scale_params
        shape_params = shape_params.unsqueeze(0) if len(shape_params.shape) == 1 else shape_params
        if expr_params is not None:
            expr_params = expr_params.unsqueeze(0)
        
        print("Generating T-pose mesh with mhr_forward...")
        print(f"  Shape params: {shape_params.shape}")
        print(f"  Scale params: {scale_params.shape}")
        print(f"  Global rot: {global_rot.shape} (all zeros)")
        print(f"  Body pose: {body_pose_params.shape} (all zeros)")
        print(f"  Hand pose: None (using zeros from body_pose_params for straight fingers)")
        
        # Call mhr_forward to generate T-pose vertices
        with torch.no_grad():
            vertices = mhr_head.mhr_forward(
                global_trans=global_trans,
                global_rot=global_rot,
                body_pose_params=body_pose_params,
                hand_pose_params=hand_pose_params,
                scale_params=scale_params,
                shape_params=shape_params,
                expr_params=expr_params,
                return_keypoints=False,
                do_pcblend=True,
                return_joint_coords=False,
                return_model_params=False,
                return_joint_rotations=False,
            )
        
        # Convert to numpy and apply camera system correction (same as in mhr_head.forward)
        vertices_np = vertices.cpu().numpy()[0]  # Remove batch dimension
        vertices_np[..., [1, 2]] *= -1  # Camera system difference
        
        print(f"Generated {len(vertices_np)} vertices in T-pose")
        print(f"  Expected vertex counts by LOD:")
        print(f"    LOD 0: ~18439 vertices (matches current output)")
        print(f"    LOD 1: ~73639 vertices")
        print(f"    LOD 2: ~294559 vertices")
        print(f"  Note: MHR model is hardcoded to lod=1 in mhr_head.py line 111, but generates {len(vertices_np)} vertices")
        print(f"  This suggests the model is actually using LOD 0, not LOD 1")
        
        # Create trimesh and save as OBJ
        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces)
        mesh.export(output_obj_path)
        print(f"Saved T-pose mesh to {output_obj_path}")
        
        return vertices_np, output_obj_path
    
    def generate_tpose(self, scale_params, shape_params, output_obj_path="tpose.obj"):
        # This is the main generate_tpose function (not generate_tpose_from_outputs_raw)
        """
        Generate a T-pose/A-pose mesh with the given body shape and scale parameters,
        but with all pose parameters set to zero (no pose).
        
        This keeps the same body shape (shape_params) and scale (scale_params) but removes
        all pose information (global_rot, body_pose_params, hand_pose_params, expr_params).
        
        Args:
            scale_params: Scale parameters (numpy array or list)
            shape_params: Shape parameters (numpy array or list)
            output_obj_path: Path to save the OBJ file
            
        Returns:
            tuple: (vertices_np, output_obj_path)
        """
        import trimesh
        
        # Get MHR head from estimator
        mhr_head = self.estimator.model.head_pose
        faces = self.estimator.faces
        
        # Log MHR model LOD information
        print("Checking MHR model LOD...")
        if hasattr(mhr_head, 'lod'):
            print(f"  MHRHead LOD: {mhr_head.lod}")
        if hasattr(mhr_head, 'mhr'):
            mhr_model = mhr_head.mhr
            # Try to get LOD from MHR model
            if hasattr(mhr_model, 'lod'):
                print(f"  MHR model LOD: {mhr_model.lod}")
            elif hasattr(mhr_model, '_lod'):
                print(f"  MHR model LOD (private): {mhr_model._lod}")
            else:
                print("  MHR model LOD: Not directly accessible from model object")
                # Check vertex count from faces to infer LOD
                if hasattr(faces, 'shape'):
                    print(f"  Faces shape: {faces.shape}")
                if hasattr(mhr_head, 'faces'):
                    print(f"  MHRHead faces shape: {mhr_head.faces.shape}")
        else:
            print("  MHR model: Not using Momentum (using torch.jit.load)")
        
        # Convert numpy arrays to torch tensors
        device = self.device
        
        # Convert to numpy arrays if needed, then to torch tensors
        if isinstance(scale_params, list):
            scale_params = np.array(scale_params)
        if isinstance(shape_params, list):
            shape_params = np.array(shape_params)
        
        scale_params = torch.from_numpy(scale_params).float().to(device)
        shape_params = torch.from_numpy(shape_params).float().to(device)
        
        # Set all pose parameters to zero (T-pose/A-pose)
        # Global rotation: zeros (no rotation)
        global_rot = torch.zeros(3).float().to(device)
        global_trans = torch.zeros(3).float().to(device)
        
        # Body pose: zeros (T-pose)
        body_pose_params = torch.zeros(133).float().to(device)
        
        # Hand pose: None (use body_pose_params zeros for hands, which should give straight fingers)
        # If we pass zeros here, it adds hand_pose_mean which gives relaxed/bent fingers
        # By passing None, we skip replace_hands_in_pose and use the zeros from body_pose_params
        hand_pose_params = None
        
        # Expression: zeros (no facial expression)
        # We always zero this out since it's per-frame
        expr_params = torch.zeros(72).float().to(device)
        
        # Add batch dimension
        global_rot = global_rot.unsqueeze(0)
        global_trans = global_trans.unsqueeze(0)
        body_pose_params = body_pose_params.unsqueeze(0)
        scale_params = scale_params.unsqueeze(0) if len(scale_params.shape) == 1 else scale_params
        shape_params = shape_params.unsqueeze(0) if len(shape_params.shape) == 1 else shape_params
        expr_params = expr_params.unsqueeze(0)
        
        print("Generating T-pose mesh with mhr_forward...")
        print(f"  Shape params: {shape_params.shape}")
        print(f"  Scale params: {scale_params.shape}")
        print(f"  Global rot: {global_rot.shape} (all zeros)")
        print(f"  Body pose: {body_pose_params.shape} (all zeros)")
        print(f"  Hand pose: None (using zeros from body_pose_params for straight fingers)")
        
        # Call mhr_forward to generate T-pose vertices
        with torch.no_grad():
            vertices = mhr_head.mhr_forward(
                global_trans=global_trans,
                global_rot=global_rot,
                body_pose_params=body_pose_params,
                hand_pose_params=hand_pose_params,
                scale_params=scale_params,
                shape_params=shape_params,
                expr_params=expr_params,
                return_keypoints=False,
                do_pcblend=True,
                return_joint_coords=False,
                return_model_params=False,
                return_joint_rotations=False,
            )
        
        # Convert to numpy and apply camera system correction (same as in mhr_head.forward)
        vertices_np = vertices.cpu().numpy()[0]  # Remove batch dimension
        vertices_np[..., [1, 2]] *= -1  # Camera system difference
        
        # Clean up GPU tensors
        del vertices, scale_params, shape_params, global_rot, global_trans, body_pose_params, expr_params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Generated {len(vertices_np)} vertices in T-pose")
        
        # Create trimesh and save as OBJ
        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces)
        mesh.export(output_obj_path)
        print(f"Saved T-pose mesh to {output_obj_path}")
        
        return vertices_np, output_obj_path