"""
Pose Estimation Manager - Handles expensive pose estimation operations.

This module is responsible for:
- Initializing and managing the SAM3DBodyEstimator
- Processing all frames to get pose estimation results
- Saving/loading estimation results as JSON files
- Camera intrinsics caching
"""
import torch
import numpy as np
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.data.utils.io import load_image
from fbxify.utils import to_serializable, extract_fbx_faces_with_blender
from fbxify import VERSION
from fbxify.i18n import Translator, DEFAULT_LANGUAGE
import json
import os
import glob
import re
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import threading


class CancelledError(Exception):
    """Raised when a running pose estimation job is cancelled."""
    pass


class PoseEstimationManager:
    """
    Manages pose estimation operations (the expensive part).
    Handles estimation for all frames and saves/loads results as JSON.
    """
    
    cached_cam_int = None  # cache for camera intrinsics to avoid re-running FOV estimator per frame
    
    def __init__(
        self,
        checkpoint_path,
        mhr_path,
        device=None,
        detector_name=None,
        detector_path=None,
        fov_name="moge2",
        fov_path=None,
        precision="fp32",
    ):
        """
        Initialize the pose estimation manager.
        
        Args:
            checkpoint_path: Path to SAM 3D Body checkpoint
            mhr_path: Path to MHR model file
            device: Device to use (cuda/cpu)
            detector_name: Name of human detector to use
            detector_path: Path to detector model
            fov_name: Name of FOV estimator to use
            fov_path: Path to FOV estimator model
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SAM 3D Body on {self.device}...")
        
        model, cfg = load_sam_3d_body(
            checkpoint_path=checkpoint_path,
            device=self.device,
            mhr_path=mhr_path
        )

        self.cfg = cfg
        
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

        self.precision = "fp32"
        self.set_inference_options(precision=precision)
        self._cancel_event = threading.Event()
        self._lod_faces_cache: Dict[int, np.ndarray] = {}
        self._lod_faces_initialized = False

    def _extract_faces_from_fbx(self, fbx_path: str) -> Optional[np.ndarray]:
        return extract_fbx_faces_with_blender(fbx_path)

    def _load_faces_from_file(self, lod: int) -> Optional[np.ndarray]:
        base_dir = os.path.dirname(__file__)
        faces_path = os.path.join(base_dir, "mapping", "mhr", f"lod{lod}_faces.npy")
        if not os.path.exists(faces_path):
            return None
        data = np.load(faces_path, allow_pickle=False)
        if isinstance(data, np.ndarray):
            return data
        if hasattr(data, "files") and data.files:
            key = "faces" if "faces" in data.files else data.files[0]
            return data[key]
        return None

    def _ensure_lod_faces_files(self) -> None:
        if self._lod_faces_initialized:
            return
        self._lod_faces_initialized = True
        
        base_dir = os.path.dirname(__file__)
        assets_dir = os.path.join(base_dir, "mapping", "mhr")
        if not os.path.isdir(assets_dir):
            print(f"  LOD faces: assets dir not found: {assets_dir}")
            return
        fbx_paths = sorted(glob.glob(os.path.join(assets_dir, "lod*.fbx")))
        if not fbx_paths:
            print(f"  LOD faces: no FBX files in {assets_dir}")
            return
        out_dir = os.path.join(os.path.dirname(__file__), "mapping", "mhr")
        os.makedirs(out_dir, exist_ok=True)
        for fbx_path in fbx_paths:
            base = os.path.basename(fbx_path)
            match = re.match(r"lod(\d+)\.fbx$", base, re.IGNORECASE)
            if not match:
                continue
            lod = int(match.group(1))
            out_path = os.path.join(out_dir, f"lod{lod}_faces.npy")
            if os.path.exists(out_path):
                continue
            faces = self._extract_faces_from_fbx(fbx_path)
            if faces is None or faces.size == 0:
                print(f"  LOD faces: failed to extract faces from {fbx_path}")
                continue
            np.save(out_path, faces.astype(np.int64))
            print(f"  LOD faces: saved {out_path} shape={faces.shape}")

    def cancel_current_job(self) -> None:
        """Signal any running estimation loop to stop."""
        self._cancel_event.set()

    def clear_cancel(self) -> None:
        """Clear the cancel signal before starting a new job."""
        self._cancel_event.clear()

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise CancelledError("Pose estimation cancelled by user.")
    
    def cache_cam_int_from_images(self, img_paths, average_of=1):
        """
        Cache camera intrinsics by averaging over sampled images.
        
        Args:
            img_paths: List of image paths to sample from
            average_of: Number of images to sample and average
        """
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

    def set_inference_options(self, precision: str = "fp32"):
        """
        Configure inference precision.
        """
        precision = (precision or "fp32").lower()
        if precision not in ["fp32", "bf16", "fp16"]:
            raise ValueError(f"Unsupported precision: {precision}")

        if precision != self.precision:
            self._apply_precision(precision)
            self.precision = precision

    def _apply_precision(self, precision: str):
        model = self.estimator.model
        if precision == "fp32":
            model.float()
            model.backbone_dtype = torch.float32
            return

        model.cfg.defrost()
        model.cfg.TRAIN.FP16_TYPE = "bfloat16" if precision == "bf16" else "float16"
        model.cfg.freeze()

        model.convert_to_fp16()
        model.backbone_dtype = torch.bfloat16 if precision == "bf16" else torch.float16


    def cache_cam_int_from_file(self, cam_int):
        """
        Cache camera intrinsics from file or array.
        
        Accepts either:
        1. MoGe format: 3x3 numpy array or torch tensor with intrinsics matrix
        2. COLMAP format: path to cameras.txt file or dict with COLMAP parameters
        
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

    def estimate_all_frames(self, frame_paths: List[str], num_people: int = 1,
                           bbox_dict: Optional[Dict[int, List[Tuple]]] = None,
                           progress_callback: Optional[callable] = None,
                           source_name: Optional[str] = None,
                           missing_bbox_behavior: str = "Run Detection",
                           lang: str = DEFAULT_LANGUAGE) -> Dict[str, Dict[str, Any]]:
        """
        Estimate poses for all frames.
        
        Args:
            frame_paths: List of frame file paths
            num_people: Number of people to detect (if bboxes not provided)
            bbox_dict: Optional dictionary mapping frame_index (1-based) to list of bbox tuples
            progress_callback: Optional callback function(progress, description)
            source_name: Optional source filename for metadata
            missing_bbox_behavior: What to do when bbox is missing for a frame.
                                  "Run Detection" (default): Use num_people to run detection.
                                  "Skip Frame": Skip pose estimation for that frame (store empty dict).
            
        Returns:
            Dictionary in format: {frame_X: {person_id: estimation_data}}
            If no people detected in a frame, frame_X will be an empty dict but present.
        """
        total_start = time.time()
        estimation_results = {}
        translator = Translator(lang)
                
        for frame_index, frame_path in enumerate(frame_paths):
            self._check_cancelled()
            # Update progress bar if callback is provided
            if progress_callback:
                progress = frame_index / len(frame_paths)
                progress_callback(progress, translator.t("progress.estimating_frame", frame_index=frame_index + 1, total_frames=len(frame_paths)))
            
            # Get bboxes for this frame (bbox_dict uses 1-based indexing for MOT format)
            bboxes = None
            bbox_missing = False
            if bbox_dict is not None:
                if (frame_index + 1) in bbox_dict:
                    bboxes = bbox_dict[frame_index + 1]
                else:
                    bbox_missing = True
            
            # Handle missing bbox based on behavior
            if bbox_missing and missing_bbox_behavior == "Skip Frame":
                print(f"  [INFO] Bbox missing for frame {frame_index + 1}, skipping (behavior: Skip Frame)")
                # Store empty dict to skip this frame
                estimation_results[str(frame_index)] = {}
                continue
            
            # Process this frame (either bbox exists, or missing_bbox_behavior is "Run Detection")
            frame_results = self._estimate_single_frame(
                frame_path,
                num_people=num_people,
                bboxes=bboxes
            )
            
            # Store results with frame index as string key
            estimation_results[str(frame_index)] = frame_results
        
        total_end = time.time()
        total_time = total_end - total_start
        avg_time_per_frame = total_time / len(frame_paths) if frame_paths else 0
        
        return estimation_results
    
    def _estimate_single_frame(self, image_path: str, num_people: int = 1,
                              bboxes: Optional[List[Tuple]] = None) -> Dict[str, Any]:
        """
        Estimate pose for a single frame.
        
        Args:
            image_path: Path to image file
            num_people: Number of people to detect (if bboxes not provided)
            bboxes: Optional list of bbox tuples (person_id, x1, y1, w, h, ...)
            
        Returns:
            Dictionary mapping person_id to estimation data
            Empty dict if no people detected
        """
        ids = []
        bboxes_numpy = None
        
        if num_people > 0 and bboxes is None:
            ids = [i for i in range(num_people)]  # without bboxes, name people 0, 1, 2, ...
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

        # IMPORTANT: process_one_image returns a list of dicts, in ORDER of bboxes passed in!
        timer_start = time.time()
        outputs_raw = self.estimator.process_one_image(image_path, bboxes=bboxes_numpy, cam_int=self.cached_cam_int)
        timer_process = time.time()
        process_time = timer_process - timer_start
        # Note: Detailed breakdown is printed inside process_one_image
        
        # Check if mesh was generated (pred_vertices present)
        mesh_generated = False
        if outputs_raw and len(outputs_raw) > 0:
            mesh_generated = "pred_vertices" in outputs_raw[0] and outputs_raw[0]["pred_vertices"] is not None
            if mesh_generated:
                vert_count = len(outputs_raw[0]["pred_vertices"]) if outputs_raw[0]["pred_vertices"] is not None else 0
                print(f"  [INFO] Mesh generated: {vert_count} vertices (may not be needed)")

        # Convert outputs to serializable format and organize by person ID
        timer_filter_start = time.time()
        frame_results = {}
        for i, person_id in enumerate(ids):
            if bboxes is not None:
                # Use the output at index i (matches bbox order)
                if i < len(outputs_raw):
                    outputs = outputs_raw[i]
                else:
                    continue  # No output for this person
            else:
                # without bboxes, pick the ith largest person in the frame
                outputs = self._pick_largest_person(outputs_raw, i)
                if outputs is None:
                    continue  # No output for this person
            
            # Filter to only required keys and convert to serializable format
            filtered_output = self._filter_estimation_output(outputs)
            estimation_data = to_serializable(filtered_output)
            frame_results[str(person_id)] = estimation_data
        
        timer_filter_end = time.time()
        filter_time = timer_filter_end - timer_filter_start
        
        # If no people detected, return empty dict (but frame is still present)
        return frame_results
    
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
        Output format: numpy array of shape (N, 4) with [x1, y1, x2, y2] format 
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
            # Convert to xyxy format: [x1, y1, x1+w, y1+h]
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
    
    def _filter_estimation_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter estimation output to only include required keys for FBX export.
        
        NOTE: process_one_image() generates MANY outputs that we don't use:
        - pred_vertices: Full mesh vertices (~18k+ vertices) - NOT USED, but generated every frame
        - pred_keypoints_2d: 2D keypoints - NOT USED
        - pred_pose_raw: Raw pose parameters - NOT USED
        - body_pose_params: Body pose params - NOT USED (we use pred_global_rots instead)
        - hand_pose_params: Hand pose params - NOT USED
        - expr_params: Expression params - NOT USED
        - mask: Segmentation mask - NOT USED
        
        Required keys (what we actually use):
        - pred_joint_coords: Joint coordinates
        - pred_global_rots: Global joint rotations
        - pred_keypoints_3d: 3D keypoints
        - global_rot: Global rotation
        - pred_cam_t: Camera translation
        - scale_params: Scale parameters (for mesh generation, only needed once per person)
        - shape_params: Shape parameters (for mesh generation, only needed once per person)
        
        OPTIMIZATION OPPORTUNITY: The mesh (pred_vertices) is generated on every frame
        but we only need it once per person for the T-pose. We could potentially:
        1. Skip mesh generation for most frames
        2. Only generate mesh when needed (first frame or when shape/scale changes)
        3. Call internal MHR functions directly to skip full inference pipeline
        
        Args:
            output: Full estimation output dictionary
            
        Returns:
            Filtered dictionary with only required keys
        """
        required_keys = [
            "pred_joint_coords",
            "pred_global_rots",
            "pred_keypoints_3d",
            "global_rot",
            "pred_cam_t",
            "scale_params",
            "shape_params"
        ]
        
        filtered = {}
        for key in required_keys:
            if key in output:
                filtered[key] = output[key]
            else:
                # Warn if required key is missing (except optional mesh params)
                if key not in ["scale_params", "shape_params"]:
                    print(f"  WARNING: Missing required key '{key}' in estimation output")
        
        return filtered
    
    def save_estimation_results(self, estimation_results: Dict[str, Dict[str, Any]], file_path: str,
                               source_name: Optional[str] = None, num_people: Optional[int] = None):
        """
        Save estimation results to JSON file with metadata wrapper.
        
        Args:
            estimation_results: Dictionary in format {frame_X: {person_id: estimation_data}}
            file_path: Path to save JSON file
            source_name: Optional source filename for metadata
            num_people: Optional number of people (from bbox file or user input)
        """
        # Count unique person IDs from estimation results if num_people not provided
        if num_people is None:
            unique_person_ids = set()
            for frame_data in estimation_results.values():
                if isinstance(frame_data, dict):
                    for person_id in frame_data.keys():
                        unique_person_ids.add(person_id)
            num_people = len(unique_person_ids) if unique_person_ids else 0
        
        # Create metadata structure
        metadata = {
            "source": source_name or "unknown",
            "creation_date": datetime.utcnow().isoformat() + "Z",
            "version": VERSION,
            "num_people": num_people,
            "frames": estimation_results
        }
        
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Calculate file size for info
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        num_frames = len(estimation_results)
        print(f"Saved estimation results to {file_path} ({file_size_mb:.2f} MB, {num_frames} frames)")
    
    def load_estimation_results(self, file_path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Load estimation results from JSON file.
        
        Handles both old format (direct frames dict) and new format (with metadata).
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Tuple of (frames_dict, metadata_dict)
            frames_dict: Dictionary in format {frame_X: {person_id: estimation_data}}
            metadata_dict: Dictionary with source, creation_date, version (may be empty for old format)
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if it's new format (has metadata keys) or old format (direct frames)
        if isinstance(data, dict) and "frames" in data and "version" in data:
            # New format with metadata
            metadata = {
                "source": data.get("source", "unknown"),
                "creation_date": data.get("creation_date", "unknown"),
                "version": data.get("version", "unknown"),
                "num_people": data.get("num_people", None)
            }
            frames = data["frames"]
            
            # Check version compatibility
            file_version = metadata.get("version", "unknown")
            if file_version != VERSION:
                warning_msg = (
                    f"WARNING: Estimation file version ({file_version}) differs from current version ({VERSION}). "
                    f"File may have been created with a different version of the software. "
                    f"Proceeding with caution - errors may occur if formats are incompatible."
                )
                print(warning_msg)
                # Store version mismatch in metadata for later error messages
                metadata["_version_mismatch"] = True
                metadata["_file_version"] = file_version
            
            print(f"Loaded estimation results from {file_path}")
            print(f"  Source: {metadata['source']}")
            print(f"  Created: {metadata['creation_date']}")
            print(f"  Version: {file_version}")
            if metadata.get("num_people") is not None:
                print(f"  Number of people: {metadata['num_people']}")
            return frames, metadata
        else:
            # Old format (direct frames dict) - wrap in metadata
            print(f"Loaded estimation results from {file_path} (old format, no metadata)")
            metadata = {
                "source": "unknown",
                "creation_date": "unknown",
                "version": "unknown",
                "num_people": None
            }
            return data, metadata
    
    def generate_tpose(self, scale_params, shape_params, output_obj_path="tpose.obj", lod: Optional[int] = None):
        """
        Generate a T-pose/A-pose mesh with the given body shape and scale parameters,
        but with all pose parameters set to zero (no pose).
        
        This keeps the same body shape (shape_params) and scale (scale_params) but removes
        all pose information (global_rot, body_pose_params, hand_pose_params, expr_params).
        
        Args:
            scale_params: Scale parameters (numpy array or list)
            shape_params: Shape parameters (numpy array or list)
            output_obj_path: Path to save the OBJ file
            lod: Optional LOD level for MHR model (0-6). If None, uses default LOD.
            
        Returns:
            tuple: (vertices_np, output_obj_path)
        """
        import trimesh
        
        # Get MHR head from estimator
        mhr_head = self.estimator.model.head_pose
        self._ensure_lod_faces_files()
        
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
        print(f"  LOD: {lod}")
        
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
                lod=lod,
            )
        
        # Convert to numpy and apply camera system correction (same as in mhr_head.forward)
        vertices_np = vertices.cpu().numpy()[0]  # Remove batch dimension
        vertices_np[..., [1, 2]] *= -1  # Camera system difference
        
        # Clean up GPU tensors
        del vertices, scale_params, shape_params, global_rot, global_trans, body_pose_params, expr_params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Generated {len(vertices_np)} vertices in T-pose")

        # Prefer cached per-LOD faces if present; fallback to checkpoint faces.
        faces_default = mhr_head.faces.cpu().numpy()
        faces_file = self._load_faces_from_file(lod) if lod is not None else None
        faces_to_use = faces_file if faces_file is not None and faces_file.size > 0 else faces_default

        # Create trimesh and save as OBJ
        print(f"  Faces: {len(faces_to_use)}")
        print(f"  vertices: {len(vertices_np)}")
        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_to_use)
        print(f"  mesh: {mesh.vertices.shape}")
        mesh.export(output_obj_path)
        print(f"Saved T-pose mesh to {output_obj_path}")
        
        return vertices_np, output_obj_path

