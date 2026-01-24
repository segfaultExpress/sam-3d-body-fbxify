"""
FbxifyManager - Core business logic for processing videos/images and exporting FBX files.
"""
import os
import cv2
import tempfile
import shutil
import subprocess
import json
import copy
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from fbxify.pose_estimation_manager import PoseEstimationManager
from fbxify.fbx_data_prep_manager import FbxDataPrepManager
from fbxify.utils import export_to_fbx, to_serializable
from fbxify.refinement.refinement_manager import RefinementManager
from fbxify.i18n import Translator, DEFAULT_LANGUAGE
from fbxify.metadata import JOINT_NAMES_TO_INDEX


@dataclass
class ProcessResult:
    """Result of processing frames."""
    joint_to_bone_mappings: Dict[str, Any]
    root_motions: Optional[Dict[str, List[Dict]]]
    visualization_data: Optional[List[Dict]]
    frame_paths: List[str]
    profile_name: str
    fps: float
    mesh_obj_path: Optional[str] = None


class FbxifyManager:
    """
    Manager class for processing videos/images and exporting FBX files.
    """
    
    def __init__(self, estimation_manager: PoseEstimationManager, data_prep_manager: FbxDataPrepManager):
        """
        Initialize the manager with pose estimation and data prep managers.
        
        Args:
            estimation_manager: PoseEstimationManager instance
            data_prep_manager: FbxDataPrepManager instance
        """
        self.estimation_manager = estimation_manager
        self.data_prep_manager = data_prep_manager
    
    def extract_frames_from_video(self, video_path: str, temp_dir: str) -> Tuple[List[str], float]:
        """
        Extract all frames from video and save to temp directory.
        
        Args:
            video_path: Path to video file
            temp_dir: Temporary directory to save frames
            
        Returns:
            Tuple of (frame_paths, fps) where fps is the video frame rate
        """
        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_count = 0
        
        # Get FPS from video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps is None:
            # Fallback to default if FPS cannot be determined
            fps = 30.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_count += 1
        
        cap.release()
        return frame_paths, fps
    
    def prepare_video(self, input_video_path: str) -> Tuple[List[str], str, float]:
        """
        Split video into frames.
        
        Args:
            input_video_path: Path to input video file
            
        Returns:
            Tuple of (frame_paths, temp_dir, fps) where temp_dir should be cleaned up later
            and fps is the video frame rate
        """
        if input_video_path is None:
            return None, None, 30.0
        
        # Create temp directory for frames
        temp_dir = tempfile.mkdtemp(prefix="sam3d_frames_")
        frame_paths, fps = self.extract_frames_from_video(input_video_path, temp_dir)
        return frame_paths, temp_dir, fps
    
    def prepare_bboxes(self, bbox_file_path: str) -> Dict[int, List[Tuple]]:
        """
        Prepare bounding boxes from file.
        
        Args:
            bbox_file_path: Path to bbox file (CSV format)
            
        Returns:
            Dictionary mapping frame_index to list of bbox tuples
        """
        if bbox_file_path is None:
            return None

        bbox_dict = {}
        with open(bbox_file_path, "r") as f:
            try:
                for line in f:
                    if line.strip() == "":
                        continue
                    frame_index, person_id, x1, y1, w, h, conf, x_world, y_world, z_world = line.strip().split(",")
                    # Convert frame_index to int to standardize keys (handles '1', '1.0', etc.)
                    frame_index_int = int(float(frame_index))
                    if frame_index_int not in bbox_dict:
                        bbox_dict[frame_index_int] = []
                    bbox_dict[frame_index_int].append((person_id, x1, y1, w, h, conf, x_world, y_world, z_world))
            except Exception as e:
                raise ValueError(f"Error preparing bboxes: {e}")
        return bbox_dict
    
    def set_camera_intrinsics(self, fov_method: str, fov_file_path: Optional[str], 
                             frame_paths: List[str], sample_number: int = 1):
        """
        Set camera intrinsics based on method.
        
        Args:
            fov_method: Method to use ("Default", "File", "Sample")
            fov_file_path: Path to FOV file (if method is "File")
            frame_paths: List of frame paths (if method is "Sample")
            sample_number: Number of samples to average (if method is "Sample")
        """
        if fov_method == "File":
            if fov_file_path is None:
                raise ValueError("Camera intrinsics file required when FOV method is 'File'")
            self.estimation_manager.cache_cam_int_from_file(fov_file_path)
        elif fov_method == "Sample":
            self.estimation_manager.cache_cam_int_from_images(frame_paths, average_of=sample_number)
        # else: "Default" - built-in FOV estimator runs every frame (default behavior)
    
    def _remove_outliers_and_average(self, body_params_list: List[Dict[str, Any]], outlier_percent: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers from body parameters and return averaged values.
        
        Args:
            body_params_list: List of parameter dicts [{scale_params, shape_params}, ...]
            outlier_percent: Percentage of outliers to remove from each tail (0-50)
            
        Returns:
            Tuple of (averaged_scale_params, averaged_shape_params) as numpy arrays
        """
        if not body_params_list:
            raise ValueError("body_params_list cannot be empty")
        
        # Convert to numpy arrays (each row is one frame's parameters)
        # Ensure all parameters are numpy arrays and have consistent shape
        scale_params_list = []
        shape_params_list = []
        for p in body_params_list:
            scale = p["scale_params"]
            shape = p["shape_params"]
            
            # Convert to numpy if needed
            if not isinstance(scale, np.ndarray):
                scale = np.array(scale)
            if not isinstance(shape, np.ndarray):
                shape = np.array(shape)
            
            # Flatten to 1D if needed
            if scale.ndim > 1:
                scale = scale.flatten()
            if shape.ndim > 1:
                shape = shape.flatten()
            
            scale_params_list.append(scale)
            shape_params_list.append(shape)
        
        # Stack into 2D arrays (frames x params)
        scale_params_array = np.array(scale_params_list)
        shape_params_array = np.array(shape_params_list)
        
        # Debug: check shapes
        print(f"  Parameter array shapes: scale={scale_params_array.shape}, shape={shape_params_array.shape}")
        
        # Remove outliers using percentile-based method
        # Remove top/bottom outlier_percent/2% from each dimension, then remove frames that have outliers in any dimension
        if outlier_percent > 0 and len(body_params_list) > 2:
            # Calculate percentiles for each dimension
            lower_percentile = outlier_percent / 2.0
            upper_percentile = 100.0 - (outlier_percent / 2.0)
            
            # For each dimension, find the valid range
            scale_valid_masks = []
            shape_valid_masks = []
            
            # Process scale_params: find valid range for each dimension
            for dim_idx in range(scale_params_array.shape[1]):
                dim_values = scale_params_array[:, dim_idx]
                lower_bound = np.percentile(dim_values, lower_percentile)
                upper_bound = np.percentile(dim_values, upper_percentile)
                # Keep values within bounds
                valid_mask = (dim_values >= lower_bound) & (dim_values <= upper_bound)
                scale_valid_masks.append(valid_mask)
            
            # Process shape_params: find valid range for each dimension
            for dim_idx in range(shape_params_array.shape[1]):
                dim_values = shape_params_array[:, dim_idx]
                lower_bound = np.percentile(dim_values, lower_percentile)
                upper_bound = np.percentile(dim_values, upper_percentile)
                # Keep values within bounds
                valid_mask = (dim_values >= lower_bound) & (dim_values <= upper_bound)
                shape_valid_masks.append(valid_mask)
            
            # Combine masks: keep frame only if ALL dimensions are valid for both scale and shape
            scale_valid_mask = np.all(scale_valid_masks, axis=0) if scale_valid_masks else np.ones(len(body_params_list), dtype=bool)
            shape_valid_mask = np.all(shape_valid_masks, axis=0) if shape_valid_masks else np.ones(len(body_params_list), dtype=bool)
            
            # Keep frames that are valid for both scale and shape
            combined_mask = scale_valid_mask & shape_valid_mask
            
            if np.sum(combined_mask) == 0:
                # If all frames are outliers, use all frames (fallback)
                print(f"  Warning: All frames were outliers, using all frames for averaging")
                combined_mask = np.ones(len(body_params_list), dtype=bool)
            
            scale_params_array = scale_params_array[combined_mask]
            shape_params_array = shape_params_array[combined_mask]
            
            print(f"  Removed {len(body_params_list) - len(scale_params_array)} outliers ({outlier_percent}% removal), keeping {len(scale_params_array)} frames")
        else:
            print(f"  No outlier removal (outlier_percent={outlier_percent} or too few frames)")
        
        # Average the remaining parameters
        scale_params_avg = np.mean(scale_params_array, axis=0)
        shape_params_avg = np.mean(shape_params_array, axis=0)
        
        return scale_params_avg, shape_params_avg
    
    def process_from_estimation_json(self, estimation_json_path: str, profile_name: str,
                                    use_root_motion: bool = True, fps: float = 30.0,
                                    refinement_config=None, progress_callback: Optional[callable] = None,
                                    lod: int = -1, use_personalized_body: bool = False,
                                    outlier_removal_percent: float = 10.0,
                                    lang: str = DEFAULT_LANGUAGE,
                                    extrinsics_file: Optional[str] = None,
                                    extrinsics_sample_rate: int = 0,
                                    extrinsics_scale: float = 0.0) -> ProcessResult:
        """
        Process from saved estimation JSON file.
        
        Args:
            estimation_json_path: Path to estimation JSON file
            profile_name: Profile name
            use_root_motion: Whether to use root motion
            fps: Frame rate
            refinement_config: Optional RefinementConfig to apply before joint mapping
            progress_callback: Optional callback function(progress, description)
            lod: Level of Detail for mesh (-1 = no mesh, 0-6 = LOD level)
            use_personalized_body: Whether to generate personalized body mesh from estimation data
            outlier_removal_percent: Percentage of outliers to remove from each tail (0-50)
            
        Returns:
            ProcessResult containing all processing results
        """
        # Load estimation results (returns tuple of frames and metadata)
        estimation_results, metadata = self.estimation_manager.load_estimation_results(estimation_json_path)

        # Apply extrinsics before refinement if provided
        if extrinsics_file:
            estimation_results = self.data_prep_manager.apply_extrinsics_to_estimation(
                estimation_results,
                extrinsics_file=extrinsics_file,
                sample_rate=extrinsics_sample_rate,
                extrinsics_scale=extrinsics_scale
            )
        
        # Check for version mismatch and show warning if present
        if metadata.get("_version_mismatch", False):
            file_version = metadata.get("_file_version", "unknown")
            from fbxify import VERSION
            warning_msg = (
                f"⚠️ Version Mismatch: File version ({file_version}) differs from current version ({VERSION}). "
                f"This may cause errors if formats are incompatible."
            )
            print(warning_msg)
            # Note: We can't use gr.Warning here since we're not in a Gradio context
            # The error will be caught and shown if validation fails
        
        # Apply refinement to estimation results if enabled (before joint mapping)
        translator = Translator(lang)
        print(f"FbxifyManager.process_from_estimation_json(): refinement_config is {'None' if refinement_config is None else 'not None'}")
        if refinement_config is not None:
            print(f"FbxifyManager.process_from_estimation_json(): Creating RefinementManager with config")
            if progress_callback:
                progress_callback(0.0, translator.t("progress.applying_refinement"))
            
            refinement_manager = RefinementManager(refinement_config, fps, lang=lang)
            estimation_results = refinement_manager.apply(
                estimation_results,
                progress_callback=progress_callback
            )
        else:
            print(f"FbxifyManager.process_from_estimation_json(): refinement_config is None, skipping refinement")
        
        # Transform estimation results into FBX-ready data
        if progress_callback:
            base_progress = 0.5 if refinement_config is not None else 0.3
            progress_callback(base_progress, translator.t("progress.preparing_fbx_data"))
        
        fbx_data = self.data_prep_manager.prepare_from_estimation(
            estimation_results,
            profile_name,
            use_root_motion=use_root_motion,
            metadata=metadata
        )
        
        # Extract frame paths from estimation results (we don't have original paths, so use frame indices)
        num_frames = len(estimation_results)
        frame_paths = [f"frame_{i:06d}" for i in range(num_frames)]
        
        # Generate mesh if enabled and personalized body is requested
        mesh_obj_path = None
        if lod >= 0 and profile_name == "mhr" and use_personalized_body:
            if progress_callback:
                progress_callback(0.9, translator.t("progress.preparing_fbx_data") + " - Generating mesh...")
            
            # Collect body parameters from estimation results
            body_params_collection = {}  # person_id -> list of {scale_params, shape_params}
            
            for frame_data in estimation_results.values():
                for person_id, estimation_data in frame_data.items():
                    if person_id not in body_params_collection:
                        body_params_collection[person_id] = []
                    
                    scale_params = estimation_data.get("scale_params")
                    shape_params = estimation_data.get("shape_params")
                    if scale_params is not None and shape_params is not None:
                        # Convert to numpy arrays if they're lists (from JSON deserialization)
                        if isinstance(scale_params, list):
                            scale_params = np.array(scale_params)
                        if isinstance(shape_params, list):
                            shape_params = np.array(shape_params)
                        
                        # Ensure they're 1D arrays
                        if scale_params.ndim > 1:
                            scale_params = scale_params.flatten()
                        if shape_params.ndim > 1:
                            shape_params = shape_params.flatten()
                        
                        body_params_collection[person_id].append({
                            "scale_params": scale_params,
                            "shape_params": shape_params
                        })
            
            # Process mesh for each person
            for identifier, params_list in body_params_collection.items():
                if not params_list:
                    continue
                
                # Remove outliers and average
                try:
                    scale_params_avg, shape_params_avg = self._remove_outliers_and_average(
                        params_list, outlier_removal_percent
                    )
                    
                    # Log parameter statistics
                    print(f"Generating T-pose mesh for person {identifier}...")
                    print(f"  Processed {len(params_list)} frames with {outlier_removal_percent}% outlier removal")
                    print(f"  Scale params shape: {scale_params_avg.shape}, mean: {np.mean(scale_params_avg):.4f}, std: {np.std(scale_params_avg):.4f}")
                    print(f"  Shape params shape: {shape_params_avg.shape}, mean: {np.mean(shape_params_avg):.4f}, std: {np.std(shape_params_avg):.4f}")
                    
                    # Generate T-pose mesh
                    temp_obj = tempfile.NamedTemporaryFile(delete=False, suffix='.obj')
                    temp_obj.close()
                    
                    _, mesh_path = self.estimation_manager.generate_tpose(
                        scale_params=scale_params_avg,
                        shape_params=shape_params_avg,
                        output_obj_path=temp_obj.name,
                        lod=lod
                    )
                    mesh_obj_path = mesh_path
                    print(f"  Generated mesh saved to: {mesh_obj_path}")
                    
                    # Only process first person for now (can be extended later)
                    break
                except Exception as e:
                    print(f"  Warning: Failed to generate mesh: {e}")
                    import traceback
                    traceback.print_exc()
                    mesh_obj_path = None
        
        if progress_callback:
            progress_callback(1.0, translator.t("progress.preprocessing_complete"))
        
        return ProcessResult(
            joint_to_bone_mappings=fbx_data["joint_to_bone_mappings"],
            root_motions=fbx_data["root_motions"],
            visualization_data=None,
            frame_paths=frame_paths,
            profile_name=profile_name,
            fps=fps,
            mesh_obj_path=mesh_obj_path
        )
        
    def export_fbx_files(self, profile_name: str, joint_to_bone_mappings: Dict[str, Any],
                         root_motions: Optional[Dict[str, List[Dict]]], frame_paths: List[str],
                         fps: float = 30.0, progress_callback: Optional[callable] = None,
                         lod: int = -1, mesh_obj_path: Optional[str] = None,
                         lod_fbx_path: Optional[str] = None, lang: str = DEFAULT_LANGUAGE) -> List[str]:
        """
        Export FBX files for each person.
        
        Args:
            profile_name: Profile name
            joint_to_bone_mappings: Dictionary mapping person ID to joint-to-bone mapping
            root_motions: Dictionary mapping person ID to root motion data
            frame_paths: List of frame paths (for metadata)
            fps: Frame rate of the animation (frames per second)
            progress_callback: Optional callback function(progress, description)
            lod: Level of Detail for mesh (-1 = no mesh, 0-6 = LOD level)
            mesh_obj_path: Path to generated mesh OBJ file
            lod_fbx_path: Path to LOD FBX file
            
        Returns:
            List of exported FBX file paths
        """
        fbx_paths = []
        num_keyframes = len(frame_paths)
        num_people = len(joint_to_bone_mappings.keys())
        
        for person_index, identifier in enumerate(joint_to_bone_mappings.keys()):
            root_motion = root_motions[identifier] if root_motions else None
            
            # Create a progress callback for this person's export
            # Weight the progress based on which person we're processing
            person_progress_callback = None
            if progress_callback:
                def make_person_callback(person_idx, total_people, main_callback):
                    # Each person gets an equal slice of progress
                    person_start = person_idx / total_people
                    person_end = (person_idx + 1) / total_people
                    person_range = person_end - person_start
                    
                    def inner_callback(normalized_progress, description):
                        # normalized_progress is 0.0 to 1.0 for this person's export
                        # Map it to this person's slice of overall progress
                        weighted_progress = person_start + (normalized_progress * person_range)
                        main_callback(weighted_progress, description)
                    return inner_callback
                person_progress_callback = make_person_callback(person_index, num_people, progress_callback)
            
            fbx_path = export_to_fbx(
                self.data_prep_manager.create_metadata(profile_name, identifier, num_keyframes=num_keyframes, fps=fps),
                joint_to_bone_mappings[identifier],
                root_motion,
                self.data_prep_manager.get_armature_rest_pose(profile_name),
                self.estimation_manager.faces,
                mesh_obj_path=mesh_obj_path if lod >= 0 and profile_name == "mhr" else None,
                lod_fbx_path=lod_fbx_path if lod >= 0 and profile_name == "mhr" else None,
                progress_callback=person_progress_callback,
                lang=lang
            )
            
            if fbx_path is not None:
                fbx_paths.append(fbx_path)

        return fbx_paths
    