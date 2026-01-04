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
from tools.vis_utils import visualize_sample_together


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
    
    def process_frames(self, frame_paths: List[str], profile_name: str, num_people: int,
                      bbox_dict: Optional[Dict[int, List[Tuple]]] = None,
                      use_root_motion: bool = True, create_visualization: bool = False,
                      fps: float = 30.0, progress_callback: Optional[callable] = None,
                      lod: int = -1, body_param_sample_num: int = 5,
                      save_estimation_json: Optional[str] = None,
                      refinement_config=None) -> ProcessResult:
        """
        Process all frames and collect results.
        
        Args:
            frame_paths: List of frame file paths
            profile_name: Profile name (e.g., "mixamo", "unity", "mhr")
            num_people: Number of people to detect
            bbox_dict: Optional bounding box dictionary
            use_root_motion: Whether to track root motion
            create_visualization: Whether to create visualization data
            fps: Frame rate of the animation (frames per second)
            progress_callback: Optional callback function(progress, description)
            lod: Level of Detail for mesh (-1 = no mesh, 0-6 = LOD level)
            body_param_sample_num: Number of frames to sample for body parameter averaging
            save_estimation_json: Optional path to save estimation results JSON
            
        Returns:
            ProcessResult containing all processing results
        """
        # Extract source name from frame paths
        # For single images, use the image filename
        # For extracted frames, try to infer from directory structure
        source_name = "unknown"
        if frame_paths:
            first_frame = frame_paths[0]
            if os.path.isfile(first_frame):
                # Check if it's a single image (not in a temp frame directory)
                frame_dir = os.path.dirname(first_frame)
                # If frame is in a temp directory (contains "sam3d_frames"), try to infer original source
                if "sam3d_frames" in frame_dir:
                    # For extracted frames, we can't easily get the original filename
                    # Use a generic name based on the temp directory
                    source_name = "extracted_video"
                else:
                    # Single image file - use its name
                    source_name = os.path.basename(first_frame)
            else:
                source_name = "unknown"
        
        # Step 1: Estimate poses for all frames (expensive operation)
        def estimation_progress(progress_value, description):
            if progress_callback:
                progress_callback(progress_value * 0.5, description)
        
        estimation_results = self.estimation_manager.estimate_all_frames(
            frame_paths,
            num_people=num_people,
            bbox_dict=bbox_dict,
            progress_callback=estimation_progress,
            source_name=source_name
        )
        
        # Save estimation results if requested
        if save_estimation_json:
            self.estimation_manager.save_estimation_results(estimation_results, save_estimation_json, source_name=source_name)
        
        # Step 2: Apply refinement to estimation results if enabled (before joint mapping)
        if refinement_config is not None:
            if progress_callback:
                progress_callback(0.5, "Applying refinement to estimation results...")
            
            refinement_manager = RefinementManager(refinement_config, fps)
            estimation_results = refinement_manager.apply(
                estimation_results,
                progress_callback=lambda p, d: progress_callback(0.5 + p * 0.1, d) if progress_callback else None
            )
        
        # Step 3: Prepare visualization data if needed
        visualization_data = [] if create_visualization else None
        if create_visualization:
            for frame_index, frame_path in enumerate(frame_paths):
                frame_key = str(frame_index)
                if frame_key in estimation_results:
                    frame_data = estimation_results[frame_key]
                    # Convert estimation data back to list format for visualization
                    outputs_list = []
                    for person_id, estimation_data in frame_data.items():
                        # Convert lists back to numpy arrays for visualization
                        output = {}
                        for key, value in estimation_data.items():
                            if isinstance(value, list):
                                output[key] = np.array(value)
                            else:
                                output[key] = value
                        outputs_list.append(output)
                    
                    visualization_data.append({
                        "image": cv2.imread(frame_path),
                        "outputs": outputs_list
                    })
        
        # Step 4: Transform estimation results into FBX-ready data
        if progress_callback:
            base_progress = 0.6 if refinement_config is not None else 0.5
            progress_callback(base_progress, "Preparing FBX data...")
        
        fbx_data = self.data_prep_manager.prepare_from_estimation(
            estimation_results,
            profile_name,
            use_root_motion=use_root_motion
        )
        
        joint_to_bone_mappings = fbx_data["joint_to_bone_mappings"]
        root_motions = fbx_data["root_motions"]
        
        # Step 4: Generate mesh if enabled
        mesh_obj_path = None
        if lod >= 0 and profile_name == "mhr":
            # Collect body parameters from estimation results
            body_params_collection = {}  # person_id -> list of {scale_params, shape_params}
            
            for frame_data in estimation_results.values():
                for person_id, estimation_data in frame_data.items():
                    if person_id not in body_params_collection:
                        body_params_collection[person_id] = []
                    
                    scale_params = estimation_data.get("scale_params")
                    shape_params = estimation_data.get("shape_params")
                    if scale_params is not None and shape_params is not None:
                        body_params_collection[person_id].append({
                            "scale_params": scale_params,
                            "shape_params": shape_params
                        })
            
            # Process mesh for each person
            for identifier, params_list in body_params_collection.items():
                if not params_list:
                    continue
                
                # Sample N frames (min of body_param_sample_num and available frames)
                num_samples = min(body_param_sample_num, len(params_list))
                sampled_params = params_list[:num_samples]  # Take first N samples
                
                # Average scale_params and shape_params
                scale_params_list = [p["scale_params"] for p in sampled_params]
                shape_params_list = [p["shape_params"] for p in sampled_params]
                
                # Convert to numpy arrays and average
                scale_params_avg = np.mean([np.array(sp) for sp in scale_params_list], axis=0)
                shape_params_avg = np.mean([np.array(sp) for sp in shape_params_list], axis=0)
                
                # Log parameter statistics
                print(f"Generating T-pose mesh for person {identifier}...")
                print(f"  Averaged {num_samples} samples out of {len(params_list)} available frames")
                
                # Generate T-pose mesh
                temp_obj = tempfile.NamedTemporaryFile(delete=False, suffix='.obj')
                temp_obj.close()
                
                try:
                    _, mesh_path = self.estimation_manager.generate_tpose(
                        scale_params=scale_params_avg,
                        shape_params=shape_params_avg,
                        output_obj_path=temp_obj.name
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
        
        # Full GPU cache clear at the end
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        return ProcessResult(
            joint_to_bone_mappings=joint_to_bone_mappings,
            root_motions=root_motions,
            visualization_data=visualization_data,
            frame_paths=frame_paths,
            profile_name=profile_name,
            fps=fps,
            mesh_obj_path=mesh_obj_path
        )
    
    def process_from_estimation_json(self, estimation_json_path: str, profile_name: str,
                                    use_root_motion: bool = True, fps: float = 30.0,
                                    refinement_config=None, progress_callback: Optional[callable] = None) -> ProcessResult:
        """
        Process from saved estimation JSON file.
        
        Args:
            estimation_json_path: Path to estimation JSON file
            profile_name: Profile name
            use_root_motion: Whether to use root motion
            fps: Frame rate
            refinement_config: Optional RefinementConfig to apply before joint mapping
            progress_callback: Optional callback function(progress, description)
            
        Returns:
            ProcessResult containing all processing results
        """
        # Load estimation results (returns tuple of frames and metadata)
        estimation_results, metadata = self.estimation_manager.load_estimation_results(estimation_json_path)
        
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
        if refinement_config is not None:
            if progress_callback:
                progress_callback(0.0, "Applying refinement to estimation results...")
            
            refinement_manager = RefinementManager(refinement_config, fps)
            estimation_results = refinement_manager.apply(
                estimation_results,
                progress_callback=progress_callback
            )
        
        # Transform estimation results into FBX-ready data
        if progress_callback:
            progress_callback(0.5 if refinement_config is not None else 0.0, "Preparing FBX data...")
        
        fbx_data = self.data_prep_manager.prepare_from_estimation(
            estimation_results,
            profile_name,
            use_root_motion=use_root_motion
        )
        
        # Extract frame paths from estimation results (we don't have original paths, so use frame indices)
        num_frames = len(estimation_results)
        frame_paths = [f"frame_{i:06d}" for i in range(num_frames)]
        
        if progress_callback:
            progress_callback(1.0, "Processing complete")
        
        return ProcessResult(
            joint_to_bone_mappings=fbx_data["joint_to_bone_mappings"],
            root_motions=fbx_data["root_motions"],
            visualization_data=None,  # Not available from JSON
            frame_paths=frame_paths,
            profile_name=profile_name,
            fps=fps,
            mesh_obj_path=None
        )
    
    def _convert_translation_to_list(self, trans_val) -> List[float]:
        """Convert translation value to list format."""
        if hasattr(trans_val, 'tolist'):
            return trans_val.tolist()
        if isinstance(trans_val, (list, tuple)):
            return list(trans_val)
        return [0.0, 0.0, 0.0]
    
    def _is_valid_3x3_matrix(self, matrix) -> bool:
        """Check if matrix is a valid 3x3 nested list."""
        if not isinstance(matrix, list) or len(matrix) != 3:
            return False
        if not isinstance(matrix[0], (list, tuple)) or len(matrix[0]) != 3:
            return False
        return True
    
    def _convert_rotation_to_list(self, rot_val) -> List[List[float]]:
        """Convert rotation value to 3x3 list format."""
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        
        if hasattr(rot_val, 'tolist'):
            rot_list = rot_val.tolist()
            if self._is_valid_3x3_matrix(rot_list):
                return rot_list
            return identity
        
        if isinstance(rot_val, list):
            if self._is_valid_3x3_matrix(rot_val):
                return rot_val
            return identity
        
        return identity
    
    def _convert_root_motion_to_refinement_format(self, root_motion_list: List[Dict]) -> Dict:
        """
        Convert root_motion from list format to refinement format.
        
        Input: [{"global_rot": [...], "pred_cam_t": [...]}, ...]
        Output: {"translation": [[...], ...], "rotation": [[[...]], ...]}
        """
        translation = []
        rotation = []
        
        for frame_data in root_motion_list:
            trans_val = frame_data.get("pred_cam_t")
            translation.append(self._convert_translation_to_list(trans_val) if trans_val is not None else [0.0, 0.0, 0.0])
            
            rot_val = frame_data.get("global_rot")
            rotation.append(self._convert_rotation_to_list(rot_val) if rot_val is not None else 
                          [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        
        return {
            "translation": translation,
            "rotation": rotation
        }
    
    def _convert_root_motion_from_refinement_format(self, refined_root_motion: Dict, num_frames: int) -> List[Dict]:
        """
        Convert root_motion from refinement format back to list format.
        
        Input: {"translation": [[...], ...], "rotation": [[[...]], ...]}
        Output: [{"global_rot": [...], "pred_cam_t": [...]}, ...]
        """
        refined_list = []
        for t in range(num_frames):
            refined_list.append({
                "global_rot": refined_root_motion["rotation"][t],
                "pred_cam_t": refined_root_motion["translation"][t]
            })
        return refined_list
        
    def export_fbx_files(self, profile_name: str, joint_to_bone_mappings: Dict[str, Any],
                         root_motions: Optional[Dict[str, List[Dict]]], frame_paths: List[str],
                         fps: float = 30.0, progress_callback: Optional[callable] = None,
                         lod: int = -1, mesh_obj_path: Optional[str] = None,
                         lod_fbx_path: Optional[str] = None) -> List[str]:
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
        
        for person_index, identifier in enumerate(joint_to_bone_mappings.keys()):
            root_motion = root_motions[identifier] if root_motions else None
            
            fbx_path = export_to_fbx(
                self.data_prep_manager.create_metadata(profile_name, identifier, num_keyframes=num_keyframes, fps=fps),
                joint_to_bone_mappings[identifier],
                root_motion,
                self.data_prep_manager.get_armature_rest_pose(profile_name),
                self.estimation_manager.faces,
                mesh_obj_path=mesh_obj_path if lod >= 0 and profile_name == "mhr" else None,
                lod_fbx_path=lod_fbx_path if lod >= 0 and profile_name == "mhr" else None
            )
            
            if fbx_path is not None:
                fbx_paths.append(fbx_path)

            if progress_callback:
                progress = (person_index + 1) / len(joint_to_bone_mappings.keys())
                progress_callback(progress, f"Processing person {person_index + 1}")

        return fbx_paths
    
    def export_visualization(self, visualization_data: List[Dict], fps: int = 30,
                            progress_callback: Optional[callable] = None) -> Optional[str]:
        """
        Create a video from visualization frames using ffmpeg.
        
        Args:
            visualization_data: List of dicts with "image" and "outputs" keys
            fps: Frame rate for output video
            progress_callback: Optional callback function(progress, description)
            
        Returns:
            Path to output video file, or None if no frames
        """
        if not visualization_data:
            return None
            
        # Create temp directory for frames
        temp_dir = tempfile.mkdtemp(prefix="sam3d_vis_frames_")
        
        try:
            if progress_callback:
                progress_callback(0.0, "Creating visualization frames")
            
            # Save all visualization frames
            frame_paths = []
            for idx, vis_data in enumerate(visualization_data):
                rend_img = visualize_sample_together(vis_data["image"], vis_data["outputs"], self.estimation_manager.faces)
                frame_path = os.path.join(temp_dir, f"frame_{idx:06d}.jpg")
                cv2.imwrite(frame_path, rend_img.astype(np.uint8))
                frame_paths.append(frame_path)
            
            if not frame_paths:
                return None

            if len(frame_paths) == 1:
                # Copy the single frame to a persistent temp file before cleanup
                output_image = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                output_image.close()
                shutil.copy2(frame_paths[0], output_image.name)
                return output_image.name
            
            # Get frame dimensions from first frame
            first_frame = cv2.imread(frame_paths[0])
            height, width = first_frame.shape[:2]
            
            # Create output video path
            output_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            output_video.close()

            # Use ffmpeg to create video
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.jpg'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                output_video.name
            ]

            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg error: {e.stderr.decode()}")
                raise
            except FileNotFoundError:
                raise RuntimeError("FFmpeg not found. Please install FFmpeg to create visualization videos.")

            if progress_callback:
                progress_callback(1.0, "Visualization created")
            
            return output_video.name
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
