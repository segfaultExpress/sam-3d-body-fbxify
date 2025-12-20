"""
FbxifyManager - Core business logic for processing videos/images and exporting FBX files.
"""
import os
import cv2
import tempfile
import shutil
import subprocess
import json
import time
import copy
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from fbxify.pose_estimator import PoseEstimator
from fbxify.utils import export_to_fbx, to_serializable
from fbxify.refinement.refinement_manager import RefinementManager
from tools.vis_utils import visualize_sample_together
import numpy as np


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


@dataclass
class DebugData:
    """Debug data structure for save/load operations."""
    joint_to_bone_mappings: Dict[str, Any]
    root_motions: Dict[str, List[Dict]]
    visualization_data: Optional[List[Dict]]
    frame_paths: List[str]
    profile_name: str
    fps: float


class FbxifyManager:
    """
    Manager class for processing videos/images and exporting FBX files.
    """
    
    def __init__(self, estimator: PoseEstimator, debug_results_dir: str = "/workspace/temp/sam3d_debug_results"):
        """
        Initialize the manager with a PoseEstimator instance.
        
        Args:
            estimator: PoseEstimator instance (dependency injection)
            debug_results_dir: Directory for saving debug results
        """
        self.estimator = estimator
        self.debug_results_dir = debug_results_dir
        os.makedirs(self.debug_results_dir, exist_ok=True)
    
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
            self.estimator.cache_cam_int_from_file(fov_file_path)
        elif fov_method == "Sample":
            self.estimator.cache_cam_int_from_images(frame_paths, average_of=sample_number)
        # else: "Default" - built-in FOV estimator runs every frame (default behavior)
    
    def process_frames(self, frame_paths: List[str], profile_name: str, num_people: int,
                      bbox_dict: Optional[Dict[int, List[Tuple]]] = None,
                      use_root_motion: bool = True, create_visualization: bool = False,
                      fps: float = 30.0, progress_callback: Optional[callable] = None,
                      lod: int = -1, body_param_sample_num: int = 5) -> ProcessResult:
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
            
        Returns:
            ProcessResult containing all processing results
        """
        joint_to_bone_mappings = {}
        root_motions = {} if use_root_motion else None
        
        # Store outputs and images for visualization if needed
        visualization_data = [] if create_visualization else None
        
        # Collect body parameters for mesh generation (if enabled)
        body_params_collection = {}  # person_id -> list of {scale_params, shape_params}
        
        # Process each frame
        for keyframe_index, frame_path in enumerate(frame_paths):
            print(f"Processing frame {keyframe_index + 1} of {len(frame_paths)}")
            # proper MOT bbox formatting starts at 1, offset keyframe index by 1
            bboxes = bbox_dict[keyframe_index + 1] if bbox_dict is not None and keyframe_index + 1 in bbox_dict else None
            
            if progress_callback:
                progress_callback((keyframe_index * 0.5) / len(frame_paths), f"Processing frame {keyframe_index + 1} of {len(frame_paths)}")
            else:
                print(f"Processing frame {keyframe_index + 1} of {len(frame_paths)}")

            try:
                results, raw_outputs = self.estimator.process_single_frame(
                    profile_name,
                    frame_path,
                    keyframe_index,
                    joint_to_bone_mappings=joint_to_bone_mappings,
                    root_motions=root_motions,
                    num_people=num_people,
                    bboxes=bboxes
                )

                if create_visualization:
                    visualization_data.append({
                        "image": cv2.imread(frame_path),
                        "outputs": raw_outputs
                    })
                
                # Only process results if any people were detected
                if results and len(results) > 0:
                    for identifier in results.keys():
                        result = results[identifier]
                        joint_to_bone_mappings[identifier] = result["joint_to_bone_mapping"]
                        
                        # Collect body parameters for mesh generation if enabled
                        if lod >= 0 and profile_name == "mhr":
                            if identifier not in body_params_collection:
                                body_params_collection[identifier] = []
                            
                            scale_params = result.get("scale_params")
                            shape_params = result.get("shape_params")
                            if scale_params is not None and shape_params is not None:
                                body_params_collection[identifier].append({
                                    "scale_params": scale_params,
                                    "shape_params": shape_params
                                })
                else:
                    print(f"  Warning: No people detected in frame {keyframe_index + 1}, skipping...")
                
                # Lightweight cleanup per frame: just delete references
                del results, raw_outputs
            except ValueError as e:
                # Handle "No people detected" or similar errors gracefully
                if "No people detected" in str(e) or "person_index" in str(e):
                    print(f"  Warning: {str(e)} in frame {keyframe_index + 1}, skipping...")
                    continue
                else:
                    # Re-raise other ValueError exceptions
                    raise

        # Full GPU cache clear at the end of all frames (for video processing)
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        # Generate mesh if enabled
        mesh_obj_path = None  # Initialize to None
        if lod >= 0 and profile_name == "mhr" and body_params_collection:
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
                
                # Log parameter statistics to check for zeros
                print(f"Generating T-pose mesh for person {identifier}...")
                print(f"  Averaged {num_samples} samples out of {len(params_list)} available frames")
                print(f"  Scale params stats: shape={scale_params_avg.shape}, min={np.min(scale_params_avg):.6f}, max={np.max(scale_params_avg):.6f}, mean={np.mean(scale_params_avg):.6f}, non-zero count={np.count_nonzero(scale_params_avg)}/{len(scale_params_avg)}")
                print(f"  Shape params stats: shape={shape_params_avg.shape}, min={np.min(shape_params_avg):.6f}, max={np.max(shape_params_avg):.6f}, mean={np.mean(shape_params_avg):.6f}, non-zero count={np.count_nonzero(shape_params_avg)}/{len(shape_params_avg)}")
                
                # Check if all zeros
                if np.allclose(scale_params_avg, 0) and np.allclose(shape_params_avg, 0):
                    print("  WARNING: Both scale_params and shape_params are all zeros!")
                elif np.allclose(scale_params_avg, 0):
                    print("  WARNING: scale_params are all zeros!")
                elif np.allclose(shape_params_avg, 0):
                    print("  WARNING: shape_params are all zeros!")
                
                # Generate T-pose mesh
                import tempfile
                import os
                temp_obj = tempfile.NamedTemporaryFile(delete=False, suffix='.obj')
                temp_obj.close()
                
                try:
                    _, mesh_path = self.estimator.generate_tpose(
                        scale_params=scale_params_avg,
                        shape_params=shape_params_avg,
                        output_obj_path=temp_obj.name
                    )
                    mesh_obj_path = mesh_path
                    print(f"  Generated mesh saved to: {mesh_obj_path}")
                    
                    # Also save to project root for inspection
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    root_obj_path = os.path.join(project_root, f"/workspace/tpose_person_{identifier}.obj")
                    import shutil
                    shutil.copyfile(mesh_path, root_obj_path)
                    print(f"  Also saved to project root: {root_obj_path}")
                    
                    # Clean up GPU memory after mesh generation
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Only process first person for now (can be extended later)
                    break
                except Exception as e:
                    print(f"  Warning: Failed to generate mesh: {e}")
                    import traceback
                    traceback.print_exc()
                    mesh_obj_path = None  # Ensure it's None on error
                    
                    # Clean up GPU memory even on error
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        return ProcessResult(
            joint_to_bone_mappings=joint_to_bone_mappings,
            root_motions=root_motions,
            visualization_data=visualization_data,
            frame_paths=frame_paths,
            profile_name=profile_name,
            fps=fps,
            mesh_obj_path=mesh_obj_path
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
    
    def apply_refinement(self, process_result: ProcessResult, refinement_config, 
                        progress_callback: Optional[callable] = None) -> ProcessResult:
        """
        Apply refinement to joint_to_bone_mappings and root_motions.
        
        Args:
            process_result: ProcessResult to refine
            refinement_config: RefinementConfig instance
            progress_callback: Optional callback function(progress, description)
            
        Returns:
            ProcessResult with refined data
        """
        if refinement_config is None:
            return process_result
        
        if progress_callback:
            progress_callback(0.0, "Applying refinement...")
        
        refinement_manager = RefinementManager(refinement_config, process_result.fps)
        
        # Apply refinement to each person's joint_to_bone_mapping
        refined_mappings = {}
        for person_index, (identifier, joint_mapping) in enumerate(process_result.joint_to_bone_mappings.items()):
            # Deep copy to avoid modifying original
            refined_mapping = copy.deepcopy(joint_mapping)
            
            # Apply refinement recursively
            refined_mapping, _ = refinement_manager.apply(refined_mapping, root_motion=None)
            refined_mappings[identifier] = refined_mapping
            
            if progress_callback:
                progress = (person_index + 1) / len(process_result.joint_to_bone_mappings)
                progress_callback(progress * 0.5, f"Refining person {person_index + 1}")
        
        # Apply refinement to root motions if present
        refined_root_motions = None
        if process_result.root_motions is not None:
            refined_root_motions = {}
            for person_index, (identifier, root_motion_list) in enumerate(process_result.root_motions.items()):
                root_motion_dict = self._convert_root_motion_to_refinement_format(root_motion_list)
                _, refined_root_motion = refinement_manager.apply(None, root_motion_dict)
                refined_list = self._convert_root_motion_from_refinement_format(
                    refined_root_motion, len(root_motion_list)
                )
                refined_root_motions[identifier] = refined_list
                
                if progress_callback:
                    progress = 0.5 + ((person_index + 1) / len(process_result.root_motions)) * 0.5
                    progress_callback(progress, f"Refining root motion for person {person_index + 1}")
        
        if progress_callback:
            progress_callback(1.0, "Refinement complete")
        
        # Return new ProcessResult with refined data
        return ProcessResult(
            joint_to_bone_mappings=refined_mappings,
            root_motions=refined_root_motions,
            visualization_data=process_result.visualization_data,
            frame_paths=process_result.frame_paths,
            profile_name=process_result.profile_name,
            fps=process_result.fps,
            mesh_obj_path=process_result.mesh_obj_path  # Preserve mesh_obj_path
        )
    
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
                self.estimator.create_metadata(profile_name, identifier, num_keyframes=num_keyframes, fps=fps),
                joint_to_bone_mappings[identifier],
                root_motion,
                self.estimator.get_armature_rest_pose(profile_name),
                self.estimator.faces,
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
                rend_img = visualize_sample_together(vis_data["image"], vis_data["outputs"], self.estimator.faces)
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
    
    def save_debug_results(self, process_result: ProcessResult) -> int:
        """
        Save processing results to debug directory.
        
        Args:
            process_result: ProcessResult to save
            
        Returns:
            Timestamp of saved results
        """
        timestamp = int(time.time())
        timestamp_dir = os.path.join(self.debug_results_dir, str(timestamp))
        os.makedirs(timestamp_dir, exist_ok=True)
        
        # Save the data files
        with open(os.path.join(timestamp_dir, "joint_to_bone_mappings.json"), "w") as f:
            json.dump(process_result.joint_to_bone_mappings, f, indent=4)
        
        if process_result.root_motions is not None:
            with open(os.path.join(timestamp_dir, "root_motions.json"), "w") as f:
                json.dump(to_serializable(process_result.root_motions), f, indent=4)
        
        # Save visualization data
        if process_result.visualization_data is not None:
            serializable_vis_data = []
            for idx, item in enumerate(process_result.visualization_data):
                # Save images as files
                img_path = os.path.join(timestamp_dir, f"vis_image_{idx:06d}.jpg")
                cv2.imwrite(img_path, item["image"])
                
                # Save outputs to separate file (convert numpy arrays to lists)
                outputs_path = os.path.join(timestamp_dir, f"vis_outputs_{idx:06d}.json")
                with open(outputs_path, "w") as f:
                    json.dump(to_serializable(item["outputs"]), f, indent=4)
                
                serializable_vis_data.append({
                    "image_shape": item["image"].shape,
                    "image_dtype": str(item["image"].dtype),
                })
            
            with open(os.path.join(timestamp_dir, "visualization_data.json"), "w") as f:
                json.dump(serializable_vis_data, f, indent=4)
        
        with open(os.path.join(timestamp_dir, "frame_paths.json"), "w") as f:
            json.dump(process_result.frame_paths, f, indent=4)
        
        # Save metadata.json with profile_name and fps
        metadata = {
            "profile_name": process_result.profile_name,
            "fps": process_result.fps
        }
        with open(os.path.join(timestamp_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        
        # Update the index file
        index_path = os.path.join(self.debug_results_dir, "sam3d_saved_results.json")
        saved_results = []
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                saved_results = json.load(f)
        saved_results.append(timestamp)
        with open(index_path, "w") as f:
            json.dump(saved_results, f, indent=4)
        
        print(f"Debug results saved with timestamp: {timestamp}")
        return timestamp
    
    def get_saved_timestamps(self) -> List[int]:
        """Get list of timestamps from sam3d_saved_results.json."""
        index_path = os.path.join(self.debug_results_dir, "sam3d_saved_results.json")
        if not os.path.exists(index_path):
            return []
        with open(index_path, "r") as f:
            saved_results = json.load(f)
        return saved_results
    
    def clear_saved_results(self):
        """Clear all saved results."""
        if os.path.exists(self.debug_results_dir):
            shutil.rmtree(self.debug_results_dir, ignore_errors=True)
        os.makedirs(self.debug_results_dir, exist_ok=True)
        index_path = os.path.join(self.debug_results_dir, "sam3d_saved_results.json")
        with open(index_path, "w") as f:
            json.dump([], f)
        print("Debug results cleared")
    
    def load_debug_results(self, timestamp: int) -> DebugData:
        """
        Load results from a specific timestamp directory.
        
        Args:
            timestamp: Timestamp of saved results
            
        Returns:
            DebugData containing loaded results
        """
        timestamp_dir = os.path.join(self.debug_results_dir, str(timestamp))
        
        if not os.path.exists(timestamp_dir):
            raise ValueError(f"No saved results found for timestamp: {timestamp}")
        
        with open(os.path.join(timestamp_dir, "joint_to_bone_mappings.json"), "r") as f:
            joint_to_bone_mappings = json.load(f)
        
        with open(os.path.join(timestamp_dir, "root_motions.json"), "r") as f:
            root_motions_raw = json.load(f)
        
        # Convert lists back to numpy arrays in root_motions
        root_motions = {}
        for identifier, motion_list in root_motions_raw.items():
            root_motions[identifier] = []
            for motion in motion_list:
                converted_motion = {}
                for key, value in motion.items():
                    if isinstance(value, list):
                        converted_motion[key] = np.array(value)
                    else:
                        converted_motion[key] = value
                root_motions[identifier].append(converted_motion)
        
        # Load visualization data
        vis_data_path = os.path.join(timestamp_dir, "visualization_data.json")
        visualization_data = None
        if os.path.exists(vis_data_path):
            with open(vis_data_path, "r") as f:
                vis_metadata = json.load(f)
            visualization_data = []
            for idx, meta in enumerate(vis_metadata):
                img_path = os.path.join(timestamp_dir, f"vis_image_{idx:06d}.jpg")
                image = cv2.imread(img_path)
                
                # Load outputs if available
                outputs_path = os.path.join(timestamp_dir, f"vis_outputs_{idx:06d}.json")
                outputs = []
                if os.path.exists(outputs_path):
                    with open(outputs_path, "r") as f:
                        outputs_list = json.load(f)
                    # Convert lists back to numpy arrays
                    for output_dict in outputs_list:
                        output = {}
                        for key, value in output_dict.items():
                            if isinstance(value, list):
                                output[key] = np.array(value)
                            else:
                                output[key] = value
                        outputs.append(output)
                
                visualization_data.append({
                    "image": image,
                    "outputs": outputs
                })
        
        with open(os.path.join(timestamp_dir, "frame_paths.json"), "r") as f:
            frame_paths = json.load(f)
        
        # Load metadata.json (new format) or fall back to profile_name.txt (old format)
        metadata_path = os.path.join(timestamp_dir, "metadata.json")
        profile_name_txt_path = os.path.join(timestamp_dir, "profile_name.txt")
        
        if os.path.exists(metadata_path):
            # New format: load from metadata.json
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            profile_name = metadata.get("profile_name", "unknown")
            fps = metadata.get("fps", 30.0)
        elif os.path.exists(profile_name_txt_path):
            # Old format: load from profile_name.txt, use default FPS
            with open(profile_name_txt_path, "r") as f:
                profile_name = f.read().strip()
            fps = 30.0  # Default for old format
        else:
            # Fallback if neither exists
            profile_name = "unknown"
            fps = 30.0
        
        return DebugData(
            joint_to_bone_mappings=joint_to_bone_mappings,
            root_motions=root_motions,
            visualization_data=visualization_data,
            frame_paths=frame_paths,
            profile_name=profile_name,
            fps=fps
        )
    
    def reexport_debug_results(self, timestamp: int, use_root_motion: bool = True,
                              create_visualization: bool = False,
                              refinement_config = None,
                              progress_callback: Optional[callable] = None) -> List[str]:
        """
        Re-run the export functions to debug blender/visualization issues.
        
        Args:
            timestamp: Timestamp of saved results to re-export
            use_root_motion: Whether to use root motion
            create_visualization: Whether to create visualization
            refinement_config: Optional RefinementConfig to apply before exporting
            progress_callback: Optional callback function(progress, description)
            
        Returns:
            List of output file paths
        """
        if progress_callback:
            progress_callback(0.0, "Re-exporting results")
        
        debug_data = self.load_debug_results(timestamp)
        
        # Convert debug_data to ProcessResult
        process_result = ProcessResult(
            joint_to_bone_mappings=debug_data.joint_to_bone_mappings,
            root_motions=debug_data.root_motions if use_root_motion else None,
            visualization_data=debug_data.visualization_data,
            frame_paths=debug_data.frame_paths,
            profile_name=debug_data.profile_name,
            fps=debug_data.fps
        )
        
        # Apply refinement if enabled
        if refinement_config is not None:
            def refinement_progress(progress_value, description):
                if progress_callback:
                    progress_callback(0.1 + progress_value * 0.2, description)
            
            # Apply refinement (FPS is already in process_result)
            process_result = self.apply_refinement(
                process_result,
                refinement_config,
                refinement_progress
            )
        
        output_files = []
        
        # Export FBX files
        def export_progress(progress_value, description):
            if progress_callback:
                base_progress = 0.3 if refinement_config is not None else 0.1
                progress_callback(base_progress + progress_value * 0.6, description)
        
        # Use FPS from process_result
        fbx_paths = self.export_fbx_files(
            process_result.profile_name,
            process_result.joint_to_bone_mappings,
            process_result.root_motions,
            process_result.frame_paths,
            process_result.fps,
            export_progress
        )
        output_files.extend(fbx_paths)
        
        # Export visualization if requested and available
        if create_visualization and process_result.visualization_data is not None:
            # Check if we have valid outputs in the visualization data
            has_valid_outputs = any(
                len(item.get("outputs", [])) > 0 
                for item in process_result.visualization_data
            )
            if has_valid_outputs:
                def vis_progress(progress_value, description):
                    if progress_callback:
                        progress_callback(0.9 + progress_value * 0.1, description)
                
                visualization_output = self.export_visualization(
                    process_result.visualization_data,
                    progress_callback=vis_progress
                )
                if visualization_output is not None:
                    output_files.append(visualization_output)
            else:
                print("Warning: Visualization data does not contain valid outputs, skipping visualization")
        
        if progress_callback:
            progress_callback(1.0, "Re-export complete")
        
        return output_files if output_files else []
