import os
import argparse
import cv2
import tempfile
import shutil
import numpy as np
import subprocess
from fbxify.pose_estimator import PoseEstimator
from fbxify.utils import export_to_fbx, to_serializable
from fbxify.metadata import PROFILES
from fbxify.i18n import Translator, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
from tools.vis_utils import visualize_sample_together
import gradio as gr
import time
import json
from datetime import datetime

VITH_CHECKPOINT_PATH = "/workspace/checkpoints/sam-3d-body-vith"
DINOV3_CHECKPOINT_PATH = "/workspace/checkpoints/sam-3d-body-dinov3"
DEBUG_RESULTS_DIR = "/workspace/temp/sam3d_debug_results"

def parse_args():
    parser = argparse.ArgumentParser()
    # can choose between vith and dinov3
    parser.add_argument("--model", type=str, default="vith", choices=["vith", "dinov3"])
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    return parser.parse_args()

def create_app(estimator):
    """Create Gradio app"""
    
    # Initialize translator with default language
    translator = Translator(DEFAULT_LANGUAGE)
        
    def extract_frames_from_video(video_path, temp_dir):
        """Extract all frames from MP4 video and save to temp directory."""
        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_count += 1
        
        cap.release()
        return frame_paths
    
    def prepare_video(input_video):
        """Split MP4 video into frames"""
        if input_video is None:
            return None
        
        # Create temp directory for frames
        temp_dir = tempfile.mkdtemp(prefix="sam3d_frames_")
        frame_paths = extract_frames_from_video(input_video.name, temp_dir)
        return frame_paths

    def prepare_bboxes(bbox_file):
        if bbox_file is None:
            return None

        bbox_dict = {}
        with open(bbox_file.name, "r") as f:
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
    
    def set_cam_int(estimator, fov_method, fov_file, frame_paths, sample_number):
        if fov_method == "File":
            if fov_file is None:
                raise ValueError(translator.t("errors.camera_intrinsics_required"))
            estimator.cache_cam_int_from_file(fov_file.name)
        elif fov_method == "Sample":
            estimator.cache_cam_int_from_images(frame_paths, average_of=sample_number)
        # else: "Default" - built-in FOV estimator runs every frame (default behavior)

    def export_results(estimator, profile_name, joint_to_bone_mappings, root_motions, frame_paths, start_progress, end_progress, progress=gr.Progress()):
        
        # Collect all FBX paths for each person
        fbx_paths = []
        num_keyframes = len(frame_paths)
        for person_index, identifier in enumerate(joint_to_bone_mappings.keys()):
            root_motion = root_motions[identifier]
            fbx_path = output({
                "metadata": estimator.create_metadata(profile_name, identifier, num_keyframes=num_keyframes),
                "joint_to_bone_mapping": joint_to_bone_mappings[identifier],
                "root_motion": root_motion,
                "rest_pose": estimator.get_armature_rest_pose(profile_name)
            })
            if fbx_path is not None:
                fbx_paths.append(fbx_path)

            multiplier = (end_progress - start_progress)

            progress(start_progress + (person_index * multiplier) / len(joint_to_bone_mappings.keys()), desc=translator.t("progress.processing_person"))

        return fbx_paths

    def export_visualization(visualization_data, faces, start_progress, end_progress, progress=gr.Progress()):
        """Create a video from visualization frames using ffmpeg."""
        # Create temp directory for frames
        temp_dir = tempfile.mkdtemp(prefix="sam3d_vis_frames_")
        
        try:
            progress(start_progress, desc=translator.t("progress.creating_visualization"))
            # Save all visualization frames
            frame_paths = []
            for idx, vis_data in enumerate(visualization_data):
                rend_img = visualize_sample_together(vis_data["image"], vis_data["outputs"], faces)
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
            # Get frame rate from original video if available, default to 30fps
            fps = 30
            
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

            progress(end_progress, desc=translator.t("progress.creating_visualization"))
            return output_video.name
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise gr.Error(translator.t("errors.error_occurred", error_type=type(e).__name__, error_msg=str(e)))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def process(input_file, profile_name, use_bbox, bbox_file, num_people, fov_method, fov_file, sample_number, use_root_motion, create_visualization, debug_save_results, progress=gr.Progress()):
        """Process image or video file."""

        output_files = []
        try:
            if input_file is None:
                return None, None

            # there must either be a valid number of people (>0) or a bbox file
            if use_bbox and bbox_file is None:
                raise ValueError(translator.t("errors.bbox_file_required"))

            bbox_dict = None
            if use_bbox:
                bbox_dict = prepare_bboxes(bbox_file)
                num_people = len(bbox_dict)
            else:
                if num_people <= 0:
                    raise ValueError(translator.t("errors.num_people_required"))

            file_path = input_file.name
            file_ext = os.path.splitext(file_path)[1].lower()
            is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            if is_video:
                frame_paths = prepare_video(input_file)
            else:
                frame_paths = [input_file.name]

            set_cam_int(estimator, fov_method, fov_file, frame_paths, sample_number)

            joint_to_bone_mappings = {}
            root_motions = {} if use_root_motion else None
            
            # Store outputs and images for visualization if needed
            visualization_data = [] if create_visualization else None
            
            # If bbox_dict exists, get sorted keys to allow starting from 0, 1, or any index
            for keyframe_index, frame_path in enumerate(frame_paths):
                # proper MOT bbox formatting starts at 1, offset keyframe index by 1 (sorry if you like 0-indexing for bboxes)
                bboxes = bbox_dict[keyframe_index + 1] if bbox_dict is not None and keyframe_index + 1 in bbox_dict else None
                print(f"Processing frame {keyframe_index + 1} of {len(frame_paths)}")

                try:
                    results, raw_outputs = estimator.process_single_frame(
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
                    else:
                        print(f"  Warning: No people detected in frame {keyframe_index + 1}, skipping...")
                except ValueError as e:
                    # Handle "No people detected" or similar errors gracefully
                    if "No people detected" in str(e) or "person_index" in str(e):
                        print(f"  Warning: {str(e)} in frame {keyframe_index + 1}, skipping...")
                        continue
                    else:
                        # Re-raise other ValueError exceptions
                        raise

                progress((keyframe_index * 0.5) / len(frame_paths), desc=translator.t("progress.processing_keyframes"))

            if debug_save_results:
                debug_save_latest_results(joint_to_bone_mappings, root_motions, visualization_data, frame_paths, profile_name)
            
            fbx_paths = export_results(estimator, 
                profile_name,
                joint_to_bone_mappings,
                root_motions,
                frame_paths,
                0.5,
                1.0 if create_visualization else 0.9,
                progress
            )
            output_files.extend(fbx_paths)

            if create_visualization:
                visualization_output = export_visualization(visualization_data, estimator.faces, 0.9, 1.0, progress)
                if visualization_output is not None:
                    output_files.append(visualization_output)
        except Exception as e:
            # Catch other exceptions and display them as well
            error_type = type(e).__name__
            error_msg = str(e)
            if error_msg:
                raise gr.Error(translator.t("errors.error_occurred", error_type=error_type, error_msg=error_msg))
            else:
                raise gr.Error(translator.t("errors.error_occurred_no_msg", error_type=error_type))
                
        # Return combined list (or None if empty)
        return output_files if output_files else None

    def debug_save_latest_results(joint_to_bone_mappings, root_motions, visualization_data, frame_paths, profile_name):
        """Save the latest results to the debug directory"""
        # Create debug directory if it doesn't exist
        os.makedirs(DEBUG_RESULTS_DIR, exist_ok=True)
        
        timestamp = int(time.time())
        timestamp_dir = os.path.join(DEBUG_RESULTS_DIR, str(timestamp))
        os.makedirs(timestamp_dir, exist_ok=True)
        
        # Save the data files
        with open(os.path.join(timestamp_dir, "joint_to_bone_mappings.json"), "w") as f:
            json.dump(joint_to_bone_mappings, f, indent=4)
        
        if root_motions is not None:
            with open(os.path.join(timestamp_dir, "root_motions.json"), "w") as f:
                json.dump(to_serializable(root_motions), f, indent=4)
        # Save visualization data
        if visualization_data is not None:
            serializable_vis_data = []
            for idx, item in enumerate(visualization_data):
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
            json.dump(frame_paths, f, indent=4)
        with open(os.path.join(timestamp_dir, "profile_name.txt"), "w") as f:
            f.write(profile_name)
        
        # Update the index file
        index_path = os.path.join(DEBUG_RESULTS_DIR, "sam3d_saved_results.json")
        saved_results = []
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                saved_results = json.load(f)
        saved_results.append(timestamp)
        with open(index_path, "w") as f:
            json.dump(saved_results, f, indent=4)
        
        print(f"Debug results saved with timestamp: {timestamp}")
        return timestamp

    def debug_get_saved_timestamps():
        """Get list of timestamps from sam3d_saved_results.json"""
        index_path = os.path.join(DEBUG_RESULTS_DIR, "sam3d_saved_results.json")
        if not os.path.exists(index_path):
            return []
        with open(index_path, "r") as f:
            saved_results = json.load(f)
        return saved_results

    def debug_clear_saved_results():
        """Clear all saved results"""
        if os.path.exists(DEBUG_RESULTS_DIR):
            shutil.rmtree(DEBUG_RESULTS_DIR, ignore_errors=True)
        os.makedirs(DEBUG_RESULTS_DIR, exist_ok=True)
        index_path = os.path.join(DEBUG_RESULTS_DIR, "sam3d_saved_results.json")
        with open(index_path, "w") as f:
            json.dump([], f)
        print("Debug results cleared")

    def debug_load_results(timestamp):
        """Load results from a specific timestamp directory"""
        timestamp_dir = os.path.join(DEBUG_RESULTS_DIR, str(timestamp))
        
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
        with open(os.path.join(timestamp_dir, "profile_name.txt"), "r") as f:
            profile_name = f.read().strip()
        
        return joint_to_bone_mappings, root_motions, visualization_data, frame_paths, profile_name

    def debug_reexport_results(timestamp, use_root_motion, create_visualization, progress=gr.Progress()):
        """
        Re-run the export functions to debug blender/visualization issues
        """
        if timestamp is None or timestamp == "" or timestamp == translator.t("ui.debug_no_saved_results"):
            raise ValueError("Please select a timestamp to re-export")
        
        progress(0.0, desc=translator.t("progress.reexporting"))
        
        timestamp = int(timestamp)
        joint_to_bone_mappings, root_motions, visualization_data, frame_paths, profile_name = debug_load_results(timestamp)
        
        output_files = []
        
        # Export FBX files
        fbx_paths = export_results(
            estimator, 
            profile_name,
            joint_to_bone_mappings,
            root_motions,
            frame_paths,
            0.0,
            0.9 if create_visualization and visualization_data is not None else 1.0,
            progress
        )
        output_files.extend(fbx_paths)
        
        # Export visualization if requested and available
        if create_visualization and visualization_data is not None:
            # Check if we have valid outputs in the visualization data
            has_valid_outputs = any(
                len(item.get("outputs", [])) > 0 
                for item in visualization_data
            )
            if has_valid_outputs:
                visualization_output = export_visualization(visualization_data, estimator.faces, 0.9, 1.0, progress)
                if visualization_output is not None:
                    output_files.append(visualization_output)
            else:
                print("Warning: Visualization data does not contain valid outputs, skipping visualization")
        
        progress(1.0, desc=translator.t("progress.reexporting"))
        
        return output_files if output_files else None

    def output(result):
        if result is None:
            return None

        fbx_path = export_to_fbx(
            result["metadata"],
            result["joint_to_bone_mapping"],
            result["root_motion"],
            result["rest_pose"],
            estimator.faces
        )
        return fbx_path
    
    def get_ui_texts(lang: str = DEFAULT_LANGUAGE):
        """Get all UI texts for a given language."""
        t = Translator(lang)
        features = t.get("app.features", [])
        usage = t.get("app.usage", [])
        
        features_text = "\n".join([f"- {f}" for f in features])
        usage_text = "\n".join([f"{i+1}. {u}" for i, u in enumerate(usage)])
        
        return {
            "title": t.t("app.title"),
            "heading": t.t("app.heading"),
            "features_title": t.t("app.features_title"),
            "features": features_text,
            "usage_title": t.t("app.usage_title"),
            "usage": usage_text,
            "profile": t.t("ui.profile"),
            "input_file": t.t("ui.input_file"),
            "use_bbox": t.t("ui.use_bbox"),
            "bbox_file": t.t("ui.bbox_file"),
            "num_people": t.t("ui.num_people"),
            "fov_method": t.t("ui.fov_method"),
            "fov_method_info": t.t("ui.fov_method_info"),
            "fov_file": t.t("ui.fov_file"),
            "sample_number": t.t("ui.sample_number"),
            "sample_number_info": t.t("ui.sample_number_info"),
            "use_root_motion": t.t("ui.use_root_motion"),
            "generate_btn": t.t("ui.generate_btn"),
            "output_files": t.t("ui.output_files"),
            "developer_options": t.t("ui.developer_options"),
            "create_visualization": t.t("ui.create_visualization"),
        }
    
    def update_ui_language(lang: str):
        """Update UI elements with new language."""
        texts = get_ui_texts(lang)
        # Update translator for process function
        nonlocal translator
        translator = Translator(lang)
        
        features = translator.get("app.features", [])
        usage = translator.get("app.usage", [])
        features_text = "\n".join([f"- {f}" for f in features])
        usage_text = "\n".join([f"{i+1}. {u}" for i, u in enumerate(usage)])
        description_text = f"### {texts['features_title']}\n{features_text}\n\n### {texts['usage_title']}\n{usage_text}"
        
        return (
            gr.update(value=f"## {texts['heading']}"),  # heading
            gr.update(value=description_text),  # description
            gr.update(label=texts["profile"]),  # profile_name
            gr.update(label=texts["input_file"]),  # input_file
            gr.update(label=texts["use_bbox"]),  # use_bbox
            gr.update(label=texts["bbox_file"]),  # bbox_file
            gr.update(label=texts["num_people"]),  # num_people
            gr.update(label=texts["fov_method"], info=texts["fov_method_info"]),  # fov_method
            gr.update(label=texts["fov_file"]),  # fov_file
            gr.update(label=texts["sample_number"], info=texts["sample_number_info"]),  # sample_number
            gr.update(label=texts["use_root_motion"]),  # use_root_motion
            gr.update(value=texts["generate_btn"]),  # generate_btn
            gr.update(label=texts["output_files"]),  # output_files
            gr.update(label=texts["create_visualization"]),  # create_visualization
            gr.update(label=texts["debug_save_results"]),  # debug_save_results
            gr.update(label=texts["debug_saved_timestamps"]),  # debug_saved_timestamps
            gr.update(value=texts["debug_reexport_btn"]),  # debug_reexport_btn
            gr.update(value=texts["debug_clear_btn"]),  # debug_clear_btn
        )
    
    with gr.Blocks(title=translator.t("app.title")) as app:
        # Title and heading
        heading_md = gr.Markdown(f"## {translator.t('app.heading')}")
        
        # Description with features and usage
        features = translator.get("app.features", [])
        usage = translator.get("app.usage", [])
        features_text = "\n".join([f"- {f}" for f in features])
        usage_text = "\n".join([f"{i+1}. {u}" for i, u in enumerate(usage)])
        description_text = f"### {translator.t('app.features_title')}\n{features_text}\n\n### {translator.t('app.usage_title')}\n{usage_text}"
        description_md = gr.Markdown(description_text)
        
        # Language selector dropdown (visible for manual override)
        lang_selector = gr.Dropdown(
            label="🌐 Language / 言語",
            choices=[("English", "en"), ("日本語", "ja")],
            value=DEFAULT_LANGUAGE,
            interactive=True
        )
        
        # Feel free to add your own! But mixamo can also do remapping post-this via the Rokoko plugin, etc.
        profile_name = gr.Dropdown(
            label=translator.t("ui.profile"),
            choices=list(PROFILES.keys()),
            value=list(PROFILES.keys())[0]
        )
        with gr.Row():
            with gr.Column():
                input_file = gr.File(
                    label=translator.t("ui.input_file"),
                    file_types=["image", "video"]
                )
                use_bbox = gr.Checkbox(
                    label=translator.t("ui.use_bbox"),
                    value=False
                )
                with gr.Row():
                    bbox_file = gr.File(
                        label=translator.t("ui.bbox_file"),
                        file_types=[".txt"],
                        visible=False
                    )
                    num_people = gr.Number(
                        label=translator.t("ui.num_people"),
                        value=1,
                        precision=0,
                        visible=True
                    )
                
                # FOV Estimation Options
                fov_method = gr.Dropdown(
                    label=translator.t("ui.fov_method"),
                    choices=["Default", "File", "Sample"],
                    value="Default",
                    info=translator.t("ui.fov_method_info")
                )
                fov_file = gr.File(
                    label=translator.t("ui.fov_file"),
                    file_types=[".txt"],
                    visible=False
                )
                sample_number = gr.Number(
                    label=translator.t("ui.sample_number"),
                    value=1,
                    precision=0,
                    minimum=1,
                    visible=False,
                    info=translator.t("ui.sample_number_info")
                )
                use_root_motion = gr.Checkbox(
                    label=translator.t("ui.use_root_motion"),
                    value=True
                )
                
                # Developer Options
                with gr.Accordion(translator.t("ui.developer_options"), open=False):
                    create_visualization = gr.Checkbox(
                        label=translator.t("ui.create_visualization"),
                        value=False
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### Debug: Save & Re-export")
                    
                    debug_save_results = gr.Checkbox(
                        label=translator.t("ui.debug_save_results"),
                        value=False
                    )
                    
                    def get_timestamp_choices():
                        timestamps = debug_get_saved_timestamps()
                        if not timestamps:
                            return [translator.t("ui.debug_no_saved_results")]
                        # Convert timestamps to readable format
                        choices = []
                        for ts in reversed(timestamps):  # Show newest first
                            dt = datetime.fromtimestamp(ts)
                            label = f"{dt.strftime('%Y-%m-%d %H:%M:%S')} ({ts})"
                            choices.append((label, str(ts)))
                        return choices
                    
                    debug_saved_timestamps = gr.Dropdown(
                        label=translator.t("ui.debug_saved_timestamps"),
                        choices=get_timestamp_choices(),
                        value=None,
                        interactive=True
                    )
                    
                    with gr.Row():
                        debug_refresh_btn = gr.Button("🔄 Refresh List", size="sm")
                        debug_clear_btn = gr.Button(translator.t("ui.debug_clear_btn"), size="sm", variant="stop")
                    
                    debug_reexport_btn = gr.Button(
                        translator.t("ui.debug_reexport_btn"),
                        variant="secondary"
                    )
                
                generate_btn = gr.Button(
                    translator.t("ui.generate_btn"),
                    variant="primary"
                )
            with gr.Column():
                output_files = gr.File(
                    label=translator.t("ui.output_files"),
                    interactive=False,
                    file_count="multiple"
                )
        
        def toggle_bbox_inputs(use_bbox_value):
            """Toggle visibility of bbox_file and num_people based on checkbox."""
            return (
                gr.update(visible=use_bbox_value),
                gr.update(visible=not use_bbox_value)
            )
        
        def toggle_fov_inputs(fov_method_value):
            """Toggle visibility of fov_file and sample_number based on FOV method selection."""
            if fov_method_value == "File":
                return (
                    gr.update(visible=True),
                    gr.update(visible=False)
                )
            elif fov_method_value == "Sample":
                return (
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
            else:  # "Default"
                return (
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
        
        # Language change handler
        def on_lang_change(lang):
            return update_ui_language(lang)
        
        # Detect browser language and update UI on page load
        def detect_and_set_language():
            """Detect browser language and return it for initial setup."""
            # This will be called on page load
            return DEFAULT_LANGUAGE
        
        # Load language on page load - detect browser language via JavaScript
        app.load(
            fn=detect_and_set_language,
            inputs=[],
            outputs=[lang_selector],
            js="""
            function() {
                const lang = navigator.language || navigator.userLanguage;
                const langCode = lang.split('-')[0].toLowerCase();
                const supportedLangs = ['en', 'ja'];
                const detectedLang = supportedLangs.includes(langCode) ? langCode : 'en';
                return detectedLang;
            }
            """
        )
        
        # Also add a custom script that runs after page load to trigger language update
        app.load(
            fn=None,
            inputs=[],
            outputs=[],
            js="""
            function() {
                // Wait for Gradio to fully initialize
                setTimeout(() => {
                    const lang = navigator.language || navigator.userLanguage;
                    const langCode = lang.split('-')[0].toLowerCase();
                    const supportedLangs = ['en', 'ja'];
                    const detectedLang = supportedLangs.includes(langCode) ? langCode : 'en';
                    
                    // Find language selector by looking for dropdown with language options
                    const selects = document.querySelectorAll('select');
                    for (const select of selects) {
                        const options = Array.from(select.options);
                        const hasEn = options.some(opt => opt.value === 'en');
                        const hasJa = options.some(opt => opt.value === 'ja');
                        if (hasEn && hasJa && options.length === 2) {
                            select.value = detectedLang;
                            // Trigger input event which Gradio listens to
                            select.dispatchEvent(new Event('input', { bubbles: true }));
                            select.dispatchEvent(new Event('change', { bubbles: true }));
                            break;
                        }
                    }
                }, 500);
            }
            """
        )
        
        lang_selector.change(
            fn=on_lang_change,
            inputs=[lang_selector],
            outputs=[heading_md, description_md, profile_name, input_file, use_bbox, bbox_file, num_people, fov_method, fov_file, sample_number, use_root_motion, generate_btn, output_files, create_visualization, debug_save_results, debug_saved_timestamps, debug_reexport_btn, debug_clear_btn]
        )
        
        use_bbox.change(fn=toggle_bbox_inputs, inputs=[use_bbox], outputs=[bbox_file, num_people])
        fov_method.change(fn=toggle_fov_inputs, inputs=[fov_method], outputs=[fov_file, sample_number])
        generate_btn.click(fn=process, inputs=[input_file, profile_name, use_bbox, bbox_file, num_people, fov_method, fov_file, sample_number, use_root_motion, create_visualization, debug_save_results], outputs=[output_files])
        
        # Debug re-export handlers
        def refresh_timestamps():
            choices = get_timestamp_choices()
            return gr.update(choices=choices, value=None)
        
        def clear_and_refresh():
            debug_clear_saved_results()
            choices = get_timestamp_choices()
            return gr.update(choices=choices, value=None)
        
        debug_refresh_btn.click(
            fn=refresh_timestamps,
            inputs=[],
            outputs=[debug_saved_timestamps]
        )
        
        debug_clear_btn.click(
            fn=clear_and_refresh,
            inputs=[],
            outputs=[debug_saved_timestamps]
        )
        
        debug_reexport_btn.click(
            fn=debug_reexport_results,
            inputs=[debug_saved_timestamps, use_root_motion, create_visualization],
            outputs=[output_files]
        )
    
    return app

if __name__ == "__main__":
    args = parse_args()
    if args.model == "vith":
        checkpoint_base_path = VITH_CHECKPOINT_PATH
    elif args.model == "dinov3":
        checkpoint_base_path = DINOV3_CHECKPOINT_PATH
    else:
        raise ValueError(f"Invalid model: {args.model}")

    checkpoint_path = os.path.join(checkpoint_base_path, "model.ckpt")
    mhr_path = os.path.join(checkpoint_base_path, "assets", "mhr_model.pt")
    
    # Get detector path from args or environment variable
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    # Get FOV path from args or environment variable
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", None)
    
    estimator = PoseEstimator(
        checkpoint_path=checkpoint_path,
        mhr_path=mhr_path,
        detector_name=args.detector_name,
        detector_path=detector_path,
        fov_name=args.fov_name,
        fov_path=fov_path
    )

    app = create_app(estimator)
    app.launch(
        server_name="0.0.0.0",
        server_port=7444,
        share=True,
    )