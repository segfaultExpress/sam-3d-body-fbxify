"""
Gradio application for SAM 3D Body to FBX conversion.

This module creates the Gradio UI using modular sections and delegates
all business logic to FbxifyManager.
"""
import os
import argparse
import tempfile
import shutil
import json
import gradio as gr
from fbxify.pose_estimation_manager import PoseEstimationManager, CancelledError
from fbxify.fbx_data_prep_manager import FbxDataPrepManager
from fbxify.fbxify_manager import FbxifyManager
from fbxify.i18n import Translator, DEFAULT_LANGUAGE
from fbxify.gradio_ui.header_section import create_header_section, update_header_language
from fbxify.gradio_ui.entry_section import toggle_bbox_inputs, toggle_fov_inputs, update_entry_language
from fbxify.gradio_ui.pose_results_section import update_pose_results_language
from fbxify.gradio_ui.fbx_processing_section import update_fbx_processing_language, toggle_generate_fbx_button
from fbxify.gradio_ui.fbx_results_section import update_fbx_results_language
from fbxify.gradio_ui.fbx_options_section import toggle_extrinsics_inputs, update_fbx_options_language
from fbxify import VERSION
from fbxify.gradio_ui.developer_section import (
    update_pose_cli_language,
    update_fbx_cli_language,
    update_pose_dev_language,
    update_fbx_dev_language,
    toggle_camera_inputs,
)
from fbxify.gradio_ui.pose_tab import create_pose_tab
from fbxify.gradio_ui.fbx_tab import create_fbx_tab

VITH_CHECKPOINT_PATH = "/workspace/checkpoints/sam-3d-body-vith"
DINOV3_CHECKPOINT_PATH = "/workspace/checkpoints/sam-3d-body-dinov3"


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


def create_app(manager: FbxifyManager):
    """Create Gradio app using modular sections and manager."""
    
    # Initialize translator with default language
    translator = Translator(DEFAULT_LANGUAGE)
    
    def estimate_pose(input_file, use_bbox, bbox_file, num_people, missing_bbox_behavior, fov_method,
                     fov_file, sample_number, precision, progress=gr.Progress()):
        """Estimate pose from image or video file - Step 1."""
        temp_dir = None
        
        try:
            manager.estimation_manager.clear_cancel()
            if input_file is None:
                return (
                    gr.update(),  # pose_json_file (pose tab)
                    gr.update(),  # pose_json_file (fbx tab)
                    None,  # pose_json_state
                    gr.update(interactive=False),  # generate_fbx_btn
                    gr.update(interactive=False)   # estimate_pose_btn
                )

            # Validate inputs
            if use_bbox and bbox_file is None:
                raise ValueError(translator.t("errors.bbox_file_required"))

            # Prepare bboxes
            bbox_dict = None
            if use_bbox:
                bbox_dict = manager.prepare_bboxes(bbox_file.name)
                # Count unique person IDs across all frames
                unique_person_ids = set()
                for bboxes in bbox_dict.values():
                    for bbox in bboxes:
                        if len(bbox) > 0:
                            unique_person_ids.add(bbox[0])
                num_people = len(unique_person_ids) if unique_person_ids else 0
            else:
                if num_people <= 0:
                    raise ValueError(translator.t("errors.num_people_required"))

            # Prepare video or image
            file_path = input_file.name
            file_ext = os.path.splitext(file_path)[1].lower()
            is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            
            fps = 30.0  # Default FPS for images
            if is_video:
                frame_paths, temp_dir, fps = manager.prepare_video(file_path)
            else:
                frame_paths = [file_path]
                temp_dir = None

            # Set camera intrinsics
            fov_file_path = fov_file.name if fov_file else None
            manager.set_camera_intrinsics(fov_method, fov_file_path, frame_paths, sample_number)

            # Apply inference options (precision + hand crop scale)
            precision_value = "fp32"
            if isinstance(precision, str):
                if precision.lower().startswith("bf16"):
                    precision_value = "bf16"
                elif precision.lower().startswith("fp16"):
                    precision_value = "fp16"
            manager.estimation_manager.set_inference_options(
                precision=precision_value
            )

            # Process frames - only estimation
            # Use Gradio's progress tracker
            def progress_callback(progress_value, description):
                if progress is not None:
                    progress(progress_value, desc=description)

            # Save estimation JSON to temp file
            estimation_json_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
            
            # Only estimate poses, don't generate FBX yet
            estimation_results = manager.estimation_manager.estimate_all_frames(
                frame_paths,
                num_people=num_people,
                bbox_dict=bbox_dict,
                progress_callback=progress_callback,
                missing_bbox_behavior=missing_bbox_behavior if use_bbox else "Run Detection",
                lang=translator.lang
            )
            
            # Extract source name for metadata
            source_name = os.path.basename(file_path)
            
            # Save estimation results
            manager.estimation_manager.save_estimation_results(
                estimation_results,
                estimation_json_path,
                source_name=source_name,
                num_people=num_people
            )

        except CancelledError:
            # Cancelled by user; return to idle state without error
            return (
                gr.update(value=None),  # pose_json_file (pose tab)
                gr.update(value=None),  # pose_json_file (fbx tab)
                None,  # pose_json_state
                gr.update(interactive=False),  # generate_fbx_btn
                gr.update(interactive=(input_file is not None))  # estimate_pose_btn
            )
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            if error_msg:
                raise gr.Error(translator.t("errors.error_occurred", error_type=error_type, error_msg=error_msg))
            else:
                raise gr.Error(translator.t("errors.error_occurred_no_msg", error_type=error_type))
        finally:
            # Clean up temp directory if created
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Return JSON file for the dropdown (update both value and enable button)
        # Re-enable buttons now that estimation is complete
        # Note: input_file.change handler will disable estimate_pose_btn if file is removed
        return (
            gr.update(value=estimation_json_path),  # pose_json_file (pose tab)
            gr.update(value=estimation_json_path),  # pose_json_file (fbx tab)
            estimation_json_path,  # pose_json_state
            gr.update(interactive=True),  # generate_fbx_btn
            gr.update(interactive=(input_file is not None))   # estimate_pose_btn (re-enable only if file still exists)
        )
    
    def generate_fbx(pose_json_file, profile_name, use_root_motion, auto_floor, include_mesh, include_extrinsics,
                    extrinsics_sample_rate, extrinsics_scale, extrinsics_invert_quaternion,
                    extrinsics_invert_translation, extrinsics_file,
                    create_camera, camera_zoom, camera_scene,
                    use_personalized_body, lod, outlier_removal_percent,
                    export_personalized_body_obj, graph_refinement,
                    input_file,
                    refinement_config,  # Single refinement config object from state
                    progress=gr.Progress()):
        """Generate FBX from pose estimation JSON - Step 2."""
        output_files = []

        
        try:
            manager.estimation_manager.clear_cancel()
            if pose_json_file is None:
                raise ValueError(translator.t("errors.pose_json_file_required"))
            
            # Get file path
            if pose_json_file is None:
                raise ValueError(translator.t("errors.pose_json_file_required"))
            json_path = pose_json_file.name if hasattr(pose_json_file, 'name') else pose_json_file
            
            # Load from estimation JSON and apply refinement if enabled (refinement happens before joint mapping)
            def processing_progress(progress_value, description):
                if progress is not None:
                    progress(progress_value * 0.3, desc=description)
            
            # Convert lod to int if it's a float from slider
            lod_int = int(lod) if lod is not None else -1

            print(f"generate_fbx(): lod_int: {lod_int}")
            # Convert outlier_removal_percent to float
            outlier_percent = float(outlier_removal_percent) if outlier_removal_percent is not None else 10.0
            
            extrinsics_file_path = extrinsics_file.name if (include_extrinsics and extrinsics_file is not None) else None
            camera_scene_path = camera_scene.name if (create_camera and camera_scene is not None) else None
            process_result = manager.process_from_estimation_json(
                json_path,
                profile_name,
                use_root_motion,
                fps=30.0,
                refinement_config=refinement_config,
                progress_callback=processing_progress,
                lod=lod_int if include_mesh else -1,
                use_personalized_body=use_personalized_body if include_mesh else False,
                outlier_removal_percent=outlier_percent if (include_mesh and use_personalized_body) else 10.0,
                lang=translator.lang,
                auto_floor=auto_floor,
                extrinsics_file=extrinsics_file_path,
                extrinsics_sample_rate=int(extrinsics_sample_rate) if extrinsics_sample_rate is not None else 0,
                extrinsics_scale=float(extrinsics_scale) if extrinsics_scale is not None else 0.0,
                extrinsics_invert_quaternion=bool(extrinsics_invert_quaternion),
                extrinsics_invert_translation=bool(extrinsics_invert_translation),
                collect_refinement_logs=bool(graph_refinement),
            )

            # Export FBX files
            # Map export progress (0-1) to the 0.3-0.9 range in overall progress
            def export_progress(progress_value, description):
                if progress is not None:
                    # progress_value is 0.0 to 1.0 from export_fbx_files
                    # Map it to 0.3-0.9 range (export takes 60% of remaining progress after processing)
                    base_progress = 0.3
                    export_range = 0.6  # 0.9 - 0.3
                    mapped_progress = base_progress + (progress_value * export_range)
                    progress(mapped_progress, desc=description)

            # Get LOD path if mesh is included
            lod_fbx_path = None
            if include_mesh and lod_int >= 0 and process_result.profile_name == "mhr":
                from fbxify.metadata import PROFILES
                profile = PROFILES.get(process_result.profile_name)
                if profile:
                    lod_key = f"lod{lod_int}_path"
                    if lod_key in profile:
                        lod_rel_path = profile[lod_key]
                        lod_fbx_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fbxify", lod_rel_path)
            
            fbx_paths = manager.export_fbx_files(
                process_result.profile_name,
                process_result.joint_to_bone_mappings,
                process_result.root_motions,
                process_result.frame_paths,
                process_result.fps,
                export_progress,
                lod=lod_int if include_mesh else -1,
                mesh_obj_paths=process_result.mesh_obj_paths,  # Use generated meshes from JSON
                lod_fbx_path=lod_fbx_path,
                lang=translator.lang,
                height_offset=process_result.height_offset,
                metadata_extras=process_result.metadata_extras,
                create_camera=bool(create_camera),
                camera_scene_path=camera_scene_path,
                camera_zoom=float(camera_zoom) if camera_zoom is not None else 0.0,
                extrinsics_file=extrinsics_file_path
            )
            output_files.extend(fbx_paths)
            if export_personalized_body_obj and process_result.mesh_obj_paths:
                for mesh_path in process_result.mesh_obj_paths.values():
                    if mesh_path and os.path.exists(mesh_path):
                        output_files.append(mesh_path)
            
            if graph_refinement and process_result.refinement_logs:
                try:
                    from fbxify.refinement.refinement_graphs import convert_refinement_logs
                    graph_files = convert_refinement_logs(process_result.refinement_logs)
                    output_files.extend(graph_files)
                except Exception as e:
                    print(f"Graph Refinement: Failed to generate graphs: {e}")
                    try:
                        gr.Warning(f"Graph Refinement failed: {e}")
                    except Exception:
                        pass

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            if error_msg:
                raise gr.Error(translator.t("errors.error_occurred", error_type=error_type, error_msg=error_msg))
            else:
                raise gr.Error(translator.t("errors.error_occurred_no_msg", error_type=error_type))
        
        # Return output files and re-enable estimate_pose_btn (only if input_file still exists)
        return (
            output_files if output_files else None,
            gr.update(interactive=(input_file is not None)),  # estimate_pose_btn (re-enable only if file still exists)
            gr.update(interactive=(pose_json_file is not None))  # generate_fbx_btn
        )

    def _format_cli_arg(value: str) -> str:
        if value is None:
            return ""
        text = str(value)
        if any(char.isspace() for char in text):
            return f"\"{text}\""
        return text

    def build_pose_cli_command(use_bbox, bbox_file, num_people, missing_bbox_behavior, fov_method,
                               fov_file, sample_number, precision):
        precision_map = {
            "FP32 (Full)": "fp32",
            "BF16 (Fast + Safer)": "bf16",
            "FP16 (Fastest)": "fp16"
        }
        precision_value = precision_map.get(precision, str(precision).lower() if precision else "fp32")

        cmd_parts = ["python", "-m", "fbxify.cli"]

        if use_bbox:
            cmd_parts += ["--bbox_file", "<BBOX_FILE>"]
            if missing_bbox_behavior:
                cmd_parts += ["--missing_bbox_behavior", _format_cli_arg(missing_bbox_behavior)]
        else:
            if num_people is not None:
                cmd_parts += ["--num_people", str(int(num_people))]

        if fov_method and fov_method != "Default":
            cmd_parts += ["--fov_method", _format_cli_arg(fov_method)]
            if fov_method == "File":
                cmd_parts += ["--fov_file", "<CAMERA_INTRINSICS_FILE>"]
            elif fov_method == "Sample":
                if sample_number is not None:
                    cmd_parts += ["--sample_number", str(int(sample_number))]

        cmd_parts += ["--precision", precision_value]
        cmd_parts += ["--save_estimation_json", "<POSE_JSON>"]
        cmd_parts.append("<INPUT_FILE>")
        return " ".join(cmd_parts)

    def build_fbx_cli_command(profile_name, use_root_motion, auto_floor):
        cmd_parts = ["python", "-m", "fbxify.cli"]

        if profile_name:
            cmd_parts += ["--profile", _format_cli_arg(profile_name)]

        if use_root_motion is False:
            cmd_parts.append("--no_root_motion")
        if auto_floor is False:
            cmd_parts.append("--no_auto_floor")

        cmd_parts += ["--load_estimation_json", "<POSE_JSON>"]
        cmd_parts.append("<INPUT_FILE>")
        return " ".join(cmd_parts)
    
    def on_lang_change(lang):
        """Update all UI components when language changes."""
        # Update translator
        nonlocal translator
        translator = Translator(lang)

        # Get updates from each section
        header_updates = update_header_language(lang)
        entry_updates = update_entry_language(lang, translator)
        pose_results_updates = update_pose_results_language(lang)
        fbx_processing_updates = update_fbx_processing_language(lang)
        fbx_options_updates = update_fbx_options_language(lang, translator)
        fbx_results_updates = update_fbx_results_language(lang)
        pose_cli_updates = update_pose_cli_language(lang)
        fbx_cli_updates = update_fbx_cli_language(lang)
        pose_dev_updates = update_pose_dev_language(lang)
        fbx_dev_updates = update_fbx_dev_language(lang)

        # Combine all updates
        return (
            *header_updates,  # heading, description, header tabs
            gr.update(label=translator.t("ui.pose_tab_label")),  # pose tab label
            gr.update(label=translator.t("ui.fbx_tab_label")),  # fbx tab label
            *entry_updates,  # pose inputs
            *pose_results_updates,  # pose results/actions
            *fbx_processing_updates,  # profile, pose_json_file
            *fbx_options_updates,  # fbx options
            *fbx_results_updates,  # fbx actions/results
            *pose_cli_updates,
            *fbx_cli_updates,
            *pose_dev_updates,
            *fbx_dev_updates,
        )

    def detect_and_set_language():
        """Detect browser language and return it for initial setup."""
        return DEFAULT_LANGUAGE

    # Create UI sections
    with gr.Blocks(title=translator.t("app.title")) as app:
        # Header section (now returns heading_md, description_md, tabs, lang_selector)
        heading_md, description_md, header_tabs, lang_selector = create_header_section(translator)

        # Pose/FBX tabs inside Program area
        with gr.Tabs() as pose_fbx_tabs:
            with gr.Tab(translator.t("ui.pose_tab_label")) as pose_tab:
                pose_tab_components = create_pose_tab(translator)
            with gr.Tab(translator.t("ui.fbx_tab_label")) as fbx_tab:
                fbx_tab_components = create_fbx_tab(translator)

        entry_components = pose_tab_components["entry"]
        pose_results_components = pose_tab_components["pose_results"]
        pose_cli_components = pose_tab_components["pose_cli"]
        pose_dev_components = pose_tab_components["pose_dev"]

        fbx_processing_components = fbx_tab_components["fbx_processing"]
        fbx_options_components = fbx_tab_components["fbx_options"]
        refinement_components = fbx_tab_components["refinement"]
        fbx_results_components = fbx_tab_components["fbx_results"]
        fbx_cli_components = fbx_tab_components["fbx_cli"]
        fbx_dev_components = fbx_tab_components["fbx_dev"]

        # Shared state
        pose_json_state = gr.State(value=None)

        def toggle_estimate_pose_button(input_file):
            """Enable/disable Estimate Pose button based on whether file is uploaded."""
            return gr.update(interactive=(input_file is not None))

        def switch_to_fbx_tab():
            """Attempt to switch to the Generate FBX tab."""
            return gr.update(selected=1)

        def sync_pose_json_to_pose(pose_json_file):
            """Sync pose JSON from FBX tab to Pose tab and state."""
            return (
                gr.update(value=pose_json_file),  # pose pose_json_file
                pose_json_file,  # pose_json_state
                toggle_generate_fbx_button(pose_json_file),
            )
        
        # Wire up event handlers
        # Language change
        lang_selector.change(
            fn=on_lang_change,
            inputs=[lang_selector],
            outputs=[
                heading_md, description_md, header_tabs,  # header
                pose_tab, fbx_tab,  # pose/fbx tabs
                entry_components['input_file'],
                entry_components['use_bbox'], entry_components['bbox_file'],
                entry_components['num_people'], entry_components['missing_bbox_behavior'], entry_components['fov_method'],
                entry_components['fov_file'], entry_components['sample_number'],
                entry_components['precision'],
                pose_results_components['pose_json_file'],
                pose_results_components['estimate_pose_btn'],
                fbx_processing_components['profile_name'],
                fbx_processing_components['pose_json_file'],
                fbx_options_components['auto_run'], fbx_options_components['use_root_motion'], fbx_options_components['auto_floor'], fbx_options_components['include_mesh'],
                fbx_options_components['include_extrinsics'], fbx_options_components['use_personalized_body'],
                fbx_options_components['lod'], fbx_options_components['outlier_removal_percent'],
                fbx_options_components['extrinsics_sample_rate'], fbx_options_components['extrinsics_scale'],
                fbx_options_components['extrinsics_invert_quaternion'], fbx_options_components['extrinsics_invert_translation'],
                fbx_options_components['extrinsics_file'],
                fbx_results_components['generate_fbx_btn'],
                fbx_results_components['output_files'],
                pose_cli_components['pose_cli_generator_accordion'], pose_cli_components['pose_cli_generator_info_md'],
                pose_cli_components['pose_generate_cli_btn'], pose_cli_components['pose_cli_command'],
                fbx_cli_components['fbx_cli_generator_accordion'], fbx_cli_components['fbx_cli_generator_info_md'],
                fbx_cli_components['fbx_generate_cli_btn'], fbx_cli_components['fbx_cli_command'],
                pose_dev_components['pose_developer_options_accordion'], pose_dev_components['pose_cancel_jobs_info_md'], pose_dev_components['pose_cancel_jobs_btn'],
                fbx_dev_components['fbx_developer_options_accordion'],
                fbx_dev_components['fbx_cancel_jobs_info_md'],
                fbx_dev_components['fbx_cancel_jobs_btn'],
                fbx_dev_components['export_personalized_body_obj'],
                fbx_dev_components['graph_refinement'],
                fbx_dev_components['create_camera'],
                fbx_dev_components['camera_zoom'],
                fbx_dev_components['camera_scene'],
            ]
        )
        
        # Bbox toggle
        entry_components['use_bbox'].change(
            fn=toggle_bbox_inputs,
            inputs=[entry_components['use_bbox']],
            outputs=[entry_components['bbox_file'], entry_components['num_people'], entry_components['missing_bbox_behavior']]
        )
        
        # FOV toggle
        entry_components['fov_method'].change(
            fn=toggle_fov_inputs,
            inputs=[entry_components['fov_method']],
            outputs=[entry_components['fov_file'], entry_components['sample_number']]
        )

        # Sync pose JSON from FBX tab to Pose tab (read-only on Pose tab)
        fbx_processing_components['pose_json_file'].change(
            fn=sync_pose_json_to_pose,
            inputs=[fbx_processing_components['pose_json_file']],
            outputs=[
                pose_results_components['pose_json_file'],
                pose_json_state,
                fbx_results_components['generate_fbx_btn'],
            ]
        )
        
        # Mesh toggle - show/hide lod and use_personalized_body
        fbx_options_components['include_mesh'].change(
            fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
            inputs=[fbx_options_components['include_mesh']],
            outputs=[fbx_options_components['lod'], fbx_options_components['use_personalized_body']]
        )

        # Extrinsics toggle - show/hide extrinsics inputs
        fbx_options_components['include_extrinsics'].change(
            fn=toggle_extrinsics_inputs,
            inputs=[fbx_options_components['include_extrinsics']],
            outputs=[
                fbx_options_components['extrinsics_sample_rate'],
                fbx_options_components['extrinsics_scale'],
                fbx_options_components['extrinsics_invert_quaternion'],
                fbx_options_components['extrinsics_invert_translation'],
                fbx_options_components['extrinsics_file']
            ]
        )
        
        # Camera toggle - show/hide camera inputs
        fbx_dev_components['create_camera'].change(
            fn=toggle_camera_inputs,
            inputs=[fbx_dev_components['create_camera']],
            outputs=[fbx_dev_components['camera_zoom'], fbx_dev_components['camera_scene']]
        )
        
        # Combined toggle for outlier removal - depends on both include_mesh and use_personalized_body
        def update_outlier_visibility(include_mesh_val, use_personalized_val):
            # Handle case where value might be a list
            if isinstance(include_mesh_val, list):
                include_mesh_val = include_mesh_val[0] if include_mesh_val else False
            if isinstance(use_personalized_val, list):
                use_personalized_val = use_personalized_val[0] if use_personalized_val else False
            return gr.update(visible=include_mesh_val and use_personalized_val)
        
        # Update outlier visibility when include_mesh changes
        fbx_options_components['include_mesh'].change(
            fn=update_outlier_visibility,
            inputs=[fbx_options_components['include_mesh'], fbx_options_components['use_personalized_body']],
            outputs=[fbx_options_components['outlier_removal_percent']]
        )
        
        # Update outlier visibility when use_personalized_body changes
        fbx_options_components['use_personalized_body'].change(
            fn=update_outlier_visibility,
            inputs=[fbx_options_components['include_mesh'], fbx_options_components['use_personalized_body']],
            outputs=[fbx_options_components['outlier_removal_percent']]
        )
        
        # Get refinement components for building config
        all_refinement_inputs = refinement_components['all_refinement_inputs']
        build_refinement_config_wrapper = refinement_components['build_refinement_config_wrapper']
        refinement_config_state = refinement_components['refinement_config_state']
        
        # Helper function to build config with logging
        def build_and_log_config(*args):
            """Helper function to build config with logging."""
            print(f"build_and_log_config(): Building refinement config from {len(args)} inputs")
            if args:
                print(f"build_and_log_config(): First input (refinement_enabled) = {args[0]}")
            config = build_refinement_config_wrapper(*args)
            print(f"build_and_log_config(): Built config is {'None' if config is None else 'not None'}")
            return config

        # Enable/disable Estimate Pose button based on file upload
        entry_components['input_file'].change(
            fn=toggle_estimate_pose_button,
            inputs=[entry_components['input_file']],
            outputs=[pose_results_components['estimate_pose_btn']]
        )
        
        # Helper function to conditionally auto-run generate_fbx
        def auto_run_generate_fbx(pose_json_file, auto_run, profile_name, use_root_motion, auto_floor, include_mesh, include_extrinsics,
                                  extrinsics_sample_rate, extrinsics_scale, extrinsics_invert_quaternion,
                                  extrinsics_invert_translation, extrinsics_file, create_camera, camera_zoom, camera_scene,
                                  use_personalized_body, lod, outlier_removal_percent, export_personalized_body_obj,
                                  graph_refinement, input_file, *refinement_inputs, progress=gr.Progress()):
            """Conditionally trigger generate_fbx if auto_run is enabled."""
            if not auto_run or pose_json_file is None:
                # Just re-enable estimate_pose_btn if input_file still exists
                return (
                    None,
                    gr.update(interactive=(input_file is not None)),
                    gr.update(interactive=(pose_json_file is not None))
                )
            
            # Build refinement config
            refinement_cfg = build_and_log_config(*refinement_inputs)
            
            # Call generate_fbx (progress will be automatically injected by Gradio)
            return generate_fbx(
                pose_json_file,
                profile_name,
                use_root_motion,
                auto_floor,
                include_mesh,
                include_extrinsics,
                extrinsics_sample_rate,
                extrinsics_scale,
                extrinsics_invert_quaternion,
                extrinsics_invert_translation,
                extrinsics_file,
                create_camera,
                camera_zoom,
                camera_scene,
                use_personalized_body,
                lod,
                outlier_removal_percent,
                export_personalized_body_obj,
                graph_refinement,
                input_file,
                refinement_cfg,
                progress=progress
            )
        
        # Estimate Pose button (Step 1)
        # Disable both Estimate Pose and Generate FBX buttons immediately when Estimate Pose is clicked
        estimate_pose_click = pose_results_components['estimate_pose_btn'].click(
            fn=lambda: (
                gr.update(value=None),
                gr.update(value=None),
                None,
                gr.update(interactive=False),
                gr.update(interactive=False),
            ),
            inputs=[],
            outputs=[
                pose_results_components['pose_json_file'],
                fbx_processing_components['pose_json_file'],
                pose_json_state,
                fbx_results_components['generate_fbx_btn'],
                pose_results_components['estimate_pose_btn'],
            ]
        ).then(
            fn=estimate_pose,
            inputs=[
                entry_components['input_file'],
                entry_components['use_bbox'],
                entry_components['bbox_file'],
                entry_components['num_people'],
                entry_components['missing_bbox_behavior'],
                entry_components['fov_method'],
                entry_components['fov_file'],
                entry_components['sample_number'],
                entry_components['precision']
            ],
            outputs=[
                pose_results_components['pose_json_file'],
                fbx_processing_components['pose_json_file'],
                pose_json_state,
                fbx_results_components['generate_fbx_btn'],
                pose_results_components['estimate_pose_btn'],
            ]
        )

        estimate_pose_click.then(
            fn=switch_to_fbx_tab,
            inputs=[],
            outputs=[pose_fbx_tabs]
        )
        
        def disable_buttons_for_auto_run(pose_json_file, auto_run):
            """Disable buttons while auto-run generate_fbx is active."""
            if auto_run and pose_json_file is not None:
                return (
                    gr.update(interactive=False),  # generate_fbx_btn
                    gr.update(interactive=False)   # estimate_pose_btn
                )
            return gr.update(), gr.update()

        # Auto-run: If auto_run is checked, automatically trigger generate_fbx after estimate_pose completes
        estimate_pose_click.then(
            fn=disable_buttons_for_auto_run,
            inputs=[fbx_processing_components['pose_json_file'], fbx_options_components['auto_run']],
            outputs=[fbx_results_components['generate_fbx_btn'], pose_results_components['estimate_pose_btn']]
        ).then(
            fn=auto_run_generate_fbx,
            inputs=[
                fbx_processing_components['pose_json_file'],
                fbx_options_components['auto_run'],
                fbx_processing_components['profile_name'],
                fbx_options_components['use_root_motion'],
                fbx_options_components['auto_floor'],
                fbx_options_components['include_mesh'],
                fbx_options_components['include_extrinsics'],
                fbx_options_components['extrinsics_sample_rate'],
                fbx_options_components['extrinsics_scale'],
                fbx_options_components['extrinsics_invert_quaternion'],
                fbx_options_components['extrinsics_invert_translation'],
                fbx_options_components['extrinsics_file'],
                fbx_dev_components['create_camera'],
                fbx_dev_components['camera_zoom'],
                fbx_dev_components['camera_scene'],
                fbx_options_components['use_personalized_body'],
                fbx_options_components['lod'],
                fbx_options_components['outlier_removal_percent'],
                fbx_dev_components['export_personalized_body_obj'],
                fbx_dev_components['graph_refinement'],
                entry_components['input_file'],
                *all_refinement_inputs
            ],
            outputs=[
                fbx_results_components['output_files'],
                pose_results_components['estimate_pose_btn'],
                fbx_results_components['generate_fbx_btn']
            ],
            show_progress=True
        )
        
        def validate_json_file_on_upload(pose_json_file):
            """
            Validate JSON file when uploaded and check version compatibility.
            Returns button state and shows warning if version mismatch.
            """
            if pose_json_file is None:
                return gr.update(interactive=False)
            
            try:
                # Get file path
                json_path = pose_json_file.name if hasattr(pose_json_file, 'name') else pose_json_file
                
                # Load and check version
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Check if it's new format (has metadata keys) or old format (direct frames)
                if isinstance(data, dict) and "frames" in data and "version" in data:
                    file_version = data.get("version", "unknown")
                    if file_version != VERSION:
                        # Show warning using Gradio's warning mechanism
                        warning_msg = (
                            f"⚠️ WARNING: Estimation file version ({file_version}) differs from current version ({VERSION}). "
                            f"File may have been created with a different version of the software. "
                            f"Proceeding with caution - errors may occur if formats are incompatible."
                        )
                        print(warning_msg)
                        # Also raise a Gradio warning that will be displayed to the user
                        gr.Warning(warning_msg)
                
            except Exception as e:
                # If validation fails, still enable the button (let generate_fbx handle the error)
                print(f"Warning: Could not validate JSON file: {e}")
            
            # Enable button if file is provided
            return gr.update(interactive=(pose_json_file is not None))
        
        # Enable/disable Generate FBX button based on JSON file and validate version
        fbx_processing_components['pose_json_file'].change(
            fn=validate_json_file_on_upload,
            inputs=[fbx_processing_components['pose_json_file']],
            outputs=[fbx_results_components['generate_fbx_btn']]
        )
        
        # Generate FBX button (Step 2)
        # Disable Estimate Pose button immediately when Generate FBX is clicked
        generate_fbx_click = fbx_results_components['generate_fbx_btn'].click(
            fn=lambda: (gr.update(interactive=False), gr.update(interactive=False)),  # Disable buttons immediately
            inputs=[],
            outputs=[fbx_results_components['generate_fbx_btn'], pose_results_components['estimate_pose_btn']]
        ).then(
            # First, build the refinement config from all inputs
            fn=lambda *args: build_and_log_config(*args),
            inputs=all_refinement_inputs,
            outputs=[refinement_config_state]
        ).then(
            # Then, call generate_fbx with the built config
            fn=generate_fbx,
            inputs=[
                fbx_processing_components['pose_json_file'],
                fbx_processing_components['profile_name'],
                fbx_options_components['use_root_motion'],
                fbx_options_components['auto_floor'],
                fbx_options_components['include_mesh'],
                fbx_options_components['include_extrinsics'],
                fbx_options_components['extrinsics_sample_rate'],
                fbx_options_components['extrinsics_scale'],
                fbx_options_components['extrinsics_invert_quaternion'],
                fbx_options_components['extrinsics_invert_translation'],
                fbx_options_components['extrinsics_file'],
                fbx_dev_components['create_camera'],
                fbx_dev_components['camera_zoom'],
                fbx_dev_components['camera_scene'],
                fbx_options_components['use_personalized_body'],
                fbx_options_components['lod'],
                fbx_options_components['outlier_removal_percent'],
                fbx_dev_components['export_personalized_body_obj'],
                fbx_dev_components['graph_refinement'],
                entry_components['input_file'],  # Add input_file to check if it still exists
                refinement_config_state,
            ],
            outputs=[
                fbx_results_components['output_files'],
                pose_results_components['estimate_pose_btn'],
                fbx_results_components['generate_fbx_btn']
            ]  # Re-enable buttons when done
        )

        pose_cli_components['pose_generate_cli_btn'].click(
            fn=build_pose_cli_command,
            inputs=[
                entry_components['use_bbox'],
                entry_components['bbox_file'],
                entry_components['num_people'],
                entry_components['missing_bbox_behavior'],
                entry_components['fov_method'],
                entry_components['fov_file'],
                entry_components['sample_number'],
                entry_components['precision'],
            ],
            outputs=[pose_cli_components['pose_cli_command']]
        )

        fbx_cli_components['fbx_generate_cli_btn'].click(
            fn=build_fbx_cli_command,
            inputs=[
                fbx_processing_components['profile_name'],
                fbx_options_components['use_root_motion'],
                fbx_options_components['auto_floor'],
            ],
            outputs=[fbx_cli_components['fbx_cli_command']]
        )

        def cancel_current_jobs(input_file, pose_json_file):
            manager.estimation_manager.cancel_current_job()
            return (
                gr.update(interactive=(input_file is not None)),  # estimate_pose_btn
                gr.update(interactive=(pose_json_file is not None))  # generate_fbx_btn
            )

        pose_dev_components['pose_cancel_jobs_btn'].click(
            fn=cancel_current_jobs,
            inputs=[entry_components['input_file'], fbx_processing_components['pose_json_file']],
            outputs=[pose_results_components['estimate_pose_btn'], fbx_results_components['generate_fbx_btn']],
            cancels=[estimate_pose_click, generate_fbx_click]
        )

        fbx_dev_components['fbx_cancel_jobs_btn'].click(
            fn=cancel_current_jobs,
            inputs=[entry_components['input_file'], fbx_processing_components['pose_json_file']],
            outputs=[pose_results_components['estimate_pose_btn'], fbx_results_components['generate_fbx_btn']],
            cancels=[estimate_pose_click, generate_fbx_click]
        )
        
        # Language detection on page load
        app.queue()

        app.load(
            fn=detect_and_set_language,
            inputs=[],
            outputs=[lang_selector],
            js="""
            function() {
                const lang = navigator.language || navigator.userLanguage;
                const langCode = lang.split('-')[0].toLowerCase();
                const supportedLangs = ['en', 'ja', 'es', 'fr'];
                const detectedLang = supportedLangs.includes(langCode) ? langCode : 'en';
                return detectedLang;
            }
            """
        )
        
        app.load(
            fn=None,
            inputs=[],
            outputs=[],
            js="""
            function() {
                setTimeout(() => {
                    const lang = navigator.language || navigator.userLanguage;
                    const langCode = lang.split('-')[0].toLowerCase();
                    const supportedLangs = ['en', 'ja', 'es', 'fr'];
                    const detectedLang = supportedLangs.includes(langCode) ? langCode : 'en';
                    
                    const selects = document.querySelectorAll('select');
                    for (const select of selects) {
                        const options = Array.from(select.options);
                        const hasEn = options.some(opt => opt.value === 'en');
                        const hasJa = options.some(opt => opt.value === 'ja');
                        const hasEs = options.some(opt => opt.value === 'es');
                        const hasFr = options.some(opt => opt.value === 'fr');
                        if (hasEn && hasJa && hasEs && hasFr && options.length === 4) {
                            select.value = detectedLang;
                            select.dispatchEvent(new Event('input', { bubbles: true }));
                            select.dispatchEvent(new Event('change', { bubbles: true }));
                            break;
                        }
                    }
                }, 500);
            }
            """
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
    
    estimation_manager = PoseEstimationManager(
        checkpoint_path=checkpoint_path,
        mhr_path=mhr_path,
        detector_name=args.detector_name,
        detector_path=detector_path,
        fov_name=args.fov_name,
        fov_path=fov_path,
        precision="fp32"
    )
    
    data_prep_manager = FbxDataPrepManager()
    
    # Create manager with both managers
    manager = FbxifyManager(estimation_manager, data_prep_manager)

    app = create_app(manager)
    app.launch(
        server_name="0.0.0.0",
        server_port=7444,
        share=True,
    )
