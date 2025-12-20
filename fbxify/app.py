"""
Gradio application for SAM 3D Body to FBX conversion.

This module creates the Gradio UI using modular sections and delegates
all business logic to FbxifyManager.
"""
import os
import argparse
import tempfile
import shutil
import gradio as gr
from fbxify.pose_estimator import PoseEstimator
from fbxify.fbxify_manager import FbxifyManager
from fbxify.i18n import Translator, DEFAULT_LANGUAGE
from fbxify.gradio_ui.header_section import create_header_section, update_header_language
from fbxify.gradio_ui.entry_section import create_entry_section, toggle_bbox_inputs, toggle_fov_inputs, toggle_mesh_inputs, update_entry_language
from fbxify.gradio_ui.output_section import create_output_section, update_output_language
from fbxify.gradio_ui.developer_section import create_developer_section, refresh_timestamps, update_developer_language
from fbxify.gradio_ui.refinement_section import create_refinement_section, build_refinement_config_from_gui

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


def create_app(manager: FbxifyManager):
    """Create Gradio app using modular sections and manager."""
    
    # Initialize translator with default language
    translator = Translator(DEFAULT_LANGUAGE)
    
    def process(input_file, profile_name, use_bbox, bbox_file, num_people, fov_method, 
                fov_file, sample_number, use_root_motion, include_mesh, lod, body_param_sample_num,
                create_visualization, debug_save_results, 
                # Refinement parameters
                refinement_enabled, interpolate_missing_keyframes,
                root_max_pos_speed, root_max_pos_accel, root_max_ang_speed_deg, root_max_ang_accel_deg,
                root_method, root_cutoff_hz, root_one_euro_min_cutoff, root_one_euro_beta, root_one_euro_d_cutoff,
                root_cutoff_xy_hz, root_cutoff_z_hz,
                hands_max_pos_speed, hands_max_pos_accel, hands_max_ang_speed_deg, hands_max_ang_accel_deg,
                hands_method, hands_cutoff_hz, hands_one_euro_min_cutoff, hands_one_euro_beta, hands_one_euro_d_cutoff,
                fingers_max_pos_speed, fingers_max_pos_accel, fingers_max_ang_speed_deg, fingers_max_ang_accel_deg,
                fingers_method, fingers_cutoff_hz, fingers_one_euro_min_cutoff, fingers_one_euro_beta, fingers_one_euro_d_cutoff,
                head_max_pos_speed, head_max_pos_accel, head_max_ang_speed_deg, head_max_ang_accel_deg,
                head_method, head_cutoff_hz, head_one_euro_min_cutoff, head_one_euro_beta, head_one_euro_d_cutoff,
                legs_max_pos_speed, legs_max_pos_accel, legs_max_ang_speed_deg, legs_max_ang_accel_deg,
                legs_method, legs_cutoff_hz, legs_one_euro_min_cutoff, legs_one_euro_beta, legs_one_euro_d_cutoff,
                default_max_pos_speed, default_max_pos_accel, default_max_ang_speed_deg, default_max_ang_accel_deg,
                default_method, default_cutoff_hz, default_one_euro_min_cutoff, default_one_euro_beta, default_one_euro_d_cutoff,
                progress=None):
        """Process image or video file using FbxifyManager."""
        output_files = []
        temp_dir = None
        
        try:
            if input_file is None:
                return None

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

            # Process frames
            def progress_callback(progress_value, description):
                if progress is not None:
                    progress(progress_value, desc=description)

            # Convert lod to int if it's a float from slider
            lod_int = int(lod) if lod is not None else -1
            body_param_sample_num_int = int(body_param_sample_num) if body_param_sample_num is not None else 5
            
            process_result = manager.process_frames(
                frame_paths,
                profile_name,
                num_people,
                bbox_dict,
                use_root_motion,
                create_visualization,
                fps,
                progress_callback,
                lod=lod_int if include_mesh else -1,
                body_param_sample_num=body_param_sample_num_int if include_mesh else 5
            )

            # Save debug results if requested
            if debug_save_results:
                manager.save_debug_results(process_result)

            # Build refinement config from GUI values
            refinement_config = build_refinement_config_from_gui(
                refinement_enabled,
                interpolate_missing_keyframes,
                translator,
                root_max_pos_speed, root_max_pos_accel, root_max_ang_speed_deg, root_max_ang_accel_deg,
                root_method, root_cutoff_hz, root_one_euro_min_cutoff, root_one_euro_beta, root_one_euro_d_cutoff,
                root_cutoff_xy_hz, root_cutoff_z_hz,
                hands_max_pos_speed, hands_max_pos_accel, hands_max_ang_speed_deg, hands_max_ang_accel_deg,
                hands_method, hands_cutoff_hz, hands_one_euro_min_cutoff, hands_one_euro_beta, hands_one_euro_d_cutoff,
                fingers_max_pos_speed, fingers_max_pos_accel, fingers_max_ang_speed_deg, fingers_max_ang_accel_deg,
                fingers_method, fingers_cutoff_hz, fingers_one_euro_min_cutoff, fingers_one_euro_beta, fingers_one_euro_d_cutoff,
                head_max_pos_speed, head_max_pos_accel, head_max_ang_speed_deg, head_max_ang_accel_deg,
                head_method, head_cutoff_hz, head_one_euro_min_cutoff, head_one_euro_beta, head_one_euro_d_cutoff,
                legs_max_pos_speed, legs_max_pos_accel, legs_max_ang_speed_deg, legs_max_ang_accel_deg,
                legs_method, legs_cutoff_hz, legs_one_euro_min_cutoff, legs_one_euro_beta, legs_one_euro_d_cutoff,
                default_max_pos_speed, default_max_pos_accel, default_max_ang_speed_deg, default_max_ang_accel_deg,
                default_method, default_cutoff_hz, default_one_euro_min_cutoff, default_one_euro_beta, default_one_euro_d_cutoff,
            )

            # Apply refinement if enabled
            if refinement_config is not None:
                def refinement_progress(progress_value, description):
                    if progress is not None:
                        progress(0.4 + progress_value * 0.1, desc=description)
                
                process_result = manager.apply_refinement(
                    process_result,
                    refinement_config,
                    refinement_progress
                )

            # Export FBX files
            def export_progress(progress_value, description):
                if progress is not None:
                    progress(0.5 + progress_value * 0.4, desc=description)

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
                mesh_obj_path=process_result.mesh_obj_path,
                lod_fbx_path=lod_fbx_path
            )
            output_files.extend(fbx_paths)

            # Export visualization if requested
            if create_visualization and process_result.visualization_data:
                def vis_progress(progress_value, description):
                    if progress is not None:
                        progress(0.9 + progress_value * 0.1, desc=description)

                visualization_output = manager.export_visualization(
                    process_result.visualization_data,
                    progress_callback=vis_progress
                )
                if visualization_output is not None:
                    output_files.append(visualization_output)

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
                
        return output_files if output_files else None

    def debug_reexport_results(timestamp, use_root_motion, create_visualization,
                                # Refinement parameters
                                refinement_enabled, interpolate_missing_keyframes,
                                root_max_pos_speed, root_max_pos_accel, root_max_ang_speed_deg, root_max_ang_accel_deg,
                                root_method, root_cutoff_hz, root_one_euro_min_cutoff, root_one_euro_beta, root_one_euro_d_cutoff,
                                root_cutoff_xy_hz, root_cutoff_z_hz,
                                hands_max_pos_speed, hands_max_pos_accel, hands_max_ang_speed_deg, hands_max_ang_accel_deg,
                                hands_method, hands_cutoff_hz, hands_one_euro_min_cutoff, hands_one_euro_beta, hands_one_euro_d_cutoff,
                                fingers_max_pos_speed, fingers_max_pos_accel, fingers_max_ang_speed_deg, fingers_max_ang_accel_deg,
                                fingers_method, fingers_cutoff_hz, fingers_one_euro_min_cutoff, fingers_one_euro_beta, fingers_one_euro_d_cutoff,
                                head_max_pos_speed, head_max_pos_accel, head_max_ang_speed_deg, head_max_ang_accel_deg,
                                head_method, head_cutoff_hz, head_one_euro_min_cutoff, head_one_euro_beta, head_one_euro_d_cutoff,
                                legs_max_pos_speed, legs_max_pos_accel, legs_max_ang_speed_deg, legs_max_ang_accel_deg,
                                legs_method, legs_cutoff_hz, legs_one_euro_min_cutoff, legs_one_euro_beta, legs_one_euro_d_cutoff,
                                default_max_pos_speed, default_max_pos_accel, default_max_ang_speed_deg, default_max_ang_accel_deg,
                                default_method, default_cutoff_hz, default_one_euro_min_cutoff, default_one_euro_beta, default_one_euro_d_cutoff,
                                progress=None):
        """Re-export results from saved debug data."""
        if timestamp is None or timestamp == "" or timestamp == translator.t("ui.debug_no_saved_results"):
            raise ValueError("Please select a timestamp to re-export")
        
        def progress_callback(progress_value, description):
            if progress is not None:
                progress(progress_value, desc=description)

        # Build refinement config from GUI values
        refinement_config = build_refinement_config_from_gui(
            refinement_enabled,
            interpolate_missing_keyframes,
            translator,
            root_max_pos_speed, root_max_pos_accel, root_max_ang_speed_deg, root_max_ang_accel_deg,
            root_method, root_cutoff_hz, root_one_euro_min_cutoff, root_one_euro_beta, root_one_euro_d_cutoff,
            root_cutoff_xy_hz, root_cutoff_z_hz,
            hands_max_pos_speed, hands_max_pos_accel, hands_max_ang_speed_deg, hands_max_ang_accel_deg,
            hands_method, hands_cutoff_hz, hands_one_euro_min_cutoff, hands_one_euro_beta, hands_one_euro_d_cutoff,
            fingers_max_pos_speed, fingers_max_pos_accel, fingers_max_ang_speed_deg, fingers_max_ang_accel_deg,
            fingers_method, fingers_cutoff_hz, fingers_one_euro_min_cutoff, fingers_one_euro_beta, fingers_one_euro_d_cutoff,
            head_max_pos_speed, head_max_pos_accel, head_max_ang_speed_deg, head_max_ang_accel_deg,
            head_method, head_cutoff_hz, head_one_euro_min_cutoff, head_one_euro_beta, head_one_euro_d_cutoff,
            legs_max_pos_speed, legs_max_pos_accel, legs_max_ang_speed_deg, legs_max_ang_accel_deg,
            legs_method, legs_cutoff_hz, legs_one_euro_min_cutoff, legs_one_euro_beta, legs_one_euro_d_cutoff,
            default_max_pos_speed, default_max_pos_accel, default_max_ang_speed_deg, default_max_ang_accel_deg,
            default_method, default_cutoff_hz, default_one_euro_min_cutoff, default_one_euro_beta, default_one_euro_d_cutoff,
        )

        output_files = manager.reexport_debug_results(
            int(timestamp),
            use_root_motion,
            create_visualization,
            refinement_config,
            progress_callback
        )
        
        return output_files if output_files else None

    def on_lang_change(lang):
        """Update all UI components when language changes."""
        # Update translator
        nonlocal translator
        translator = Translator(lang)
        
        # Get updates from each section
        header_updates = update_header_language(lang)
        entry_updates = update_entry_language(lang, translator)
        output_update = update_output_language(lang)
        developer_updates = update_developer_language(lang)
        
        # Combine all updates
        return (
            *header_updates,  # heading, description
            *entry_updates,   # profile, input_file, use_bbox, bbox_file, num_people, fov_method, fov_file, sample_number, use_root_motion, include_mesh, lod, body_param_sample_num, generate_btn
            output_update,    # output_files
            *developer_updates,  # create_visualization, debug_save_results, debug_saved_timestamps, debug_reexport_btn, debug_clear_btn
        )

    def detect_and_set_language():
        """Detect browser language and return it for initial setup."""
        return DEFAULT_LANGUAGE

    # Create UI sections
    with gr.Blocks(title=translator.t("app.title")) as app:
        # Header section
        heading_md, description_md, lang_selector = create_header_section(translator)
        
        # Layout: Two columns
        with gr.Row():
            with gr.Column():
                # Entry section - create components within column
                entry_components = create_entry_section(translator)
                
                # Developer section - creates its own accordion
                dev_components = create_developer_section(
                    translator,
                    lambda: manager.get_saved_timestamps()
                )
                
            with gr.Column():
                # Output section
                output_files = create_output_section(translator)
                
                # Refinement section
                refinement_components = create_refinement_section(translator)
        
        # Wire up event handlers
        # Language change
        lang_selector.change(
            fn=on_lang_change,
            inputs=[lang_selector],
            outputs=[
                heading_md, description_md,  # header
                entry_components['profile_name'], entry_components['input_file'],
                entry_components['use_bbox'], entry_components['bbox_file'],
                entry_components['num_people'], entry_components['fov_method'],
                entry_components['fov_file'], entry_components['sample_number'],
                entry_components['use_root_motion'], entry_components['include_mesh'],
                entry_components['lod'], entry_components['body_param_sample_num'],
                entry_components['generate_btn'],  # entry
                output_files,  # output
                dev_components['create_visualization'], dev_components['debug_save_results'],
                dev_components['debug_saved_timestamps'], dev_components['debug_reexport_btn'],
                dev_components['debug_clear_btn']  # developer
            ]
        )
        
        # Bbox toggle
        entry_components['use_bbox'].change(
            fn=toggle_bbox_inputs,
            inputs=[entry_components['use_bbox']],
            outputs=[entry_components['bbox_file'], entry_components['num_people']]
        )
        
        # FOV toggle
        entry_components['fov_method'].change(
            fn=toggle_fov_inputs,
            inputs=[entry_components['fov_method']],
            outputs=[entry_components['fov_file'], entry_components['sample_number']]
        )
        
        # Mesh toggle
        entry_components['include_mesh'].change(
            fn=toggle_mesh_inputs,
            inputs=[entry_components['include_mesh']],
            outputs=[entry_components['lod'], entry_components['body_param_sample_num']]
        )
        
        # Collect refinement inputs
        refinement_inputs = [
            refinement_components['refinement_enabled'],
            refinement_components['interpolate_missing_keyframes'],
            refinement_components['profile_components']['root']['max_pos_speed'],
            refinement_components['profile_components']['root']['max_pos_accel'],
            refinement_components['profile_components']['root']['max_ang_speed_deg'],
            refinement_components['profile_components']['root']['max_ang_accel_deg'],
            refinement_components['profile_components']['root']['method'],
            refinement_components['profile_components']['root']['cutoff_hz'],
            refinement_components['profile_components']['root']['one_euro_min_cutoff'],
            refinement_components['profile_components']['root']['one_euro_beta'],
            refinement_components['profile_components']['root']['one_euro_d_cutoff'],
            refinement_components['profile_components']['root']['root_cutoff_xy_hz'],
            refinement_components['profile_components']['root']['root_cutoff_z_hz'],
            refinement_components['profile_components']['hands']['max_pos_speed'],
            refinement_components['profile_components']['hands']['max_pos_accel'],
            refinement_components['profile_components']['hands']['max_ang_speed_deg'],
            refinement_components['profile_components']['hands']['max_ang_accel_deg'],
            refinement_components['profile_components']['hands']['method'],
            refinement_components['profile_components']['hands']['cutoff_hz'],
            refinement_components['profile_components']['hands']['one_euro_min_cutoff'],
            refinement_components['profile_components']['hands']['one_euro_beta'],
            refinement_components['profile_components']['hands']['one_euro_d_cutoff'],
            refinement_components['profile_components']['fingers']['max_pos_speed'],
            refinement_components['profile_components']['fingers']['max_pos_accel'],
            refinement_components['profile_components']['fingers']['max_ang_speed_deg'],
            refinement_components['profile_components']['fingers']['max_ang_accel_deg'],
            refinement_components['profile_components']['fingers']['method'],
            refinement_components['profile_components']['fingers']['cutoff_hz'],
            refinement_components['profile_components']['fingers']['one_euro_min_cutoff'],
            refinement_components['profile_components']['fingers']['one_euro_beta'],
            refinement_components['profile_components']['fingers']['one_euro_d_cutoff'],
            refinement_components['profile_components']['head']['max_pos_speed'],
            refinement_components['profile_components']['head']['max_pos_accel'],
            refinement_components['profile_components']['head']['max_ang_speed_deg'],
            refinement_components['profile_components']['head']['max_ang_accel_deg'],
            refinement_components['profile_components']['head']['method'],
            refinement_components['profile_components']['head']['cutoff_hz'],
            refinement_components['profile_components']['head']['one_euro_min_cutoff'],
            refinement_components['profile_components']['head']['one_euro_beta'],
            refinement_components['profile_components']['head']['one_euro_d_cutoff'],
            refinement_components['profile_components']['legs']['max_pos_speed'],
            refinement_components['profile_components']['legs']['max_pos_accel'],
            refinement_components['profile_components']['legs']['max_ang_speed_deg'],
            refinement_components['profile_components']['legs']['max_ang_accel_deg'],
            refinement_components['profile_components']['legs']['method'],
            refinement_components['profile_components']['legs']['cutoff_hz'],
            refinement_components['profile_components']['legs']['one_euro_min_cutoff'],
            refinement_components['profile_components']['legs']['one_euro_beta'],
            refinement_components['profile_components']['legs']['one_euro_d_cutoff'],
            refinement_components['profile_components']['default']['max_pos_speed'],
            refinement_components['profile_components']['default']['max_pos_accel'],
            refinement_components['profile_components']['default']['max_ang_speed_deg'],
            refinement_components['profile_components']['default']['max_ang_accel_deg'],
            refinement_components['profile_components']['default']['method'],
            refinement_components['profile_components']['default']['cutoff_hz'],
            refinement_components['profile_components']['default']['one_euro_min_cutoff'],
            refinement_components['profile_components']['default']['one_euro_beta'],
            refinement_components['profile_components']['default']['one_euro_d_cutoff'],
        ]

        # Generate button
        entry_components['generate_btn'].click(
            fn=process,
            inputs=[
                entry_components['input_file'],
                entry_components['profile_name'],
                entry_components['use_bbox'],
                entry_components['bbox_file'],
                entry_components['num_people'],
                entry_components['fov_method'],
                entry_components['fov_file'],
                entry_components['sample_number'],
                entry_components['use_root_motion'],
                entry_components['include_mesh'],
                entry_components['lod'],
                entry_components['body_param_sample_num'],
                dev_components['create_visualization'],
                dev_components['debug_save_results']
            ] + refinement_inputs,
            outputs=[output_files]
        )
        
        # Debug handlers
        def refresh_timestamps_wrapper():
            return refresh_timestamps(
                lambda: manager.get_saved_timestamps(),
                translator
            )
        
        def clear_and_refresh():
            manager.clear_saved_results()
            return refresh_timestamps_wrapper()
        
        dev_components['debug_refresh_btn'].click(
            fn=refresh_timestamps_wrapper,
            inputs=[],
            outputs=[dev_components['debug_saved_timestamps']]
        )
        
        dev_components['debug_clear_btn'].click(
            fn=clear_and_refresh,
            inputs=[],
            outputs=[dev_components['debug_saved_timestamps']]
        )
        
        dev_components['debug_reexport_btn'].click(
            fn=debug_reexport_results,
            inputs=[
                dev_components['debug_saved_timestamps'],
                entry_components['use_root_motion'],
                dev_components['create_visualization']
            ] + refinement_inputs,
            outputs=[output_files]
        )
        
        # Language detection on page load
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
    
    estimator = PoseEstimator(
        checkpoint_path=checkpoint_path,
        mhr_path=mhr_path,
        detector_name=args.detector_name,
        detector_path=detector_path,
        fov_name=args.fov_name,
        fov_path=fov_path
    )
    
    # Create manager with estimator
    manager = FbxifyManager(estimator, DEBUG_RESULTS_DIR)

    app = create_app(manager)
    app.launch(
        server_name="0.0.0.0",
        server_port=7444,
        share=True,
    )
