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
from fbxify.pose_estimation_manager import PoseEstimationManager
from fbxify.fbx_data_prep_manager import FbxDataPrepManager
from fbxify.fbxify_manager import FbxifyManager
from fbxify.i18n import Translator, DEFAULT_LANGUAGE
from fbxify.gradio_ui.header_section import create_header_section, update_header_language
from fbxify.gradio_ui.entry_section import create_entry_section, toggle_bbox_inputs, toggle_fov_inputs, update_entry_language
from fbxify.gradio_ui.fbx_processing_section import create_fbx_processing_section, update_fbx_processing_language, toggle_generate_fbx_button
from fbxify.gradio_ui.fbx_options_section import create_fbx_options_section, toggle_mesh_inputs, toggle_personalized_body, update_fbx_options_language
from fbxify import VERSION
from fbxify.gradio_ui.developer_section import create_developer_section, update_developer_language
from fbxify.gradio_ui.refinement_section import create_refinement_section, build_refinement_config_from_gui

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
    
    def estimate_pose(input_file, profile_name, use_bbox, bbox_file, num_people, missing_bbox_behavior, fov_method, 
                     fov_file, sample_number, progress=gr.Progress()):
        """Estimate pose from image or video file - Step 1."""
        temp_dir = None
        
        try:
            if input_file is None:
                return (
                    gr.update(),  # pose_json_file
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
            gr.update(value=estimation_json_path),  # pose_json_file
            gr.update(interactive=True),  # generate_fbx_btn
            gr.update(interactive=(input_file is not None))   # estimate_pose_btn (re-enable only if file still exists)
        )
    
    def generate_fbx(pose_json_file, profile_name, use_root_motion, include_mesh, use_personalized_body, lod, outlier_removal_percent,
                    input_file,
                    refinement_config,  # Single refinement config object from state
                    progress=gr.Progress()):
        """Generate FBX from pose estimation JSON - Step 2."""
        output_files = []

        
        try:
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
            # Convert outlier_removal_percent to float
            outlier_percent = float(outlier_removal_percent) if outlier_removal_percent is not None else 10.0
            
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
                lang=translator.lang
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
                mesh_obj_path=process_result.mesh_obj_path,  # Use generated mesh from JSON
                lod_fbx_path=lod_fbx_path,
                lang=translator.lang
            )
            output_files.extend(fbx_paths)

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
            gr.update(interactive=(input_file is not None))  # estimate_pose_btn (re-enable only if file still exists)
        )
    
    def on_lang_change(lang):
        """Update all UI components when language changes."""
        # Update translator
        nonlocal translator
        translator = Translator(lang)
        
        # Get updates from each section
        header_updates = update_header_language(lang)
        entry_updates = update_entry_language(lang, translator)
        fbx_processing_updates = update_fbx_processing_language(lang)
        fbx_options_updates = update_fbx_options_language(lang, translator)
        developer_updates = update_developer_language(lang)
        
        # Combine all updates
        return (
            *header_updates,  # heading, description, tabs
            *entry_updates,   # input_file, use_bbox, bbox_file, num_people, missing_bbox_behavior, fov_method, fov_file, sample_number, estimate_pose_btn
            *fbx_processing_updates,    # profile_name, pose_json_file, generate_fbx_btn, output_files
            *fbx_options_updates,  # use_root_motion, include_mesh, use_personalized_body, lod, outlier_removal_percent
            *developer_updates,  # (empty now)
        )

    def detect_and_set_language():
        """Detect browser language and return it for initial setup."""
        return DEFAULT_LANGUAGE

    # Create UI sections
    with gr.Blocks(title=translator.t("app.title")) as app:
        # Header section (now returns heading_md, description_md, tabs, lang_selector)
        heading_md, description_md, tabs, lang_selector = create_header_section(translator)
        
        # Layout: Two columns
        with gr.Row():
            with gr.Column():
                # Entry section - create components within column
                entry_components = create_entry_section(translator)
                
                # Developer section - creates its own accordion
                dev_components = create_developer_section(translator)
                
            with gr.Column():
                # FBX processing section (returns dict with profile_name, pose_json_file, generate_fbx_btn, output_files)
                fbx_processing_components = create_fbx_processing_section(translator)
                
                # FBX options section
                fbx_options_components = create_fbx_options_section(translator)
                
                # Refinement section
                refinement_components = create_refinement_section(translator)
        
        # Wire up event handlers
        # Language change
        lang_selector.change(
            fn=on_lang_change,
            inputs=[lang_selector],
            outputs=[
                heading_md, description_md, tabs,  # header (heading, description, tabs)
                entry_components['input_file'],
                entry_components['use_bbox'], entry_components['bbox_file'],
                entry_components['num_people'], entry_components['missing_bbox_behavior'], entry_components['fov_method'],
                entry_components['fov_file'], entry_components['sample_number'],
                entry_components['estimate_pose_btn'],  # entry
                fbx_processing_components['profile_name'], fbx_processing_components['pose_json_file'], fbx_processing_components['generate_fbx_btn'], fbx_processing_components['output_files'],  # fbx processing
                fbx_options_components['auto_run'], fbx_options_components['use_root_motion'], fbx_options_components['include_mesh'],
                fbx_options_components['use_personalized_body'], fbx_options_components['lod'], fbx_options_components['outlier_removal_percent']  # fbx options
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
        
        # Mesh toggle - show/hide lod and use_personalized_body
        fbx_options_components['include_mesh'].change(
            fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
            inputs=[fbx_options_components['include_mesh']],
            outputs=[fbx_options_components['lod'], fbx_options_components['use_personalized_body']]
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

        def toggle_estimate_pose_button(input_file):
            """Enable/disable Estimate Pose button based on whether file is uploaded."""
            return gr.update(interactive=(input_file is not None))
        
        # Enable/disable Estimate Pose button based on file upload
        entry_components['input_file'].change(
            fn=toggle_estimate_pose_button,
            inputs=[entry_components['input_file']],
            outputs=[entry_components['estimate_pose_btn']]
        )
        
        # Helper function to conditionally auto-run generate_fbx
        def auto_run_generate_fbx(pose_json_file, auto_run, profile_name, use_root_motion, include_mesh, use_personalized_body, lod, 
                                  outlier_removal_percent, input_file, *refinement_inputs, progress=gr.Progress()):
            """Conditionally trigger generate_fbx if auto_run is enabled."""
            if not auto_run or pose_json_file is None:
                # Just re-enable estimate_pose_btn if input_file still exists
                return None, gr.update(interactive=(input_file is not None))
            
            # Build refinement config
            refinement_cfg = build_and_log_config(*refinement_inputs)
            
            # Call generate_fbx (progress will be automatically injected by Gradio)
            return generate_fbx(
                pose_json_file,
                profile_name,
                use_root_motion,
                include_mesh,
                use_personalized_body,
                lod,
                outlier_removal_percent,
                input_file,
                refinement_cfg,
                progress=progress
            )
        
        # Estimate Pose button (Step 1)
        # Disable both Estimate Pose and Generate FBX buttons immediately when Estimate Pose is clicked
        estimate_pose_click = entry_components['estimate_pose_btn'].click(
            fn=lambda: (gr.update(), gr.update(interactive=False), gr.update(interactive=False)),  # Disable both buttons immediately
            inputs=[],
            outputs=[fbx_processing_components['pose_json_file'], fbx_processing_components['generate_fbx_btn'], entry_components['estimate_pose_btn']]
        ).then(
            fn=estimate_pose,
            inputs=[
                entry_components['input_file'],
                fbx_processing_components['profile_name'],
                entry_components['use_bbox'],
                entry_components['bbox_file'],
                entry_components['num_people'],
                entry_components['missing_bbox_behavior'],
                entry_components['fov_method'],
                entry_components['fov_file'],
                entry_components['sample_number']
            ],
            outputs=[fbx_processing_components['pose_json_file'], fbx_processing_components['generate_fbx_btn'], entry_components['estimate_pose_btn']]  # Update file and re-enable buttons when done
        )
        
        # Auto-run: If auto_run is checked, automatically trigger generate_fbx after estimate_pose completes
        estimate_pose_click.then(
            fn=auto_run_generate_fbx,
            inputs=[
                fbx_processing_components['pose_json_file'],
                fbx_options_components['auto_run'],
                fbx_processing_components['profile_name'],
                fbx_options_components['use_root_motion'],
                fbx_options_components['include_mesh'],
                fbx_options_components['use_personalized_body'],
                fbx_options_components['lod'],
                fbx_options_components['outlier_removal_percent'],
                entry_components['input_file'],
                *all_refinement_inputs
            ],
            outputs=[fbx_processing_components['output_files'], entry_components['estimate_pose_btn']],
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
            outputs=[fbx_processing_components['generate_fbx_btn']]
        )
        
        # Generate FBX button (Step 2)
        # Disable Estimate Pose button immediately when Generate FBX is clicked
        fbx_processing_components['generate_fbx_btn'].click(
            fn=lambda: gr.update(interactive=False),  # Disable Estimate Pose button immediately
            inputs=[],
            outputs=[entry_components['estimate_pose_btn']]
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
                fbx_options_components['include_mesh'],
                fbx_options_components['use_personalized_body'],
                fbx_options_components['lod'],
                fbx_options_components['outlier_removal_percent'],
                entry_components['input_file'],  # Add input_file to check if it still exists
                refinement_config_state,
            ],
            outputs=[fbx_processing_components['output_files'], entry_components['estimate_pose_btn']]  # Re-enable Estimate Pose when done
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
    
    estimation_manager = PoseEstimationManager(
        checkpoint_path=checkpoint_path,
        mhr_path=mhr_path,
        detector_name=args.detector_name,
        detector_path=detector_path,
        fov_name=args.fov_name,
        fov_path=fov_path
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
