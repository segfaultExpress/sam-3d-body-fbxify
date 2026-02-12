"""
Gradio application for SAM 3D Body to FBX conversion.

This module creates the Gradio UI using modular sections and delegates
all business logic to a Backend (LocalBackend or RemoteBackend).
"""
import os
import argparse
import json
import gradio as gr
from fbxify.i18n import Translator, DEFAULT_LANGUAGE
from fbxify.utils import _is_video_path
from fbxify.gradio_ui.header_section import create_header_section, update_header_language
from fbxify.gradio_ui.entry_section import (
    toggle_tracking_inputs,
    toggle_fov_inputs,
    update_entry_language,
    TRACKING_MODE_BBOX,
    TRACKING_MODE_COUNT,
    TRACKING_MODE_INFERENCE,
    TRACKING_MODE_INFERENCE_BBOX,
)
from fbxify.gradio_ui.tracking_section import (
    build_tracking_config_from_gui,
    load_tracking_configuration,
    save_tracking_configuration,
    update_tracking_language,
)
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


def create_app(backend):
    """Create Gradio app using modular sections and backend (LocalBackend or RemoteBackend)."""
    
    # Initialize translator with default language
    translator = Translator(DEFAULT_LANGUAGE)
    
    def estimate_pose(
        input_file,
        tracking_mode,
        bbox_file,
        num_people,
        missing_bbox_behavior,
        fov_method,
        fov_file,
        sample_number,
        precision,
        output_tracking_bbox,
        tracking_enabled,
        max_gap_frames,
        merge_max_gap_frames,
        min_tracklet_length,
        min_similarity,
        shape_distance_threshold,
        cam_distance_threshold,
        min_cam_similarity,
        pose_distance_threshold,
        min_pose_similarity,
        shape_maturity_frames,
        high_shape_override_cam,
        high_shape_threshold,
        iou_distance_threshold,
        shape_weight,
        cam_weight,
        pose_weight,
        iou_weight,
        use_shape_params,
        use_pred_cam_t,
        use_pose_aux,
        use_bbox_iou,
        export_frame_assignments,
        export_tracklet_detections,
        export_mot_bboxes,
        # Background filtering
        bg_filter_enabled,
        bg_min_bbox_height_px,
        bg_min_bbox_area_px2,
        bg_depth_max_z,
        bg_keep_nearest_z_quantile,
        bg_size_auto_method,
        bg_size_feature,
        bg_size_percentile,
        bg_tracklet_score_enabled,
        bg_tracklet_score_threshold,
        bg_min_tracklet_frames_for_scoring,
        bg_w_length,
        bg_w_size,
        bg_w_size_stability,
        bg_w_centering,
        bg_auto_roi_enabled,
        bg_auto_roi_window_frames,
        bg_auto_roi_point,
        bg_auto_roi_mad_k,
        bg_auto_roi_min_radius_px,
        bg_auto_roi_smoothing_alpha,
        bg_refine_second_pass,
        frame_batch_size,
        detection_batch_size,
        progress=gr.Progress(),
    ):
        """Estimate pose from image or video file - Step 1."""
        def progress_callback(progress_value, description):
            if progress is not None:
                if "Detecting" in (description or ""):
                    mapped = progress_value * 0.5
                else:
                    mapped = 0.5 + progress_value * 0.45
                progress(mapped, desc=description)

        tracking_config = None
        if isinstance(tracking_mode, list):
            tm = tracking_mode[0] if tracking_mode else TRACKING_MODE_COUNT
        else:
            tm = tracking_mode
        if tm == TRACKING_MODE_INFERENCE or tm == TRACKING_MODE_INFERENCE_BBOX:
            tracking_config = build_tracking_config_from_gui(
                tracking_enabled,
                max_gap_frames,
                merge_max_gap_frames,
                min_tracklet_length,
                min_similarity,
                shape_distance_threshold,
                cam_distance_threshold,
                min_cam_similarity,
                pose_distance_threshold,
                min_pose_similarity,
                shape_maturity_frames,
                high_shape_override_cam,
                high_shape_threshold,
                iou_distance_threshold,
                shape_weight,
                cam_weight,
                pose_weight,
                iou_weight,
                use_shape_params,
                use_pred_cam_t,
                use_pose_aux,
                use_bbox_iou,
                export_frame_assignments,
                export_tracklet_detections,
                export_mot_bboxes,
                bg_filter_enabled,
                bg_min_bbox_height_px,
                bg_min_bbox_area_px2,
                bg_depth_max_z,
                bg_keep_nearest_z_quantile,
                bg_size_auto_method,
                bg_size_feature,
                bg_size_percentile,
                bg_tracklet_score_enabled,
                bg_tracklet_score_threshold,
                bg_min_tracklet_frames_for_scoring,
                bg_w_length,
                bg_w_size,
                bg_w_size_stability,
                bg_w_centering,
                bg_auto_roi_enabled,
                bg_auto_roi_window_frames,
                bg_auto_roi_point,
                bg_auto_roi_mad_k,
                bg_auto_roi_min_radius_px,
                bg_auto_roi_smoothing_alpha,
                bg_refine_second_pass,
            )

        try:
            return backend.run_pose_estimation(
                input_file,
                tracking_mode,
                bbox_file,
                num_people,
                missing_bbox_behavior,
                fov_method,
                fov_file,
                sample_number,
                precision,
                output_tracking_bbox,
                tracking_config,
                frame_batch_size,
                detection_batch_size,
                translator.lang,
                progress_callback,
            )
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            if error_msg:
                raise gr.Error(translator.t("errors.error_occurred", error_type=error_type, error_msg=error_msg))
            raise gr.Error(translator.t("errors.error_occurred_no_msg", error_type=error_type))
    
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
        def processing_progress(progress_value, description):
            if progress is not None:
                progress(progress_value, desc=description)

        try:
            return backend.run_fbx_generation(
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
                refinement_config,
                translator.lang,
                processing_progress,
            )
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            if error_msg:
                raise gr.Error(translator.t("errors.error_occurred", error_type=error_type, error_msg=error_msg))
            raise gr.Error(translator.t("errors.error_occurred_no_msg", error_type=error_type))

    def _format_cli_arg(value: str) -> str:
        if value is None:
            return ""
        text = str(value)
        if any(char.isspace() for char in text):
            return f"\"{text}\""
        return text

    def build_pose_cli_command(tracking_mode, bbox_file, num_people, missing_bbox_behavior, fov_method,
                               fov_file, sample_number, precision, tracking_config_file, frame_batch_size, detection_batch_size):
        precision_map = {
            "FP32 (Full)": "fp32",
            "BF16 (Fast + Safer)": "bf16",
            "FP16 (Fastest)": "fp16"
        }
        precision_value = precision_map.get(precision, str(precision).lower() if precision else "fp32")
        tracking_mode_map = {
            TRACKING_MODE_BBOX: "bbox",
            TRACKING_MODE_COUNT: "count",
            TRACKING_MODE_INFERENCE: "inference",
            TRACKING_MODE_INFERENCE_BBOX: "inference_bbox",
        }
        tracking_mode_value = tracking_mode_map.get(tracking_mode, "count")

        cmd_parts = ["python", "-m", "fbxify.cli"]
        cmd_parts += ["--tracking_mode", tracking_mode_value]

        if tracking_mode_value == "bbox":
            cmd_parts += ["--bbox_file", "<BBOX_FILE>"]
            if missing_bbox_behavior:
                cmd_parts += ["--missing_bbox_behavior", _format_cli_arg(missing_bbox_behavior)]
        elif tracking_mode_value == "inference_bbox":
            cmd_parts += ["--bbox_file", "<BBOX_FILE>"]
            if missing_bbox_behavior:
                cmd_parts += ["--missing_bbox_behavior", _format_cli_arg(missing_bbox_behavior)]
            if tracking_config_file:
                cmd_parts += ["--tracking_config", _format_cli_arg(tracking_config_file)]
        else:
            if num_people is not None:
                cmd_parts += ["--num_people", str(int(num_people))]
            if tracking_mode_value == "inference" and tracking_config_file:
                cmd_parts += ["--tracking_config", _format_cli_arg(tracking_config_file)]

        if fov_method and fov_method != "Default":
            cmd_parts += ["--fov_method", _format_cli_arg(fov_method)]
            if fov_method == "File":
                cmd_parts += ["--fov_file", "<CAMERA_INTRINSICS_FILE>"]
            elif fov_method == "Sample":
                if sample_number is not None:
                    cmd_parts += ["--sample_number", str(int(sample_number))]

        cmd_parts += ["--precision", precision_value]
        if frame_batch_size is not None and int(frame_batch_size) > 1:
            cmd_parts += ["--frame_batch_size", str(int(frame_batch_size))]
        if detection_batch_size is not None and int(detection_batch_size) > 1:
            cmd_parts += ["--detection_batch_size", str(int(detection_batch_size))]
        cmd_parts += ["--save_estimation_json", "<POSE_JSON>"]
        cmd_parts.append("<INPUT_FILE>")
        return " ".join(cmd_parts)

    def build_fbx_cli_command(profile_name, use_root_motion, auto_floor, include_extrinsics,
                              extrinsics_file, extrinsics_sample_rate, extrinsics_scale,
                              extrinsics_invert_quaternion, extrinsics_invert_translation):
        cmd_parts = ["python", "-m", "fbxify.cli_fbx_generation"]

        if profile_name:
            cmd_parts += ["--profile", _format_cli_arg(profile_name)]

        if use_root_motion is False:
            cmd_parts.append("--no_root_motion")
        if auto_floor is False:
            cmd_parts.append("--no_auto_floor")

        cmd_parts += ["--output_dir", "<OUTPUT_DIR>"]
        cmd_parts += ["--refinement_config", "<REFINEMENT_JSON>"]
        if include_extrinsics:
            cmd_parts += ["--extrinsics_file", "<EXTRINSICS_FILE>"]
            if extrinsics_sample_rate is not None and int(extrinsics_sample_rate) != 0:
                cmd_parts += ["--extrinsics_sample_rate", str(int(extrinsics_sample_rate))]
            if extrinsics_scale is not None and float(extrinsics_scale) != 0.0:
                cmd_parts += ["--extrinsics_scale", str(float(extrinsics_scale))]
            if extrinsics_invert_quaternion:
                cmd_parts.append("--extrinsics_invert_quaternion")
            if extrinsics_invert_translation:
                cmd_parts.append("--extrinsics_invert_translation")

        cmd_parts.append("<POSE_JSON>")
        return " ".join(cmd_parts)
    
    def on_lang_change(lang):
        """Update all UI components when language changes."""
        # Update translator
        nonlocal translator
        translator = Translator(lang)

        # Get updates from each section
        header_updates = update_header_language(lang)
        entry_updates = update_entry_language(lang, translator)
        tracking_updates = update_tracking_language(lang, translator)
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
            *tracking_updates,  # tracking inputs
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

        def _extract_first_json_path(pose_json_file):
            if pose_json_file is None:
                return None
            if isinstance(pose_json_file, list):
                for item in pose_json_file:
                    path = item.name if hasattr(item, "name") else item
                    if isinstance(path, str) and path.lower().endswith(".json"):
                        return path
                return None
            path = pose_json_file.name if hasattr(pose_json_file, "name") else pose_json_file
            if isinstance(path, str) and path.lower().endswith(".json"):
                return path
            return None

        def sync_pose_json_to_pose(pose_json_file):
            """Sync pose JSON from FBX tab to Pose tab and state."""
            selected_json = _extract_first_json_path(pose_json_file)
            pose_update = gr.update()
            if isinstance(pose_json_file, list):
                pose_update = gr.update(value=pose_json_file)
            return (
                pose_update,  # pose pose_json_file
                selected_json,  # pose_json_state
                toggle_generate_fbx_button(selected_json),
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
                entry_components['tracking_mode'], entry_components['bbox_file'],
                entry_components['num_people'], entry_components['missing_bbox_behavior'], entry_components['fov_method'],
                entry_components['fov_file'], entry_components['sample_number'],
                entry_components['precision'],
                entry_components['frame_batch_size'],
                entry_components['detection_batch_size'],
                entry_components['tracking_group'],
                entry_components['tracking_enabled'],
                entry_components['max_gap_frames'],
                entry_components['merge_max_gap_frames'],
                entry_components['min_tracklet_length'],
                entry_components['min_similarity'],
                entry_components['shape_distance_threshold'],
                entry_components['cam_distance_threshold'],
                entry_components['min_cam_similarity'],
                entry_components['pose_distance_threshold'],
                entry_components['min_pose_similarity'],
                entry_components['shape_maturity_frames'],
                entry_components['high_shape_override_cam'],
                entry_components['high_shape_threshold'],
                entry_components['iou_distance_threshold'],
                entry_components['shape_weight'],
                entry_components['cam_weight'],
                entry_components['pose_weight'],
                entry_components['iou_weight'],
                entry_components['use_shape_params'],
                entry_components['use_pred_cam_t'],
                entry_components['use_pose_aux'],
                entry_components['use_bbox_iou'],
                entry_components['export_frame_assignments'],
                entry_components['export_tracklet_detections'],
                entry_components['export_mot_bboxes'],
                entry_components['bg_filter_enabled'],
                entry_components['bg_min_bbox_height_px'],
                entry_components['bg_min_bbox_area_px2'],
                entry_components['bg_depth_max_z'],
                entry_components['bg_keep_nearest_z_quantile'],
                entry_components['bg_size_auto_method'],
                entry_components['bg_size_feature'],
                entry_components['bg_size_percentile'],
                entry_components['bg_tracklet_score_enabled'],
                entry_components['bg_tracklet_score_threshold'],
                entry_components['bg_min_tracklet_frames_for_scoring'],
                entry_components['bg_w_length'],
                entry_components['bg_w_size'],
                entry_components['bg_w_size_stability'],
                entry_components['bg_w_centering'],
                entry_components['bg_auto_roi_enabled'],
                entry_components['bg_auto_roi_window_frames'],
                entry_components['bg_auto_roi_point'],
                entry_components['bg_auto_roi_mad_k'],
                entry_components['bg_auto_roi_min_radius_px'],
                entry_components['bg_auto_roi_smoothing_alpha'],
                entry_components['bg_refine_second_pass'],
                entry_components['tracking_config_upload'],
                entry_components['tracking_save_config_btn'],
                entry_components['tracking_config_download'],
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
                pose_dev_components['pose_output_tracking_bbox'],
                pose_dev_components['pose_run_bbox_detection_btn'],
                pose_dev_components['pose_run_bbox_detection_info'],
                pose_dev_components['pose_bbox_detection_output'],
                pose_dev_components['pose_rerun_tracking_file'],
                pose_dev_components['pose_rerun_tracking_btn'],
                pose_dev_components['pose_rerun_tracking_step_through'],
                pose_dev_components['pose_rerun_tracking_debug_start_frame'],
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
        
        # Tracking mode toggle
        entry_components['tracking_mode'].change(
            fn=toggle_tracking_inputs,
            inputs=[entry_components['tracking_mode']],
            outputs=[
                entry_components['bbox_file'],
                entry_components['num_people'],
                entry_components['missing_bbox_behavior'],
                entry_components['tracking_group'],
            ],
        )
        
        # FOV toggle
        entry_components['fov_method'].change(
            fn=toggle_fov_inputs,
            inputs=[entry_components['fov_method']],
            outputs=[entry_components['fov_file'], entry_components['sample_number']]
        )

        # Tracking config load/save
        entry_components['tracking_config_upload'].change(
            fn=load_tracking_configuration,
            inputs=[entry_components['tracking_config_upload']],
            outputs=[
                entry_components['tracking_enabled'],
                entry_components['max_gap_frames'],
                entry_components['merge_max_gap_frames'],
                entry_components['min_tracklet_length'],
                entry_components['min_similarity'],
                entry_components['shape_distance_threshold'],
                entry_components['cam_distance_threshold'],
                entry_components['min_cam_similarity'],
                entry_components['pose_distance_threshold'],
                entry_components['min_pose_similarity'],
                entry_components['shape_maturity_frames'],
                entry_components['high_shape_override_cam'],
                entry_components['high_shape_threshold'],
                entry_components['iou_distance_threshold'],
                entry_components['shape_weight'],
                entry_components['cam_weight'],
                entry_components['pose_weight'],
                entry_components['iou_weight'],
                entry_components['use_shape_params'],
                entry_components['use_pred_cam_t'],
                entry_components['use_pose_aux'],
                entry_components['use_bbox_iou'],
                entry_components['export_frame_assignments'],
                entry_components['export_tracklet_detections'],
                entry_components['export_mot_bboxes'],
                entry_components['bg_filter_enabled'],
                entry_components['bg_min_bbox_height_px'],
                entry_components['bg_min_bbox_area_px2'],
                entry_components['bg_depth_max_z'],
                entry_components['bg_keep_nearest_z_quantile'],
                entry_components['bg_size_auto_method'],
                entry_components['bg_size_feature'],
                entry_components['bg_size_percentile'],
                entry_components['bg_tracklet_score_enabled'],
                entry_components['bg_tracklet_score_threshold'],
                entry_components['bg_min_tracklet_frames_for_scoring'],
                entry_components['bg_w_length'],
                entry_components['bg_w_size'],
                entry_components['bg_w_size_stability'],
                entry_components['bg_w_centering'],
                entry_components['bg_auto_roi_enabled'],
                entry_components['bg_auto_roi_window_frames'],
                entry_components['bg_auto_roi_point'],
                entry_components['bg_auto_roi_mad_k'],
                entry_components['bg_auto_roi_min_radius_px'],
                entry_components['bg_auto_roi_smoothing_alpha'],
                entry_components['bg_refine_second_pass'],
            ],
        )

        entry_components['tracking_save_config_btn'].click(
            fn=save_tracking_configuration,
            inputs=[
                entry_components['tracking_enabled'],
                entry_components['max_gap_frames'],
                entry_components['merge_max_gap_frames'],
                entry_components['min_tracklet_length'],
                entry_components['min_similarity'],
                entry_components['shape_distance_threshold'],
                entry_components['cam_distance_threshold'],
                entry_components['min_cam_similarity'],
                entry_components['pose_distance_threshold'],
                entry_components['min_pose_similarity'],
                entry_components['shape_maturity_frames'],
                entry_components['high_shape_override_cam'],
                entry_components['high_shape_threshold'],
                entry_components['iou_distance_threshold'],
                entry_components['shape_weight'],
                entry_components['cam_weight'],
                entry_components['pose_weight'],
                entry_components['iou_weight'],
                entry_components['use_shape_params'],
                entry_components['use_pred_cam_t'],
                entry_components['use_pose_aux'],
                entry_components['use_bbox_iou'],
                entry_components['export_frame_assignments'],
                entry_components['export_tracklet_detections'],
                entry_components['export_mot_bboxes'],
                entry_components['bg_filter_enabled'],
                entry_components['bg_min_bbox_height_px'],
                entry_components['bg_min_bbox_area_px2'],
                entry_components['bg_depth_max_z'],
                entry_components['bg_keep_nearest_z_quantile'],
                entry_components['bg_size_auto_method'],
                entry_components['bg_size_feature'],
                entry_components['bg_size_percentile'],
                entry_components['bg_tracklet_score_enabled'],
                entry_components['bg_tracklet_score_threshold'],
                entry_components['bg_min_tracklet_frames_for_scoring'],
                entry_components['bg_w_length'],
                entry_components['bg_w_size'],
                entry_components['bg_w_size_stability'],
                entry_components['bg_w_centering'],
                entry_components['bg_auto_roi_enabled'],
                entry_components['bg_auto_roi_window_frames'],
                entry_components['bg_auto_roi_point'],
                entry_components['bg_auto_roi_mad_k'],
                entry_components['bg_auto_roi_min_radius_px'],
                entry_components['bg_auto_roi_smoothing_alpha'],
                entry_components['bg_refine_second_pass'],
            ],
            outputs=[entry_components['tracking_config_download']],
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
                entry_components['tracking_mode'],
                entry_components['bbox_file'],
                entry_components['num_people'],
                entry_components['missing_bbox_behavior'],
                entry_components['fov_method'],
                entry_components['fov_file'],
                entry_components['sample_number'],
                entry_components['precision'],
                pose_dev_components['pose_output_tracking_bbox'],
                entry_components['tracking_enabled'],
                entry_components['max_gap_frames'],
                entry_components['merge_max_gap_frames'],
                entry_components['min_tracklet_length'],
                entry_components['min_similarity'],
                entry_components['shape_distance_threshold'],
                entry_components['cam_distance_threshold'],
                entry_components['min_cam_similarity'],
                entry_components['pose_distance_threshold'],
                entry_components['min_pose_similarity'],
                entry_components['shape_maturity_frames'],
                entry_components['high_shape_override_cam'],
                entry_components['high_shape_threshold'],
                entry_components['iou_distance_threshold'],
                entry_components['shape_weight'],
                entry_components['cam_weight'],
                entry_components['pose_weight'],
                entry_components['iou_weight'],
                entry_components['use_shape_params'],
                entry_components['use_pred_cam_t'],
                entry_components['use_pose_aux'],
                entry_components['use_bbox_iou'],
                entry_components['export_frame_assignments'],
                entry_components['export_tracklet_detections'],
                entry_components['export_mot_bboxes'],
                entry_components['bg_filter_enabled'],
                entry_components['bg_min_bbox_height_px'],
                entry_components['bg_min_bbox_area_px2'],
                entry_components['bg_depth_max_z'],
                entry_components['bg_keep_nearest_z_quantile'],
                entry_components['bg_size_auto_method'],
                entry_components['bg_size_feature'],
                entry_components['bg_size_percentile'],
                entry_components['bg_tracklet_score_enabled'],
                entry_components['bg_tracklet_score_threshold'],
                entry_components['bg_min_tracklet_frames_for_scoring'],
                entry_components['bg_w_length'],
                entry_components['bg_w_size'],
                entry_components['bg_w_size_stability'],
                entry_components['bg_w_centering'],
                entry_components['bg_auto_roi_enabled'],
                entry_components['bg_auto_roi_window_frames'],
                entry_components['bg_auto_roi_point'],
                entry_components['bg_auto_roi_mad_k'],
                entry_components['bg_auto_roi_min_radius_px'],
                entry_components['bg_auto_roi_smoothing_alpha'],
                entry_components['bg_refine_second_pass'],
                entry_components['frame_batch_size'],
                entry_components['detection_batch_size'],
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
            json_path = _extract_first_json_path(pose_json_file)
            if json_path is None:
                return gr.update(interactive=False)
            
            try:
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
                entry_components['tracking_mode'],
                entry_components['bbox_file'],
                entry_components['num_people'],
                entry_components['missing_bbox_behavior'],
                entry_components['fov_method'],
                entry_components['fov_file'],
                entry_components['sample_number'],
                entry_components['precision'],
                entry_components['tracking_config_upload'],
                entry_components['frame_batch_size'],
                entry_components['detection_batch_size'],
            ],
            outputs=[pose_cli_components['pose_cli_command']]
        )

        fbx_cli_components['fbx_generate_cli_btn'].click(
            fn=build_fbx_cli_command,
            inputs=[
                fbx_processing_components['profile_name'],
                fbx_options_components['use_root_motion'],
                fbx_options_components['auto_floor'],
                fbx_options_components['include_extrinsics'],
                fbx_options_components['extrinsics_file'],
                fbx_options_components['extrinsics_sample_rate'],
                fbx_options_components['extrinsics_scale'],
                fbx_options_components['extrinsics_invert_quaternion'],
                fbx_options_components['extrinsics_invert_translation'],
            ],
            outputs=[fbx_cli_components['fbx_cli_command']]
        )

        def cancel_current_jobs(input_file, pose_json_file):
            backend.cancel_current_job()
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

        def run_bbox_detection_now(input_file, detection_batch_size, progress=gr.Progress()):
            """Run detection on all frames and export MOT file for download."""
            def prog(progress_value, desc):
                if progress:
                    progress(progress_value, desc=desc)
            try:
                mot_path = backend.run_detection_only(input_file, detection_batch_size, prog)
                return gr.update(value=mot_path, visible=True) if mot_path else gr.update(visible=False)
            except Exception as e:
                print(f"[Export detection to MOT] Error: {e}")
                raise

        pose_dev_components['pose_run_bbox_detection_btn'].click(
            fn=run_bbox_detection_now,
            inputs=[entry_components['input_file'], entry_components['detection_batch_size']],
            outputs=[pose_dev_components['pose_bbox_detection_output']],
        )

        def rerun_tracking(
            estimation_file,
            step_through,
            debug_start_frame,
            tracking_enabled,
            max_gap_frames,
            merge_max_gap_frames,
            min_tracklet_length,
            min_similarity,
            shape_distance_threshold,
            cam_distance_threshold,
            min_cam_similarity,
            pose_distance_threshold,
            min_pose_similarity,
            shape_maturity_frames,
            high_shape_override_cam,
            high_shape_threshold,
            iou_distance_threshold,
            shape_weight,
            cam_weight,
            pose_weight,
            iou_weight,
            use_shape_params,
            use_pred_cam_t,
            use_pose_aux,
            use_bbox_iou,
            export_frame_assignments,
            export_tracklet_detections,
            export_mot_bboxes,
            bg_filter_enabled,
            bg_min_bbox_height_px,
            bg_min_bbox_area_px2,
            bg_depth_max_z,
            bg_keep_nearest_z_quantile,
            bg_size_auto_method,
            bg_size_feature,
            bg_size_percentile,
            bg_tracklet_score_enabled,
            bg_tracklet_score_threshold,
            bg_min_tracklet_frames_for_scoring,
            bg_w_length,
            bg_w_size,
            bg_w_size_stability,
            bg_w_centering,
            bg_auto_roi_enabled,
            bg_auto_roi_window_frames,
            bg_auto_roi_point,
            bg_auto_roi_mad_k,
            bg_auto_roi_min_radius_px,
            bg_auto_roi_smoothing_alpha,
            bg_refine_second_pass,
        ):
            """Load estimation JSON, re-run tracking with debug, remap IDs, save and set as pose output."""
            tracking_config_params = {
                "tracking_enabled": True,  # force enabled so tracking runs
                "max_gap_frames": max_gap_frames,
                "merge_max_gap_frames": merge_max_gap_frames,
                "min_tracklet_length": min_tracklet_length,
                "min_similarity": min_similarity,
                "shape_distance_threshold": shape_distance_threshold,
                "cam_distance_threshold": cam_distance_threshold,
                "min_cam_similarity": min_cam_similarity,
                "pose_distance_threshold": pose_distance_threshold,
                "min_pose_similarity": min_pose_similarity,
                "shape_maturity_frames": shape_maturity_frames,
                "high_shape_override_cam": high_shape_override_cam,
                "high_shape_threshold": high_shape_threshold,
                "iou_distance_threshold": iou_distance_threshold,
                "shape_weight": shape_weight,
                "cam_weight": cam_weight,
                "pose_weight": pose_weight,
                "iou_weight": iou_weight,
                "use_shape_params": use_shape_params,
                "use_pred_cam_t": use_pred_cam_t,
                "use_pose_aux": use_pose_aux,
                "use_bbox_iou": use_bbox_iou,
                "export_frame_assignments": export_frame_assignments,
                "export_tracklet_detections": export_tracklet_detections,
                "export_mot_bboxes": export_mot_bboxes,
                "bg_filter_enabled": bg_filter_enabled,
                "bg_min_bbox_height_px": bg_min_bbox_height_px,
                "bg_min_bbox_area_px2": bg_min_bbox_area_px2,
                "bg_depth_max_z": bg_depth_max_z,
                "bg_keep_nearest_z_quantile": bg_keep_nearest_z_quantile,
                "bg_size_auto_method": bg_size_auto_method,
                "bg_size_feature": bg_size_feature,
                "bg_size_percentile": bg_size_percentile,
                "bg_tracklet_score_enabled": bg_tracklet_score_enabled,
                "bg_tracklet_score_threshold": bg_tracklet_score_threshold,
                "bg_min_tracklet_frames_for_scoring": bg_min_tracklet_frames_for_scoring,
                "bg_w_length": bg_w_length,
                "bg_w_size": bg_w_size,
                "bg_w_size_stability": bg_w_size_stability,
                "bg_w_centering": bg_w_centering,
                "bg_auto_roi_enabled": bg_auto_roi_enabled,
                "bg_auto_roi_window_frames": bg_auto_roi_window_frames,
                "bg_auto_roi_point": bg_auto_roi_point,
                "bg_auto_roi_mad_k": bg_auto_roi_mad_k,
                "bg_auto_roi_min_radius_px": bg_auto_roi_min_radius_px,
                "bg_auto_roi_smoothing_alpha": bg_auto_roi_smoothing_alpha,
                "bg_refine_second_pass": bg_refine_second_pass,
            }
            try:
                return backend.rerun_tracking(
                    estimation_file,
                    step_through,
                    debug_start_frame,
                    tracking_config_params,
                )
            except Exception as e:
                print(f"[Re-run tracking] Error: {e}")
                raise

        pose_dev_components['pose_rerun_tracking_btn'].click(
            fn=rerun_tracking,
            inputs=[
                pose_dev_components['pose_rerun_tracking_file'],
                pose_dev_components['pose_rerun_tracking_step_through'],
                pose_dev_components['pose_rerun_tracking_debug_start_frame'],
                entry_components['tracking_enabled'],
                entry_components['max_gap_frames'],
                entry_components['merge_max_gap_frames'],
                entry_components['min_tracklet_length'],
                entry_components['min_similarity'],
                entry_components['shape_distance_threshold'],
                entry_components['cam_distance_threshold'],
                entry_components['min_cam_similarity'],
                entry_components['pose_distance_threshold'],
                entry_components['min_pose_similarity'],
                entry_components['shape_maturity_frames'],
                entry_components['high_shape_override_cam'],
                entry_components['high_shape_threshold'],
                entry_components['iou_distance_threshold'],
                entry_components['shape_weight'],
                entry_components['cam_weight'],
                entry_components['pose_weight'],
                entry_components['iou_weight'],
                entry_components['use_shape_params'],
                entry_components['use_pred_cam_t'],
                entry_components['use_pose_aux'],
                entry_components['use_bbox_iou'],
                entry_components['export_frame_assignments'],
                entry_components['export_tracklet_detections'],
                entry_components['export_mot_bboxes'],
                entry_components['bg_filter_enabled'],
                entry_components['bg_min_bbox_height_px'],
                entry_components['bg_min_bbox_area_px2'],
                entry_components['bg_depth_max_z'],
                entry_components['bg_keep_nearest_z_quantile'],
                entry_components['bg_size_auto_method'],
                entry_components['bg_size_feature'],
                entry_components['bg_size_percentile'],
                entry_components['bg_tracklet_score_enabled'],
                entry_components['bg_tracklet_score_threshold'],
                entry_components['bg_min_tracklet_frames_for_scoring'],
                entry_components['bg_w_length'],
                entry_components['bg_w_size'],
                entry_components['bg_w_size_stability'],
                entry_components['bg_w_centering'],
                entry_components['bg_auto_roi_enabled'],
                entry_components['bg_auto_roi_window_frames'],
                entry_components['bg_auto_roi_point'],
                entry_components['bg_auto_roi_mad_k'],
                entry_components['bg_auto_roi_min_radius_px'],
                entry_components['bg_auto_roi_smoothing_alpha'],
                entry_components['bg_refine_second_pass'],
            ],
            outputs=[
                pose_results_components['pose_json_file'],
                fbx_processing_components['pose_json_file'],
                pose_json_state,
                fbx_results_components['generate_fbx_btn'],
            ],
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
    remote_url = os.environ.get("FBXIFY_REMOTE_WORKER_URL")
    if remote_url:
        from fbxify.backend import RemoteBackend
        backend = RemoteBackend(remote_url)
    else:
        from fbxify.pose_estimation_manager import PoseEstimationManager
        from fbxify.fbxify_manager import FbxifyManager
        from fbxify.fbxify_manager import FbxDataPrepManager
        from fbxify.tracking.tracking_manager import TrackingManager

        args = parse_args()
        if args.model == "vith":
            checkpoint_base_path = VITH_CHECKPOINT_PATH
        elif args.model == "dinov3":
            checkpoint_base_path = DINOV3_CHECKPOINT_PATH
        else:
            raise ValueError(f"Invalid model: {args.model}")

        checkpoint_path = os.path.join(checkpoint_base_path, "model.ckpt")
        mhr_path = os.path.join(checkpoint_base_path, "assets", "mhr_model.pt")

        detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
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
        manager = FbxifyManager(estimation_manager, data_prep_manager)
        tracking_manager = TrackingManager()
        from fbxify.backend import LocalBackend
        backend = LocalBackend(manager, tracking_manager)

    app = create_app(backend)
    app.launch(
        server_name="0.0.0.0",
        server_port=7444,
        share=True,
    )
