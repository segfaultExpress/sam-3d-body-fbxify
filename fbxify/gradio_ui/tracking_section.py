"""
Gradio UI components for inference tracking configuration.
"""
import json
import tempfile
import gradio as gr
from typing import Dict, Any, Tuple, List
from fbxify.i18n import Translator
from fbxify.tracking.tracking_config import TrackingConfig


def create_tracking_section(translator: Translator) -> Dict[str, Any]:
    components: Dict[str, Any] = {}

    with gr.Accordion(label=translator.t("ui.tracking.config_title"), open=False, visible=False) as tracking_group:
        components["tracking_group"] = tracking_group

        components["tracking_enabled"] = gr.Checkbox(
            label=translator.t("ui.tracking.enabled"),
            value=True,
            info=translator.t("ui.tracking.enabled_info"),
        )

        with gr.Group():
            gr.Markdown(f"### {translator.t('ui.tracking.thresholds_title')}")
            with gr.Row():
                components["max_gap_frames"] = gr.Number(
                    label=translator.t("ui.tracking.max_gap_frames"),
                    value=5,
                    precision=0,
                    minimum=0,
                    info=translator.t("ui.tracking.max_gap_frames_info"),
                )
                components["merge_max_gap_frames"] = gr.Number(
                    label=translator.t("ui.tracking.merge_max_gap_frames"),
                    value=12,
                    precision=0,
                    minimum=0,
                    info=translator.t("ui.tracking.merge_max_gap_frames_info"),
                )
                components["min_tracklet_length"] = gr.Number(
                    label=translator.t("ui.tracking.min_tracklet_length"),
                    value=2,
                    precision=0,
                    minimum=1,
                    info=translator.t("ui.tracking.min_tracklet_length_info"),
                )

            components["min_similarity"] = gr.Number(
                label=translator.t("ui.tracking.min_similarity"),
                value=0.4,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                info=translator.t("ui.tracking.min_similarity_info"),
            )

            with gr.Row():
                components["shape_distance_threshold"] = gr.Number(
                    label=translator.t("ui.tracking.shape_distance_threshold"),
                    value=1.2,
                    minimum=0.0,
                    step=0.05,
                    info=translator.t("ui.tracking.shape_distance_threshold_info"),
                )
                components["cam_distance_threshold"] = gr.Number(
                    label=translator.t("ui.tracking.cam_distance_threshold"),
                    value=0.8,
                    minimum=0.0,
                    step=0.05,
                    info=translator.t("ui.tracking.cam_distance_threshold_info"),
                )
                components["min_cam_similarity"] = gr.Number(
                    label=translator.t("ui.tracking.min_cam_similarity"),
                    value=0.01,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    info=translator.t("ui.tracking.min_cam_similarity_info"),
                )
                components["pose_distance_threshold"] = gr.Number(
                    label=translator.t("ui.tracking.pose_distance_threshold"),
                    value=0.9,
                    minimum=0.0,
                    step=0.05,
                    info=translator.t("ui.tracking.pose_distance_threshold_info"),
                )
                components["min_pose_similarity"] = gr.Number(
                    label=translator.t("ui.tracking.min_pose_similarity"),
                    value=0.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    info=translator.t("ui.tracking.min_pose_similarity_info"),
                )

            with gr.Row():
                components["shape_maturity_frames"] = gr.Number(
                    label=translator.t("ui.tracking.shape_maturity_frames"),
                    value=20,
                    precision=0,
                    minimum=0,
                    info=translator.t("ui.tracking.shape_maturity_frames_info"),
                )
                components["high_shape_override_cam"] = gr.Checkbox(
                    label=translator.t("ui.tracking.high_shape_override_cam"),
                    value=True,
                    info=translator.t("ui.tracking.high_shape_override_cam_info"),
                )
                components["high_shape_threshold"] = gr.Number(
                    label=translator.t("ui.tracking.high_shape_threshold"),
                    value=0.95,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    info=translator.t("ui.tracking.high_shape_threshold_info"),
                )

            components["iou_distance_threshold"] = gr.Number(
                label=translator.t("ui.tracking.iou_distance_threshold"),
                value=0.5,
                minimum=0.0,
                step=0.05,
                info=translator.t("ui.tracking.iou_distance_threshold_info"),
            )

        with gr.Group():
            gr.Markdown(f"### {translator.t('ui.tracking.weights_title')}")
            with gr.Row():
                components["shape_weight"] = gr.Number(
                    label=translator.t("ui.tracking.shape_weight"),
                    value=0.5,
                    minimum=0.0,
                    step=0.05,
                    info=translator.t("ui.tracking.shape_weight_info"),
                )
                components["cam_weight"] = gr.Number(
                    label=translator.t("ui.tracking.cam_weight"),
                    value=0.3,
                    minimum=0.0,
                    step=0.05,
                    info=translator.t("ui.tracking.cam_weight_info"),
                )
                components["pose_weight"] = gr.Number(
                    label=translator.t("ui.tracking.pose_weight"),
                    value=0.2,
                    minimum=0.0,
                    step=0.05,
                    info=translator.t("ui.tracking.pose_weight_info"),
                )
                components["iou_weight"] = gr.Number(
                    label=translator.t("ui.tracking.iou_weight"),
                    value=0.0,
                    minimum=0.0,
                    step=0.05,
                    info=translator.t("ui.tracking.iou_weight_info"),
                )

        with gr.Group():
            gr.Markdown(f"### {translator.t('ui.tracking.features_title')}")
            with gr.Row():
                components["use_shape_params"] = gr.Checkbox(
                    label=translator.t("ui.tracking.use_shape_params"),
                    value=True,
                )
                components["use_pred_cam_t"] = gr.Checkbox(
                    label=translator.t("ui.tracking.use_pred_cam_t"),
                    value=True,
                )
                components["use_pose_aux"] = gr.Checkbox(
                    label=translator.t("ui.tracking.use_pose_aux"),
                    value=True,
                )
                components["use_bbox_iou"] = gr.Checkbox(
                    label=translator.t("ui.tracking.use_bbox_iou"),
                    value=False,
                )

            with gr.Row():
                components["export_frame_assignments"] = gr.Checkbox(
                    label=translator.t("ui.tracking.export_frame_assignments"),
                    value=True,
                )
                components["export_tracklet_detections"] = gr.Checkbox(
                    label=translator.t("ui.tracking.export_tracklet_detections"),
                    value=True,
                )
                components["export_mot_bboxes"] = gr.Checkbox(
                    label=translator.t("ui.tracking.export_mot_bboxes"),
                    value=False,
                )

        # Background Filtering (optional)
        with gr.Group():
            gr.Markdown(f"### {translator.t('ui.tracking.bg_title')}")
            components["bg_filter_enabled"] = gr.Checkbox(
                label=translator.t("ui.tracking.bg_enabled"),
                value=False,
                info=translator.t("ui.tracking.bg_enabled_info"),
            )

            with gr.Accordion(label=translator.t("ui.tracking.bg_detection_gates_title"), open=False):
                with gr.Row():
                    components["bg_min_bbox_height_px"] = gr.Number(
                        label=translator.t("ui.tracking.bg_min_bbox_height_px"),
                        value=0.0,
                        minimum=0.0,
                        step=1.0,
                        info=translator.t("ui.tracking.bg_min_bbox_height_px_info"),
                    )
                    components["bg_min_bbox_area_px2"] = gr.Number(
                        label=translator.t("ui.tracking.bg_min_bbox_area_px2"),
                        value=0.0,
                        minimum=0.0,
                        step=100.0,
                        info=translator.t("ui.tracking.bg_min_bbox_area_px2_info"),
                    )
                with gr.Row():
                    components["bg_depth_max_z"] = gr.Number(
                        label=translator.t("ui.tracking.bg_depth_max_z"),
                        value=0.0,
                        minimum=0.0,
                        step=0.1,
                        info=translator.t("ui.tracking.bg_depth_max_z_info"),
                    )
                    components["bg_keep_nearest_z_quantile"] = gr.Number(
                        label=translator.t("ui.tracking.bg_keep_nearest_z_quantile"),
                        value=0.0,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        info=translator.t("ui.tracking.bg_keep_nearest_z_quantile_info"),
                    )

            with gr.Accordion(label=translator.t("ui.tracking.bg_auto_size_title"), open=False):
                with gr.Row():
                    components["bg_size_auto_method"] = gr.Dropdown(
                        label=translator.t("ui.tracking.bg_size_auto_method"),
                        choices=["none", "percentile", "otsu", "gmm2"],
                        value="none",
                        info=translator.t("ui.tracking.bg_size_auto_method_info"),
                    )
                    components["bg_size_feature"] = gr.Dropdown(
                        label=translator.t("ui.tracking.bg_size_feature"),
                        choices=["bbox_height", "bbox_area"],
                        value="bbox_height",
                        info=translator.t("ui.tracking.bg_size_feature_info"),
                    )
                components["bg_size_percentile"] = gr.Number(
                    label=translator.t("ui.tracking.bg_size_percentile"),
                    value=0.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    info=translator.t("ui.tracking.bg_size_percentile_info"),
                )

            with gr.Accordion(label=translator.t("ui.tracking.bg_tracklet_scoring_title"), open=False):
                components["bg_tracklet_score_enabled"] = gr.Checkbox(
                    label=translator.t("ui.tracking.bg_tracklet_score_enabled"),
                    value=True,
                    info=translator.t("ui.tracking.bg_tracklet_score_enabled_info"),
                )
                with gr.Row():
                    components["bg_tracklet_score_threshold"] = gr.Number(
                        label=translator.t("ui.tracking.bg_tracklet_score_threshold"),
                        value=0.0,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        info=translator.t("ui.tracking.bg_tracklet_score_threshold_info"),
                    )
                    components["bg_min_tracklet_frames_for_scoring"] = gr.Number(
                        label=translator.t("ui.tracking.bg_min_tracklet_frames_for_scoring"),
                        value=10,
                        precision=0,
                        minimum=0,
                        info=translator.t("ui.tracking.bg_min_tracklet_frames_for_scoring_info"),
                    )
                with gr.Row():
                    components["bg_w_length"] = gr.Number(
                        label=translator.t("ui.tracking.bg_w_length"),
                        value=0.45,
                        minimum=0.0,
                        step=0.05,
                        info=translator.t("ui.tracking.bg_w_length_info"),
                    )
                    components["bg_w_size"] = gr.Number(
                        label=translator.t("ui.tracking.bg_w_size"),
                        value=0.25,
                        minimum=0.0,
                        step=0.05,
                        info=translator.t("ui.tracking.bg_w_size_info"),
                    )
                    components["bg_w_size_stability"] = gr.Number(
                        label=translator.t("ui.tracking.bg_w_size_stability"),
                        value=0.15,
                        minimum=0.0,
                        step=0.05,
                        info=translator.t("ui.tracking.bg_w_size_stability_info"),
                    )
                    components["bg_w_centering"] = gr.Number(
                        label=translator.t("ui.tracking.bg_w_centering"),
                        value=0.15,
                        minimum=0.0,
                        step=0.05,
                        info=translator.t("ui.tracking.bg_w_centering_info"),
                    )

            with gr.Accordion(label=translator.t("ui.tracking.bg_auto_roi_title"), open=False):
                components["bg_auto_roi_enabled"] = gr.Checkbox(
                    label=translator.t("ui.tracking.bg_auto_roi_enabled"),
                    value=False,
                    info=translator.t("ui.tracking.bg_auto_roi_enabled_info"),
                )
                with gr.Row():
                    components["bg_auto_roi_window_frames"] = gr.Number(
                        label=translator.t("ui.tracking.bg_auto_roi_window_frames"),
                        value=30,
                        precision=0,
                        minimum=1,
                        info=translator.t("ui.tracking.bg_auto_roi_window_frames_info"),
                    )
                    components["bg_auto_roi_point"] = gr.Dropdown(
                        label=translator.t("ui.tracking.bg_auto_roi_point"),
                        choices=["bottom_center", "center"],
                        value="bottom_center",
                        info=translator.t("ui.tracking.bg_auto_roi_point_info"),
                    )
                with gr.Row():
                    components["bg_auto_roi_mad_k"] = gr.Number(
                        label=translator.t("ui.tracking.bg_auto_roi_mad_k"),
                        value=3.0,
                        minimum=0.0,
                        step=0.1,
                        info=translator.t("ui.tracking.bg_auto_roi_mad_k_info"),
                    )
                    components["bg_auto_roi_min_radius_px"] = gr.Number(
                        label=translator.t("ui.tracking.bg_auto_roi_min_radius_px"),
                        value=60.0,
                        minimum=0.0,
                        step=5.0,
                        info=translator.t("ui.tracking.bg_auto_roi_min_radius_px_info"),
                    )
                    components["bg_auto_roi_smoothing_alpha"] = gr.Number(
                        label=translator.t("ui.tracking.bg_auto_roi_smoothing_alpha"),
                        value=0.2,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        info=translator.t("ui.tracking.bg_auto_roi_smoothing_alpha_info"),
                    )

            components["bg_refine_second_pass"] = gr.Checkbox(
                label=translator.t("ui.tracking.bg_refine_second_pass"),
                value=False,
                info=translator.t("ui.tracking.bg_refine_second_pass_info"),
            )

        components["tracking_config_upload"] = gr.File(
            label=translator.t("ui.tracking.load_config"),
            file_types=[".json"],
            type="filepath",
        )
        components["tracking_save_config_btn"] = gr.Button(
            translator.t("ui.tracking.save_config_btn"),
            variant="secondary",
        )
        components["tracking_config_download"] = gr.File(
            label=translator.t("ui.tracking.save_config"),
            interactive=False,
            visible=False,
        )

    return components


def build_tracking_config_from_gui(
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
) -> TrackingConfig:
    return TrackingConfig(
        enabled=bool(tracking_enabled),
        max_gap_frames=int(max_gap_frames),
        merge_max_gap_frames=int(merge_max_gap_frames),
        min_tracklet_length=int(min_tracklet_length),
        min_similarity=float(min_similarity),
        shape_distance_threshold=float(shape_distance_threshold),
        cam_distance_threshold=float(cam_distance_threshold),
        min_cam_similarity=float(min_cam_similarity),
        pose_distance_threshold=float(pose_distance_threshold),
        min_pose_similarity=float(min_pose_similarity),
        shape_maturity_frames=int(shape_maturity_frames),
        high_shape_override_cam=bool(high_shape_override_cam),
        high_shape_threshold=float(high_shape_threshold),
        iou_distance_threshold=float(iou_distance_threshold),
        shape_weight=float(shape_weight),
        cam_weight=float(cam_weight),
        pose_weight=float(pose_weight),
        iou_weight=float(iou_weight),
        use_shape_params=bool(use_shape_params),
        use_pred_cam_t=bool(use_pred_cam_t),
        use_pose_aux=bool(use_pose_aux),
        use_bbox_iou=bool(use_bbox_iou),
        export_frame_assignments=bool(export_frame_assignments),
        export_tracklet_detections=bool(export_tracklet_detections),
        export_mot_bboxes=bool(export_mot_bboxes),
        bg_filter_enabled=bool(bg_filter_enabled),
        bg_min_bbox_height_px=float(bg_min_bbox_height_px),
        bg_min_bbox_area_px2=float(bg_min_bbox_area_px2),
        bg_depth_max_z=float(bg_depth_max_z),
        bg_keep_nearest_z_quantile=float(bg_keep_nearest_z_quantile),
        bg_size_auto_method=str(bg_size_auto_method),
        bg_size_feature=str(bg_size_feature),
        bg_size_percentile=float(bg_size_percentile),
        bg_tracklet_score_enabled=bool(bg_tracklet_score_enabled),
        bg_tracklet_score_threshold=float(bg_tracklet_score_threshold),
        bg_min_tracklet_frames_for_scoring=int(bg_min_tracklet_frames_for_scoring),
        bg_w_length=float(bg_w_length),
        bg_w_size=float(bg_w_size),
        bg_w_size_stability=float(bg_w_size_stability),
        bg_w_centering=float(bg_w_centering),
        bg_auto_roi_enabled=bool(bg_auto_roi_enabled),
        bg_auto_roi_window_frames=int(bg_auto_roi_window_frames),
        bg_auto_roi_point=str(bg_auto_roi_point),
        bg_auto_roi_mad_k=float(bg_auto_roi_mad_k),
        bg_auto_roi_min_radius_px=float(bg_auto_roi_min_radius_px),
        bg_auto_roi_smoothing_alpha=float(bg_auto_roi_smoothing_alpha),
        bg_refine_second_pass=bool(bg_refine_second_pass),
    )


def _updates_from_config(config: TrackingConfig) -> List[gr.update]:
    return [
        gr.update(value=config.enabled),
        gr.update(value=config.max_gap_frames),
        gr.update(value=config.merge_max_gap_frames),
        gr.update(value=config.min_tracklet_length),
        gr.update(value=config.min_similarity),
        gr.update(value=config.shape_distance_threshold),
        gr.update(value=config.cam_distance_threshold),
        gr.update(value=getattr(config, "min_cam_similarity", 0.01)),
        gr.update(value=config.pose_distance_threshold),
        gr.update(value=getattr(config, "min_pose_similarity", 0.0)),
        gr.update(value=getattr(config, "shape_maturity_frames", 20)),
        gr.update(value=getattr(config, "high_shape_override_cam", True)),
        gr.update(value=getattr(config, "high_shape_threshold", 0.95)),
        gr.update(value=config.iou_distance_threshold),
        gr.update(value=config.shape_weight),
        gr.update(value=config.cam_weight),
        gr.update(value=config.pose_weight),
        gr.update(value=config.iou_weight),
        gr.update(value=config.use_shape_params),
        gr.update(value=config.use_pred_cam_t),
        gr.update(value=config.use_pose_aux),
        gr.update(value=config.use_bbox_iou),
        gr.update(value=config.export_frame_assignments),
        gr.update(value=config.export_tracklet_detections),
        gr.update(value=config.export_mot_bboxes),
        gr.update(value=getattr(config, "bg_filter_enabled", False)),
        gr.update(value=getattr(config, "bg_min_bbox_height_px", 0.0)),
        gr.update(value=getattr(config, "bg_min_bbox_area_px2", 0.0)),
        gr.update(value=getattr(config, "bg_depth_max_z", 0.0)),
        gr.update(value=getattr(config, "bg_keep_nearest_z_quantile", 0.0)),
        gr.update(value=getattr(config, "bg_size_auto_method", "none")),
        gr.update(value=getattr(config, "bg_size_feature", "bbox_height")),
        gr.update(value=getattr(config, "bg_size_percentile", 0.0)),
        gr.update(value=getattr(config, "bg_tracklet_score_enabled", True)),
        gr.update(value=getattr(config, "bg_tracklet_score_threshold", 0.0)),
        gr.update(value=getattr(config, "bg_min_tracklet_frames_for_scoring", 10)),
        gr.update(value=getattr(config, "bg_w_length", 0.45)),
        gr.update(value=getattr(config, "bg_w_size", 0.25)),
        gr.update(value=getattr(config, "bg_w_size_stability", 0.15)),
        gr.update(value=getattr(config, "bg_w_centering", 0.15)),
        gr.update(value=getattr(config, "bg_auto_roi_enabled", False)),
        gr.update(value=getattr(config, "bg_auto_roi_window_frames", 30)),
        gr.update(value=getattr(config, "bg_auto_roi_point", "bottom_center")),
        gr.update(value=getattr(config, "bg_auto_roi_mad_k", 3.0)),
        gr.update(value=getattr(config, "bg_auto_roi_min_radius_px", 60.0)),
        gr.update(value=getattr(config, "bg_auto_roi_smoothing_alpha", 0.2)),
        gr.update(value=getattr(config, "bg_refine_second_pass", False)),
    ]


def load_tracking_configuration(file_path: str) -> List[gr.update]:
    if not file_path:
        return _updates_from_config(TrackingConfig())
    with open(file_path, "r") as f:
        data = json.load(f)
    config = TrackingConfig.from_dict(data if isinstance(data, dict) else {})
    return _updates_from_config(config)


def save_tracking_configuration(
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
):
    config = build_tracking_config_from_gui(
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
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.close()
    config.save_json(tmp.name)
    return gr.update(value=tmp.name, visible=True)


def update_tracking_language(lang: str, translator: Translator) -> Tuple[Any, ...]:
    t = Translator(lang)
    return (
        gr.update(label=t.t("ui.tracking.config_title")),  # tracking_group
        gr.update(label=t.t("ui.tracking.enabled"), info=t.t("ui.tracking.enabled_info")),  # tracking_enabled
        gr.update(label=t.t("ui.tracking.max_gap_frames"), info=t.t("ui.tracking.max_gap_frames_info")),  # max_gap_frames
        gr.update(label=t.t("ui.tracking.merge_max_gap_frames"), info=t.t("ui.tracking.merge_max_gap_frames_info")),  # merge_max_gap_frames
        gr.update(label=t.t("ui.tracking.min_tracklet_length"), info=t.t("ui.tracking.min_tracklet_length_info")),  # min_tracklet_length
        gr.update(label=t.t("ui.tracking.min_similarity"), info=t.t("ui.tracking.min_similarity_info")),  # min_similarity
        gr.update(label=t.t("ui.tracking.shape_distance_threshold"), info=t.t("ui.tracking.shape_distance_threshold_info")),  # shape_distance_threshold
        gr.update(label=t.t("ui.tracking.cam_distance_threshold"), info=t.t("ui.tracking.cam_distance_threshold_info")),  # cam_distance_threshold
        gr.update(label=t.t("ui.tracking.min_cam_similarity"), info=t.t("ui.tracking.min_cam_similarity_info")),  # min_cam_similarity
        gr.update(label=t.t("ui.tracking.pose_distance_threshold"), info=t.t("ui.tracking.pose_distance_threshold_info")),  # pose_distance_threshold
        gr.update(label=t.t("ui.tracking.min_pose_similarity"), info=t.t("ui.tracking.min_pose_similarity_info")),  # min_pose_similarity
        gr.update(label=t.t("ui.tracking.shape_maturity_frames"), info=t.t("ui.tracking.shape_maturity_frames_info")),  # shape_maturity_frames
        gr.update(label=t.t("ui.tracking.high_shape_override_cam"), info=t.t("ui.tracking.high_shape_override_cam_info")),  # high_shape_override_cam
        gr.update(label=t.t("ui.tracking.high_shape_threshold"), info=t.t("ui.tracking.high_shape_threshold_info")),  # high_shape_threshold
        gr.update(label=t.t("ui.tracking.iou_distance_threshold"), info=t.t("ui.tracking.iou_distance_threshold_info")),  # iou_distance_threshold
        gr.update(label=t.t("ui.tracking.shape_weight"), info=t.t("ui.tracking.shape_weight_info")),  # shape_weight
        gr.update(label=t.t("ui.tracking.cam_weight"), info=t.t("ui.tracking.cam_weight_info")),  # cam_weight
        gr.update(label=t.t("ui.tracking.pose_weight"), info=t.t("ui.tracking.pose_weight_info")),  # pose_weight
        gr.update(label=t.t("ui.tracking.iou_weight"), info=t.t("ui.tracking.iou_weight_info")),  # iou_weight
        gr.update(label=t.t("ui.tracking.use_shape_params")),  # use_shape_params
        gr.update(label=t.t("ui.tracking.use_pred_cam_t")),  # use_pred_cam_t
        gr.update(label=t.t("ui.tracking.use_pose_aux")),  # use_pose_aux
        gr.update(label=t.t("ui.tracking.use_bbox_iou")),  # use_bbox_iou
        gr.update(label=t.t("ui.tracking.export_frame_assignments")),  # export_frame_assignments
        gr.update(label=t.t("ui.tracking.export_tracklet_detections")),  # export_tracklet_detections
        gr.update(label=t.t("ui.tracking.export_mot_bboxes")),  # export_mot_bboxes
        gr.update(label=t.t("ui.tracking.bg_enabled"), info=t.t("ui.tracking.bg_enabled_info")),  # bg_filter_enabled
        gr.update(label=t.t("ui.tracking.bg_min_bbox_height_px"), info=t.t("ui.tracking.bg_min_bbox_height_px_info")),  # bg_min_bbox_height_px
        gr.update(label=t.t("ui.tracking.bg_min_bbox_area_px2"), info=t.t("ui.tracking.bg_min_bbox_area_px2_info")),  # bg_min_bbox_area_px2
        gr.update(label=t.t("ui.tracking.bg_depth_max_z"), info=t.t("ui.tracking.bg_depth_max_z_info")),  # bg_depth_max_z
        gr.update(label=t.t("ui.tracking.bg_keep_nearest_z_quantile"), info=t.t("ui.tracking.bg_keep_nearest_z_quantile_info")),  # bg_keep_nearest_z_quantile
        gr.update(label=t.t("ui.tracking.bg_size_auto_method"), info=t.t("ui.tracking.bg_size_auto_method_info")),  # bg_size_auto_method
        gr.update(label=t.t("ui.tracking.bg_size_feature"), info=t.t("ui.tracking.bg_size_feature_info")),  # bg_size_feature
        gr.update(label=t.t("ui.tracking.bg_size_percentile"), info=t.t("ui.tracking.bg_size_percentile_info")),  # bg_size_percentile
        gr.update(label=t.t("ui.tracking.bg_tracklet_score_enabled"), info=t.t("ui.tracking.bg_tracklet_score_enabled_info")),  # bg_tracklet_score_enabled
        gr.update(label=t.t("ui.tracking.bg_tracklet_score_threshold"), info=t.t("ui.tracking.bg_tracklet_score_threshold_info")),  # bg_tracklet_score_threshold
        gr.update(label=t.t("ui.tracking.bg_min_tracklet_frames_for_scoring"), info=t.t("ui.tracking.bg_min_tracklet_frames_for_scoring_info")),  # bg_min_tracklet_frames_for_scoring
        gr.update(label=t.t("ui.tracking.bg_w_length"), info=t.t("ui.tracking.bg_w_length_info")),  # bg_w_length
        gr.update(label=t.t("ui.tracking.bg_w_size"), info=t.t("ui.tracking.bg_w_size_info")),  # bg_w_size
        gr.update(label=t.t("ui.tracking.bg_w_size_stability"), info=t.t("ui.tracking.bg_w_size_stability_info")),  # bg_w_size_stability
        gr.update(label=t.t("ui.tracking.bg_w_centering"), info=t.t("ui.tracking.bg_w_centering_info")),  # bg_w_centering
        gr.update(label=t.t("ui.tracking.bg_auto_roi_enabled"), info=t.t("ui.tracking.bg_auto_roi_enabled_info")),  # bg_auto_roi_enabled
        gr.update(label=t.t("ui.tracking.bg_auto_roi_window_frames"), info=t.t("ui.tracking.bg_auto_roi_window_frames_info")),  # bg_auto_roi_window_frames
        gr.update(label=t.t("ui.tracking.bg_auto_roi_point"), info=t.t("ui.tracking.bg_auto_roi_point_info")),  # bg_auto_roi_point
        gr.update(label=t.t("ui.tracking.bg_auto_roi_mad_k"), info=t.t("ui.tracking.bg_auto_roi_mad_k_info")),  # bg_auto_roi_mad_k
        gr.update(label=t.t("ui.tracking.bg_auto_roi_min_radius_px"), info=t.t("ui.tracking.bg_auto_roi_min_radius_px_info")),  # bg_auto_roi_min_radius_px
        gr.update(label=t.t("ui.tracking.bg_auto_roi_smoothing_alpha"), info=t.t("ui.tracking.bg_auto_roi_smoothing_alpha_info")),  # bg_auto_roi_smoothing_alpha
        gr.update(label=t.t("ui.tracking.bg_refine_second_pass"), info=t.t("ui.tracking.bg_refine_second_pass_info")),  # bg_refine_second_pass
        gr.update(label=t.t("ui.tracking.load_config")),  # tracking_config_upload
        gr.update(value=t.t("ui.tracking.save_config_btn")),  # tracking_save_config_btn
        gr.update(label=t.t("ui.tracking.save_config")),  # tracking_config_download
    )
