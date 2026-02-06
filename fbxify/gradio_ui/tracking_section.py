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
        gr.update(label=t.t("ui.tracking.load_config")),  # tracking_config_upload
        gr.update(value=t.t("ui.tracking.save_config_btn")),  # tracking_save_config_btn
        gr.update(label=t.t("ui.tracking.save_config")),  # tracking_config_download
    )
