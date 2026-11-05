"""
Gradio UI components for entry section.

This module provides UI components for input controls including
file upload, bbox options, FOV options, and precision selection.
"""
from __future__ import annotations

import gradio as gr
from typing import Dict, Any, Tuple
from fbxify.i18n import Translator
from fbxify.gradio_ui.tracking_section import create_tracking_section


TRACKING_MODE_BBOX = "bbox"
TRACKING_MODE_COUNT = "count"
TRACKING_MODE_INFERENCE = "inference"
TRACKING_MODE_INFERENCE_BBOX = "inference_bbox"


MODEL_SECTIONS = (("body_model", "body_models"), ("encoder", "encoders"), ("detector", "detectors"))


def model_dropdown_choices(models: Dict[str, Any] | None, section: str) -> Tuple[list, Any]:
    """(choices, value) for one model dropdown from a GET /models style payload.

    Only enabled entries are offered; entries whose files are not on disk are labelled so the
    user knows a fetch is needed. The value is what is loaded, else the catalogue default.
    """
    if not models:
        return [], None
    key = {"body_models": "body_model", "encoders": "encoder", "detectors": "detector"}[section]
    choices = []
    for entry in models.get(section, []) or []:
        if not entry.get("enabled", False):
            continue
        label = entry.get("display_name") or entry["name"]
        status = entry.get("status", "")
        if status not in ("ready", "ready-unverified", "disabled"):
            label = f"{label} [{status}]"
        choices.append((label, entry["name"]))
    names = [c[1] for c in choices]
    loaded = (models.get("loaded") or {}).get(key)
    default = (models.get("defaults") or {}).get(key)
    value = loaded if loaded in names else (default if default in names else (names[0] if names else None))
    return choices, value


def create_entry_section(translator: Translator, models: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Create the entry section UI components.
    
    Args:
        translator: Translator instance for i18n
        models: optional model catalogue payload (backend.list_models()) feeding the
            body-model / encoder / detector dropdowns
        
    Returns:
        Dictionary of component names to Gradio components
    """
    components = {}
    
    # Input file
    components['input_file'] = gr.File(
        label=translator.t("ui.input_file"),
        file_types=["image", "video"]
    )
    
    # Estimation Options section header
    with gr.Group():
        gr.Markdown(f"## {translator.t('ui.estimation_options_title')}")

    # Model selection (from the model catalogue; swapping frees the previous model first)
    with gr.Row():
        for key, section in MODEL_SECTIONS:
            choices, value = model_dropdown_choices(models, section)
            components[key] = gr.Dropdown(
                label=translator.t(f"ui.{key}"),
                choices=choices,
                value=value,
                info=translator.t(f"ui.{key}_info"),
                allow_custom_value=True,
                interactive=True,
            )
    
    # Tracking options
    components['tracking_mode'] = gr.Dropdown(
        label=translator.t("ui.tracking.mode_label"),
        choices=[
            (translator.t("ui.tracking.mode_bbox"), TRACKING_MODE_BBOX),
            (translator.t("ui.tracking.mode_count"), TRACKING_MODE_COUNT),
            (translator.t("ui.tracking.mode_inference"), TRACKING_MODE_INFERENCE),
            (translator.t("ui.tracking.mode_inference_bbox"), TRACKING_MODE_INFERENCE_BBOX),
        ],
        value=TRACKING_MODE_COUNT,
        info=translator.t("ui.tracking.mode_info")
    )
    
    with gr.Row():
        components['bbox_file'] = gr.File(
            label=translator.t("ui.bbox_file"),
            file_types=[".txt"],
            visible=False
        )
        components['num_people'] = gr.Number(
            label=translator.t("ui.num_people"),
            value=1,
            precision=0,
            visible=True
        )
    
    # Missing bbox behavior option (only visible when use_bbox is checked)
    components['missing_bbox_behavior'] = gr.Dropdown(
        label=translator.t("ui.missing_bbox_behavior"),
        choices=["Run Detection", "Skip Frame"],
        value="Run Detection",
        visible=False,
        info=translator.t("ui.missing_bbox_behavior_info")
    )

    # Tracking config (only visible for inference tracking)
    tracking_components = create_tracking_section(translator)
    components.update(tracking_components)
    
    # FOV Estimation Options
    components['fov_method'] = gr.Dropdown(
        label=translator.t("ui.fov_method"),
        choices=["Default", "File", "Sample"],
        value="Default",
        info=translator.t("ui.fov_method_info")
    )
    components['fov_file'] = gr.File(
        label=translator.t("ui.fov_file"),
        file_types=[".txt"],
        visible=False
    )
    components['sample_number'] = gr.Number(
        label=translator.t("ui.sample_number"),
        value=1,
        precision=0,
        minimum=1,
        visible=False,
        info=translator.t("ui.sample_number_info")
    )

    # Precision options
    components['precision'] = gr.Dropdown(
        label=translator.t("ui.precision"),
        choices=["FP32 (Full)", "BF16 (Fast + Safer)", "FP16 (Fastest)"],
        value="FP32 (Full)",
        info=translator.t("ui.precision_info")
    )
    components['fast_inference'] = gr.Checkbox(
        label=translator.t("ui.fast_inference"),
        value=False,
        info=translator.t("ui.fast_inference_info"),
    )

    # Frame batching (run pose model once per N frames for speed)
    components['frame_batch_size'] = gr.Number(
        label=translator.t("ui.frame_batch_size"),
        value=1,
        precision=0,
        minimum=1,
        maximum=32,
        info=translator.t("ui.frame_batch_size_info")
    )

    # Detection batching (when no bbox file: run detector on N images per batch)
    components['detection_batch_size'] = gr.Number(
        label=translator.t("ui.detection_batch_size"),
        value=1,
        precision=0,
        minimum=1,
        maximum=32,
        info=translator.t("ui.detection_batch_size_info")
    )

    return components


def toggle_tracking_inputs(tracking_mode_value: str) -> Tuple[Any, Any, Any, Any]:
    """
    Toggle visibility of bbox_file, num_people, missing_bbox_behavior, and tracking config.
    - bbox_file + missing_bbox_behavior: visible for BBOX and Inference + BBOX File
    - num_people: visible only for Count
    - tracking_group: visible for Inference and Inference + BBOX File
    """
    if isinstance(tracking_mode_value, list):
        tracking_mode_value = tracking_mode_value[0] if tracking_mode_value else TRACKING_MODE_COUNT
    use_bbox = tracking_mode_value == TRACKING_MODE_BBOX
    use_inference_bbox = tracking_mode_value == TRACKING_MODE_INFERENCE_BBOX
    use_count = tracking_mode_value == TRACKING_MODE_COUNT
    use_tracking = tracking_mode_value == TRACKING_MODE_INFERENCE or use_inference_bbox
    show_bbox_panel = use_bbox or use_inference_bbox
    return (
        gr.update(visible=show_bbox_panel),
        gr.update(visible=use_count),
        gr.update(visible=show_bbox_panel),
        gr.update(visible=use_tracking),
    )


def toggle_fov_inputs(fov_method_value: str) -> Tuple[Any, Any]:
    """
    Toggle visibility of fov_file and sample_number based on FOV method selection.
    
    Args:
        fov_method_value: Selected FOV method
        
    Returns:
        Tuple of updates for fov_file and sample_number
    """
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


def update_entry_language(lang: str, translator: Translator) -> Tuple[Any, ...]:
    """
    Update entry section components with new language.
    
    Args:
        lang: Language code
        translator: Translator instance (will be updated)
        
    Returns:
        Tuple of updates for all entry components
    """
    t = Translator(lang)
    return (
        gr.update(label=t.t("ui.input_file")),  # input_file
        gr.update(label=t.t("ui.body_model"), info=t.t("ui.body_model_info")),  # body_model
        gr.update(label=t.t("ui.encoder"), info=t.t("ui.encoder_info")),  # encoder
        gr.update(label=t.t("ui.detector"), info=t.t("ui.detector_info")),  # detector
        gr.update(label=t.t("ui.tracking.mode_label"), info=t.t("ui.tracking.mode_info")),  # tracking_mode
        gr.update(label=t.t("ui.bbox_file")),  # bbox_file
        gr.update(label=t.t("ui.num_people")),  # num_people
        gr.update(label=t.t("ui.missing_bbox_behavior"), info=t.t("ui.missing_bbox_behavior_info")),  # missing_bbox_behavior
        gr.update(label=t.t("ui.fov_method"), info=t.t("ui.fov_method_info")),  # fov_method
        gr.update(label=t.t("ui.fov_file")),  # fov_file
        gr.update(label=t.t("ui.sample_number"), info=t.t("ui.sample_number_info")),  # sample_number
        gr.update(label=t.t("ui.precision"), info=t.t("ui.precision_info")),  # precision
        gr.update(label=t.t("ui.fast_inference"), info=t.t("ui.fast_inference_info")),  # fast_inference
        gr.update(label=t.t("ui.frame_batch_size"), info=t.t("ui.frame_batch_size_info")),  # frame_batch_size
        gr.update(label=t.t("ui.detection_batch_size"), info=t.t("ui.detection_batch_size_info")),  # detection_batch_size
    )
