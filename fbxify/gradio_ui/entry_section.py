"""
Gradio UI components for entry section.

This module provides UI components for input controls including
file upload, profile selection, bbox options, FOV options, and generate button.
"""
import gradio as gr
from typing import Dict, Any, Tuple
from fbxify.i18n import Translator


def create_entry_section(translator: Translator) -> Dict[str, Any]:
    """
    Create the entry section UI components.
    
    Args:
        translator: Translator instance for i18n
        
    Returns:
        Dictionary of component names to Gradio components
    """
    components = {}
    
    # Input file
    components['input_file'] = gr.File(
        label=translator.t("ui.input_file"),
        file_types=["image", "video"]
    )
    
    # Estimate Pose button (Step 1) - moved here, right after input file
    components['estimate_pose_btn'] = gr.Button(
        translator.t("ui.step_1_estimate_pose"),
        variant="primary",
        interactive=False  # Disabled by default until file is uploaded
    )
    
    # Estimation Options section header
    with gr.Group():
        gr.Markdown(f"## {translator.t('ui.estimation_options_title')}")
    
    # Bbox options
    components['use_bbox'] = gr.Checkbox(
        label=translator.t("ui.use_bbox"),
        value=False
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
    
    return components


def toggle_bbox_inputs(use_bbox_value: bool) -> Tuple[Any, Any]:
    """
    Toggle visibility of bbox_file and num_people based on checkbox.
    
    Args:
        use_bbox_value: Whether to use bbox file
        
    Returns:
        Tuple of updates for bbox_file and num_people
    """
    return (
        gr.update(visible=use_bbox_value),
        gr.update(visible=not use_bbox_value)
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
        gr.update(label=t.t("ui.use_bbox")),  # use_bbox
        gr.update(label=t.t("ui.bbox_file")),  # bbox_file
        gr.update(label=t.t("ui.num_people")),  # num_people
        gr.update(label=t.t("ui.fov_method"), info=t.t("ui.fov_method_info")),  # fov_method
        gr.update(label=t.t("ui.fov_file")),  # fov_file
        gr.update(label=t.t("ui.sample_number"), info=t.t("ui.sample_number_info")),  # sample_number
        gr.update(value=t.t("ui.step_1_estimate_pose")),  # estimate_pose_btn
    )
