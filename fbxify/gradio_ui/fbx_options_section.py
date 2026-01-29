"""
Gradio UI components for FBX options section.

This module provides UI components for FBX generation options including
root motion, mesh inclusion, LOD, and body parameter sampling.
"""
import gradio as gr
from typing import Dict, Any, Tuple
from fbxify.i18n import Translator


def create_fbx_options_section(translator: Translator) -> Dict[str, Any]:
    """
    Create the FBX options section UI components.
    
    Args:
        translator: Translator instance for i18n
        
    Returns:
        Dictionary of component names to Gradio components
    """
    components = {}
    
    # FBX Options section header
    with gr.Group():
        gr.Markdown(f"## {translator.t('ui.fbx_options_title')}")
    
    # Auto-run checkbox
    components['auto_run'] = gr.Checkbox(
        label=translator.t("ui.auto_run"),
        value=True
    )
    
    # Root motion checkbox
    components['use_root_motion'] = gr.Checkbox(
        label=translator.t("ui.use_root_motion"),
        value=True
    )
    components['auto_floor'] = gr.Checkbox(
        label=translator.t("ui.auto_floor"),
        value=True,
        info=translator.t("ui.auto_floor_info")
    )
    components['include_extrinsics'] = gr.Checkbox(
        label=translator.t("ui.include_extrinsics"),
        value=False
    )
    
    components['include_mesh'] = gr.Checkbox(
        label=translator.t("ui.include_mesh"),
        value=False
    )

    components['use_personalized_body'] = gr.Checkbox(
        label=translator.t("ui.use_personalized_body"),
        value=True,
        visible=False,
        info=translator.t("ui.use_personalized_body_info")
    )
    
    with gr.Row():
        components['lod'] = gr.Slider(
            label=translator.t("ui.lod"),
            minimum=0,
            maximum=6,
            step=1,
            value=1,
            visible=False,
            info=translator.t("ui.lod_info")
        )
        components['outlier_removal_percent'] = gr.Number(
            label=translator.t("ui.outlier_removal_percent"),
            minimum=0,
            maximum=50,
            step=0.1,
            value=10.0,
            precision=1,
            visible=False,
            info=translator.t("ui.outlier_removal_percent_info")
        )

    with gr.Row():
        components['extrinsics_sample_rate'] = gr.Number(
            label=translator.t("ui.extrinsics_sample_rate"),
            value=0,
            precision=0,
            minimum=0,
            step=1,
            visible=False,
            info=translator.t("ui.extrinsics_sample_rate_info")
        )
        components['extrinsics_scale'] = gr.Number(
            label=translator.t("ui.extrinsics_scale"),
            value=1,
            minimum=0,
            step=0.01,
            visible=False,
            info=translator.t("ui.extrinsics_scale_info")
        )
        components['extrinsics_invert_quaternion'] = gr.Checkbox(
            label=translator.t("ui.extrinsics_invert_quaternion"),
            value=False,
            visible=False,
            info=translator.t("ui.extrinsics_invert_quaternion_info")
        )
        components['extrinsics_invert_translation'] = gr.Checkbox(
            label=translator.t("ui.extrinsics_invert_translation"),
            value=False,
            visible=False,
            info=translator.t("ui.extrinsics_invert_translation_info")
        )
    components['extrinsics_file'] = gr.File(
        label=translator.t("ui.extrinsics_file"),
        file_types=[".txt"],
        visible=False
    )
    
    return components


def toggle_mesh_inputs(include_mesh_value: bool, use_personalized_body_value: bool = False) -> Tuple[Any, Any]:
    """
    Toggle visibility of lod and outlier_removal_percent based on checkboxes.
    Note: use_personalized_body visibility is handled separately.
    
    Args:
        include_mesh_value: Whether to include mesh
        use_personalized_body_value: Whether to use personalized body (only shown if include_mesh is True)
        
    Returns:
        Tuple of updates for lod and outlier_removal_percent
    """
    # Handle case where value might be a list (from Gradio's internal processing)
    if isinstance(include_mesh_value, list):
        include_mesh_value = include_mesh_value[0] if include_mesh_value else False
    if isinstance(use_personalized_body_value, list):
        use_personalized_body_value = use_personalized_body_value[0] if use_personalized_body_value else False
    
    return (
        gr.update(visible=include_mesh_value),  # lod
        gr.update(visible=include_mesh_value and use_personalized_body_value)  # outlier_removal_percent
    )


def toggle_personalized_body(include_mesh_value: bool, use_personalized_body_value: bool) -> Tuple[Any,]:
    """
    Toggle visibility of outlier_removal_percent based on both include_mesh and use_personalized_body.
    
    Args:
        include_mesh_value: Whether to include mesh
        use_personalized_body_value: Whether to use personalized body
        
    Returns:
        Tuple of update for outlier_removal_percent
    """
    # Handle case where value might be a list (from Gradio's internal processing)
    if isinstance(include_mesh_value, list):
        include_mesh_value = include_mesh_value[0] if include_mesh_value else False
    if isinstance(use_personalized_body_value, list):
        use_personalized_body_value = use_personalized_body_value[0] if use_personalized_body_value else False
    
    return (
        gr.update(visible=include_mesh_value and use_personalized_body_value),
    )


def toggle_extrinsics_inputs(include_extrinsics_value: bool) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Toggle visibility of extrinsics inputs based on checkbox.
    
    Args:
        include_extrinsics_value: Whether to include extrinsics
        
    Returns:
        Tuple of updates for extrinsics_sample_rate, extrinsics_scale,
        extrinsics_invert_quaternion, extrinsics_invert_translation, extrinsics_file
    """
    # Handle case where value might be a list (from Gradio's internal processing)
    if isinstance(include_extrinsics_value, list):
        include_extrinsics_value = include_extrinsics_value[0] if include_extrinsics_value else False
    
    return (
        gr.update(visible=include_extrinsics_value),
        gr.update(visible=include_extrinsics_value),
        gr.update(visible=include_extrinsics_value),
        gr.update(visible=include_extrinsics_value),
        gr.update(visible=include_extrinsics_value)
    )


def update_fbx_options_language(lang: str, translator: Translator) -> Tuple[Any, ...]:
    """
    Update FBX options section components with new language.
    
    Args:
        lang: Language code
        translator: Translator instance (will be updated)
        
    Returns:
        Tuple of updates for all FBX options components
    """
    t = Translator(lang)
    return (
        gr.update(label=t.t("ui.auto_run")),  # auto_run
        gr.update(label=t.t("ui.use_root_motion")),  # use_root_motion
        gr.update(label=t.t("ui.auto_floor"), info=t.t("ui.auto_floor_info")),  # auto_floor
        gr.update(label=t.t("ui.include_mesh")),  # include_mesh
        gr.update(label=t.t("ui.include_extrinsics")),  # include_extrinsics
        gr.update(label=t.t("ui.use_personalized_body"), info=t.t("ui.use_personalized_body_info")),  # use_personalized_body
        gr.update(label=t.t("ui.lod"), info=t.t("ui.lod_info")),  # lod
        gr.update(label=t.t("ui.outlier_removal_percent"), info=t.t("ui.outlier_removal_percent_info")),  # outlier_removal_percent
        gr.update(label=t.t("ui.extrinsics_sample_rate"), info=t.t("ui.extrinsics_sample_rate_info")),  # extrinsics_sample_rate
        gr.update(label=t.t("ui.extrinsics_scale"), info=t.t("ui.extrinsics_scale_info")),  # extrinsics_scale
        gr.update(label=t.t("ui.extrinsics_invert_quaternion"), info=t.t("ui.extrinsics_invert_quaternion_info")),  # extrinsics_invert_quaternion
        gr.update(label=t.t("ui.extrinsics_invert_translation"), info=t.t("ui.extrinsics_invert_translation_info")),  # extrinsics_invert_translation
        gr.update(label=t.t("ui.extrinsics_file")),  # extrinsics_file
    )

