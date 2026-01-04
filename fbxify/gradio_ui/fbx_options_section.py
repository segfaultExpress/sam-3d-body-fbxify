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
    
    # Root motion checkbox
    components['use_root_motion'] = gr.Checkbox(
        label=translator.t("ui.use_root_motion"),
        value=True
    )
    
    # Mesh options
    components['include_mesh'] = gr.Checkbox(
        label=translator.t("ui.include_mesh"),
        value=False
    )
    
    with gr.Row():
        components['lod'] = gr.Slider(
            label=translator.t("ui.lod"),
            minimum=0,
            maximum=6,
            step=1,
            value=0,
            visible=False,
            info=translator.t("ui.lod_info")
        )
        components['body_param_sample_num'] = gr.Number(
            label=translator.t("ui.body_param_sample_num"),
            value=5,
            precision=0,
            minimum=1,
            visible=False,
            info=translator.t("ui.body_param_sample_num_info")
        )
    
    return components


def toggle_mesh_inputs(include_mesh_value: bool) -> Tuple[Any, Any]:
    """
    Toggle visibility of lod and body_param_sample_num based on checkbox.
    
    Args:
        include_mesh_value: Whether to include mesh
        
    Returns:
        Tuple of updates for lod and body_param_sample_num
    """
    return (
        gr.update(visible=include_mesh_value),
        gr.update(visible=include_mesh_value)
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
        gr.update(label=t.t("ui.use_root_motion")),  # use_root_motion
        gr.update(label=t.t("ui.include_mesh")),  # include_mesh
        gr.update(label=t.t("ui.lod"), info=t.t("ui.lod_info")),  # lod
        gr.update(label=t.t("ui.body_param_sample_num"), info=t.t("ui.body_param_sample_num_info")),  # body_param_sample_num
    )

