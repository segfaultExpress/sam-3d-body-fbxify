"""
Generate FBX tab layout for the Gradio UI.
"""
import gradio as gr
from typing import Dict, Any
from fbxify.i18n import Translator
from fbxify.gradio_ui.fbx_processing_section import create_fbx_processing_section
from fbxify.gradio_ui.fbx_options_section import create_fbx_options_section
from fbxify.gradio_ui.refinement_section import create_refinement_section
from fbxify.gradio_ui.fbx_results_section import create_fbx_results_section
from fbxify.gradio_ui.developer_section import create_fbx_cli_section, create_fbx_dev_section


def create_fbx_tab(translator: Translator) -> Dict[str, Any]:
    """
    Build the Generate FBX tab layout and return components.
    """
    components: Dict[str, Any] = {}
    with gr.Row():
        with gr.Column():
            components["fbx_processing"] = create_fbx_processing_section(translator)
            components["fbx_options"] = create_fbx_options_section(translator)
            components["refinement"] = create_refinement_section(translator)
            components["fbx_cli"] = create_fbx_cli_section(translator)
            components["fbx_dev"] = create_fbx_dev_section(translator)
        with gr.Column():
            components["fbx_results"] = create_fbx_results_section(translator)
    return components
