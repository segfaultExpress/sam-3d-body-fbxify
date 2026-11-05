"""
Pose Estimation tab layout for the Gradio UI.
"""
from __future__ import annotations

import gradio as gr
from typing import Dict, Any
from fbxify.i18n import Translator
from fbxify.gradio_ui.entry_section import create_entry_section
from fbxify.gradio_ui.pose_results_section import create_pose_results_section
from fbxify.gradio_ui.developer_section import create_pose_cli_section, create_pose_dev_section


def create_pose_tab(translator: Translator, models: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Build the Pose Estimation tab layout and return components.

    models: optional model catalogue payload for the model dropdowns (backend.list_models()).
    """
    components: Dict[str, Any] = {}
    with gr.Row():
        with gr.Column():
            components["entry"] = create_entry_section(translator, models=models)
            components["pose_cli"] = create_pose_cli_section(translator)
            components["pose_dev"] = create_pose_dev_section(translator)
        with gr.Column():
            components["pose_results"] = create_pose_results_section(translator)
    return components
