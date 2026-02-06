"""
Gradio UI components for pose estimation results/actions.

This module provides UI components for the pose JSON output and
the Estimate Pose action button.
"""
import gradio as gr
from typing import Dict, Any, Tuple
from fbxify.i18n import Translator


def create_pose_results_section(translator: Translator) -> Dict[str, Any]:
    """
    Create the pose results section UI components.

    Args:
        translator: Translator instance for i18n

    Returns:
        Dictionary of component names to Gradio components
    """
    components = {}

    components["pose_json_file"] = gr.File(
        label=translator.t("ui.pose_json_file"),
        file_types=[".json", ".mp4", ".txt", ".csv", ".png", ".jpg"],
        file_count="multiple",
        type="filepath",
        interactive=False
    )

    components["estimate_pose_btn"] = gr.Button(
        translator.t("ui.estimate_pose_btn"),
        variant="primary",
        interactive=False  # Disabled by default until file is uploaded
    )

    return components


def update_pose_results_language(lang: str) -> Tuple[Any, Any]:
    """
    Update pose results components with new language.

    Args:
        lang: Language code

    Returns:
        Tuple of updates for pose_json_file and estimate_pose_btn
    """
    t = Translator(lang)
    return (
        gr.update(label=t.t("ui.pose_json_file")),  # pose_json_file
        gr.update(value=t.t("ui.estimate_pose_btn")),  # estimate_pose_btn
    )
