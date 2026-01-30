"""
Gradio UI components for FBX generation actions/results.

This module provides UI components for the generate button and output files.
"""
import gradio as gr
from typing import Dict, Any, Tuple
from fbxify.i18n import Translator


def create_fbx_results_section(translator: Translator) -> Dict[str, Any]:
    """
    Create the FBX results section UI components.

    Args:
        translator: Translator instance for i18n

    Returns:
        Dictionary of component names to Gradio components
    """
    components = {}

    components["output_files"] = gr.File(
        label=translator.t("ui.output_files"),
        interactive=False,
        file_count="multiple"
    )

    components["generate_fbx_btn"] = gr.Button(
        translator.t("ui.generate_fbx_btn"),
        variant="primary",
        interactive=False  # Disabled by default
    )

    return components


def update_fbx_results_language(lang: str) -> Tuple[Any, Any]:
    """
    Update FBX results components with new language.

    Args:
        lang: Language code

    Returns:
        Tuple of updates for generate_fbx_btn and output_files
    """
    t = Translator(lang)
    return (
        gr.update(value=t.t("ui.generate_fbx_btn")),  # generate_fbx_btn
        gr.update(label=t.t("ui.output_files")),  # output_files
    )
