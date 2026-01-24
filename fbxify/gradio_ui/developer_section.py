"""
Gradio UI components for developer section.

This module provides UI components for developer options and tooling.
"""
import gradio as gr
from typing import Dict, Any, Tuple
from fbxify.i18n import Translator


def create_developer_section(translator: Translator) -> Dict[str, Any]:
    """
    Create the developer section UI components.
    
    Args:
        translator: Translator instance for i18n
        
    Returns:
        Dictionary of component names to Gradio components
    """
    components = {}

    with gr.Accordion(label=translator.t("ui.cli_generator_title"), open=False) as cli_accordion:
        components["cli_generator_accordion"] = cli_accordion
        components["cli_generator_info_md"] = gr.Markdown(translator.t("ui.cli_generator_info"))
        components["generate_cli_btn"] = gr.Button(
            translator.t("ui.generate_cli_btn"),
            variant="secondary"
        )
        components["cli_command"] = gr.Textbox(
            label=translator.t("ui.cli_command_label"),
            info=translator.t("ui.cli_command_info"),
            lines=3,
            interactive=False
        )

    with gr.Accordion(label=translator.t("ui.developer_options"), open=False) as developer_accordion:
        components["developer_options_accordion"] = developer_accordion
        components["cancel_jobs_info_md"] = gr.Markdown(translator.t("ui.cancel_current_jobs_info"))
        components["cancel_jobs_btn"] = gr.Button(
            translator.t("ui.cancel_current_jobs"),
            variant="stop",
            interactive=True
        )
    
    return components


def update_developer_language(lang: str) -> Tuple[Any, ...]:
    """
    Update developer section components with new language.
    
    Args:
        lang: Language code
        
    Returns:
        Tuple of updates for all developer components
    """
    t = Translator(lang)
    return (
        gr.update(label=t.t("ui.cli_generator_title")),  # cli_generator_accordion
        gr.update(value=t.t("ui.cli_generator_info")),  # cli_generator_info_md
        gr.update(value=t.t("ui.generate_cli_btn")),  # generate_cli_btn
        gr.update(label=t.t("ui.cli_command_label"), info=t.t("ui.cli_command_info")),  # cli_command
        gr.update(label=t.t("ui.developer_options")),  # developer_options_accordion
        gr.update(value=t.t("ui.cancel_current_jobs_info")),  # cancel_jobs_info_md
        gr.update(value=t.t("ui.cancel_current_jobs"))  # cancel_jobs_btn
    )
