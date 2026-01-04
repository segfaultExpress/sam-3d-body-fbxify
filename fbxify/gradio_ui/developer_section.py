"""
Gradio UI components for developer section.

This module provides UI components for developer options including visualization.
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
    
    with gr.Accordion(translator.t("ui.developer_options"), open=False):
        components['create_visualization'] = gr.Checkbox(
            label=translator.t("ui.create_visualization"),
            value=False
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
        gr.update(label=t.t("ui.create_visualization")),  # create_visualization
    )
