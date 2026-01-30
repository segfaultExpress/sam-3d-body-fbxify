"""
Gradio UI components for FBX processing section.

This module provides UI components for FBX inputs like profile and pose JSON.
"""
import gradio as gr
from typing import Dict, Any, Tuple
from fbxify.i18n import Translator
from fbxify.metadata import PROFILES


def create_fbx_processing_section(translator: Translator) -> Dict[str, Any]:
    """
    Create the FBX processing section UI components.
    
    Args:
        translator: Translator instance for i18n
        
    Returns:
        Dictionary of component names to Gradio components
    """
    components = {}
    
    # Pose JSON file upload/dropdown
    components['pose_json_file'] = gr.File(
        label=translator.t("ui.pose_json_file"),
        file_types=[".json"],
        interactive=True
    )
    
    # Profile dropdown (armature model selection)
    components['profile_name'] = gr.Dropdown(
        label=translator.t("ui.profile"),
        choices=list(PROFILES.keys()),
        value=list(PROFILES.keys())[0]
    )
    
    return components


def update_fbx_processing_language(lang: str) -> Tuple[Any, Any]:
    """
    Update FBX processing section components with new language.
    
    Args:
        lang: Language code
        
    Returns:
        Tuple of updates for profile_name, pose_json_file, generate_fbx_btn, and output_files
    """
    t = Translator(lang)
    return (
        gr.update(label=t.t("ui.profile")),  # profile_name
        gr.update(label=t.t("ui.pose_json_file")),  # pose_json_file
    )


def toggle_generate_fbx_button(pose_json_file) -> Any:
    """
    Enable/disable Generate FBX button based on whether JSON file is provided.
    
    Args:
        pose_json_file: The pose JSON file (can be None)
        
    Returns:
        Update for generate_fbx_btn
    """
    return gr.update(interactive=(pose_json_file is not None))

