"""
Gradio UI components for FBX processing section.

This module provides UI components for pose JSON selection, FBX generation, and output files.
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
    
    # Profile dropdown (armature model selection)
    components['profile_name'] = gr.Dropdown(
        label=translator.t("ui.profile"),
        choices=list(PROFILES.keys()),
        value=list(PROFILES.keys())[0]
    )
    
    # Pose JSON file upload/dropdown
    components['pose_json_file'] = gr.File(
        label=translator.t("ui.pose_json_file"),
        file_types=[".json"],
        interactive=True
    )
    
    # Generate FBX button (Step 2) (disabled until JSON is available)
    components['generate_fbx_btn'] = gr.Button(
        translator.t("ui.step_2_generate_fbx"),
        variant="primary",
        interactive=False  # Disabled by default
    )
    
    # Output files (only FBX and visualization)
    components['output_files'] = gr.File(
        label=translator.t("ui.output_files"),
        interactive=False,
        file_count="multiple"
    )
    
    return components


def update_fbx_processing_language(lang: str) -> Tuple[Any, Any, Any, Any]:
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
        gr.update(value=t.t("ui.step_2_generate_fbx")),  # generate_fbx_btn
        gr.update(label=t.t("ui.output_files"))  # output_files
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

