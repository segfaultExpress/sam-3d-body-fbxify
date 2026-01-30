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
        components["export_personalized_body_obj"] = gr.Checkbox(
            label=translator.t("ui.export_personalized_body_obj"),
            value=False
        )
        components["create_camera"] = gr.Checkbox(
            label=translator.t("ui.create_camera"),
            value=False
        )
        components["camera_zoom"] = gr.Number(
            label=translator.t("ui.camera_zoom"),
            value=-5.0,
            visible=False
        )
        components["camera_scene"] = gr.File(
            label=translator.t("ui.camera_scene"),
            file_types=["image", "video"],
            visible=False
        )
    
    return components


def create_pose_cli_section(translator: Translator) -> Dict[str, Any]:
    """Create the Pose Estimation CLI generator section."""
    components = {}
    with gr.Accordion(label=translator.t("ui.pose_cli_generator_title"), open=False) as cli_accordion:
        components["pose_cli_generator_accordion"] = cli_accordion
        components["pose_cli_generator_info_md"] = gr.Markdown(translator.t("ui.pose_cli_generator_info"))
        components["pose_generate_cli_btn"] = gr.Button(
            translator.t("ui.pose_generate_cli_btn"),
            variant="secondary"
        )
        components["pose_cli_command"] = gr.Textbox(
            label=translator.t("ui.pose_cli_command_label"),
            info=translator.t("ui.pose_cli_command_info"),
            lines=3,
            interactive=False
        )
    return components


def create_fbx_cli_section(translator: Translator) -> Dict[str, Any]:
    """Create the Generate FBX CLI generator section."""
    components = {}
    with gr.Accordion(label=translator.t("ui.fbx_cli_generator_title"), open=False) as cli_accordion:
        components["fbx_cli_generator_accordion"] = cli_accordion
        components["fbx_cli_generator_info_md"] = gr.Markdown(translator.t("ui.fbx_cli_generator_info"))
        components["fbx_generate_cli_btn"] = gr.Button(
            translator.t("ui.fbx_generate_cli_btn"),
            variant="secondary"
        )
        components["fbx_cli_command"] = gr.Textbox(
            label=translator.t("ui.fbx_cli_command_label"),
            info=translator.t("ui.fbx_cli_command_info"),
            lines=3,
            interactive=False
        )
    return components


def create_pose_dev_section(translator: Translator) -> Dict[str, Any]:
    """Create Pose Estimation developer options (cancel-only)."""
    components = {}
    with gr.Accordion(label=translator.t("ui.developer_options"), open=False) as developer_accordion:
        components["pose_developer_options_accordion"] = developer_accordion
        components["pose_cancel_jobs_info_md"] = gr.Markdown(translator.t("ui.cancel_current_jobs_info"))
        components["pose_cancel_jobs_btn"] = gr.Button(
            translator.t("ui.cancel_current_jobs"),
            variant="stop",
            interactive=True
        )
    return components


def create_fbx_dev_section(translator: Translator) -> Dict[str, Any]:
    """Create Generate FBX developer options (no cancel)."""
    components = {}
    with gr.Accordion(label=translator.t("ui.developer_options"), open=False) as developer_accordion:
        components["fbx_developer_options_accordion"] = developer_accordion
        components["fbx_cancel_jobs_info_md"] = gr.Markdown(translator.t("ui.cancel_current_jobs_info"))
        components["fbx_cancel_jobs_btn"] = gr.Button(
            translator.t("ui.cancel_current_jobs"),
            variant="stop",
            interactive=True
        )
        components["export_personalized_body_obj"] = gr.Checkbox(
            label=translator.t("ui.export_personalized_body_obj"),
            value=False
        )
        components["create_camera"] = gr.Checkbox(
            label=translator.t("ui.create_camera"),
            value=False
        )
        components["camera_zoom"] = gr.Number(
            label=translator.t("ui.camera_zoom"),
            value=-5.0,
            visible=False
        )
        components["camera_scene"] = gr.File(
            label=translator.t("ui.camera_scene"),
            file_types=["image", "video"],
            visible=False
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
        gr.update(value=t.t("ui.cancel_current_jobs")),  # cancel_jobs_btn
        gr.update(label=t.t("ui.export_personalized_body_obj")),  # export_personalized_body_obj
        gr.update(label=t.t("ui.create_camera")),  # create_camera
        gr.update(label=t.t("ui.camera_zoom")),  # camera_zoom
        gr.update(label=t.t("ui.camera_scene")),  # camera_scene
    )


def update_pose_cli_language(lang: str) -> Tuple[Any, ...]:
    """Update Pose CLI generator components with new language."""
    t = Translator(lang)
    return (
        gr.update(label=t.t("ui.pose_cli_generator_title")),
        gr.update(value=t.t("ui.pose_cli_generator_info")),
        gr.update(value=t.t("ui.pose_generate_cli_btn")),
        gr.update(label=t.t("ui.pose_cli_command_label"), info=t.t("ui.pose_cli_command_info")),
    )


def update_fbx_cli_language(lang: str) -> Tuple[Any, ...]:
    """Update FBX CLI generator components with new language."""
    t = Translator(lang)
    return (
        gr.update(label=t.t("ui.fbx_cli_generator_title")),
        gr.update(value=t.t("ui.fbx_cli_generator_info")),
        gr.update(value=t.t("ui.fbx_generate_cli_btn")),
        gr.update(label=t.t("ui.fbx_cli_command_label"), info=t.t("ui.fbx_cli_command_info")),
    )


def update_pose_dev_language(lang: str) -> Tuple[Any, ...]:
    """Update pose developer options language."""
    t = Translator(lang)
    return (
        gr.update(label=t.t("ui.developer_options")),
        gr.update(value=t.t("ui.cancel_current_jobs_info")),
        gr.update(value=t.t("ui.cancel_current_jobs")),
    )


def update_fbx_dev_language(lang: str) -> Tuple[Any, ...]:
    """Update FBX developer options language."""
    t = Translator(lang)
    return (
        gr.update(label=t.t("ui.developer_options")),
        gr.update(value=t.t("ui.cancel_current_jobs_info")),
        gr.update(value=t.t("ui.cancel_current_jobs")),
        gr.update(label=t.t("ui.export_personalized_body_obj")),
        gr.update(label=t.t("ui.create_camera")),
        gr.update(label=t.t("ui.camera_zoom")),
        gr.update(label=t.t("ui.camera_scene")),
    )


def toggle_camera_inputs(create_camera_value: bool) -> Tuple[Any, Any]:
    """
    Toggle visibility of camera inputs based on checkbox.
    """
    if isinstance(create_camera_value, list):
        create_camera_value = create_camera_value[0] if create_camera_value else False
    return (
        gr.update(visible=create_camera_value),
        gr.update(visible=create_camera_value),
    )
