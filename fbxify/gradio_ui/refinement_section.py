"""
Gradio UI components for refinement configuration.

This module provides UI components for configuring refinement profiles,
including base FilterProfile settings and root-specific settings.
"""
import json
import tempfile
import gradio as gr
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import asdict
from fbxify.refinement.profiles.filter_profile import FilterProfile
from fbxify.refinement.profiles.root_profile import ROOT_PROFILE
from fbxify.refinement.profiles.hands_profile import HANDS_PROFILE
from fbxify.refinement.profiles.fingers_profile import FINGERS_PROFILE
from fbxify.refinement.profiles.head_profile import HEAD_PROFILE
from fbxify.refinement.profiles.legs_profile import LEGS_PROFILE
from fbxify.refinement.profiles.arms_profile import ARMS_PROFILE
from fbxify.refinement.profiles.default_profile import DEFAULT_PROFILE
from fbxify.refinement.refinement_config import RefinementConfig
from fbxify.i18n import Translator


def create_filter_profile_ui(profile: FilterProfile, section_name: str, translator: Translator, is_root: bool = False) -> Dict[str, gr.Component]:
    """
    Create Gradio UI components for a FilterProfile.
    
    Args:
        profile: The FilterProfile instance to create UI for
        section_name: Display name for this section (e.g., "Fingers", "Hands")
        is_root: Whether this is a root profile (adds root-specific fields)
    
    Returns:
        Dictionary of component names to Gradio components
    """
    components = {}
    
    # Method choices with translations
    method_choices = [
        translator.t("ui.refinement.method_one_euro"),
        translator.t("ui.refinement.method_ema"),
        translator.t("ui.refinement.method_butterworth"),
    ]
    
    with gr.Group():
        gr.Markdown(f"### {section_name}", elem_classes=["refinement-profile-header"])
        
        # Spike thresholds
        components['max_pos_speed'] = gr.Number(
            label=translator.t("ui.refinement.max_pos_speed"),
            value=profile.max_pos_speed,
            minimum=0.0,
            step=0.1,
            info=translator.t("ui.refinement.max_pos_speed_info")
        )
        
        components['max_pos_accel'] = gr.Number(
            label=translator.t("ui.refinement.max_pos_accel"),
            value=profile.max_pos_accel,
            minimum=0.0,
            step=0.1,
            info=translator.t("ui.refinement.max_pos_accel_info")
        )
        
        components['max_ang_speed_deg'] = gr.Number(
            label=translator.t("ui.refinement.max_ang_speed_deg"),
            value=profile.max_ang_speed_deg,
            minimum=0.0,
            step=1.0,
            info=translator.t("ui.refinement.max_ang_speed_deg_info")
        )
        
        components['max_ang_accel_deg'] = gr.Number(
            label=translator.t("ui.refinement.max_ang_accel_deg"),
            value=profile.max_ang_accel_deg,
            minimum=0.0,
            step=1.0,
            info=translator.t("ui.refinement.max_ang_accel_deg_info")
        )
        
        # Smoothing method
        method_value = profile.method.title() if profile.method else translator.t("ui.refinement.method_one_euro")
        # Map method value to translated choice
        method_map = {
            "one_euro": translator.t("ui.refinement.method_one_euro"),
            "ema": translator.t("ui.refinement.method_ema"),
            "butterworth": translator.t("ui.refinement.method_butterworth"),
        }
        if profile.method and profile.method.lower() in method_map:
            method_value = method_map[profile.method.lower()]
        else:
            method_value = method_choices[0]  # Default to first choice
        
        components['method'] = gr.Dropdown(
            label=translator.t("ui.refinement.method"),
            choices=method_choices,
            value=method_value,
            info=translator.t("ui.refinement.method_info")
        )
        
        # Method-specific parameters (conditionally visible)
        method_lower = profile.method.lower() if profile.method else "one_euro"
        components['cutoff_hz'] = gr.Number(
            label=translator.t("ui.refinement.cutoff_hz"),
            value=profile.cutoff_hz,
            minimum=0.01,
            step=0.1,
            visible=method_lower in ["ema", "butterworth"],
            info=translator.t("ui.refinement.cutoff_hz_info")
        )
        
        components['one_euro_min_cutoff'] = gr.Number(
            label=translator.t("ui.refinement.one_euro_min_cutoff"),
            value=profile.one_euro_min_cutoff,
            minimum=0.01,
            step=0.1,
            visible=method_lower == "one_euro",
            info=translator.t("ui.refinement.one_euro_min_cutoff_info")
        )
        
        components['one_euro_beta'] = gr.Number(
            label=translator.t("ui.refinement.one_euro_beta"),
            value=profile.one_euro_beta,
            minimum=0.0,
            step=0.1,
            visible=method_lower == "one_euro",
            info=translator.t("ui.refinement.one_euro_beta_info")
        )
        
        components['one_euro_d_cutoff'] = gr.Number(
            label=translator.t("ui.refinement.one_euro_d_cutoff"),
            value=profile.one_euro_d_cutoff,
            minimum=0.01,
            step=0.1,
            visible=method_lower == "one_euro",
            info=translator.t("ui.refinement.one_euro_d_cutoff_info")
        )
        
        # Root-specific fields
        if is_root:
            components['root_cutoff_xy_hz'] = gr.Number(
                label=translator.t("ui.refinement.root_cutoff_xy_hz"),
                value=profile.root_cutoff_xy_hz,
                minimum=0.01,
                step=0.1,
                info=translator.t("ui.refinement.root_cutoff_xy_hz_info")
            )
            
            components['root_cutoff_z_hz'] = gr.Number(
                label=translator.t("ui.refinement.root_cutoff_z_hz"),
                value=profile.root_cutoff_z_hz,
                minimum=0.01,
                step=0.1,
                info=translator.t("ui.refinement.root_cutoff_z_hz_info")
            )
    
    return components


def build_refinement_config_from_gui(
    enabled: bool,
    interpolate_missing_keyframes: bool,
    use_foot_planting: bool,
    translator: Translator,
    # Root profile values
    root_max_pos_speed, root_max_pos_accel, root_max_ang_speed_deg, root_max_ang_accel_deg,
    root_method, root_cutoff_hz, root_one_euro_min_cutoff, root_one_euro_beta, root_one_euro_d_cutoff,
    root_cutoff_xy_hz, root_cutoff_z_hz,
    # Hands profile values
    hands_max_pos_speed, hands_max_pos_accel, hands_max_ang_speed_deg, hands_max_ang_accel_deg,
    hands_method, hands_cutoff_hz, hands_one_euro_min_cutoff, hands_one_euro_beta, hands_one_euro_d_cutoff,
    # Fingers profile values
    fingers_max_pos_speed, fingers_max_pos_accel, fingers_max_ang_speed_deg, fingers_max_ang_accel_deg,
    fingers_method, fingers_cutoff_hz, fingers_one_euro_min_cutoff, fingers_one_euro_beta, fingers_one_euro_d_cutoff,
    # Head profile values
    head_max_pos_speed, head_max_pos_accel, head_max_ang_speed_deg, head_max_ang_accel_deg,
    head_method, head_cutoff_hz, head_one_euro_min_cutoff, head_one_euro_beta, head_one_euro_d_cutoff,
    # Legs profile values
    legs_max_pos_speed, legs_max_pos_accel, legs_max_ang_speed_deg, legs_max_ang_accel_deg,
    legs_method, legs_cutoff_hz, legs_one_euro_min_cutoff, legs_one_euro_beta, legs_one_euro_d_cutoff,
    # Arms profile values
    arms_max_pos_speed, arms_max_pos_accel, arms_max_ang_speed_deg, arms_max_ang_accel_deg,
    arms_method, arms_cutoff_hz, arms_one_euro_min_cutoff, arms_one_euro_beta, arms_one_euro_d_cutoff,
    # Default profile values
    default_max_pos_speed, default_max_pos_accel, default_max_ang_speed_deg, default_max_ang_accel_deg,
    default_method, default_cutoff_hz, default_one_euro_min_cutoff, default_one_euro_beta, default_one_euro_d_cutoff,
    # Foot planting values
    foot_planting_velocity_threshold=None, foot_planting_min_height=None,
    foot_planting_contact_window=None, foot_planting_blend_factor=None,
    foot_planting_root_window=None, foot_planting_use_mid_foot=None,
) -> Optional[RefinementConfig]:
    """
    Build a RefinementConfig from GUI values.
    Returns None if refinement is disabled.
    """
    print(f"build_refinement_config_from_gui(): enabled={enabled}")
    if not enabled:
        print("build_refinement_config_from_gui(): enabled is False, returning None")
        return None
    print("build_refinement_config_from_gui(): enabled is True, building config...")
    
    def translate_method_to_internal(method_str):
        """Convert translated method name to internal value."""
        method_one_euro = translator.t("ui.refinement.method_one_euro")
        method_ema = translator.t("ui.refinement.method_ema")
        method_butterworth = translator.t("ui.refinement.method_butterworth")
        
        if method_str == method_one_euro:
            return "one_euro"
        elif method_str == method_ema:
            return "ema"
        elif method_str == method_butterworth:
            return "butterworth"
        else:
            return "one_euro"  # Default
    
    def create_profile(values, is_root=False):
        """Helper to create FilterProfile from parameter values."""
        method = translate_method_to_internal(values.get('method', translator.t("ui.refinement.method_one_euro")))
        profile = FilterProfile(
            max_pos_speed=values.get('max_pos_speed', 3.0),
            max_pos_accel=values.get('max_pos_accel', 30.0),
            max_ang_speed_deg=values.get('max_ang_speed_deg', 720.0),
            max_ang_accel_deg=values.get('max_ang_accel_deg', 7200.0),
            method=method,
            cutoff_hz=values.get('cutoff_hz', 4.0),
            one_euro_min_cutoff=values.get('one_euro_min_cutoff', 1.5),
            one_euro_beta=values.get('one_euro_beta', 0.5),
            one_euro_d_cutoff=values.get('one_euro_d_cutoff', 1.0),
        )
        if is_root:
            profile.root_cutoff_xy_hz = values.get('root_cutoff_xy_hz', 2.0)
            profile.root_cutoff_z_hz = values.get('root_cutoff_z_hz', 0.8)
        return profile
    
    # Build config from function parameters
    root_values = {
        'max_pos_speed': root_max_pos_speed, 'max_pos_accel': root_max_pos_accel,
        'max_ang_speed_deg': root_max_ang_speed_deg, 'max_ang_accel_deg': root_max_ang_accel_deg,
        'method': root_method, 'cutoff_hz': root_cutoff_hz,
        'one_euro_min_cutoff': root_one_euro_min_cutoff, 'one_euro_beta': root_one_euro_beta,
        'one_euro_d_cutoff': root_one_euro_d_cutoff,
        'root_cutoff_xy_hz': root_cutoff_xy_hz, 'root_cutoff_z_hz': root_cutoff_z_hz,
    }
    
    hands_values = {
        'max_pos_speed': hands_max_pos_speed, 'max_pos_accel': hands_max_pos_accel,
        'max_ang_speed_deg': hands_max_ang_speed_deg, 'max_ang_accel_deg': hands_max_ang_accel_deg,
        'method': hands_method, 'cutoff_hz': hands_cutoff_hz,
        'one_euro_min_cutoff': hands_one_euro_min_cutoff, 'one_euro_beta': hands_one_euro_beta,
        'one_euro_d_cutoff': hands_one_euro_d_cutoff,
    }
    
    fingers_values = {
        'max_pos_speed': fingers_max_pos_speed, 'max_pos_accel': fingers_max_pos_accel,
        'max_ang_speed_deg': fingers_max_ang_speed_deg, 'max_ang_accel_deg': fingers_max_ang_accel_deg,
        'method': fingers_method, 'cutoff_hz': fingers_cutoff_hz,
        'one_euro_min_cutoff': fingers_one_euro_min_cutoff, 'one_euro_beta': fingers_one_euro_beta,
        'one_euro_d_cutoff': fingers_one_euro_d_cutoff,
    }
    
    head_values = {
        'max_pos_speed': head_max_pos_speed, 'max_pos_accel': head_max_pos_accel,
        'max_ang_speed_deg': head_max_ang_speed_deg, 'max_ang_accel_deg': head_max_ang_accel_deg,
        'method': head_method, 'cutoff_hz': head_cutoff_hz,
        'one_euro_min_cutoff': head_one_euro_min_cutoff, 'one_euro_beta': head_one_euro_beta,
        'one_euro_d_cutoff': head_one_euro_d_cutoff,
    }
    
    legs_values = {
        'max_pos_speed': legs_max_pos_speed, 'max_pos_accel': legs_max_pos_accel,
        'max_ang_speed_deg': legs_max_ang_speed_deg, 'max_ang_accel_deg': legs_max_ang_accel_deg,
        'method': legs_method, 'cutoff_hz': legs_cutoff_hz,
        'one_euro_min_cutoff': legs_one_euro_min_cutoff, 'one_euro_beta': legs_one_euro_beta,
        'one_euro_d_cutoff': legs_one_euro_d_cutoff,
    }
    
    arms_values = {
        'max_pos_speed': arms_max_pos_speed, 'max_pos_accel': arms_max_pos_accel,
        'max_ang_speed_deg': arms_max_ang_speed_deg, 'max_ang_accel_deg': arms_max_ang_accel_deg,
        'method': arms_method, 'cutoff_hz': arms_cutoff_hz,
        'one_euro_min_cutoff': arms_one_euro_min_cutoff, 'one_euro_beta': arms_one_euro_beta,
        'one_euro_d_cutoff': arms_one_euro_d_cutoff,
    }
    
    default_values = {
        'max_pos_speed': default_max_pos_speed, 'max_pos_accel': default_max_pos_accel,
        'max_ang_speed_deg': default_max_ang_speed_deg, 'max_ang_accel_deg': default_max_ang_accel_deg,
        'method': default_method, 'cutoff_hz': default_cutoff_hz,
        'one_euro_min_cutoff': default_one_euro_min_cutoff, 'one_euro_beta': default_one_euro_beta,
        'one_euro_d_cutoff': default_one_euro_d_cutoff,
    }
    
    config = RefinementConfig()
    config.profiles = {
        "root": create_profile(root_values, is_root=True),
        "*hand*": create_profile(hands_values, is_root=False),
        "*finger*": create_profile(fingers_values, is_root=False),
        "*head*": create_profile(head_values, is_root=False),
        "*leg*": create_profile(legs_values, is_root=False),
        "*arm*": create_profile(arms_values, is_root=False),
        "*wrist*": create_profile(arms_values, is_root=False),
        "*clavicle*": create_profile(arms_values, is_root=False),
        "*": create_profile(default_values, is_root=False),   # default
    }
    config.do_spike_fix = True
    config.do_rotation_smoothing = True
    config.do_vector_smoothing = True
    config.do_root_motion_fix = True
    config.do_interpolate_missing_keyframes = interpolate_missing_keyframes
    config.do_foot_planting = use_foot_planting
    
    # Set foot planting config if enabled
    if use_foot_planting:
        from fbxify.refinement.foot_planting_config import FootPlantingConfig
        config.foot_planting_config = FootPlantingConfig(
            foot_contact_velocity_threshold=foot_planting_velocity_threshold if foot_planting_velocity_threshold is not None else 0.15,
            foot_contact_min_height=foot_planting_min_height if foot_planting_min_height is not None else 0.10,
            contact_smoothing_window=int(foot_planting_contact_window) if foot_planting_contact_window is not None else 3,
            blend_factor=foot_planting_blend_factor if foot_planting_blend_factor is not None else 0.3,
            root_smoothing_window=int(foot_planting_root_window) if foot_planting_root_window is not None else 5,
            use_mid_foot=foot_planting_use_mid_foot if foot_planting_use_mid_foot is not None else True,
        )
    
    return config


def update_method_visibility(method: str, translator: Translator) -> Tuple[gr.update, gr.update, gr.update, gr.update]:
    """
    Update visibility of method-specific parameters based on selected method.
    
    Args:
        method: Selected method (translated string)
        translator: Translator instance for method name mapping
    
    Returns:
        Tuple of updates for cutoff_hz, one_euro_min_cutoff, one_euro_beta, one_euro_d_cutoff
    """
    # Map translated method names back to internal values
    method_one_euro = translator.t("ui.refinement.method_one_euro")
    method_ema = translator.t("ui.refinement.method_ema")
    method_butterworth = translator.t("ui.refinement.method_butterworth")
    
    # Determine which method was selected
    if method == method_one_euro:
        method_lower = "one_euro"
    elif method == method_ema:
        method_lower = "ema"
    elif method == method_butterworth:
        method_lower = "butterworth"
    else:
        method_lower = "one_euro"  # Default
    
    cutoff_visible = method_lower in ["ema", "butterworth"]
    one_euro_visible = method_lower == "one_euro"
    
    return (
        gr.update(visible=cutoff_visible),  # cutoff_hz
        gr.update(visible=one_euro_visible),  # one_euro_min_cutoff
        gr.update(visible=one_euro_visible),  # one_euro_beta
        gr.update(visible=one_euro_visible),  # one_euro_d_cutoff
    )


def create_refinement_section(translator: Translator) -> Dict[str, Any]:
    """
    Create the complete refinement section UI.
    This function should be called within a Gradio Blocks context.
    
    Returns:
        Dictionary of component names to Gradio components
    """
    with gr.Group():
        gr.Markdown(f"## {translator.t('ui.refinement.title')}", elem_classes=["refinement-main-header"])
    
    
    # Refinement Global Settings group
    with gr.Group():
        gr.Markdown(f"### {translator.t('ui.refinement.global_settings')}", elem_classes=["refinement-profile-header"])

        # Global controls
        with gr.Row():
            with gr.Column():
                refinement_enabled = gr.Checkbox(
                    label=translator.t("ui.refinement.enabled"),
                    value=True,
                    info=translator.t("ui.refinement.enabled_info")
                )
            
                interpolate_missing_keyframes = gr.Checkbox(
                    label=translator.t("ui.refinement.interpolate_missing_keyframes"),
                    value=False,
                    info=translator.t("ui.refinement.interpolate_missing_keyframes_info")
                )
                
                use_foot_planting = gr.Checkbox(
                    label=translator.t("ui.refinement.use_foot_planting"),
                    value=True,
                    info=translator.t("ui.refinement.use_foot_planting_info")
                )
            
            config_file_upload = gr.File(
                label=translator.t("ui.refinement.load_config"),
                file_types=[".json"],
                type="filepath"
            )
        
        save_config_btn = gr.Button(translator.t("ui.refinement.save_config_btn"), variant="secondary")
        
        config_file_download = gr.File(
            label=translator.t("ui.refinement.save_config"),
            interactive=False,
            visible=False
        )
    
    # Foot Planting section (conditionally visible)
    from fbxify.refinement.foot_planting_config import FootPlantingConfig
    default_foot_planting_config = FootPlantingConfig()
    
    with gr.Group(visible=True) as foot_planting_group:
        gr.Markdown(f"### {translator.t('ui.refinement.foot_planting.title')}", elem_classes=["refinement-profile-header"])
        
        foot_planting_velocity_threshold = gr.Number(
            label=translator.t("ui.refinement.foot_planting.foot_contact_velocity_threshold"),
            value=default_foot_planting_config.foot_contact_velocity_threshold,
            minimum=0.001,
            step=0.01,
            info=translator.t("ui.refinement.foot_planting.foot_contact_velocity_threshold_info")
        )
        
        foot_planting_min_height = gr.Number(
            label=translator.t("ui.refinement.foot_planting.foot_contact_min_height"),
            value=default_foot_planting_config.foot_contact_min_height,
            minimum=0.0,
            step=0.01,
            info=translator.t("ui.refinement.foot_planting.foot_contact_min_height_info")
        )
        
        foot_planting_contact_window = gr.Number(
            label=translator.t("ui.refinement.foot_planting.contact_smoothing_window"),
            value=default_foot_planting_config.contact_smoothing_window,
            minimum=1,
            step=1,
            precision=0,
            info=translator.t("ui.refinement.foot_planting.contact_smoothing_window_info")
        )
        
        foot_planting_blend_factor = gr.Slider(
            label=translator.t("ui.refinement.foot_planting.blend_factor"),
            value=default_foot_planting_config.blend_factor,
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            info=translator.t("ui.refinement.foot_planting.blend_factor_info")
        )
        
        foot_planting_root_window = gr.Number(
            label=translator.t("ui.refinement.foot_planting.root_smoothing_window"),
            value=default_foot_planting_config.root_smoothing_window,
            minimum=1,
            step=1,
            precision=0,
            info=translator.t("ui.refinement.foot_planting.root_smoothing_window_info")
        )
        
        foot_planting_use_mid_foot = gr.Checkbox(
            label=translator.t("ui.refinement.foot_planting.use_mid_foot"),
            value=default_foot_planting_config.use_mid_foot,
            info=translator.t("ui.refinement.foot_planting.use_mid_foot_info")
        )
    
    # Show/hide foot planting section based on checkbox
    def toggle_foot_planting(use_foot_planting_value):
        return gr.update(visible=use_foot_planting_value)
    
    use_foot_planting.change(
        fn=toggle_foot_planting,
        inputs=[use_foot_planting],
        outputs=[foot_planting_group]
    )
    
    # Create profile sections in a row layout
    profile_components = {}
    
    # Root profile (with root-specific fields)
    root_components = create_filter_profile_ui(
        ROOT_PROFILE, 
        translator.t("ui.refinement.sections.root"), 
        translator,
        is_root=True
    )
    profile_components["root"] = root_components
    
    # Hands profile
    hands_components = create_filter_profile_ui(
        HANDS_PROFILE, 
        translator.t("ui.refinement.sections.hands"), 
        translator,
        is_root=False
    )
    profile_components["hands"] = hands_components
    
    # Fingers profile
    fingers_components = create_filter_profile_ui(
        FINGERS_PROFILE, 
        translator.t("ui.refinement.sections.fingers"), 
        translator,
        is_root=False
    )
    profile_components["fingers"] = fingers_components
    
    # Head profile
    head_components = create_filter_profile_ui(
        HEAD_PROFILE, 
        translator.t("ui.refinement.sections.head"), 
        translator,
        is_root=False
    )
    profile_components["head"] = head_components
    
    # Legs profile
    legs_components = create_filter_profile_ui(
        LEGS_PROFILE, 
        translator.t("ui.refinement.sections.legs"), 
        translator,
        is_root=False
    )
    profile_components["legs"] = legs_components
    
    # Arms profile
    arms_components = create_filter_profile_ui(
        ARMS_PROFILE, 
        translator.t("ui.refinement.sections.arms"), 
        translator,
        is_root=False
    )
    profile_components["arms"] = arms_components
    
    # Default profile
    default_components = create_filter_profile_ui(
        DEFAULT_PROFILE, 
        translator.t("ui.refinement.sections.default"), 
        translator,
        is_root=False
    )
    profile_components["default"] = default_components
    
    # Wire up method change handlers for each profile
    for profile_name, components in profile_components.items():
        if 'method' in components:
            method_dropdown = components['method']
            cutoff_hz = components.get('cutoff_hz')
            one_euro_min_cutoff = components.get('one_euro_min_cutoff')
            one_euro_beta = components.get('one_euro_beta')
            one_euro_d_cutoff = components.get('one_euro_d_cutoff')
            
            if cutoff_hz and one_euro_min_cutoff and one_euro_beta and one_euro_d_cutoff:
                # Create a closure to capture translator
                def make_update_fn(t):
                    return lambda method: update_method_visibility(method, t)
                
                method_dropdown.change(
                    fn=make_update_fn(translator),
                    inputs=[method_dropdown],
                    outputs=[cutoff_hz, one_euro_min_cutoff, one_euro_beta, one_euro_d_cutoff]
                )
    
    # Function to enable/disable all refinement inputs based on checkbox
    def toggle_refinement_inputs(enabled_value):
        """Enable or disable all refinement inputs based on the enabled checkbox."""
        # Collect all outputs that need to be updated
        outputs_list = []
        
        # Add global checkboxes
        outputs_list.append(gr.update(interactive=enabled_value))  # interpolate_missing_keyframes
        outputs_list.append(gr.update(interactive=enabled_value))  # use_foot_planting
        
        # Add foot planting inputs
        outputs_list.append(gr.update(interactive=enabled_value))  # foot_planting_velocity_threshold
        outputs_list.append(gr.update(interactive=enabled_value))  # foot_planting_min_height
        outputs_list.append(gr.update(interactive=enabled_value))  # foot_planting_contact_window
        outputs_list.append(gr.update(interactive=enabled_value))  # foot_planting_blend_factor
        outputs_list.append(gr.update(interactive=enabled_value))  # foot_planting_root_window
        outputs_list.append(gr.update(interactive=enabled_value))  # foot_planting_use_mid_foot
        
        # Add all profile components
        for profile_name, components in profile_components.items():
            for key in ['max_pos_speed', 'max_pos_accel', 'max_ang_speed_deg', 'max_ang_accel_deg',
                        'method', 'cutoff_hz', 'one_euro_min_cutoff', 'one_euro_beta', 'one_euro_d_cutoff']:
                if key in components:
                    outputs_list.append(gr.update(interactive=enabled_value))
            # Add root-specific fields if this is the root profile
            if profile_name == "root":
                for key in ['root_cutoff_xy_hz', 'root_cutoff_z_hz']:
                    if key in components:
                        outputs_list.append(gr.update(interactive=enabled_value))
        
        return outputs_list
    
    # Collect all outputs for the toggle function
    toggle_outputs = [
        interpolate_missing_keyframes,
        use_foot_planting,
        foot_planting_velocity_threshold,
        foot_planting_min_height,
        foot_planting_contact_window,
        foot_planting_blend_factor,
        foot_planting_root_window,
        foot_planting_use_mid_foot,
    ]
    
    # Add all profile components to outputs
    for profile_name, components in profile_components.items():
        for key in ['max_pos_speed', 'max_pos_accel', 'max_ang_speed_deg', 'max_ang_accel_deg',
                    'method', 'cutoff_hz', 'one_euro_min_cutoff', 'one_euro_beta', 'one_euro_d_cutoff']:
            if key in components:
                toggle_outputs.append(components[key])
        # Add root-specific fields if this is the root profile
        if profile_name == "root":
            for key in ['root_cutoff_xy_hz', 'root_cutoff_z_hz']:
                if key in components:
                    toggle_outputs.append(components[key])
    
    # Wire up the refinement_enabled checkbox to toggle all inputs
    refinement_enabled.change(
        fn=toggle_refinement_inputs,
        inputs=[refinement_enabled],
        outputs=toggle_outputs
    )
    
    # Save configuration function
    def save_configuration(
        enabled: bool,
        interpolate_missing_keyframes: bool,
        use_foot_planting: bool,
        # Root profile values
        root_max_pos_speed, root_max_pos_accel, root_max_ang_speed_deg, root_max_ang_accel_deg,
        root_method, root_cutoff_hz, root_one_euro_min_cutoff, root_one_euro_beta, root_one_euro_d_cutoff,
        root_cutoff_xy_hz, root_cutoff_z_hz,
        # Hands profile values
        hands_max_pos_speed, hands_max_pos_accel, hands_max_ang_speed_deg, hands_max_ang_accel_deg,
        hands_method, hands_cutoff_hz, hands_one_euro_min_cutoff, hands_one_euro_beta, hands_one_euro_d_cutoff,
        # Fingers profile values
        fingers_max_pos_speed, fingers_max_pos_accel, fingers_max_ang_speed_deg, fingers_max_ang_accel_deg,
        fingers_method, fingers_cutoff_hz, fingers_one_euro_min_cutoff, fingers_one_euro_beta, fingers_one_euro_d_cutoff,
        # Head profile values
        head_max_pos_speed, head_max_pos_accel, head_max_ang_speed_deg, head_max_ang_accel_deg,
        head_method, head_cutoff_hz, head_one_euro_min_cutoff, head_one_euro_beta, head_one_euro_d_cutoff,
        # Legs profile values
        legs_max_pos_speed, legs_max_pos_accel, legs_max_ang_speed_deg, legs_max_ang_accel_deg,
        legs_method, legs_cutoff_hz, legs_one_euro_min_cutoff, legs_one_euro_beta, legs_one_euro_d_cutoff,
        # Arms profile values
        arms_max_pos_speed, arms_max_pos_accel, arms_max_ang_speed_deg, arms_max_ang_accel_deg,
        arms_method, arms_cutoff_hz, arms_one_euro_min_cutoff, arms_one_euro_beta, arms_one_euro_d_cutoff,
        # Default profile values
        default_max_pos_speed, default_max_pos_accel, default_max_ang_speed_deg, default_max_ang_accel_deg,
        default_method, default_cutoff_hz, default_one_euro_min_cutoff, default_one_euro_beta, default_one_euro_d_cutoff,
        # Foot planting values
        foot_planting_velocity_threshold=None, foot_planting_min_height=None,
        foot_planting_contact_window=None, foot_planting_blend_factor=None,
        foot_planting_root_window=None, foot_planting_use_mid_foot=None,
    ) -> str:
        """Save current configuration to JSON."""
        def translate_method_to_internal(method_str):
            """Convert translated method name to internal value."""
            method_one_euro = translator.t("ui.refinement.method_one_euro")
            method_ema = translator.t("ui.refinement.method_ema")
            method_butterworth = translator.t("ui.refinement.method_butterworth")
            
            if method_str == method_one_euro:
                return "one_euro"
            elif method_str == method_ema:
                return "ema"
            elif method_str == method_butterworth:
                return "butterworth"
            else:
                return "one_euro"  # Default
        
        def create_profile_dict(values, is_root=False):
            """Helper to create profile dict from parameter values."""
            method = translate_method_to_internal(values.get('method', translator.t("ui.refinement.method_one_euro")))
            profile_dict = {
                "max_pos_speed": values.get('max_pos_speed', 3.0),
                "max_pos_accel": values.get('max_pos_accel', 30.0),
                "max_ang_speed_deg": values.get('max_ang_speed_deg', 720.0),
                "max_ang_accel_deg": values.get('max_ang_accel_deg', 7200.0),
                "method": method,
                "cutoff_hz": values.get('cutoff_hz', 4.0),
                "one_euro_min_cutoff": values.get('one_euro_min_cutoff', 1.5),
                "one_euro_beta": values.get('one_euro_beta', 0.5),
                "one_euro_d_cutoff": values.get('one_euro_d_cutoff', 1.0),
            }
            if is_root:
                profile_dict["root_cutoff_xy_hz"] = values.get('root_cutoff_xy_hz', 2.0)
                profile_dict["root_cutoff_z_hz"] = values.get('root_cutoff_z_hz', 0.8)
            return profile_dict
        
        # Build config from function parameters
        root_values = {
            'max_pos_speed': root_max_pos_speed, 'max_pos_accel': root_max_pos_accel,
            'max_ang_speed_deg': root_max_ang_speed_deg, 'max_ang_accel_deg': root_max_ang_accel_deg,
            'method': root_method, 'cutoff_hz': root_cutoff_hz,
            'one_euro_min_cutoff': root_one_euro_min_cutoff, 'one_euro_beta': root_one_euro_beta,
            'one_euro_d_cutoff': root_one_euro_d_cutoff,
            'root_cutoff_xy_hz': root_cutoff_xy_hz, 'root_cutoff_z_hz': root_cutoff_z_hz,
        }
        
        hands_values = {
            'max_pos_speed': hands_max_pos_speed, 'max_pos_accel': hands_max_pos_accel,
            'max_ang_speed_deg': hands_max_ang_speed_deg, 'max_ang_accel_deg': hands_max_ang_accel_deg,
            'method': hands_method, 'cutoff_hz': hands_cutoff_hz,
            'one_euro_min_cutoff': hands_one_euro_min_cutoff, 'one_euro_beta': hands_one_euro_beta,
            'one_euro_d_cutoff': hands_one_euro_d_cutoff,
        }
        
        fingers_values = {
            'max_pos_speed': fingers_max_pos_speed, 'max_pos_accel': fingers_max_pos_accel,
            'max_ang_speed_deg': fingers_max_ang_speed_deg, 'max_ang_accel_deg': fingers_max_ang_accel_deg,
            'method': fingers_method, 'cutoff_hz': fingers_cutoff_hz,
            'one_euro_min_cutoff': fingers_one_euro_min_cutoff, 'one_euro_beta': fingers_one_euro_beta,
            'one_euro_d_cutoff': fingers_one_euro_d_cutoff,
        }
        
        head_values = {
            'max_pos_speed': head_max_pos_speed, 'max_pos_accel': head_max_pos_accel,
            'max_ang_speed_deg': head_max_ang_speed_deg, 'max_ang_accel_deg': head_max_ang_accel_deg,
            'method': head_method, 'cutoff_hz': head_cutoff_hz,
            'one_euro_min_cutoff': head_one_euro_min_cutoff, 'one_euro_beta': head_one_euro_beta,
            'one_euro_d_cutoff': head_one_euro_d_cutoff,
        }
        
        legs_values = {
            'max_pos_speed': legs_max_pos_speed, 'max_pos_accel': legs_max_pos_accel,
            'max_ang_speed_deg': legs_max_ang_speed_deg, 'max_ang_accel_deg': legs_max_ang_accel_deg,
            'method': legs_method, 'cutoff_hz': legs_cutoff_hz,
            'one_euro_min_cutoff': legs_one_euro_min_cutoff, 'one_euro_beta': legs_one_euro_beta,
            'one_euro_d_cutoff': legs_one_euro_d_cutoff,
        }
        
        arms_values = {
            'max_pos_speed': arms_max_pos_speed, 'max_pos_accel': arms_max_pos_accel,
            'max_ang_speed_deg': arms_max_ang_speed_deg, 'max_ang_accel_deg': arms_max_ang_accel_deg,
            'method': arms_method, 'cutoff_hz': arms_cutoff_hz,
            'one_euro_min_cutoff': arms_one_euro_min_cutoff, 'one_euro_beta': arms_one_euro_beta,
            'one_euro_d_cutoff': arms_one_euro_d_cutoff,
        }
        
        default_values = {
            'max_pos_speed': default_max_pos_speed, 'max_pos_accel': default_max_pos_accel,
            'max_ang_speed_deg': default_max_ang_speed_deg, 'max_ang_accel_deg': default_max_ang_accel_deg,
            'method': default_method, 'cutoff_hz': default_cutoff_hz,
            'one_euro_min_cutoff': default_one_euro_min_cutoff, 'one_euro_beta': default_one_euro_beta,
            'one_euro_d_cutoff': default_one_euro_d_cutoff,
        }
        
        config = {
            "enabled": enabled,
            "profiles": {
                "root": create_profile_dict(root_values, is_root=True),
                "*hand*": create_profile_dict(hands_values, is_root=False),
                "*finger*": create_profile_dict(fingers_values, is_root=False),
                "*head*": create_profile_dict(head_values, is_root=False),
                "*leg*": create_profile_dict(legs_values, is_root=False),
                "*arm*": create_profile_dict(arms_values, is_root=False),
                "*wrist*": create_profile_dict(arms_values, is_root=False),
                "*clavicle*": create_profile_dict(arms_values, is_root=False),
                "*": create_profile_dict(default_values, is_root=False),
            },
            "do_spike_fix": True,
            "do_rotation_smoothing": True,
            "do_vector_smoothing": True,
            "do_root_motion_fix": True,
            "do_interpolate_missing_keyframes": interpolate_missing_keyframes,
            "do_foot_planting": use_foot_planting,
        }
        
        # Add foot planting config if enabled
        if use_foot_planting:
            from fbxify.refinement.foot_planting_config import FootPlantingConfig
            foot_planting_config = FootPlantingConfig(
                foot_contact_velocity_threshold=foot_planting_velocity_threshold if foot_planting_velocity_threshold is not None else 0.05,
                foot_contact_min_height=foot_planting_min_height if foot_planting_min_height is not None else 0.02,
                contact_smoothing_window=int(foot_planting_contact_window) if foot_planting_contact_window is not None else 3,
                blend_factor=foot_planting_blend_factor if foot_planting_blend_factor is not None else 0.3,
                root_smoothing_window=int(foot_planting_root_window) if foot_planting_root_window is not None else 5,
                use_mid_foot=foot_planting_use_mid_foot if foot_planting_use_mid_foot is not None else True,
            )
            config["foot_planting_config"] = foot_planting_config.to_dict()
        
        # Create a temporary file with the JSON
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, temp_file, indent=2)
        temp_file.close()
        return temp_file.name
    
    # Collect all inputs for save function
    save_inputs = [refinement_enabled, interpolate_missing_keyframes, use_foot_planting]
    
    # Add root profile inputs
    save_inputs.extend([
        root_components['max_pos_speed'], root_components['max_pos_accel'],
        root_components['max_ang_speed_deg'], root_components['max_ang_accel_deg'],
        root_components['method'], root_components['cutoff_hz'],
        root_components['one_euro_min_cutoff'], root_components['one_euro_beta'],
        root_components['one_euro_d_cutoff'],
        root_components['root_cutoff_xy_hz'], root_components['root_cutoff_z_hz'],
    ])
    
    # Add hands profile inputs
    save_inputs.extend([
        hands_components['max_pos_speed'], hands_components['max_pos_accel'],
        hands_components['max_ang_speed_deg'], hands_components['max_ang_accel_deg'],
        hands_components['method'], hands_components['cutoff_hz'],
        hands_components['one_euro_min_cutoff'], hands_components['one_euro_beta'],
        hands_components['one_euro_d_cutoff'],
    ])
    
    # Add fingers profile inputs
    save_inputs.extend([
        fingers_components['max_pos_speed'], fingers_components['max_pos_accel'],
        fingers_components['max_ang_speed_deg'], fingers_components['max_ang_accel_deg'],
        fingers_components['method'], fingers_components['cutoff_hz'],
        fingers_components['one_euro_min_cutoff'], fingers_components['one_euro_beta'],
        fingers_components['one_euro_d_cutoff'],
    ])
    
    # Add head profile inputs
    save_inputs.extend([
        head_components['max_pos_speed'], head_components['max_pos_accel'],
        head_components['max_ang_speed_deg'], head_components['max_ang_accel_deg'],
        head_components['method'], head_components['cutoff_hz'],
        head_components['one_euro_min_cutoff'], head_components['one_euro_beta'],
        head_components['one_euro_d_cutoff'],
    ])
    
    # Add legs profile inputs
    save_inputs.extend([
        legs_components['max_pos_speed'], legs_components['max_pos_accel'],
        legs_components['max_ang_speed_deg'], legs_components['max_ang_accel_deg'],
        legs_components['method'], legs_components['cutoff_hz'],
        legs_components['one_euro_min_cutoff'], legs_components['one_euro_beta'],
        legs_components['one_euro_d_cutoff'],
    ])
    
    # Add arms profile inputs
    save_inputs.extend([
        arms_components['max_pos_speed'], arms_components['max_pos_accel'],
        arms_components['max_ang_speed_deg'], arms_components['max_ang_accel_deg'],
        arms_components['method'], arms_components['cutoff_hz'],
        arms_components['one_euro_min_cutoff'], arms_components['one_euro_beta'],
        arms_components['one_euro_d_cutoff'],
    ])
    
    # Add default profile inputs
    save_inputs.extend([
        default_components['max_pos_speed'], default_components['max_pos_accel'],
        default_components['max_ang_speed_deg'], default_components['max_ang_accel_deg'],
        default_components['method'], default_components['cutoff_hz'],
        default_components['one_euro_min_cutoff'], default_components['one_euro_beta'],
        default_components['one_euro_d_cutoff'],
    ])
    
    # Add foot planting inputs
    save_inputs.extend([
        foot_planting_velocity_threshold,
        foot_planting_min_height,
        foot_planting_contact_window,
        foot_planting_blend_factor,
        foot_planting_root_window,
        foot_planting_use_mid_foot,
    ])
    
    # Wire up save button - show config file download when clicked
    def save_and_show_config(*args):
        """Save configuration and show the download file."""
        file_path = save_configuration(*args)
        return gr.update(value=file_path, visible=True)
    
    save_config_btn.click(
        fn=save_and_show_config,
        inputs=save_inputs,
        outputs=[config_file_download]
    )
    
    # Load configuration function
    def load_configuration(file_path: str) -> List[gr.update]:
        """Load configuration from JSON file and return updates for all components."""
        if file_path is None or file_path == "":
            return [gr.update()] * (1 + len(save_inputs) - 1)  # enabled + all profile fields
        
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return [gr.update()] * (1 + len(save_inputs) - 1)
        
        updates = []
        
        # Update enabled checkbox
        updates.append(gr.update(value=config.get("enabled", True)))
        
        # Update interpolate_missing_keyframes checkbox
        updates.append(gr.update(value=config.get("do_interpolate_missing_keyframes", False)))
        
        # Update use_foot_planting checkbox
        updates.append(gr.update(value=config.get("do_foot_planting", False)))
        
        # Update each profile
        profiles = config.get("profiles", {})
        
        def translate_method_to_ui(method_str):
            """Convert internal method value to translated UI value."""
            method_lower = method_str.lower() if method_str else "one_euro"
            method_map = {
                "one_euro": translator.t("ui.refinement.method_one_euro"),
                "ema": translator.t("ui.refinement.method_ema"),
                "butterworth": translator.t("ui.refinement.method_butterworth"),
            }
            return method_map.get(method_lower, method_map["one_euro"])
        
        def get_profile_updates(profile_key: str, component_keys: List[str], is_root: bool = False):
            """Get updates for a profile from config, matching the order of component_keys."""
            profile_data = profiles.get(profile_key, {})
            if not profile_data:
                return [gr.update()] * len(component_keys)
            
            profile_updates = []
            for key in component_keys:
                if key in profile_data:
                    value = profile_data[key]
                    # Convert method to translated value
                    if key == 'method':
                        value = translate_method_to_ui(value)
                    profile_updates.append(gr.update(value=value))
                else:
                    profile_updates.append(gr.update())
            
            return profile_updates
        
        # Root profile (matching save_inputs order)
        root_keys = ['max_pos_speed', 'max_pos_accel', 'max_ang_speed_deg', 'max_ang_accel_deg',
                    'method', 'cutoff_hz', 'one_euro_min_cutoff', 'one_euro_beta', 'one_euro_d_cutoff',
                    'root_cutoff_xy_hz', 'root_cutoff_z_hz']
        updates.extend(get_profile_updates("root", root_keys, is_root=True))
        
        # Hands profile
        hands_keys = ['max_pos_speed', 'max_pos_accel', 'max_ang_speed_deg', 'max_ang_accel_deg',
                     'method', 'cutoff_hz', 'one_euro_min_cutoff', 'one_euro_beta', 'one_euro_d_cutoff']
        updates.extend(get_profile_updates("*hand*", hands_keys, is_root=False))
        
        # Fingers profile
        fingers_keys = ['max_pos_speed', 'max_pos_accel', 'max_ang_speed_deg', 'max_ang_accel_deg',
                       'method', 'cutoff_hz', 'one_euro_min_cutoff', 'one_euro_beta', 'one_euro_d_cutoff']
        updates.extend(get_profile_updates("*finger*", fingers_keys, is_root=False))
        
        # Head profile
        head_keys = ['max_pos_speed', 'max_pos_accel', 'max_ang_speed_deg', 'max_ang_accel_deg',
                    'method', 'cutoff_hz', 'one_euro_min_cutoff', 'one_euro_beta', 'one_euro_d_cutoff']
        updates.extend(get_profile_updates("*head*", head_keys, is_root=False))
        
        # Legs profile
        legs_keys = ['max_pos_speed', 'max_pos_accel', 'max_ang_speed_deg', 'max_ang_accel_deg',
                    'method', 'cutoff_hz', 'one_euro_min_cutoff', 'one_euro_beta', 'one_euro_d_cutoff']
        updates.extend(get_profile_updates("*leg*", legs_keys, is_root=False))
        
        # Arms profile
        arms_keys = ['max_pos_speed', 'max_pos_accel', 'max_ang_speed_deg', 'max_ang_accel_deg',
                    'method', 'cutoff_hz', 'one_euro_min_cutoff', 'one_euro_beta', 'one_euro_d_cutoff']
        updates.extend(get_profile_updates("*arm*", arms_keys, is_root=False))
        
        # Default profile
        default_keys = ['max_pos_speed', 'max_pos_accel', 'max_ang_speed_deg', 'max_ang_accel_deg',
                       'method', 'cutoff_hz', 'one_euro_min_cutoff', 'one_euro_beta', 'one_euro_d_cutoff']
        updates.extend(get_profile_updates("*", default_keys, is_root=False))
        
        # Foot planting parameters
        foot_planting_config = config.get("foot_planting_config", {})
        updates.append(gr.update(value=foot_planting_config.get("foot_contact_velocity_threshold", 0.15)))
        updates.append(gr.update(value=foot_planting_config.get("foot_contact_min_height", 0.10)))
        updates.append(gr.update(value=foot_planting_config.get("contact_smoothing_window", 3)))
        updates.append(gr.update(value=foot_planting_config.get("blend_factor", 0.3)))
        updates.append(gr.update(value=foot_planting_config.get("root_smoothing_window", 5)))
        updates.append(gr.update(value=foot_planting_config.get("use_mid_foot", True)))
        
        return updates
    
    # Wire up load file
    load_outputs = [refinement_enabled, interpolate_missing_keyframes, use_foot_planting] + save_inputs[3:]  # enabled, interpolate_missing_keyframes, use_foot_planting, then all profile fields
    config_file_upload.change(
        fn=load_configuration,
        inputs=[config_file_upload],
        outputs=load_outputs
    )
    
    # Create a hidden State component to store the built config
    # We'll use a wrapper function to build the config from all inputs
    refinement_config_state = gr.State(value=None)
    
    # Function to build config from all refinement inputs
    def build_refinement_config_wrapper(*args):
        """Wrapper that takes all refinement inputs and builds the config."""
        # Unpack all arguments in the correct order
        (enabled, interpolate_missing_keyframes, use_foot_planting,
         root_max_pos_speed, root_max_pos_accel, root_max_ang_speed_deg, root_max_ang_accel_deg,
         root_method, root_cutoff_hz, root_one_euro_min_cutoff, root_one_euro_beta, root_one_euro_d_cutoff,
         root_cutoff_xy_hz, root_cutoff_z_hz,
         hands_max_pos_speed, hands_max_pos_accel, hands_max_ang_speed_deg, hands_max_ang_accel_deg,
         hands_method, hands_cutoff_hz, hands_one_euro_min_cutoff, hands_one_euro_beta, hands_one_euro_d_cutoff,
         fingers_max_pos_speed, fingers_max_pos_accel, fingers_max_ang_speed_deg, fingers_max_ang_accel_deg,
         fingers_method, fingers_cutoff_hz, fingers_one_euro_min_cutoff, fingers_one_euro_beta, fingers_one_euro_d_cutoff,
         head_max_pos_speed, head_max_pos_accel, head_max_ang_speed_deg, head_max_ang_accel_deg,
         head_method, head_cutoff_hz, head_one_euro_min_cutoff, head_one_euro_beta, head_one_euro_d_cutoff,
         legs_max_pos_speed, legs_max_pos_accel, legs_max_ang_speed_deg, legs_max_ang_accel_deg,
         legs_method, legs_cutoff_hz, legs_one_euro_min_cutoff, legs_one_euro_beta, legs_one_euro_d_cutoff,
         arms_max_pos_speed, arms_max_pos_accel, arms_max_ang_speed_deg, arms_max_ang_accel_deg,
         arms_method, arms_cutoff_hz, arms_one_euro_min_cutoff, arms_one_euro_beta, arms_one_euro_d_cutoff,
         default_max_pos_speed, default_max_pos_accel, default_max_ang_speed_deg, default_max_ang_accel_deg,
         default_method, default_cutoff_hz, default_one_euro_min_cutoff, default_one_euro_beta, default_one_euro_d_cutoff,
         foot_planting_velocity_threshold, foot_planting_min_height,
         foot_planting_contact_window, foot_planting_blend_factor,
         foot_planting_root_window, foot_planting_use_mid_foot) = args
        
        return build_refinement_config_from_gui(
            enabled, interpolate_missing_keyframes, use_foot_planting, translator,
            root_max_pos_speed, root_max_pos_accel, root_max_ang_speed_deg, root_max_ang_accel_deg,
            root_method, root_cutoff_hz, root_one_euro_min_cutoff, root_one_euro_beta, root_one_euro_d_cutoff,
            root_cutoff_xy_hz, root_cutoff_z_hz,
            hands_max_pos_speed, hands_max_pos_accel, hands_max_ang_speed_deg, hands_max_ang_accel_deg,
            hands_method, hands_cutoff_hz, hands_one_euro_min_cutoff, hands_one_euro_beta, hands_one_euro_d_cutoff,
            fingers_max_pos_speed, fingers_max_pos_accel, fingers_max_ang_speed_deg, fingers_max_ang_accel_deg,
            fingers_method, fingers_cutoff_hz, fingers_one_euro_min_cutoff, fingers_one_euro_beta, fingers_one_euro_d_cutoff,
            head_max_pos_speed, head_max_pos_accel, head_max_ang_speed_deg, head_max_ang_accel_deg,
            head_method, head_cutoff_hz, head_one_euro_min_cutoff, head_one_euro_beta, head_one_euro_d_cutoff,
            legs_max_pos_speed, legs_max_pos_accel, legs_max_ang_speed_deg, legs_max_ang_accel_deg,
            legs_method, legs_cutoff_hz, legs_one_euro_min_cutoff, legs_one_euro_beta, legs_one_euro_d_cutoff,
            arms_max_pos_speed, arms_max_pos_accel, arms_max_ang_speed_deg, arms_max_ang_accel_deg,
            arms_method, arms_cutoff_hz, arms_one_euro_min_cutoff, arms_one_euro_beta, arms_one_euro_d_cutoff,
            default_max_pos_speed, default_max_pos_accel, default_max_ang_speed_deg, default_max_ang_accel_deg,
            default_method, default_cutoff_hz, default_one_euro_min_cutoff, default_one_euro_beta, default_one_euro_d_cutoff,
            foot_planting_velocity_threshold, foot_planting_min_height,
            foot_planting_contact_window, foot_planting_blend_factor,
            foot_planting_root_window, foot_planting_use_mid_foot,
        )
    
    # Collect all refinement inputs in the correct order
    all_refinement_inputs = [
        refinement_enabled, interpolate_missing_keyframes, use_foot_planting,
        root_components['max_pos_speed'], root_components['max_pos_accel'],
        root_components['max_ang_speed_deg'], root_components['max_ang_accel_deg'],
        root_components['method'], root_components['cutoff_hz'],
        root_components['one_euro_min_cutoff'], root_components['one_euro_beta'],
        root_components['one_euro_d_cutoff'],
        root_components['root_cutoff_xy_hz'], root_components['root_cutoff_z_hz'],
        hands_components['max_pos_speed'], hands_components['max_pos_accel'],
        hands_components['max_ang_speed_deg'], hands_components['max_ang_accel_deg'],
        hands_components['method'], hands_components['cutoff_hz'],
        hands_components['one_euro_min_cutoff'], hands_components['one_euro_beta'],
        hands_components['one_euro_d_cutoff'],
        fingers_components['max_pos_speed'], fingers_components['max_pos_accel'],
        fingers_components['max_ang_speed_deg'], fingers_components['max_ang_accel_deg'],
        fingers_components['method'], fingers_components['cutoff_hz'],
        fingers_components['one_euro_min_cutoff'], fingers_components['one_euro_beta'],
        fingers_components['one_euro_d_cutoff'],
        head_components['max_pos_speed'], head_components['max_pos_accel'],
        head_components['max_ang_speed_deg'], head_components['max_ang_accel_deg'],
        head_components['method'], head_components['cutoff_hz'],
        head_components['one_euro_min_cutoff'], head_components['one_euro_beta'],
        head_components['one_euro_d_cutoff'],
        legs_components['max_pos_speed'], legs_components['max_pos_accel'],
        legs_components['max_ang_speed_deg'], legs_components['max_ang_accel_deg'],
        legs_components['method'], legs_components['cutoff_hz'],
        legs_components['one_euro_min_cutoff'], legs_components['one_euro_beta'],
        legs_components['one_euro_d_cutoff'],
        arms_components['max_pos_speed'], arms_components['max_pos_accel'],
        arms_components['max_ang_speed_deg'], arms_components['max_ang_accel_deg'],
        arms_components['method'], arms_components['cutoff_hz'],
        arms_components['one_euro_min_cutoff'], arms_components['one_euro_beta'],
        arms_components['one_euro_d_cutoff'],
        default_components['max_pos_speed'], default_components['max_pos_accel'],
        default_components['max_ang_speed_deg'], default_components['max_ang_accel_deg'],
        default_components['method'], default_components['cutoff_hz'],
        default_components['one_euro_min_cutoff'], default_components['one_euro_beta'],
        default_components['one_euro_d_cutoff'],
        foot_planting_velocity_threshold, foot_planting_min_height,
        foot_planting_contact_window, foot_planting_blend_factor,
        foot_planting_root_window, foot_planting_use_mid_foot,
    ]
    
    # Return components dictionary for integration
    return {
        "all_refinement_inputs": all_refinement_inputs,
        "refinement_config_state": refinement_config_state,
        "build_refinement_config_wrapper": build_refinement_config_wrapper,
        "refinement_enabled": refinement_enabled,
        "interpolate_missing_keyframes": interpolate_missing_keyframes,
        "use_foot_planting": use_foot_planting,
        "foot_planting_velocity_threshold": foot_planting_velocity_threshold,
        "foot_planting_min_height": foot_planting_min_height,
        "foot_planting_contact_window": foot_planting_contact_window,
        "foot_planting_blend_factor": foot_planting_blend_factor,
        "foot_planting_root_window": foot_planting_root_window,
        "foot_planting_use_mid_foot": foot_planting_use_mid_foot,
        "config_file_upload": config_file_upload,
        "config_file_download": config_file_download,
        "save_config_btn": save_config_btn,
        "profile_components": profile_components,
    }


def create_refinement_section_ui(translator: Translator = None) -> gr.Blocks:
    """
    Create a standalone Gradio interface for the refinement section.
    Useful for testing or as a separate tab.
    
    Args:
        translator: Translator instance. If None, uses default language.
    """
    from fbxify.i18n import Translator as TranslatorClass, DEFAULT_LANGUAGE
    if translator is None:
        translator = TranslatorClass(DEFAULT_LANGUAGE)
    
    with gr.Blocks(title=translator.t("ui.refinement.title")) as interface:
        gr.Markdown(f"# {translator.t('ui.refinement.title')}")
        components = create_refinement_section(translator)
    
    return interface


if __name__ == "__main__":
    # For testing
    interface = create_refinement_section_ui()
    interface.launch()
