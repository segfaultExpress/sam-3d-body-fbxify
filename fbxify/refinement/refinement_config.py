import json
import os
from typing import Any, Dict, Optional

from fbxify.refinement.profiles.filter_profile import FilterProfile
from fbxify.refinement.profiles.root_profile import ROOT_PROFILE
from fbxify.refinement.profiles.hands_profile import HANDS_PROFILE
from fbxify.refinement.profiles.fingers_profile import FINGERS_PROFILE
from fbxify.refinement.profiles.head_profile import HEAD_PROFILE
from fbxify.refinement.profiles.legs_profile import LEGS_PROFILE
from fbxify.refinement.profiles.arms_profile import ARMS_PROFILE
from fbxify.refinement.profiles.default_profile import DEFAULT_PROFILE
from fbxify.refinement.foot_planting_config import FootPlantingConfig


def _filter_profile_from_dict(d: Dict[str, Any], is_root: bool = False) -> FilterProfile:
    """Build a FilterProfile from a dictionary (e.g. from JSON)."""
    method = d.get("method", "one_euro")
    if isinstance(method, str) and method not in ("one_euro", "ema", "butterworth"):
        method = "one_euro"
    p = FilterProfile(
        max_pos_speed=float(d.get("max_pos_speed", 3.0)),
        max_pos_accel=float(d.get("max_pos_accel", 30.0)),
        max_ang_speed_deg=float(d.get("max_ang_speed_deg", 720.0)),
        max_ang_accel_deg=float(d.get("max_ang_accel_deg", 7200.0)),
        method=method,
        cutoff_hz=float(d.get("cutoff_hz", 4.0)),
        one_euro_min_cutoff=float(d.get("one_euro_min_cutoff", 1.5)),
        one_euro_beta=float(d.get("one_euro_beta", 0.5)),
        one_euro_d_cutoff=float(d.get("one_euro_d_cutoff", 1.0)),
        root_cutoff_xy_hz=float(d.get("root_cutoff_xy_hz", 2.0)),
        root_cutoff_z_hz=float(d.get("root_cutoff_z_hz", 0.8)),
    )
    return p


class RefinementConfig:
    """
    RefinementConfig is a class that defines the configuration for the refinement manager.
    - profiles: a dictionary of bone-name matching -> FilterProfile
    - do_spike_fix: a boolean flag to enable spike fix
    - do_rotation_smoothing: a boolean flag to enable rotation smoothing
    - do_vector_smoothing: a boolean flag to enable vector smoothing
    - do_root_motion_fix: a boolean flag to enable root motion fix
    - do_foot_planting: a boolean flag to enable foot planting
    - foot_planting_config: FootPlantingConfig instance with foot planting parameters
    Note: FPS is no longer part of this config - it should be passed separately to RefinementManager
    """
    # bone-name matching -> FilterProfile
    profiles = {
        "root": ROOT_PROFILE,
        "*hand*": HANDS_PROFILE,
        "*wrist*": HANDS_PROFILE,    # Match wrist bones
        "*finger*": FINGERS_PROFILE,
        "*head*": HEAD_PROFILE,
        "*leg*": LEGS_PROFILE,
        "*arm*": ARMS_PROFILE,      # Match uparm, lowarm, etc.
        "*clavicle*": ARMS_PROFILE, # Match clavicle bones
        "*": DEFAULT_PROFILE,   # default
    }

    # feature toggles
    do_spike_fix = True
    do_rotation_smoothing = True
    do_vector_smoothing = True
    do_root_motion_fix = True
    do_interpolate_missing_keyframes = False
    do_foot_planting = True
    foot_planting_config = FootPlantingConfig()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RefinementConfig":
        """
        Build a RefinementConfig from a dictionary (e.g. from JSON).
        Expected keys: profiles (dict of pattern -> FilterProfile-like dict),
        do_spike_fix, do_rotation_smoothing, do_vector_smoothing, do_root_motion_fix,
        do_interpolate_missing_keyframes, do_foot_planting, foot_planting_config (optional dict).
        Profile keys not present in JSON keep the class defaults.
        """
        config = cls()
        # Start with default profiles; JSON overrides a subset if provided
        config.profiles = dict(config.profiles)
        profiles_data = data.get("profiles")
        if profiles_data and isinstance(profiles_data, dict):
            for pattern, pdata in profiles_data.items():
                if isinstance(pdata, dict):
                    is_root = str(pattern).strip().lower() == "root"
                    config.profiles[pattern] = _filter_profile_from_dict(pdata, is_root=is_root)
                # non-dict values (e.g. from JSON) are ignored; keep default for that pattern
        if "do_spike_fix" in data:
            config.do_spike_fix = bool(data["do_spike_fix"])
        if "do_rotation_smoothing" in data:
            config.do_rotation_smoothing = bool(data["do_rotation_smoothing"])
        if "do_vector_smoothing" in data:
            config.do_vector_smoothing = bool(data["do_vector_smoothing"])
        if "do_root_motion_fix" in data:
            config.do_root_motion_fix = bool(data["do_root_motion_fix"])
        if "do_interpolate_missing_keyframes" in data:
            config.do_interpolate_missing_keyframes = bool(data["do_interpolate_missing_keyframes"])
        if "do_foot_planting" in data:
            config.do_foot_planting = bool(data["do_foot_planting"])
        fp_data = data.get("foot_planting_config")
        if fp_data and isinstance(fp_data, dict):
            config.foot_planting_config = FootPlantingConfig.from_dict(fp_data)
        return config

    @classmethod
    def from_json_file(cls, path: str) -> "RefinementConfig":
        """Load a RefinementConfig from a JSON file."""
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Refinement config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)