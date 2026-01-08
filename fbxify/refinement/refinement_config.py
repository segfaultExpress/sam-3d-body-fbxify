from fbxify.refinement.profiles.filter_profile import FilterProfile
from fbxify.refinement.profiles.root_profile import ROOT_PROFILE
from fbxify.refinement.profiles.hands_profile import HANDS_PROFILE
from fbxify.refinement.profiles.fingers_profile import FINGERS_PROFILE
from fbxify.refinement.profiles.head_profile import HEAD_PROFILE
from fbxify.refinement.profiles.legs_profile import LEGS_PROFILE
from fbxify.refinement.profiles.arms_profile import ARMS_PROFILE
from fbxify.refinement.profiles.default_profile import DEFAULT_PROFILE
from fbxify.refinement.foot_planting_config import FootPlantingConfig

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