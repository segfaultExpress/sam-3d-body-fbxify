from fbxify.refinement.profiles.filter_profile import FilterProfile

HANDS_PROFILE = FilterProfile( # hands typically has roll instability
    max_pos_speed=1.0,
    max_pos_accel=10.0,
    max_ang_speed_deg=360.0,
    max_ang_accel_deg=3600.0,
    method="one_euro",
    one_euro_min_cutoff=25.0,
    one_euro_beta=0.4,
)
