from fbxify.refinement.profiles.filter_profile import FilterProfile

HEAD_PROFILE = FilterProfile( # head typically has yaw flicker
    max_pos_speed=2.0,
    max_pos_accel=20.0,
    max_ang_speed_deg=240.0,
    max_ang_accel_deg=2400.0,
    method="one_euro",
    one_euro_min_cutoff=18.0,
    one_euro_beta=0.6,
)