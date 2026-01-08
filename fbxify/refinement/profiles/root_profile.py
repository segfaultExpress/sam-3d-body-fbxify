from fbxify.refinement.profiles.filter_profile import FilterProfile

ROOT_PROFILE = FilterProfile(
    max_pos_speed=1.0,
    max_pos_accel=8.0,
    max_ang_speed_deg=180.0,
    max_ang_accel_deg=1800.0,
    method="ema",
    cutoff_hz=1.5,
    root_cutoff_xy_hz=1.5,
    root_cutoff_z_hz=0.2,
)
