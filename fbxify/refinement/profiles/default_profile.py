from fbxify.refinement.profiles.filter_profile import FilterProfile

# Default profile for bones that don't match other patterns (spine, neck, shoulder, etc.)
# Uses low settings since it applies to the majority of bones
DEFAULT_PROFILE = FilterProfile(
    max_pos_speed=2.0,          # Lower than default 3.0
    max_pos_accel=20.0,         # Lower than default 30.0
    max_ang_speed_deg=360.0,    # Restored to original - spike detection not the issue
    max_ang_accel_deg=3600.0,   # Restored to original - spike detection not the issue
    method="one_euro",
    one_euro_min_cutoff=25.0,   # Increased significantly to reduce smoothing aggressiveness - goal is subtle refinement (<10°)
    one_euro_beta=0.3,          # Lower than default 0.5
    one_euro_d_cutoff=0.8,      # Lower than default 1.0
)
