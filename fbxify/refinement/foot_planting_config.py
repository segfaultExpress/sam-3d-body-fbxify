"""
Foot Planting Configuration

Configuration parameters for foot planting root motion adjustment.
"""
from dataclasses import dataclass


@dataclass
class FootPlantingConfig:
    """
    Configuration for foot planting root motion adjustment.
    
    Foot planting adjusts root motion based on foot contact to reduce jitter
    and create more natural walking/running motion.
    """
    # Contact detection parameters
    foot_contact_velocity_threshold: float = 0.05  # m/s - foot is "planted" if velocity < this
    foot_contact_min_height: float = 0.02  # meters - foot must be near ground (Y coordinate)
    contact_smoothing_window: int = 3  # frames - window for smoothing contact detection
    
    # Root motion adjustment parameters
    blend_factor: float = 0.3  # How much to blend foot-based root vs original (0-1)
    root_smoothing_window: int = 5  # frames - window for smoothing adjusted root motion
    
    # Foot position computation
    use_mid_foot: bool = True  # Use mid_foot (average of heel/toes) vs ankle
    apply_before_root_smoothing: bool = False  # Apply foot planting before or after root smoothing
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "foot_contact_velocity_threshold": self.foot_contact_velocity_threshold,
            "foot_contact_min_height": self.foot_contact_min_height,
            "contact_smoothing_window": self.contact_smoothing_window,
            "blend_factor": self.blend_factor,
            "root_smoothing_window": self.root_smoothing_window,
            "use_mid_foot": self.use_mid_foot,
            "apply_before_root_smoothing": self.apply_before_root_smoothing,
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            foot_contact_velocity_threshold=data.get("foot_contact_velocity_threshold", 0.05),
            foot_contact_min_height=data.get("foot_contact_min_height", 0.02),
            contact_smoothing_window=data.get("contact_smoothing_window", 3),
            blend_factor=data.get("blend_factor", 0.3),
            root_smoothing_window=data.get("root_smoothing_window", 5),
            use_mid_foot=data.get("use_mid_foot", True),
            apply_before_root_smoothing=data.get("apply_before_root_smoothing", False),
        )

