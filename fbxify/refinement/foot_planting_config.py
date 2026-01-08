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
    foot_contact_velocity_threshold: float = 0.5  # m/s - foot is "planted" if velocity < this
    foot_contact_min_height: float = 0.10  # meters - foot must be near ground (Y coordinate in joint space)
    contact_smoothing_window: int = 3  # frames - window for smoothing contact detection
    
    # Root motion adjustment parameters
    blend_factor: float = 0.3  # How much to blend foot-based root vs original (0-1)
    root_smoothing_window: int = 5  # frames - window for smoothing adjusted root motion
    
    # Foot position computation
    use_mid_foot: bool = False  # Use mid_foot (average of heel/toes) vs ankle
    
    # Coordinate system: height direction vector (normalized)
    # Joint space: [0, -1, 0] (negative Y is "up")
    # Camera space: [0, 1, 0] (positive Y is "up")
    height_direction: tuple = (0, -1, 0)  # Direction vector for "up" in the coordinate system being used
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "foot_contact_velocity_threshold": self.foot_contact_velocity_threshold,
            "foot_contact_min_height": self.foot_contact_min_height,
            "contact_smoothing_window": self.contact_smoothing_window,
            "blend_factor": self.blend_factor,
            "root_smoothing_window": self.root_smoothing_window,
            "use_mid_foot": self.use_mid_foot,
            "height_direction": self.height_direction,
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        height_direction = data.get("height_direction", (0, -1, 0))
        if isinstance(height_direction, list):
            height_direction = tuple(height_direction)
        return cls(
            foot_contact_velocity_threshold=data.get("foot_contact_velocity_threshold", 0.5),
            foot_contact_min_height=data.get("foot_contact_min_height", 0.1),
            contact_smoothing_window=data.get("contact_smoothing_window", 3),
            blend_factor=data.get("blend_factor", 0.3),
            root_smoothing_window=data.get("root_smoothing_window", 5),
            use_mid_foot=data.get("use_mid_foot", True),
            height_direction=height_direction,
        )

