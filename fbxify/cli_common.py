"""
Shared CLI constants and helpers for pose estimation and FBX generation CLIs.
"""
import os

VITH_CHECKPOINT_PATH = "/workspace/checkpoints/sam-3d-body-vith"
DINOV3_CHECKPOINT_PATH = "/workspace/checkpoints/sam-3d-body-dinov3"


def get_checkpoint_paths(model: str) -> tuple:
    """
    Return (checkpoint_path, mhr_path) for the given model.
    Raises ValueError if model is invalid or paths do not exist.
    """
    if model == "vith":
        base = VITH_CHECKPOINT_PATH
    elif model == "dinov3":
        base = DINOV3_CHECKPOINT_PATH
    else:
        raise ValueError(f"Invalid model: {model}")
    checkpoint_path = os.path.join(base, "model.ckpt")
    mhr_path = os.path.join(base, "assets", "mhr_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(mhr_path):
        raise FileNotFoundError(f"MHR model not found: {mhr_path}")
    return checkpoint_path, mhr_path
