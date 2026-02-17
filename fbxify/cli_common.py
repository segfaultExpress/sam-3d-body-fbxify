"""
Shared CLI constants and helpers for pose estimation and FBX generation CLIs.
"""
import os

_CHECKPOINTS_BASE = os.environ.get("CHECKPOINTS_DIR", "/workspace/checkpoints").rstrip("/")
VITH_CHECKPOINT_PATH = os.path.join(_CHECKPOINTS_BASE, "sam-3d-body-vith")
DINOV3_CHECKPOINT_PATH = os.path.join(_CHECKPOINTS_BASE, "sam-3d-body-dinov3")


def checkpoints_available(model: str) -> bool:
    """
    Return True if checkpoint files exist for the given model.
    Never raises; use for conditional loading (e.g. worker waiting mode).
    """
    if model == "vith":
        base = VITH_CHECKPOINT_PATH
    elif model == "dinov3":
        base = DINOV3_CHECKPOINT_PATH
    else:
        return False
    checkpoint_path = os.path.join(base, "model.ckpt")
    mhr_path = os.path.join(base, "assets", "mhr_model.pt")
    return os.path.exists(checkpoint_path) and os.path.exists(mhr_path)


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
