"""
Shared CLI constants and helpers for pose estimation and FBX generation CLIs.
"""
import os

_CHECKPOINTS_BASE = os.environ.get("CHECKPOINTS_DIR", "/fbxify/checkpoints").rstrip("/")
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
        print(f"checkpoints_available: invalid model {model!r} -> False", flush=True)
        return False
    checkpoint_path = os.path.join(base, "model.ckpt")
    mhr_path = os.path.join(base, "assets", "mhr_model.pt")
    ckpt_exists = os.path.exists(checkpoint_path)
    mhr_exists = os.path.exists(mhr_path)
    result = ckpt_exists and mhr_exists
    print(f"checkpoints_available: {model!r} -> {result}", flush=True)
    return result

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


def resolve_lod_fbx_path(profile_name: str, lod_int: int) -> str | None:
    """
    Resolve the path to lod{N}.fbx for the mhr profile.
    Checks: repo fbxify/mapping/mhr/, then CACHE_DIR/mhr_assets/.
    Returns None if not found.
    """
    if profile_name != "mhr" or lod_int < 0:
        return None
    from fbxify.metadata import PROFILES
    profile = PROFILES.get(profile_name)
    if not profile:
        return None
    lod_key = f"lod{lod_int}_path"
    lod_rel = profile.get(lod_key)
    if not lod_rel:
        return None
    # 1) Repo path: fbxify/mapping/mhr/lod1.fbx
    _fbxify_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_fbxify_dir)
    repo_path = os.path.join(_repo_root, "fbxify", lod_rel)
    if os.path.exists(repo_path) and os.path.getsize(repo_path) > 0:
        return repo_path
    # 2) Cache path: CACHE_DIR/mhr_assets/lod1.fbx or .../mapping/mhr/lod1.fbx
    cache_base = os.environ.get("CACHE_DIR", "/fbxify/cache").rstrip("/")
    mhr_assets = os.path.join(cache_base, "mhr_assets")
    for p in (
        os.path.join(mhr_assets, f"lod{lod_int}.fbx"),
        os.path.join(mhr_assets, "mapping", "mhr", f"lod{lod_int}.fbx"),
    ):
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None
