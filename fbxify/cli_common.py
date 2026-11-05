"""
Shared CLI constants and helpers for pose estimation and FBX generation CLIs.
"""
import os

_CHECKPOINTS_BASE = os.environ.get("CHECKPOINTS_DIR", "/fbxify/checkpoints").rstrip("/")
# Kept for callers that still import them; the catalogue (fbxify/models.json) is the source of truth.
VITH_CHECKPOINT_PATH = os.path.join(_CHECKPOINTS_BASE, "sam-3d-body-vith")
DINOV3_CHECKPOINT_PATH = os.path.join(_CHECKPOINTS_BASE, "sam-3d-body-dinov3")


def body_model_choices() -> list:
    """Body-model names from the model catalogue (for argparse choices). Falls back to vith/dinov3."""
    try:
        from fbxify.model_catalog import load_catalog
        return load_catalog().names("body_models", enabled_only=False)
    except Exception as exc:  # broken user models.json must not kill --help
        print(f"model_catalog: {exc}", flush=True)
        return ["vith", "dinov3"]


def checkpoints_available(model: str) -> bool:
    """
    Return True if checkpoint files exist for the given model (a catalogue body-model name).
    Never raises; use for conditional loading (e.g. worker waiting mode).
    """
    try:
        _entry, checkpoint_path, mhr_path = _resolve_body(model)
    except Exception as exc:
        print(f"checkpoints_available: {model!r} -> False ({exc})", flush=True)
        return False
    result = os.path.exists(checkpoint_path) and os.path.exists(mhr_path)
    print(f"checkpoints_available: {model!r} -> {result}", flush=True)
    return result


def get_checkpoint_paths(model: str) -> tuple:
    """
    Return (checkpoint_path, mhr_path) for the given catalogue body-model name.
    Raises ValueError if the name is unknown and FileNotFoundError if files are missing.
    """
    from fbxify.model_catalog import CatalogError

    try:
        _entry, checkpoint_path, mhr_path = _resolve_body(model)
    except CatalogError as exc:
        raise ValueError(str(exc)) from exc
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(mhr_path):
        raise FileNotFoundError(f"MHR model not found: {mhr_path}")
    return checkpoint_path, mhr_path


def _resolve_body(model: str):
    from fbxify.model_catalog import load_catalog

    return load_catalog().resolve_body(model)


def add_model_args(parser) -> None:
    """Shared --model / --encoder / --detector / legacy detector+fov+quantized flags."""
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("FBXIFY_MODEL") or None,
        choices=body_model_choices(),
        help="Body model name from the model catalogue (fbxify/models.json). Default: catalogue default or FBXIFY_MODEL.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=os.environ.get("FBXIFY_ENCODER") or None,
        help="Encoder name from the catalogue (fp32, embedl_int8_trt, embedl_int8_pt2, ...). Overrides --quantized_backbone.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default=os.environ.get("FBXIFY_DETECTOR") or None,
        help="Detector name from the catalogue (vitdet, or one from your models.json overlay). 'none' disables detection. Default: catalogue default.",
    )
    parser.add_argument(
        "--detector_name",
        default=None,
        type=str,
        help="Legacy: detector type (vitdet) loaded from --detector_path / SAM3D_DETECTOR_PATH instead of the catalogue.",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Legacy: folder with the detector weights for --detector_name (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument("--fov_name", default="moge2", type=str, help="FOV estimation model (default: moge2)")
    parser.add_argument("--fov_path", default="", type=str, help="Path to FOV estimation model folder (or set SAM3D_FOV_PATH)")
    parser.add_argument(
        "--quantized_backbone",
        type=str,
        default=os.environ.get("FBXIFY_QUANTIZED_BACKBONE", "none"),
        choices=["none", "pt2", "trt"],
        help="Legacy alias for --encoder: none -> fp32, pt2 -> embedl_int8_pt2, trt -> embedl_int8_trt.",
    )


def build_estimation_manager(args, precision: str = "fp32"):
    """Build a PoseEstimationManager from parsed CLI args via the model catalogue.

    Raises ValueError / FileNotFoundError with a readable message when a name is unknown or
    its files are missing (run ``python -m fbxify.model_catalog fetch`` first).
    """
    from fbxify.model_catalog import CatalogError, encoder_name_for_quantized_mode, load_catalog
    from fbxify.pose_estimation_manager import PoseEstimationManager

    fov_path = getattr(args, "fov_path", "") or os.environ.get("SAM3D_FOV_PATH", None)
    fov_name = getattr(args, "fov_name", "moge2")
    encoder = getattr(args, "encoder", None) or encoder_name_for_quantized_mode(getattr(args, "quantized_backbone", "none"))
    detector = getattr(args, "detector", None)
    if isinstance(detector, str) and detector.strip().lower() in ("none", "off", ""):
        detector = ""
    legacy_detector = getattr(args, "detector_name", None)

    try:
        catalog = load_catalog()
        catalog.force_enable("body_models", getattr(args, "model", None))
        catalog.force_enable("encoders", encoder)
        if detector:
            catalog.force_enable("detectors", detector)
        body_entry, checkpoint_path, mhr_path = catalog.resolve_body(getattr(args, "model", None))
        if not os.path.exists(checkpoint_path) or not os.path.exists(mhr_path):
            raise FileNotFoundError(
                f"body model {body_entry.name!r} is not on disk ({checkpoint_path}). "
                "Run: python -m fbxify.model_catalog fetch"
            )
        if legacy_detector:
            # Old flags: explicit detector type + folder, bypassing the catalogue.
            enc_entry, quantized_mode, quantized_dir = catalog.resolve_encoder(encoder, body=body_entry.name)
            manager = PoseEstimationManager(
                checkpoint_path=checkpoint_path,
                mhr_path=mhr_path,
                detector_name=legacy_detector,
                detector_path=getattr(args, "detector_path", "") or os.environ.get("SAM3D_DETECTOR_PATH", ""),
                fov_name=fov_name,
                fov_path=fov_path,
                precision=precision,
                quantized_backbone=quantized_mode,
                quantized_dir=quantized_dir,
            )
            manager.catalog = catalog
            manager.selection.update({"body_model": body_entry.name, "encoder": enc_entry.name})
            return manager
        return PoseEstimationManager.from_catalog(
            catalog,
            body_model=body_entry.name,
            encoder=encoder,
            detector=detector,
            fov_name=fov_name,
            fov_path=fov_path,
            precision=precision,
        )
    except CatalogError as exc:
        raise ValueError(str(exc)) from exc


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
