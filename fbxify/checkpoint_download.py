"""
Download SAM 3D Body checkpoints from HuggingFace when missing and HF_TOKEN is set.
Download MHR assets (lod0.fbx, lod1.fbx, etc.) from GitHub when missing.
"""
import os
import zipfile
import tempfile
import shutil
from urllib.request import urlretrieve

HF_REPO_MAP = {
    "vith": "facebook/sam-3d-body-vith",
    "dinov3": "facebook/sam-3d-body-dinov3",
}


def download_checkpoints_if_missing(model: str, checkpoints_dir: str) -> bool:
    """
    If checkpoints for the given model are missing and HF_TOKEN is set,
    attempt to download from HuggingFace into checkpoints_dir.
    Returns True if checkpoints are now available (either existed or downloaded).
    """
    checkpoints_dir = checkpoints_dir.rstrip("/")
    local_dir = os.path.join(checkpoints_dir, f"sam-3d-body-{model}")
    checkpoint_path = os.path.join(local_dir, "model.ckpt")
    mhr_path = os.path.join(local_dir, "assets", "mhr_model.pt")

    if os.path.exists(checkpoint_path) and os.path.exists(mhr_path):
        return True

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return False

    repo_id = HF_REPO_MAP.get(model)
    if not repo_id:
        print(f"checkpoint_download: unknown model {model!r}", flush=True)
        return False

    try:
        from huggingface_hub import snapshot_download

        print(f"Downloading {repo_id} from HuggingFace to {local_dir}...", flush=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=hf_token,
        )
        if os.path.exists(checkpoint_path) and os.path.exists(mhr_path):
            print(f"Successfully downloaded {model} checkpoints.", flush=True)
            return True
        print(f"Download completed but expected files missing: {checkpoint_path}, {mhr_path}", flush=True)
        return False
    except Exception as e:
        print(f"Failed to download {repo_id}: {e}", flush=True)
        return False


MHR_ASSETS_URL = "https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip"


def download_mhr_assets_if_missing(mhr_assets_dir: str) -> bool:
    """
    If MHR assets (lod0.fbx, lod1.fbx, etc.) are missing, download from GitHub.
    Returns True if assets are now available (either existed or downloaded).
    """
    mhr_assets_dir = os.path.normpath(mhr_assets_dir.rstrip("/"))
    lod1_path = os.path.join(mhr_assets_dir, "lod1.fbx")

    print(f"mhr_assets: checking lod1.fbx at {lod1_path!r} exists={os.path.exists(lod1_path)}", flush=True)
    if os.path.exists(lod1_path):
        print("mhr_assets: lod1.fbx found, skipping download.", flush=True)
        return True

    try:
        print(f"mhr_assets: lod1.fbx missing, downloading from {MHR_ASSETS_URL}...", flush=True)
        os.makedirs(mhr_assets_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            urlretrieve(MHR_ASSETS_URL, tmp_path)
            with zipfile.ZipFile(tmp_path, "r") as zf:
                for name in zf.namelist():
                    if name.startswith("assets/") and not name.endswith("/"):
                        rel = name[len("assets/"):]
                        if not rel:
                            continue
                        # Skip large corrective blendshapes (not needed for FBX export)
                        if "corrective_blendshapes_lod" in rel and rel.endswith(".npz"):
                            continue
                        dest = os.path.join(mhr_assets_dir, rel)
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        with zf.open(name) as src, open(dest, "wb") as out:
                            shutil.copyfileobj(src, out)
            if os.path.exists(lod1_path):
                print("mhr_assets: successfully downloaded.", flush=True)
                return True
            print(f"mhr_assets: download completed but lod1.fbx still missing in {mhr_assets_dir}", flush=True)
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        print(f"mhr_assets: failed to download: {e}", flush=True)
        return False
