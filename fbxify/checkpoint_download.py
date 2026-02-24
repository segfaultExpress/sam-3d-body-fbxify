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
MHR_LOD1_MIN_BYTES = 1000  # lod1.fbx should be at least 1KB; catch corrupted/incomplete


def download_mhr_assets_if_missing(mhr_assets_dir: str) -> bool:
    """
    If MHR assets (lod0.fbx, lod1.fbx, etc.) are missing or corrupted, download from GitHub.
    Returns True if assets are now available (either existed or downloaded).
    """
    mhr_assets_dir = os.path.normpath(mhr_assets_dir.rstrip("/"))
    lod1_path = os.path.join(mhr_assets_dir, "lod1.fbx")

    def _lod1_valid():
        return os.path.exists(lod1_path) and os.path.getsize(lod1_path) >= MHR_LOD1_MIN_BYTES

    print(f"mhr_assets: checking lod1.fbx at {lod1_path!r} exists={os.path.exists(lod1_path)}", flush=True)
    if _lod1_valid():
        print("mhr_assets: lod1.fbx found, skipping download.", flush=True)
        return True

    # Stale/corrupt cache: remove partial dir so we get a clean download
    if os.path.exists(mhr_assets_dir) and not _lod1_valid():
        print("mhr_assets: lod1.fbx missing or too small, clearing partial cache.", flush=True)
        try:
            shutil.rmtree(mhr_assets_dir)
        except OSError as e:
            print(f"mhr_assets: could not clear cache: {e}", flush=True)

    for attempt in range(2):
        try:
            print(f"mhr_assets: downloading from {MHR_ASSETS_URL} (attempt {attempt + 1}/2)...", flush=True)
            os.makedirs(mhr_assets_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                urlretrieve(MHR_ASSETS_URL, tmp_path)
                if os.path.getsize(tmp_path) < 10000:
                    raise OSError(f"Downloaded zip too small ({os.path.getsize(tmp_path)} bytes), likely failed")
                with zipfile.ZipFile(tmp_path, "r") as zf:
                    for name in zf.namelist():
                        if name.startswith("assets/") and not name.endswith("/"):
                            rel = name[len("assets/"):]
                            if not rel:
                                continue
                            dest = os.path.join(mhr_assets_dir, rel)
                            os.makedirs(os.path.dirname(dest), exist_ok=True)
                            with zf.open(name) as src, open(dest, "wb") as out:
                                shutil.copyfileobj(src, out)
                if _lod1_valid():
                    print("mhr_assets: successfully downloaded.", flush=True)
                    return True
                if attempt == 0:
                    print("mhr_assets: lod1.fbx missing after extract, retrying...", flush=True)
                    shutil.rmtree(mhr_assets_dir, ignore_errors=True)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except Exception as e:
            print(f"mhr_assets: attempt {attempt + 1} failed: {e}", flush=True)
            try:
                shutil.rmtree(mhr_assets_dir, ignore_errors=True)
            except OSError:
                pass
            if attempt == 1:
                print(f"mhr_assets: failed after 2 attempts.", flush=True)
                return False

    return False
