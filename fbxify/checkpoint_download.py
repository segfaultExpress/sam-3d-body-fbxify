"""
Download SAM 3D Body checkpoints from HuggingFace when missing and HF_TOKEN is set.
"""
import os


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
