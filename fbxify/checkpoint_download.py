"""
Download SAM 3D Body checkpoints from HuggingFace when missing and HF_TOKEN is set.
Download MHR assets (lod1.fbx and the files MHR.from_files needs) from GitHub when missing.
"""
import hashlib
import importlib.util
import os
import re
import stat
import tempfile
import zipfile
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


# Official v1.0.1 release asset. v1.0.0/assets.zip returns HTTP 404.
# Digest and size are from https://api.github.com/repos/facebookresearch/MHR/releases/tags/v1.0.1
# lod1.fbx inside that archive matches the previously working local mesh byte for byte.
MHR_ASSETS_URL = "https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip"
MHR_ASSETS_SHA256 = "e4f4f205cd87c0fa106577ba1de4fc763e4eb197c924461d2ef7e6944e9d6b94"
MHR_ASSETS_SIZE = 198_943_157
MHR_LOD1_SHA256 = "d66fbca815bcde6532f728f1f63071003c5d43ff44f56e263a0807baec1ae055"
MHR_LOD1_EXPECTED_BYTES = 7_884_560
MHR_LOD1_MIN_BYTES = 1_000_000
MHR_COMPANION_MIN_BYTES = 1000
MHR_ARCHIVE_MAX_UNCOMPRESSED = 8 * 1024 * 1024 * 1024
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


def _required_assets() -> dict:
    """Files MHR.from_files(lod=1) opens next to the installed package."""
    return {
        "lod1.fbx": MHR_LOD1_MIN_BYTES,
        "compact_v6_1.model": MHR_COMPANION_MIN_BYTES,
        "corrective_blendshapes_lod1.npz": MHR_COMPANION_MIN_BYTES,
        "corrective_activation.npz": MHR_COMPANION_MIN_BYTES,
    }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assets_valid(directory: str) -> bool:
    if not directory or not os.path.isdir(directory):
        return False
    for name, min_bytes in _required_assets().items():
        path = os.path.join(directory, name)
        try:
            if not os.path.isfile(path) or os.path.getsize(path) < min_bytes:
                return False
        except OSError:
            return False
    return True


def _is_mounted(path: str) -> bool:
    """True when path is a mount point. Never used as a reason to delete it."""
    try:
        if os.path.ismount(path):
            return True
    except OSError:
        return False
    try:
        wanted = os.path.abspath(path)
        with open("/proc/mounts", "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 2 and os.path.abspath(parts[1]) == wanted:
                    return True
    except OSError:
        pass
    return False


def _member_rel(name: str) -> str:
    """Return a safe path relative to the asset root, or raise ValueError."""
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or _DRIVE_PREFIX.match(normalized):
        raise ValueError(f"unsafe absolute path in MHR archive: {name}")
    parts = []
    for part in normalized.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError(f"unsafe path in MHR archive: {name}")
        parts.append(part)
    if parts and parts[0] == "assets":
        parts = parts[1:]
    return "/".join(parts)


def _validate_zip_members(zf: zipfile.ZipFile) -> None:
    total = 0
    found = set()
    for info in zf.infolist():
        rel = _member_rel(info.filename)
        if info.is_dir() or not rel:
            continue
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise ValueError(f"symlink member refused in MHR archive: {info.filename}")
        total += info.file_size
        if total > MHR_ARCHIVE_MAX_UNCOMPRESSED:
            raise ValueError("MHR archive uncompressed size exceeds the safety limit")
        found.add(rel)
    missing = [name for name in _required_assets() if name not in found]
    if missing:
        raise ValueError(f"MHR archive is missing required files: {', '.join(missing)}")


def verify_mhr_archive(path: str) -> None:
    """Reject a truncated or substituted assets.zip before anything is published."""
    size = os.path.getsize(path)
    if size != MHR_ASSETS_SIZE:
        raise OSError(f"MHR archive size {size} != expected {MHR_ASSETS_SIZE}")
    digest = _sha256_file(path)
    if digest != MHR_ASSETS_SHA256:
        raise OSError(f"MHR archive sha256 {digest} != expected {MHR_ASSETS_SHA256}")
    with zipfile.ZipFile(path, "r") as zf:
        _validate_zip_members(zf)


def extract_mhr_archive(zip_path: str, dest: str) -> None:
    """Extract a layout-checked archive into an empty staging directory."""
    dest_abs = os.path.abspath(dest)
    os.makedirs(dest_abs, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        _validate_zip_members(zf)
        for info in zf.infolist():
            rel = _member_rel(info.filename)
            if info.is_dir() or not rel:
                continue
            target = os.path.abspath(os.path.join(dest_abs, rel))
            if os.path.commonpath([dest_abs, target]) != dest_abs:
                raise ValueError(f"MHR archive path escaped destination: {info.filename}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info, "r") as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def _assert_known_lod1(directory: str) -> None:
    path = os.path.join(directory, "lod1.fbx")
    size = os.path.getsize(path)
    if size != MHR_LOD1_EXPECTED_BYTES:
        raise OSError(f"lod1.fbx size {size} != expected {MHR_LOD1_EXPECTED_BYTES}")
    digest = _sha256_file(path)
    if digest != MHR_LOD1_SHA256:
        raise OSError(f"lod1.fbx sha256 {digest} != expected {MHR_LOD1_SHA256}")


def _copy_file_atomic(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    temporary = dst + ".partial"
    shutil.copy2(src, temporary)
    os.replace(temporary, dst)


def _copy_missing_files(src: str, dest: str) -> None:
    """Copy files that are not already present. Does not overwrite or delete."""
    if not os.path.isdir(src):
        return
    for root, _dirs, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        out_root = dest if rel_root == "." else os.path.join(dest, rel_root)
        os.makedirs(out_root, exist_ok=True)
        for name in files:
            if name.endswith(".partial"):
                continue
            target = os.path.join(out_root, name)
            if os.path.lexists(target):
                continue
            _copy_file_atomic(os.path.join(root, name), target)


def _publish_assets(staging: str, dest: str) -> None:
    """Copy a validated staging tree onto dest. Never deletes dest."""
    if _assets_valid(dest):
        print("mhr_assets: destination already usable; leaving it untouched.", flush=True)
        return
    os.makedirs(dest, exist_ok=True)
    for root, _dirs, files in os.walk(staging):
        rel_root = os.path.relpath(root, staging)
        out_root = dest if rel_root == "." else os.path.join(dest, rel_root)
        os.makedirs(out_root, exist_ok=True)
        for name in files:
            if name.endswith(".partial"):
                continue
            _copy_file_atomic(os.path.join(root, name), os.path.join(out_root, name))
    if not _assets_valid(dest):
        raise OSError(f"MHR assets at {dest} failed validation after publish")


def _network_download_allowed() -> bool:
    if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("FBXIFY_ALLOW_ASSET_DOWNLOAD") != "1":
        return False
    return True


def download_mhr_assets_if_missing(mhr_assets_dir: str) -> bool:
    """
    If MHR assets are missing, download the pinned v1.0.1 archive into mhr_assets_dir.

    A failed or rejected download leaves whatever is already in the directory.
    A usable directory is not replaced.
    """
    mhr_assets_dir = os.path.normpath(mhr_assets_dir)
    print(
        f"mhr_assets: checking {mhr_assets_dir!r} usable={_assets_valid(mhr_assets_dir)}",
        flush=True,
    )
    if _assets_valid(mhr_assets_dir):
        print("mhr_assets: usable assets found, skipping download.", flush=True)
        return True
    if not _network_download_allowed():
        print("mhr_assets: skipping network download under pytest.", flush=True)
        return False

    for attempt in range(2):
        tmp_path = None
        staging = None
        try:
            print(
                f"mhr_assets: downloading from {MHR_ASSETS_URL} (attempt {attempt + 1}/2)...",
                flush=True,
            )
            fd, tmp_path = tempfile.mkstemp(suffix=".zip")
            os.close(fd)
            urlretrieve(MHR_ASSETS_URL, tmp_path)
            verify_mhr_archive(tmp_path)
            staging = tempfile.mkdtemp(prefix="mhr_assets_")
            extract_mhr_archive(tmp_path, staging)
            _assert_known_lod1(staging)
            _publish_assets(staging, mhr_assets_dir)
            if _assets_valid(mhr_assets_dir):
                print("mhr_assets: successfully downloaded.", flush=True)
                return True
            raise OSError("lod1.fbx still missing after a verified extract")
        except Exception as exc:
            print(f"mhr_assets: attempt {attempt + 1} failed: {exc}", flush=True)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            if staging:
                shutil.rmtree(staging, ignore_errors=True)
    print("mhr_assets: failed after 2 attempts.", flush=True)
    return False


def runtime_mhr_assets_dir(cache_assets_dir: str) -> str:
    """Directory MHR.from_files reads: site-packages/assets, or the cache when mhr is absent."""
    override = os.environ.get("FBXIFY_MHR_RUNTIME_ASSETS", "").strip()
    if override:
        return os.path.normpath(override)
    spec = None
    try:
        spec = importlib.util.find_spec("mhr")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        return os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(spec.origin)), "assets"))
    return os.path.normpath(cache_assets_dir)


def _same_path(left: str, right: str) -> bool:
    try:
        return os.path.normpath(os.path.realpath(left)) == os.path.normpath(os.path.realpath(right))
    except OSError:
        return os.path.normpath(os.path.abspath(left)) == os.path.normpath(os.path.abspath(right))


def ensure_runtime_assets_link(runtime_path: str, cache_assets_dir: str) -> None:
    """Point a stale runtime asset symlink at the cache.

    Mounted directories and a usable asset tree are left in place. A stale
    symlink is replaced without deleting its old target.
    """
    runtime_path = os.path.normpath(runtime_path)
    cache_assets_dir = os.path.normpath(cache_assets_dir)
    os.makedirs(cache_assets_dir, exist_ok=True)
    if os.path.abspath(runtime_path) == os.path.abspath(cache_assets_dir):
        return
    if _is_mounted(runtime_path):
        print(f"mhr_assets: {runtime_path} is mounted; leaving it unchanged.", flush=True)
        return
    if os.path.islink(runtime_path):
        if _assets_valid(runtime_path):
            print(f"mhr_assets: runtime symlink {runtime_path} already has usable assets.", flush=True)
            return
        print(f"mhr_assets: replacing stale symlink {runtime_path} -> {cache_assets_dir}", flush=True)
        os.unlink(runtime_path)
        os.symlink(cache_assets_dir, runtime_path, target_is_directory=True)
        return
    if os.path.isdir(runtime_path):
        if _assets_valid(runtime_path):
            print(f"mhr_assets: runtime directory {runtime_path} already has usable assets.", flush=True)
            return
        if _same_path(runtime_path, cache_assets_dir):
            return
        print(f"mhr_assets: linking runtime directory {runtime_path} -> {cache_assets_dir}", flush=True)
        _copy_missing_files(runtime_path, cache_assets_dir)
        shutil.rmtree(runtime_path)
        os.symlink(cache_assets_dir, runtime_path, target_is_directory=True)
        return
    parent = os.path.dirname(runtime_path)
    if not parent or not os.path.isdir(parent) or _is_mounted(runtime_path):
        return
    os.symlink(cache_assets_dir, runtime_path, target_is_directory=True)


def ensure_mhr_assets(cache_assets_dir: str) -> bool:
    """Make the runtime mesh path usable, downloading into the cache when needed."""
    cache_assets_dir = os.path.normpath(cache_assets_dir)
    runtime = runtime_mhr_assets_dir(cache_assets_dir)
    print(f"mhr_assets: cache={cache_assets_dir!r} runtime={runtime!r}", flush=True)
    try:
        if os.path.abspath(runtime) != os.path.abspath(cache_assets_dir):
            ensure_runtime_assets_link(runtime, cache_assets_dir)
    except OSError as exc:
        print(f"mhr_assets: could not repair runtime link: {exc}", flush=True)
    if _assets_valid(runtime):
        return True
    if not download_mhr_assets_if_missing(cache_assets_dir):
        return False
    if not _same_path(runtime, cache_assets_dir):
        try:
            _publish_assets(cache_assets_dir, runtime)
        except OSError as exc:
            print(f"mhr_assets: could not publish assets to runtime path: {exc}", flush=True)
            return False
    return _assets_valid(runtime) or _assets_valid(cache_assets_dir)


def mhr_runtime_assets_ready() -> bool:
    cache_dir = os.environ.get("CACHE_DIR", "/fbxify/cache").rstrip("/") or "/fbxify/cache"
    cache_assets = os.path.join(cache_dir, "mhr_assets")
    return _assets_valid(runtime_mhr_assets_dir(cache_assets))
