"""
FastAPI worker for pose estimation and FBX generation.

Loads SAM 3D Body, detector, FOV once at startup.
Exposes /jobs/pose, /jobs/fbx, /jobs/detection, /jobs/rerun_tracking.
"""
from __future__ import annotations

import errno
import os
import json
import traceback
import uuid
import shutil
import tempfile
import threading
from typing import Any, Dict, Optional
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTasks


def _verify_auth(request: Request) -> None:
    """Require FBXIFY_SHARED_SECRET when set. Accepts Authorization: Bearer or X-API-Key."""
    secret = os.environ.get("FBXIFY_SHARED_SECRET", "").strip()
    if not secret:
        return
    auth = request.headers.get("Authorization")
    api_key = request.headers.get("X-API-Key")
    token = None
    if auth and auth.startswith("Bearer "):
        token = auth[7:]
    elif api_key:
        token = api_key
    if token != secret:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# Lazy init of heavy modules (done at startup)
_manager: Optional[Any] = None
_tracking_manager: Optional[Any] = None

# Job storage: job_id -> {status, output_files, output_dir, error, progress, message}
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# Mount storage: mount_id -> {mount_id, filename, path, size_bytes, created_at}
_mounts: Dict[str, Dict[str, Any]] = {}
_mounts_lock = threading.Lock()

_MOUNT_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB streaming write chunks


def _get_mounts_dir() -> str:
    """Return the persistent mounts directory, creating it if needed."""
    d = os.environ.get("FBXIFY_MOUNTS_DIR", "").strip()
    if not d:
        d = "/fbxify/mounts" if os.path.isdir("/fbxify") else os.path.join(os.getcwd(), "fbxify_mounts")
    os.makedirs(d, exist_ok=True)
    return d


def _init_mounts() -> None:
    """Scan FBXIFY_MOUNTS_DIR and rebuild the in-memory registry from disk."""
    mounts_dir = _get_mounts_dir()
    with _mounts_lock:
        _mounts.clear()
        if not os.path.isdir(mounts_dir):
            return
        for mount_id in os.listdir(mounts_dir):
            mount_path = os.path.join(mounts_dir, mount_id)
            if not os.path.isdir(mount_path):
                continue
            files = os.listdir(mount_path)
            if not files:
                continue
            filename = files[0]
            file_path = os.path.join(mount_path, filename)
            try:
                stat = os.stat(file_path)
                _mounts[mount_id] = {
                    "mount_id": mount_id,
                    "filename": filename,
                    "path": file_path,
                    "size_bytes": stat.st_size,
                    "created_at": stat.st_ctime,
                }
            except OSError:
                continue
    count = len(_mounts)
    if count:
        print(f"[mounts] Restored {count} mount(s) from {mounts_dir}")


def _lookup_mount(mount_id: str) -> Dict[str, Any]:
    """Look up a mount by ID; raises HTTPException 404 if not found."""
    with _mounts_lock:
        entry = _mounts.get(mount_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Mount '{mount_id}' not found")
    if not os.path.isfile(entry["path"]):
        with _mounts_lock:
            _mounts.pop(mount_id, None)
        raise HTTPException(status_code=404, detail=f"Mount '{mount_id}' file missing from disk")
    return entry


async def _resolve_mount_or_upload(
    upload: Optional[UploadFile],
    mount_id: Optional[str],
    dest_path: str,
    param_name: str,
) -> str:
    """Resolve a file from either an upload or a mount reference.

    Returns the path to use for processing.  When a mount is used the file is
    symlinked (or copied on Windows) into *dest_path* so job runners see a
    local file in their output_dir.
    """
    has_upload = upload is not None and upload.filename
    has_mount = mount_id is not None and mount_id.strip()

    if has_upload and has_mount:
        raise HTTPException(
            status_code=422,
            detail=f"Provide either '{param_name}' upload or '{param_name}_mount_id', not both.",
        )
    if not has_upload and not has_mount:
        raise HTTPException(
            status_code=422,
            detail=f"Either '{param_name}' upload or '{param_name}_mount_id' is required.",
        )

    if has_mount:
        entry = _lookup_mount(mount_id.strip())
        src = entry["path"]
        try:
            os.symlink(src, dest_path)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dest_path)
        return dest_path

    # Normal upload path
    try:
        with open(dest_path, "wb") as f:
            while True:
                chunk = await upload.read(_MOUNT_CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
    except OSError as e:
        if e.errno == errno.ENOSPC:
            raise HTTPException(
                status_code=507,
                detail="Insufficient storage: disk full. Free disk space or run cleanup and retry.",
            ) from e
        raise
    return dest_path


def _get_manager():
    global _manager
    if _manager is None:
        from fbxify.pose_estimation_manager import PoseEstimationManager
        from fbxify.fbx_data_prep_manager import FbxDataPrepManager
        from fbxify.fbxify_manager import FbxifyManager
        from fbxify.cli_common import get_checkpoint_paths

        model = os.environ.get("FBXIFY_MODEL", "vith")
        checkpoint_path, mhr_path = get_checkpoint_paths(model)
        detector_path = os.environ.get("SAM3D_DETECTOR_PATH", "")
        fov_path = os.environ.get("SAM3D_FOV_PATH", "")

        estimation_manager = PoseEstimationManager(
            checkpoint_path=checkpoint_path,
            mhr_path=mhr_path,
            detector_name=os.environ.get("FBXIFY_DETECTOR", "vitdet"),
            detector_path=detector_path,
            fov_name=os.environ.get("FBXIFY_FOV", "moge2"),
            fov_path=fov_path,
            precision=os.environ.get("FBXIFY_PRECISION", "fp32"),
        )
        data_prep_manager = FbxDataPrepManager()
        _manager = FbxifyManager(estimation_manager, data_prep_manager)
    return _manager


def _get_tracking_manager():
    global _tracking_manager
    if _tracking_manager is None:
        from fbxify.tracking.tracking_manager import TrackingManager
        _tracking_manager = TrackingManager()
    return _tracking_manager


def _run_pose_job(job_id: str, input_path: str, bbox_path: Optional[str], fov_path: Optional[str], params: Dict[str, Any]):
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["message"] = "Initializing..."

        manager = _get_manager()
        tracking_manager = _get_tracking_manager()

        file_ext = os.path.splitext(input_path)[1].lower()
        is_video = file_ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        fps = 30.0
        temp_dir = None

        if is_video:
            frame_paths, temp_dir, fps = manager.prepare_video(input_path)
        else:
            frame_paths = [input_path]

        fov_method = params.get("fov_method", "Default")
        if fov_method == "File" and fov_path:
            manager.set_camera_intrinsics("File", fov_path, frame_paths, 1)
        elif fov_method == "Sample" and frame_paths:
            manager.set_camera_intrinsics("Sample", None, frame_paths, int(params.get("sample_number", 1)))

        precision = params.get("precision", "fp32")
        manager.estimation_manager.set_inference_options(precision=precision)

        bbox_dict = None
        if bbox_path:
            bbox_dict = manager.prepare_bboxes(bbox_path)
            num_people = len(set(b[0] for bboxes in bbox_dict.values() for b in bboxes if len(b) > 0))
        else:
            num_people = int(params.get("num_people", 1))

        tracking_mode = params.get("tracking_mode", "count")
        tracking_config = None
        if params.get("tracking_config"):
            try:
                tc_data = json.loads(params["tracking_config"]) if isinstance(params["tracking_config"], str) else params["tracking_config"]
                from fbxify.tracking.tracking_config import TrackingConfig
                tracking_config = TrackingConfig.from_dict(tc_data)
                print(f"[pose-job {job_id[:8]}] Tracking config parsed OK: enabled={tracking_config.enabled}, "
                      f"min_similarity={tracking_config.min_similarity}, shape_weight={tracking_config.shape_weight}")
            except Exception as e:
                print(f"[pose-job {job_id[:8]}] WARNING: Failed to parse tracking_config: {e}")
                print(f"[pose-job {job_id[:8]}] Raw tracking_config value ({type(params['tracking_config']).__name__}): "
                      f"{str(params['tracking_config'])[:200]}")
        else:
            print(f"[pose-job {job_id[:8]}] No tracking_config in params (tracking_mode={tracking_mode})")

        def prog(p, desc):
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id]["progress"] = int(p * 100)
                    _jobs[job_id]["message"] = desc or ""

        estimation_results = manager.estimation_manager.estimate_all_frames(
            frame_paths,
            num_people=num_people,
            bbox_dict=bbox_dict,
            progress_callback=prog,
            missing_bbox_behavior=params.get("missing_bbox_behavior", "Run Detection"),
            lang=params.get("lang", "en"),
            frame_batch_size=int(params.get("frame_batch_size", 1)) or None,
            detection_batch_size=int(params.get("detection_batch_size", 1)) or 1,
        )

        tracking_metadata = None
        if tracking_mode in ("inference", "inference_bbox"):
            if tracking_config:
                mode_label = "Inference Tracking + BBOX File" if tracking_mode == "inference_bbox" else "Inference Tracking"
                print(f"[pose-job {job_id[:8]}] Running tracking ({mode_label}) with config")
                tracking_metadata = tracking_manager.run(estimation_results, tracking_config, mode=mode_label)
            else:
                print(f"[pose-job {job_id[:8]}] WARNING: tracking_mode={tracking_mode} but tracking_config is None — tracking SKIPPED")

        output_dir = _jobs[job_id]["output_dir"]
        output_id = job_id[:8]
        estimation_json_path = os.path.join(output_dir, f"pose_outputs_{output_id}.json")
        output_files = [estimation_json_path]

        manager.estimation_manager.save_estimation_results(
            estimation_results,
            estimation_json_path,
            source_name=os.path.basename(input_path),
            num_people=num_people,
            tracking_metadata=tracking_metadata,
            output_files=output_files,
            fps=fps,
        )

        if params.get("output_tracking_bbox"):
            from fbxify.utils import render_tracking_bbox_overlay, _is_video_path
            overlay_suffix = ".mp4" if is_video else ".png"
            overlay_path = os.path.join(output_dir, f"pose_outputs_{output_id}_overlay{overlay_suffix}")
            out = render_tracking_bbox_overlay(frame_paths, estimation_results, overlay_path, fps=fps)
            if out:
                output_files.append(out)

        if tracking_config and tracking_config.export_mot_bboxes:
            from fbxify.utils import export_mot_bboxes as write_mot_bboxes
            mot_path = os.path.join(output_dir, f"pose_outputs_{output_id}_mot.txt")
            write_mot_bboxes(estimation_results, mot_path)
            output_files.append(mot_path)

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["output_files"] = [os.path.basename(f) for f in output_files]
            _jobs[job_id]["progress"] = 100
            _jobs[job_id]["message"] = "Done"

        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        traceback.print_exc()
        tb_str = traceback.format_exc()
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(e)
                _jobs[job_id]["traceback"] = tb_str
                _jobs[job_id]["message"] = str(e)
        raise


def _run_fbx_job(job_id: str, pose_json_path: str, extras: Dict[str, str], params: Dict[str, Any]):
    """extras: paths to saved files (refinement_config, extrinsics_file, camera_scene)."""
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["message"] = "Processing..."

        manager = _get_manager()

        refinement_config = None
        if "refinement_config" in extras and extras["refinement_config"]:
            try:
                with open(extras["refinement_config"], "r") as f:
                    data = json.load(f)
                from fbxify.refinement.refinement_config import RefinementConfig
                refinement_config = RefinementConfig.from_dict(data)
            except Exception:
                pass

        extrinsics_path = extras.get("extrinsics_file")
        camera_scene_path = extras.get("camera_scene")

        def prog(p, desc):
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id]["progress"] = int(p * 100)
                    _jobs[job_id]["message"] = desc or ""

        process_result = manager.process_from_estimation_json(
            pose_json_path,
            params.get("profile_name", "mhr"),
            use_root_motion=params.get("use_root_motion", True),
            fps=None,
            refinement_config=refinement_config,
            progress_callback=prog,
            lod=int(params.get("lod", -1)),
            use_personalized_body=params.get("use_personalized_body", False),
            outlier_removal_percent=float(params.get("outlier_removal_percent", 10)),
            lang=params.get("lang", "en"),
            auto_floor=params.get("auto_floor", True),
            extrinsics_file=extrinsics_path,
            extrinsics_sample_rate=int(params.get("extrinsics_sample_rate", 0)),
            extrinsics_scale=float(params.get("extrinsics_scale", 0)),
            extrinsics_invert_quaternion=params.get("extrinsics_invert_quaternion", False),
            extrinsics_invert_translation=params.get("extrinsics_invert_translation", False),
            collect_refinement_logs=params.get("graph_refinement", False),
        )

        output_dir = _jobs[job_id]["output_dir"]
        lod_int = int(params.get("lod", -1))
        lod_fbx_path = None
        if lod_int >= 0:
            from fbxify.cli_common import resolve_lod_fbx_path
            lod_fbx_path = resolve_lod_fbx_path(process_result.profile_name, lod_int)

        fbx_paths = manager.export_fbx_files(
            process_result.profile_name,
            process_result.joint_to_bone_mappings,
            process_result.root_motions,
            process_result.frame_paths,
            process_result.fps,
            prog,
            lod=lod_int,
            mesh_obj_paths=process_result.mesh_obj_paths,
            lod_fbx_path=lod_fbx_path,
            lang=params.get("lang", "en"),
            height_offset=process_result.height_offset,
            metadata_extras=process_result.metadata_extras,
            create_camera=params.get("create_camera", False),
            camera_scene_path=camera_scene_path,
            camera_zoom=float(params.get("camera_zoom", 0)),
            extrinsics_file=extrinsics_path,
        )

        output_files = list(fbx_paths or [])
        if params.get("export_personalized_body_obj") and process_result.mesh_obj_paths:
            for p in process_result.mesh_obj_paths.values():
                if p and os.path.exists(p):
                    output_files.append(p)

        if params.get("graph_refinement") and process_result.refinement_logs:
            try:
                from fbxify.refinement.refinement_graphs import convert_refinement_logs
                output_files.extend(convert_refinement_logs(process_result.refinement_logs))
            except Exception:
                pass

        # Copy into job output_dir so /jobs/{id}/files/{filename} can serve them (export writes to /tmp)
        output_dir = _jobs[job_id]["output_dir"]
        names_in_dir = []
        for p in output_files:
            if p and os.path.isfile(p):
                name = os.path.basename(p)
                dest = os.path.join(output_dir, name)
                if os.path.realpath(p) != os.path.realpath(dest):
                    shutil.copy2(p, dest)
                names_in_dir.append(name)
        output_files = names_in_dir

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["output_files"] = output_files
            _jobs[job_id]["progress"] = 100
            _jobs[job_id]["message"] = "Done"
    except Exception as e:
        traceback.print_exc()
        tb_str = traceback.format_exc()
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(e)
                _jobs[job_id]["traceback"] = tb_str
                _jobs[job_id]["message"] = str(e)
        raise


def _run_detection_job(job_id: str, input_path: str, batch_size: int):
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"

        manager = _get_manager()
        file_ext = os.path.splitext(input_path)[1].lower()
        is_video = file_ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        temp_dir = None
        if is_video:
            frame_paths, temp_dir, _ = manager.prepare_video(input_path)
        else:
            frame_paths = [input_path]
        output_dir = _jobs[job_id]["output_dir"]
        mot_path = os.path.join(output_dir, f"detection_{job_id[:8]}.txt")
        manager.estimation_manager.run_detection_only(frame_paths, mot_path, detection_batch_size=batch_size)

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["mot_path"] = os.path.basename(mot_path)
            _jobs[job_id]["output_files"] = [os.path.basename(mot_path)]

        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        traceback.print_exc()
        tb_str = traceback.format_exc()
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(e)
                _jobs[job_id]["traceback"] = tb_str
        raise


def _run_rerun_tracking_job(job_id: str, estimation_path: str, params: Dict[str, Any]):
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"

        manager = _get_manager()
        tracking_manager = _get_tracking_manager()
        from fbxify.gradio_ui.tracking_section import build_tracking_config_from_gui
        from fbxify.tracking.tracking_manager import apply_frame_assignments

        estimation_results, metadata = manager.estimation_manager.load_estimation_results(estimation_path)
        step_through = params.get("step_through", False)
        debug_start_frame = int(params.get("debug_start_frame", 0))
        tc_raw = params.get("tracking_config")
        if tc_raw and tc_raw.strip():
            tc_data = json.loads(tc_raw) if isinstance(tc_raw, str) else tc_raw
            from fbxify.tracking.tracking_config import TrackingConfig
            config = TrackingConfig.from_dict(tc_data)
            print(f"[rerun-tracking {job_id[:8]}] Config from JSON: enabled={config.enabled}, "
                  f"min_similarity={config.min_similarity}, shape_weight={config.shape_weight}")
        else:
            config_params = {k: v for k, v in params.items() if k not in ("step_through", "debug_start_frame", "tracking_config")}
            config = build_tracking_config_from_gui(**config_params)
            print(f"[rerun-tracking {job_id[:8]}] Config from individual params (fallback)")
        tracking_metadata = tracking_manager.run(
            estimation_results,
            config,
            mode="Re-run Tracking (debug)",
            debug_per_frame=True,
            step_through=step_through,
            debug_start_frame=debug_start_frame,
        )
        frame_assignments = tracking_metadata.get("frame_assignments") or {}
        remapped_results = apply_frame_assignments(estimation_results, frame_assignments)
        output_dir = _jobs[job_id]["output_dir"]
        new_path = os.path.join(output_dir, f"pose_outputs_rerun_{job_id[:8]}.json")
        save_tracking = dict(tracking_metadata)
        if "tracklets" in save_tracking:
            save_tracking["tracklets"] = [{k: v for k, v in t.items() if k != "detections"} for t in save_tracking["tracklets"]]
        manager.estimation_manager.save_estimation_results(
            remapped_results,
            new_path,
            source_name=metadata.get("source", "unknown"),
            num_people=metadata.get("num_people"),
            tracking_metadata=save_tracking,
            output_files=[new_path],
            fps=metadata.get("fps"),
        )

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["output_path"] = new_path
            _jobs[job_id]["output_files"] = [os.path.basename(new_path)]
    except Exception as e:
        traceback.print_exc()
        tb_str = traceback.format_exc()
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(e)
                _jobs[job_id]["traceback"] = tb_str
        raise


app = FastAPI(title="FBXify Worker API", version="1.0.0")


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    """Return 503 when checkpoints are missing (worker in waiting mode)."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc) + ". Upload checkpoints to CHECKPOINTS_DIR, then POST /reload."},
    )


@app.on_event("startup")
async def startup():
    """Load models on startup. If checkpoints are missing, try HF download (when HF_TOKEN set), else stay in waiting mode."""
    _init_mounts()

    from fbxify.cli_common import checkpoints_available
    from fbxify.checkpoint_download import download_checkpoints_if_missing, download_mhr_assets_if_missing

    model = os.environ.get("FBXIFY_MODEL", "vith")
    checkpoints_dir = os.environ.get("CHECKPOINTS_DIR", "/fbxify/checkpoints").rstrip("/")
    cache_dir = os.environ.get("CACHE_DIR", "/fbxify/cache").rstrip("/")
    # Download to CACHE_DIR/mhr_assets so host mounts (e.g. -v host/cache:/workspace/cache) receive the files.
    # Entrypoint symlinks /opt/venv/.../assets -> CACHE_DIR/mhr_assets so the app finds them.
    mhr_assets_dir = os.path.join(cache_dir, "mhr_assets")
    print(f"mhr_assets: CACHE_DIR={cache_dir!r} -> downloading to {mhr_assets_dir!r}", flush=True)

    download_mhr_assets_if_missing(mhr_assets_dir)

    available = checkpoints_available(model)
    if not available and os.environ.get("HF_TOKEN"):
        print("Checkpoints missing; attempting download from HuggingFace (HF_TOKEN set)...", flush=True)
        download_checkpoints_if_missing(model, checkpoints_dir)
        available = checkpoints_available(model)
    if available:
        try:
            print("Loading SAM 3D Body (GPU)...")
            _get_manager()
            _get_tracking_manager()
            print("Worker ready.")
        except FileNotFoundError as e:
            print(f"Checkpoints not found: {e}")
            print("Running in waiting mode.")
            print("Upload checkpoints to CHECKPOINTS_DIR, then POST /reload to load models.")
    else:
        print("Checkpoints not found. Running in waiting mode.")
        print("Upload checkpoints to CHECKPOINTS_DIR, then POST /reload to load models.")


@app.post("/mount")
async def mount_file(
    _auth: None = Depends(_verify_auth),
    file: UploadFile = File(...),
):
    """Upload a file to persistent mount storage.

    The file is streamed to disk in chunks so even multi-GB uploads stay
    memory-efficient.  Returns a ``mount_id`` that can be passed to job
    endpoints in place of a file upload.
    """
    import time as _time

    mount_id = uuid.uuid4().hex
    mounts_dir = _get_mounts_dir()
    mount_dir = os.path.join(mounts_dir, mount_id)
    os.makedirs(mount_dir, exist_ok=True)

    filename = file.filename or "uploaded_file"
    file_path = os.path.join(mount_dir, filename)
    size = 0
    try:
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(_MOUNT_CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                size += len(chunk)
    except OSError as e:
        shutil.rmtree(mount_dir, ignore_errors=True)
        if e.errno == errno.ENOSPC:
            raise HTTPException(
                status_code=507,
                detail="Insufficient storage: disk full. Free disk space or run cleanup and retry.",
            ) from e
        raise

    entry = {
        "mount_id": mount_id,
        "filename": filename,
        "path": file_path,
        "size_bytes": size,
        "created_at": _time.time(),
    }
    with _mounts_lock:
        _mounts[mount_id] = entry

    return {"mount_id": mount_id, "filename": filename, "size_bytes": size}


@app.get("/mounts")
async def list_mounts(_auth: None = Depends(_verify_auth)):
    """List all currently mounted files."""
    with _mounts_lock:
        entries = list(_mounts.values())
    return [
        {
            "mount_id": e["mount_id"],
            "filename": e["filename"],
            "size_bytes": e["size_bytes"],
            "created_at": e["created_at"],
        }
        for e in entries
    ]


@app.delete("/mounts/{mount_id}")
async def delete_mount(mount_id: str, _auth: None = Depends(_verify_auth)):
    """Remove a single mounted file."""
    with _mounts_lock:
        entry = _mounts.pop(mount_id, None)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Mount '{mount_id}' not found")
    mount_dir = os.path.dirname(entry["path"])
    if os.path.isdir(mount_dir):
        shutil.rmtree(mount_dir, ignore_errors=True)
    return {"status": "ok", "mount_id": mount_id}


@app.post("/jobs/pose")
async def create_pose_job(
    background_tasks: BackgroundTasks,
    _auth: None = Depends(_verify_auth),
    input_file: Optional[UploadFile] = File(None),
    input_mount_id: Optional[str] = Form(None),
    bbox_file: Optional[UploadFile] = File(None),
    fov_file: Optional[UploadFile] = File(None),
    tracking_mode: str = Form("count"),
    num_people: int = Form(1),
    missing_bbox_behavior: str = Form("Run Detection"),
    fov_method: str = Form("Default"),
    sample_number: int = Form(1),
    precision: str = Form("fp32"),
    output_tracking_bbox: bool = Form(False),
    frame_batch_size: int = Form(1),
    detection_batch_size: int = Form(1),
    lang: str = Form("en"),
    tracking_config: Optional[str] = Form(None),
):
    job_id = uuid.uuid4().hex
    output_dir = tempfile.mkdtemp(prefix="fbxify_job_")
    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "output_dir": output_dir, "output_files": [], "progress": 0, "message": ""}

    if input_mount_id and input_mount_id.strip():
        mount_entry = _lookup_mount(input_mount_id.strip())
        ext = os.path.splitext(mount_entry["filename"])[1] or ".jpg"
    elif input_file and input_file.filename:
        ext = os.path.splitext(input_file.filename)[1] or ".jpg"
    else:
        ext = ".jpg"
    input_path = os.path.join(output_dir, "input" + ext)
    try:
        await _resolve_mount_or_upload(input_file, input_mount_id, input_path, "input_file")
    except HTTPException:
        with _jobs_lock:
            _jobs.pop(job_id, None)
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    bbox_path = None
    if bbox_file and bbox_file.filename:
        bbox_path = os.path.join(output_dir, "bbox.csv")
        with open(bbox_path, "wb") as f:
            f.write(await bbox_file.read())

    fov_path = None
    if fov_file and fov_file.filename:
        fov_path = os.path.join(output_dir, "fov.txt")
        with open(fov_path, "wb") as f:
            f.write(await fov_file.read())

    params = {
        "tracking_mode": tracking_mode,
        "num_people": num_people,
        "missing_bbox_behavior": missing_bbox_behavior,
        "fov_method": fov_method,
        "sample_number": sample_number,
        "precision": precision,
        "output_tracking_bbox": output_tracking_bbox,
        "frame_batch_size": frame_batch_size,
        "detection_batch_size": detection_batch_size,
        "lang": lang,
        "tracking_config": tracking_config,
    }

    def run():
        try:
            _run_pose_job(job_id, input_path, bbox_path, fov_path, params)
        except Exception:
            pass

    background_tasks.add_task(run)
    return {"job_id": job_id}


@app.post("/jobs/fbx")
async def create_fbx_job(
    background_tasks: BackgroundTasks,
    _auth: None = Depends(_verify_auth),
    pose_json_file: Optional[UploadFile] = File(None),
    pose_json_mount_id: Optional[str] = Form(None),
    profile_name: str = Form("mhr"),
    use_root_motion: bool = Form(True),
    auto_floor: bool = Form(True),
    include_mesh: bool = Form(False),
    include_extrinsics: bool = Form(False),
    extrinsics_sample_rate: int = Form(0),
    extrinsics_scale: float = Form(0.0),
    extrinsics_invert_quaternion: bool = Form(False),
    extrinsics_invert_translation: bool = Form(False),
    create_camera: bool = Form(False),
    camera_zoom: float = Form(0.0),
    use_personalized_body: bool = Form(False),
    lod: int = Form(-1),
    outlier_removal_percent: float = Form(10.0),
    export_personalized_body_obj: bool = Form(False),
    graph_refinement: bool = Form(False),
    lang: str = Form("en"),
    extrinsics_file: Optional[UploadFile] = File(None),
    camera_scene: Optional[UploadFile] = File(None),
    refinement_config: Optional[UploadFile] = File(None),
):
    job_id = uuid.uuid4().hex
    output_dir = tempfile.mkdtemp(prefix="fbxify_job_")
    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "output_dir": output_dir, "output_files": [], "progress": 0, "message": ""}

    pose_path = os.path.join(output_dir, "estimation.json")
    try:
        await _resolve_mount_or_upload(pose_json_file, pose_json_mount_id, pose_path, "pose_json_file")
    except HTTPException:
        with _jobs_lock:
            _jobs.pop(job_id, None)
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    extras: Dict[str, str] = {}
    if extrinsics_file and extrinsics_file.filename:
        extrinsics_path = os.path.join(output_dir, "extrinsics.txt")
        with open(extrinsics_path, "wb") as f:
            f.write(await extrinsics_file.read())
        extras["extrinsics_file"] = extrinsics_path
    if camera_scene and camera_scene.filename:
        camera_scene_path = os.path.join(output_dir, "camera_scene.png")
        with open(camera_scene_path, "wb") as f:
            f.write(await camera_scene.read())
        extras["camera_scene"] = camera_scene_path
    if refinement_config and refinement_config.filename:
        refinement_path = os.path.join(output_dir, "refinement_config.json")
        with open(refinement_path, "wb") as f:
            f.write(await refinement_config.read())
        extras["refinement_config"] = refinement_path

    params = {
        "profile_name": profile_name,
        "use_root_motion": use_root_motion,
        "auto_floor": auto_floor,
        "include_mesh": include_mesh,
        "include_extrinsics": include_extrinsics,
        "extrinsics_sample_rate": extrinsics_sample_rate,
        "extrinsics_scale": extrinsics_scale,
        "extrinsics_invert_quaternion": extrinsics_invert_quaternion,
        "extrinsics_invert_translation": extrinsics_invert_translation,
        "create_camera": create_camera,
        "camera_zoom": camera_zoom,
        "use_personalized_body": use_personalized_body,
        "lod": lod,
        "outlier_removal_percent": outlier_removal_percent,
        "export_personalized_body_obj": export_personalized_body_obj,
        "graph_refinement": graph_refinement,
        "lang": lang,
    }

    def run():
        try:
            _run_fbx_job(job_id, pose_path, extras, params)
        except Exception:
            pass

    background_tasks.add_task(run)
    return {"job_id": job_id}


@app.post("/jobs/detection")
async def create_detection_job(
    background_tasks: BackgroundTasks,
    _auth: None = Depends(_verify_auth),
    input_file: UploadFile = File(...),
    detection_batch_size: int = Form(1),
):
    job_id = uuid.uuid4().hex
    output_dir = tempfile.mkdtemp(prefix="fbxify_job_")
    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "output_dir": output_dir, "output_files": [], "mot_path": None}

    input_path = os.path.join(output_dir, "input" + os.path.splitext(input_file.filename or "")[1] or ".jpg")
    with open(input_path, "wb") as f:
        f.write(await input_file.read())

    def run():
        try:
            _run_detection_job(job_id, input_path, detection_batch_size)
        except Exception:
            pass

    background_tasks.add_task(run)
    return {"job_id": job_id}


@app.post("/jobs/rerun_tracking")
async def create_rerun_tracking_job(
    background_tasks: BackgroundTasks,
    _auth: None = Depends(_verify_auth),
    estimation_file: Optional[UploadFile] = File(None),
    estimation_mount_id: Optional[str] = Form(None),
    tracking_config_file: Optional[UploadFile] = File(None),
    step_through: bool = Form(False),
    debug_start_frame: int = Form(0),
    # Legacy: individual params (used when tracking_config_file not provided)
    tracking_enabled: bool = Form(True),
    max_gap_frames: int = Form(5),
    merge_max_gap_frames: int = Form(12),
    min_tracklet_length: int = Form(2),
    min_similarity: float = Form(0.4),
    shape_distance_threshold: float = Form(1.2),
    cam_distance_threshold: float = Form(0.8),
    min_cam_similarity: float = Form(0.01),
    pose_distance_threshold: float = Form(0.9),
    min_pose_similarity: float = Form(0.0),
    shape_maturity_frames: int = Form(20),
    high_shape_override_cam: bool = Form(True),
    high_shape_threshold: float = Form(0.95),
    iou_distance_threshold: float = Form(0.5),
    shape_weight: float = Form(0.5),
    cam_weight: float = Form(0.3),
    pose_weight: float = Form(0.2),
    iou_weight: float = Form(0.0),
    use_shape_params: bool = Form(True),
    use_pred_cam_t: bool = Form(True),
    use_pose_aux: bool = Form(True),
    use_bbox_iou: bool = Form(False),
    export_frame_assignments: bool = Form(True),
    export_tracklet_detections: bool = Form(True),
    export_mot_bboxes: bool = Form(False),
    bg_filter_enabled: bool = Form(False),
    bg_min_bbox_height_px: float = Form(0.0),
    bg_min_bbox_area_px2: float = Form(0.0),
    bg_depth_max_z: float = Form(0.0),
    bg_keep_nearest_z_quantile: float = Form(0.0),
    bg_size_auto_method: str = Form("none"),
    bg_size_feature: str = Form("bbox_height"),
    bg_size_percentile: float = Form(0.0),
    bg_tracklet_score_enabled: bool = Form(True),
    bg_tracklet_score_threshold: float = Form(0.0),
    bg_min_tracklet_frames_for_scoring: int = Form(10),
    bg_w_length: float = Form(0.45),
    bg_w_size: float = Form(0.25),
    bg_w_size_stability: float = Form(0.15),
    bg_w_centering: float = Form(0.15),
    bg_auto_roi_enabled: bool = Form(False),
    bg_auto_roi_window_frames: int = Form(30),
    bg_auto_roi_point: str = Form("bottom_center"),
    bg_auto_roi_mad_k: float = Form(3.0),
    bg_auto_roi_min_radius_px: float = Form(60.0),
    bg_auto_roi_smoothing_alpha: float = Form(0.2),
    bg_refine_second_pass: bool = Form(False),
):
    job_id = uuid.uuid4().hex
    output_dir = tempfile.mkdtemp(prefix="fbxify_job_")
    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "output_dir": output_dir, "output_files": [], "output_path": None}

    est_path = os.path.join(output_dir, "estimation.json")
    try:
        await _resolve_mount_or_upload(estimation_file, estimation_mount_id, est_path, "estimation_file")
    except HTTPException:
        with _jobs_lock:
            _jobs.pop(job_id, None)
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    tc_json = None
    if tracking_config_file and tracking_config_file.filename:
        tc_json = (await tracking_config_file.read()).decode("utf-8")

    params = {
        "tracking_config": tc_json,
        "tracking_enabled": tracking_enabled,
        "max_gap_frames": max_gap_frames,
        "merge_max_gap_frames": merge_max_gap_frames,
        "min_tracklet_length": min_tracklet_length,
        "min_similarity": min_similarity,
        "shape_distance_threshold": shape_distance_threshold,
        "cam_distance_threshold": cam_distance_threshold,
        "min_cam_similarity": min_cam_similarity,
        "pose_distance_threshold": pose_distance_threshold,
        "min_pose_similarity": min_pose_similarity,
        "shape_maturity_frames": shape_maturity_frames,
        "high_shape_override_cam": high_shape_override_cam,
        "high_shape_threshold": high_shape_threshold,
        "iou_distance_threshold": iou_distance_threshold,
        "shape_weight": shape_weight,
        "cam_weight": cam_weight,
        "pose_weight": pose_weight,
        "iou_weight": iou_weight,
        "use_shape_params": use_shape_params,
        "use_pred_cam_t": use_pred_cam_t,
        "use_pose_aux": use_pose_aux,
        "use_bbox_iou": use_bbox_iou,
        "export_frame_assignments": export_frame_assignments,
        "export_tracklet_detections": export_tracklet_detections,
        "export_mot_bboxes": export_mot_bboxes,
        "bg_filter_enabled": bg_filter_enabled,
        "bg_min_bbox_height_px": bg_min_bbox_height_px,
        "bg_min_bbox_area_px2": bg_min_bbox_area_px2,
        "bg_depth_max_z": bg_depth_max_z,
        "bg_keep_nearest_z_quantile": bg_keep_nearest_z_quantile,
        "bg_size_auto_method": bg_size_auto_method,
        "bg_size_feature": bg_size_feature,
        "bg_size_percentile": bg_size_percentile,
        "bg_tracklet_score_enabled": bg_tracklet_score_enabled,
        "bg_tracklet_score_threshold": bg_tracklet_score_threshold,
        "bg_min_tracklet_frames_for_scoring": bg_min_tracklet_frames_for_scoring,
        "bg_w_length": bg_w_length,
        "bg_w_size": bg_w_size,
        "bg_w_size_stability": bg_w_size_stability,
        "bg_w_centering": bg_w_centering,
        "bg_auto_roi_enabled": bg_auto_roi_enabled,
        "bg_auto_roi_window_frames": bg_auto_roi_window_frames,
        "bg_auto_roi_point": bg_auto_roi_point,
        "bg_auto_roi_mad_k": bg_auto_roi_mad_k,
        "bg_auto_roi_min_radius_px": bg_auto_roi_min_radius_px,
        "bg_auto_roi_smoothing_alpha": bg_auto_roi_smoothing_alpha,
        "bg_refine_second_pass": bg_refine_second_pass,
        "step_through": step_through,
        "debug_start_frame": debug_start_frame,
    }

    def run():
        try:
            _run_rerun_tracking_job(job_id, est_path, params)
        except Exception:
            pass

    background_tasks.add_task(run)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str, _auth: None = Depends(_verify_auth)):
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        j = _jobs[job_id]
    resp = {
        "status": j["status"],
        "progress": j.get("progress", 0),
        "message": j.get("message", ""),
        "output_files": j.get("output_files", []),
        "output_path": j.get("output_path"),
        "mot_path": j.get("mot_path"),
        "error": j.get("error"),
    }
    if os.environ.get("FBXIFY_DEBUG", ""):
        resp["traceback"] = j.get("traceback")
    return resp


@app.get("/jobs/{job_id}/files/{filename}")
async def download_job_file(job_id: str, filename: str, _auth: None = Depends(_verify_auth)):
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        output_dir = _jobs[job_id].get("output_dir")
    if not output_dir or not os.path.isdir(output_dir):
        raise HTTPException(status_code=404, detail="Job output not found")
    file_path = os.path.join(output_dir, filename)
    if not os.path.isfile(file_path) or not file_path.startswith(os.path.abspath(output_dir)):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)


@app.post("/jobs/cancel")
async def cancel_jobs(_auth: None = Depends(_verify_auth)):
    """Signal cancel - worker will check cancel flag in running jobs."""
    manager = _get_manager()
    manager.estimation_manager.cancel_current_job()
    return {"status": "ok"}


@app.post("/reload")
async def reload_models(_auth: None = Depends(_verify_auth)):
    """Reload models (e.g. after uploading checkpoints via SSH). Clears cached managers and reinitializes."""
    global _manager, _tracking_manager
    _manager = None
    _tracking_manager = None
    try:
        _get_manager()
        _get_tracking_manager()
        return {"status": "ok", "message": "Models loaded"}
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Checkpoints not found: {e}. Upload to CHECKPOINTS_DIR and retry.",
        )


@app.post("/cleanup")
async def cleanup_temp(
    _auth: None = Depends(_verify_auth),
    max_age_hours: float = 0,
    unmount_all: bool = False,
):
    """Remove temp artifacts for completed/failed jobs and orphaned temp dirs.

    - max_age_hours=0 (default): flush ALL completed/failed jobs regardless of age.
    - max_age_hours>0: only flush jobs/dirs older than that many hours.
    - unmount_all=true: also remove ALL mounted files (default false).
    Actively running jobs are never touched.
    """
    import time, glob as _glob

    now = time.time()
    age_threshold = max_age_hours * 3600
    removed_dirs: list[str] = []
    errors: list[str] = []

    active_output_dirs: set[str] = set()
    purged_job_ids: list[str] = []

    with _jobs_lock:
        for jid, jdata in list(_jobs.items()):
            if jdata["status"] in ("completed", "failed"):
                odir = jdata.get("output_dir")
                if odir and os.path.isdir(odir):
                    dir_age = now - os.path.getmtime(odir)
                    if age_threshold == 0 or dir_age >= age_threshold:
                        try:
                            shutil.rmtree(odir, ignore_errors=True)
                            removed_dirs.append(odir)
                        except Exception as exc:
                            errors.append(f"{odir}: {exc}")
                        purged_job_ids.append(jid)
                else:
                    purged_job_ids.append(jid)
            elif jdata["status"] in ("pending", "running"):
                odir = jdata.get("output_dir")
                if odir:
                    active_output_dirs.add(os.path.realpath(odir))
        for jid in purged_job_ids:
            del _jobs[jid]

    tmp_root = tempfile.gettempdir()
    for pattern in ("sam3d_frames_*", "fbxify_job_*"):
        for d in _glob.glob(os.path.join(tmp_root, pattern)):
            if not os.path.isdir(d):
                continue
            if os.path.realpath(d) in active_output_dirs:
                continue
            dir_age = now - os.path.getmtime(d)
            if age_threshold > 0 and dir_age < age_threshold:
                continue
            try:
                shutil.rmtree(d, ignore_errors=True)
                removed_dirs.append(d)
            except Exception as exc:
                errors.append(f"{d}: {exc}")

    unmounted = 0
    if unmount_all:
        with _mounts_lock:
            unmounted = len(_mounts)
            _mounts.clear()
        mounts_dir = _get_mounts_dir()
        if os.path.isdir(mounts_dir):
            for name in os.listdir(mounts_dir):
                d = os.path.join(mounts_dir, name)
                if os.path.isdir(d):
                    try:
                        shutil.rmtree(d, ignore_errors=True)
                    except Exception as exc:
                        errors.append(f"{d}: {exc}")

    return {
        "status": "ok",
        "purged_jobs": len(purged_job_ids),
        "removed_dirs": len(removed_dirs),
        "removed_paths": removed_dirs,
        "unmounted": unmounted,
        "errors": errors,
    }


@app.get("/storage")
async def storage_info(_auth: None = Depends(_verify_auth)):
    """Report disk usage for /tmp and counts of fbxify temp artifacts."""
    import glob as _glob

    tmp_root = tempfile.gettempdir()
    usage = shutil.disk_usage(tmp_root)

    frame_dirs = [d for d in _glob.glob(os.path.join(tmp_root, "sam3d_frames_*")) if os.path.isdir(d)]
    job_dirs = [d for d in _glob.glob(os.path.join(tmp_root, "fbxify_job_*")) if os.path.isdir(d)]

    def _dir_size(path: str) -> int:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        return total

    frame_bytes = sum(_dir_size(d) for d in frame_dirs)
    job_bytes = sum(_dir_size(d) for d in job_dirs)

    with _jobs_lock:
        active = sum(1 for j in _jobs.values() if j["status"] in ("pending", "running"))
        done = sum(1 for j in _jobs.values() if j["status"] in ("completed", "failed"))

    mounts_dir = _get_mounts_dir()
    with _mounts_lock:
        mounts_count = len(_mounts)
        mounts_bytes = sum(e.get("size_bytes", 0) for e in _mounts.values())

    return {
        "tmp_dir": tmp_root,
        "disk_total_gb": round(usage.total / (1 << 30), 2),
        "disk_used_gb": round(usage.used / (1 << 30), 2),
        "disk_free_gb": round(usage.free / (1 << 30), 2),
        "disk_free_pct": round(usage.free / usage.total * 100, 1),
        "sam3d_frame_dirs": len(frame_dirs),
        "sam3d_frame_dirs_gb": round(frame_bytes / (1 << 30), 2),
        "fbxify_job_dirs": len(job_dirs),
        "fbxify_job_dirs_gb": round(job_bytes / (1 << 30), 2),
        "jobs_active": active,
        "jobs_done": done,
        "mounts_dir": mounts_dir,
        "mounts_count": mounts_count,
        "mounts_total_gb": round(mounts_bytes / (1 << 30), 2),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
