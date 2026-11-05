"""Benchmark one SAM 3D Body configuration: stock fp32 vs embedl INT8 encoder, normal vs Fast.

Protocol (one process = one configuration, so GPU memory numbers are clean):
  1. load the model (optionally swapping in the INT8 encoder)
  2. pass A: run every frame of --pass_a (default input/20f.mp4, 20 frames, bboxes from file)
  3. flush: gc + torch.cuda.empty_cache + reset peak-memory stats, drop cached intrinsics
  4. pass B: run --frames evenly spaced frames of --pass_b (default input/ai.mp4, detector)
  5. write results.json (timings, memory) + poses_{a,b}.npz (vertices, keypoints, cam) + frames

Run the four configurations and then fbxify.cli_compare_quantized to get the quality report:
  python -m fbxify.cli_bench_quantized --backbone fp32
  python -m fbxify.cli_bench_quantized --backbone fp32 --fast_inference
  python -m fbxify.cli_bench_quantized --backbone trt
  python -m fbxify.cli_bench_quantized --backbone trt  --fast_inference
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import sys
import time
from typing import Dict, List, Optional

import numpy as np

POSE_KEYS = (
    "pred_vertices",
    "pred_keypoints_3d",
    "pred_keypoints_2d",
    "pred_cam_t",
    "focal_length",
    "bbox",
    "pred_joint_coords",
    "pred_global_rots",
    "global_rot",
    "body_pose_params",
    "scale_params",
    "shape_params",
)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark SAM 3D Body: fp32 vs embedl INT8, normal vs Fast")
    p.add_argument("--backbone", default="fp32", choices=["fp32", "pt2", "trt"],
                   help="fp32 = stock DINOv3; pt2/trt = embedl INT8 encoder (torch.export / TensorRT)")
    p.add_argument("--fast_inference", action="store_true", help="Fast SAM 3D Body: skip keypoint-prompt pass")
    p.add_argument("--model", default="dinov3", choices=["vith", "dinov3"])
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--pass_a", default="input/20f.mp4")
    p.add_argument("--pass_a_bboxes", default="input/20f_bboxes.txt",
                   help="MOT bbox file for pass A ('' to use the detector)")
    p.add_argument("--pass_b", default="input/ai.mp4")
    p.add_argument("--pass_b_bboxes", default="")
    p.add_argument("--frames", type=int, default=20, help="Frames per pass (evenly spaced)")
    p.add_argument("--person_id", type=int, default=0, help="Track id to keep when a bbox file is given")
    p.add_argument("--output_dir", default="workspace/quant_bench")
    p.add_argument("--config_name", default="", help="Defaults to <backbone>-<normal|fast>")
    p.add_argument("--quantized_dir", default="", help="Folder with embedl files (default CHECKPOINTS_DIR/embedl-sam-3d-body)")
    p.add_argument("--detector_name", default="vitdet")
    p.add_argument("--detector_path", default="")
    p.add_argument("--fov_name", default="moge2")
    p.add_argument("--fov_path", default="")
    p.add_argument("--trt_rebuild", action="store_true", help="Force rebuilding the TensorRT engine")
    return p.parse_args()


def _mb(x: int) -> float:
    return round(x / (1024 ** 2), 1)


def _gpu_mem() -> Dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": _mb(torch.cuda.memory_allocated()),
        "reserved_mb": _mb(torch.cuda.memory_reserved()),
        "max_allocated_mb": _mb(torch.cuda.max_memory_allocated()),
        "max_reserved_mb": _mb(torch.cuda.max_memory_reserved()),
    }


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _instrument_backbone(model, sink: List[float]):
    """Time every backbone call with CUDA events; append ms to sink."""
    import torch

    backbone = model.backbone
    orig = backbone.forward

    def timed(x, extra_embed=None):
        if not torch.cuda.is_available():
            t0 = time.perf_counter()
            y = orig(x, extra_embed=extra_embed)
            sink.append((time.perf_counter() - t0) * 1000.0)
            return y
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y = orig(x, extra_embed=extra_embed)
        end.record()
        end.synchronize()
        sink.append(start.elapsed_time(end))
        return y

    backbone.forward = timed


def _pick_frames(frame_paths: List[str], n: int) -> List[int]:
    if n <= 0 or n >= len(frame_paths):
        return list(range(len(frame_paths)))
    return [int(round(i)) for i in np.linspace(0, len(frame_paths) - 1, n)]


def _bboxes_for_frame(bbox_dict, frame_index_1based: int, person_id: int):
    if bbox_dict is None:
        return None
    rows = bbox_dict.get(frame_index_1based)
    if not rows:
        return None
    rows = [r for r in rows if int(float(r[0])) == person_id] or rows[:1]
    return rows


def _stats(ms: List[float]) -> Dict[str, float]:
    if not ms:
        return {}
    arr = np.asarray(ms, dtype=np.float64)
    warm = arr[1:] if len(arr) > 1 else arr
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "min_ms": round(float(arr.min()), 2),
        "max_ms": round(float(arr.max()), 2),
        "first_ms": round(float(arr[0]), 2),
        "mean_excl_first_ms": round(float(warm.mean()), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "n": int(len(arr)),
    }


def _run_pass(name, estimation_manager, fbx_manager, video, bbox_file, n_frames, person_id,
              out_dir, encoder_sink: List[float]):
    import torch

    frame_paths, temp_dir, fps = fbx_manager.prepare_video(video)
    bbox_dict = fbx_manager.prepare_bboxes(bbox_file) if bbox_file else None
    indices = _pick_frames(frame_paths, n_frames)
    frames_dir = os.path.join(out_dir, f"frames_{name}")
    os.makedirs(frames_dir, exist_ok=True)

    # Deterministic intrinsics: FOV from the first selected frame, reused for the pass
    # (mirrors the app's "cache intrinsics" path without random sampling).
    estimation_manager.cached_cam_int = None
    first_img = frame_paths[indices[0]]
    if estimation_manager.estimator.fov_estimator is not None:
        import cv2

        img = cv2.cvtColor(cv2.imread(first_img), cv2.COLOR_BGR2RGB)
        cam_int = estimation_manager.estimator.fov_estimator.get_cam_intrinsics(img)
        cam_int = torch.as_tensor(np.asarray(cam_int) if not torch.is_tensor(cam_int) else cam_int)
        estimation_manager.cache_cam_int_from_file(cam_int.detach().cpu().float().reshape(1, 3, 3))

    est = estimation_manager.estimator
    frame_ms: List[float] = []
    encoder_ms_per_frame: List[float] = []
    poses: Dict[str, list] = {k: [] for k in POSE_KEYS}
    kept: List[str] = []
    misses: List[int] = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mem_before = _gpu_mem()

    for j, idx in enumerate(indices):
        path = frame_paths[idx]
        rows = _bboxes_for_frame(bbox_dict, idx + 1, person_id)
        bboxes_numpy = estimation_manager._convert_bboxes_to_numpy(rows) if rows else None

        est.model.head_pose.lod = 1
        est.model.head_pose_hand.lod = 1
        n_enc_before = len(encoder_sink)
        _sync()
        t0 = time.perf_counter()
        outputs = est.process_one_image(path, bboxes=bboxes_numpy, cam_int=estimation_manager.cached_cam_int)
        _sync()
        dt = (time.perf_counter() - t0) * 1000.0
        frame_ms.append(dt)
        encoder_ms_per_frame.append(float(sum(encoder_sink[n_enc_before:])))

        person = None
        if outputs:
            person = outputs[0] if rows else estimation_manager._pick_largest_person(outputs, 0)
        if person is None:
            misses.append(idx)
            print(f"[{name}] frame {idx:04d}: no person  ({dt:7.1f} ms)", flush=True)
            continue
        for k in POSE_KEYS:
            poses[k].append(np.asarray(person[k]))
        dst = os.path.join(frames_dir, f"{j:03d}_src{idx:04d}.jpg")
        shutil.copyfile(path, dst)
        kept.append(os.path.relpath(dst, out_dir))
        print(f"[{name}] frame {idx:04d}: {dt:7.1f} ms  (encoder {encoder_ms_per_frame[-1]:6.1f} ms)", flush=True)

    mem_peak = _gpu_mem()
    np.savez_compressed(
        os.path.join(out_dir, f"poses_{name}.npz"),
        frame_files=np.array(kept),
        source_indices=np.array([i for i in indices if i not in misses]),
        **{k: np.stack(v) if v else np.zeros((0,)) for k, v in poses.items()},
    )
    shutil.rmtree(temp_dir, ignore_errors=True)
    return {
        "video": video,
        "fps": fps,
        "bbox_file": bbox_file or None,
        "total_frames_in_video": len(frame_paths),
        "selected_indices": indices,
        "missed_indices": misses,
        "frame_ms": [round(x, 2) for x in frame_ms],
        "encoder_ms": [round(x, 2) for x in encoder_ms_per_frame],
        "frame": _stats(frame_ms),
        "encoder": _stats(encoder_ms_per_frame),
        "mem_before": mem_before,
        "mem_peak": mem_peak,
    }


def main():
    args = parse_args()
    import torch

    from fbxify.cli_common import get_checkpoint_paths
    from fbxify.fbx_data_prep_manager import FbxDataPrepManager
    from fbxify.fbxify_manager import FbxifyManager
    from fbxify.pose_estimation_manager import PoseEstimationManager

    config_name = args.config_name or f"{args.backbone}-{'fast' if args.fast_inference else 'normal'}"
    out_dir = os.path.join(args.output_dir, config_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== {config_name} -> {out_dir}", flush=True)

    checkpoint_path, mhr_path = get_checkpoint_paths(args.model)
    quantized = None if args.backbone == "fp32" else args.backbone

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    estimation_manager = PoseEstimationManager(
        checkpoint_path=checkpoint_path,
        mhr_path=mhr_path,
        detector_name=args.detector_name,
        detector_path=args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", ""),
        fov_name=args.fov_name,
        fov_path=args.fov_path or os.environ.get("SAM3D_FOV_PATH", None),
        precision=args.precision,
        quantized_backbone=quantized,
        quantized_dir=args.quantized_dir or None,
    )
    load_s = time.perf_counter() - t0
    estimation_manager.set_inference_options(precision=args.precision, skip_keypoint_prompt=args.fast_inference)
    fbx_manager = FbxifyManager(estimation_manager, FbxDataPrepManager())
    mem_after_load = _gpu_mem()
    print(f"Loaded in {load_s:.1f}s; GPU {mem_after_load}", flush=True)

    encoder_sink: List[float] = []
    _instrument_backbone(estimation_manager.estimator.model, encoder_sink)
    faces_path = os.path.join(args.output_dir, "faces.npy")
    if not os.path.exists(faces_path):
        np.save(faces_path, np.asarray(estimation_manager.estimator.faces, dtype=np.int64))

    backbone = estimation_manager.estimator.model.backbone
    env = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
        "backbone_class": type(backbone).__name__,
        "backbone_source": getattr(backbone, "source_path", "checkpoint"),
    }
    if quantized == "trt":
        import tensorrt as trt

        env["tensorrt"] = trt.__version__
        env["engine"] = getattr(backbone._runner, "engine_path", None)

    results = {
        "config": config_name,
        "backbone": args.backbone,
        "fast_inference": bool(args.fast_inference),
        "model": args.model,
        "precision": args.precision,
        "load_seconds": round(load_s, 2),
        "mem_after_load": mem_after_load,
        "env": env,
        "passes": {},
    }

    results["passes"]["a"] = _run_pass("a", estimation_manager, fbx_manager, args.pass_a, args.pass_a_bboxes,
                                       args.frames, args.person_id, out_dir, encoder_sink)

    # ---- flush between passes ----
    flush = {"before": _gpu_mem()}
    estimation_manager.cached_cam_int = None
    gc.collect()
    _sync()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    flush["after"] = _gpu_mem()
    results["flush"] = flush
    print(f"Flushed cache: {flush}", flush=True)

    results["passes"]["b"] = _run_pass("b", estimation_manager, fbx_manager, args.pass_b, args.pass_b_bboxes,
                                       args.frames, args.person_id, out_dir, encoder_sink)

    all_frames = results["passes"]["a"]["frame_ms"] + results["passes"]["b"]["frame_ms"]
    all_enc = results["passes"]["a"]["encoder_ms"] + results["passes"]["b"]["encoder_ms"]
    results["overall"] = {"frame": _stats(all_frames), "encoder": _stats(all_enc)}

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"{config_name}: load {load_s:.1f}s | "
          f"pass A {results['passes']['a']['frame'].get('mean_ms')} ms/frame | "
          f"pass B {results['passes']['b']['frame'].get('mean_ms')} ms/frame | "
          f"encoder {results['overall']['encoder'].get('mean_ms')} ms/frame | "
          f"peak {results['passes']['b']['mem_peak'].get('max_allocated_mb')} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
