"""A/B pose latency: stock 3DB vs skip_keypoint_prompt (Fast SAM 3D Body first slice).

Requires a loaded checkpoint and a GPU. Times process_one_image only (no FBX).

Example:
  python -m fbxify.cli_bench_fast_inference --model dinov3 path/to/frame.jpg --repeats 8
  python -m fbxify.cli_bench_fast_inference --model dinov3 path/to/clip.mp4 --max_frames 12 --bbox_file boxes.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark stock vs Fast (skip keypoint-prompt) SAM 3D Body inference"
    )
    p.add_argument("input_file", help="Image or video")
    p.add_argument("--model", default="vith", choices=["vith", "dinov3"])
    p.add_argument("--bbox_file", default=None)
    p.add_argument("--max_frames", type=int, default=8, help="Frames to time (video)")
    p.add_argument("--repeats", type=int, default=1, help="Repeats per image (images only)")
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--detector_name", default="vitdet")
    p.add_argument("--detector_path", default="")
    p.add_argument("--fov_name", default="moge2")
    p.add_argument("--fov_path", default="")
    return p.parse_args()


def _ms(seconds: float) -> float:
    return seconds * 1000.0


def _time_frames(manager, frame_paths, bbox_dict, skip: bool) -> list[float]:
    manager.set_inference_options(
        precision=manager.precision, skip_keypoint_prompt=skip
    )
    times = []
    for i, path in enumerate(frame_paths):
        bboxes = None
        if bbox_dict is not None and (i + 1) in bbox_dict:
            bboxes = bbox_dict[i + 1]
        t0 = time.perf_counter()
        manager._estimate_single_frame(path, num_people=1, bboxes=bboxes)
        if manager.device != "cpu":
            import torch

            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def main():
    args = parse_args()
    from fbxify.cli_common import get_checkpoint_paths
    from fbxify.pose_estimation_manager import PoseEstimationManager
    from fbxify.fbx_data_prep_manager import FbxDataPrepManager
    from fbxify.fbxify_manager import FbxifyManager

    checkpoint_path, mhr_path = get_checkpoint_paths(args.model)
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", None)

    print("Loading SAM 3D Body...")
    estimation_manager = PoseEstimationManager(
        checkpoint_path=checkpoint_path,
        mhr_path=mhr_path,
        detector_name=args.detector_name,
        detector_path=detector_path,
        fov_name=args.fov_name,
        fov_path=fov_path,
        precision=args.precision,
    )
    data_prep = FbxDataPrepManager()
    manager = FbxifyManager(estimation_manager, data_prep)

    ext = os.path.splitext(args.input_file)[1].lower()
    temp_dir = None
    if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        frame_paths, temp_dir, _fps = manager.prepare_video(args.input_file)
        frame_paths = frame_paths[: max(1, args.max_frames)]
    else:
        frame_paths = [args.input_file] * max(1, args.repeats)

    bbox_dict = None
    if args.bbox_file:
        bbox_dict = manager.prepare_bboxes(args.bbox_file)

    warmup_n = min(args.warmup, len(frame_paths))
    print(f"Warmup ({warmup_n} frames, stock)...")
    _time_frames(estimation_manager, frame_paths[:warmup_n], bbox_dict, skip=False)

    print(f"Timing stock 3DB on {len(frame_paths)} frame(s)...")
    stock = _time_frames(estimation_manager, frame_paths, bbox_dict, skip=False)
    print(f"Timing fast (skip_keypoint_prompt) on {len(frame_paths)} frame(s)...")
    fast = _time_frames(estimation_manager, frame_paths, bbox_dict, skip=True)

    def summarize(name, xs):
        arr = np.array(xs, dtype=np.float64)
        print(
            f"{name:28s}  mean={_ms(arr.mean()):8.1f} ms  "
            f"median={_ms(np.median(arr)):8.1f} ms  "
            f"min={_ms(arr.min()):8.1f}  max={_ms(arr.max()):8.1f}  "
            f"n={len(arr)}"
        )

    print()
    summarize("stock (full 3DB)", stock)
    summarize("fast (skip 2nd decoder)", fast)
    speedup = np.mean(stock) / max(np.mean(fast), 1e-9)
    print(f"speedup {speedup:.2f}x  (this is only SKIP_KEYPOINT_PROMPT, not TensorRT/YOLO)")

    if temp_dir:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
