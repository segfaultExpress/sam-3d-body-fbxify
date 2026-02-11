"""
Command-line interface for pose estimation only.

Runs pose estimation on an image or video and writes estimation.json.
Does not run FBX generation. Use cli_fbx_generation to create FBX from the JSON.
"""
import os
import argparse
import sys
import shutil

from fbxify.cli_common import get_checkpoint_paths
from fbxify.pose_estimation_manager import PoseEstimationManager
from fbxify.fbx_data_prep_manager import FbxDataPrepManager
from fbxify.fbxify_manager import FbxifyManager
from fbxify.tracking.tracking_config import TrackingConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run pose estimation on image/video and save estimation.json (no FBX export)"
    )
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="vith",
        choices=["vith", "dinov3"],
        help="Model to use (default: vith)",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model (default: vitdet)",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model (default: moge2)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to FOV estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="Inference precision (default: fp32)",
    )
    # Input / output
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input image or video file",
    )
    parser.add_argument(
        "--output",
        "--estimation_json",
        dest="estimation_json",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to save estimation results JSON file",
    )
    # Tracking
    parser.add_argument(
        "--tracking_mode",
        type=str,
        default="count",
        choices=["bbox", "count", "inference", "inference_bbox"],
        help="Tracking mode (default: count)",
    )
    parser.add_argument(
        "--tracking_config",
        type=str,
        default=None,
        help="Path to tracking config JSON (used in inference tracking mode)",
    )
    parser.add_argument(
        "--num_people",
        type=int,
        default=1,
        help="Number of people to detect (default: 1, ignored if --bbox_file is provided)",
    )
    parser.add_argument(
        "--bbox_file",
        type=str,
        default=None,
        help="Path to bounding box file (CSV format)",
    )
    parser.add_argument(
        "--missing_bbox_behavior",
        type=str,
        default="Run Detection",
        choices=["Run Detection", "Skip Frame"],
        help="When bbox data is missing: Run Detection or Skip Frame (default: Run Detection)",
    )
    parser.add_argument(
        "--frame_batch_size",
        type=int,
        default=1,
        help="Run pose model once per N frames (default: 1)",
    )
    parser.add_argument(
        "--detection_batch_size",
        type=int,
        default=1,
        help="Images per detector batch when no bbox file (default: 1)",
    )
    # FOV method
    parser.add_argument(
        "--fov_method",
        type=str,
        default="Default",
        choices=["Default", "File", "Sample"],
        help="FOV estimation method (default: Default)",
    )
    parser.add_argument(
        "--fov_file",
        type=str,
        default=None,
        help="Path to camera intrinsics file (required if --fov_method is File)",
    )
    parser.add_argument(
        "--sample_number",
        type=int,
        default=1,
        help="Number of images to sample for FOV (default: 1, used if --fov_method is Sample)",
    )
    return parser.parse_args()


def main():
    """Main entry point for pose-estimation-only CLI."""
    args = parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    if args.fov_method == "File" and args.fov_file is None:
        print("Error: --fov_file is required when --fov_method is File")
        sys.exit(1)
    if args.fov_method == "File" and not os.path.exists(args.fov_file):
        print(f"Error: FOV file not found: {args.fov_file}")
        sys.exit(1)
    if args.bbox_file and not os.path.exists(args.bbox_file):
        print(f"Error: Bbox file not found: {args.bbox_file}")
        sys.exit(1)
    if args.tracking_config and not os.path.exists(args.tracking_config):
        print(f"Error: Tracking config file not found: {args.tracking_config}")
        sys.exit(1)

    try:
        checkpoint_path, mhr_path = get_checkpoint_paths(args.model)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", None)

    print("Initializing SAM 3D Body estimator...")
    try:
        estimation_manager = PoseEstimationManager(
            checkpoint_path=checkpoint_path,
            mhr_path=mhr_path,
            detector_name=args.detector_name,
            detector_path=detector_path,
            fov_name=args.fov_name,
            fov_path=fov_path,
            precision=args.precision,
        )
    except Exception as e:
        print(f"Error initializing estimator: {e}")
        sys.exit(1)

    data_prep_manager = FbxDataPrepManager()
    manager = FbxifyManager(estimation_manager, data_prep_manager)

    print(f"Processing: {args.input_file}")
    file_ext = os.path.splitext(args.input_file)[1].lower()
    is_video = file_ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    temp_dir = None
    try:
        fps = 30.0
        if is_video:
            print("Extracting frames from video...")
            frame_paths, temp_dir, fps = manager.prepare_video(args.input_file)
            print(f"Video FPS: {fps}")
        else:
            frame_paths = [args.input_file]

        print(f"Processing {len(frame_paths)} frame(s)...")

        tracking_mode = args.tracking_mode
        if tracking_mode != "bbox" and args.bbox_file:
            print("Info: bbox file provided, switching tracking_mode to bbox")
            tracking_mode = "bbox"

        # Resolve tracking config: from file when provided, else default for inference mode
        tracking_config = None
        if args.tracking_config:
            tracking_config = TrackingConfig.load_json(args.tracking_config)
            print(f"Loaded tracking config from: {args.tracking_config}")
        elif tracking_mode in ("inference", "inference_bbox"):
            tracking_config = TrackingConfig()

        bbox_dict = None
        num_people = args.num_people
        if tracking_mode == "bbox":
            if not args.bbox_file:
                print("Error: --bbox_file is required when --tracking_mode is bbox")
                sys.exit(1)
            print("Loading bounding boxes...")
            bbox_dict = manager.prepare_bboxes(args.bbox_file)
            unique_person_ids = set()
            for bboxes in bbox_dict.values():
                for bbox in bboxes:
                    if len(bbox) > 0:
                        unique_person_ids.add(bbox[0])
            num_people = len(unique_person_ids) if unique_person_ids else 0
            print(f"Found {num_people} unique person(s) in bbox file")
        else:
            if num_people <= 0:
                print("Error: --num_people must be greater than 0")
                sys.exit(1)
            if args.fov_method != "Default":
                print(f"Setting camera intrinsics (method: {args.fov_method})...")
                fov_file_path = args.fov_file if args.fov_method == "File" else None
                manager.set_camera_intrinsics(
                    args.fov_method,
                    fov_file_path,
                    frame_paths,
                    args.sample_number,
                )

        manager.run_pose_estimation_only(
            frame_paths,
            num_people,
            bbox_dict,
            fps,
            args.estimation_json,
            progress_callback=None,
            missing_bbox_behavior=args.missing_bbox_behavior,
            frame_batch_size=args.frame_batch_size,
            detection_batch_size=args.detection_batch_size,
            tracking_mode=tracking_mode,
            tracking_config=tracking_config,
        )
        print(f"Estimation results saved to: {args.estimation_json}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("Cleaned up temporary files")


if __name__ == "__main__":
    main()
