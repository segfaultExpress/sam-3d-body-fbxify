"""
Command-line interface for SAM 3D Body to FBX conversion.

Unified entry point: dispatches to pose-only or FBX-only logic, or runs full pipeline.
For pose-only use: python -m fbxify.cli_pose_estimation --output <JSON> <INPUT>.
For FBX-only use: python -m fbxify.cli_fbx_generation [options] <ESTIMATION_JSON>.
"""
import os
import argparse
import sys
import shutil
from fbxify.cli_common import get_checkpoint_paths
from fbxify.pose_estimation_manager import PoseEstimationManager
from fbxify.fbx_data_prep_manager import FbxDataPrepManager
from fbxify.fbxify_manager import FbxifyManager


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert images/videos to Unity Humanoid FBX files using SAM 3D Body"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="vith",
        choices=["vith", "dinov3"],
        help="Model to use (default: vith)"
    )
    
    # Detector options
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model (default: vitdet)"
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)"
    )
    
    # FOV estimator options
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model (default: moge2)"
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to FOV estimation model folder (or set SAM3D_FOV_PATH)"
    )
    
    # Input/Output
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input image or video file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for FBX files (default: same as input file directory)"
    )
    
    # Processing options
    parser.add_argument(
        "--profile",
        type=str,
        default="mhr",
        choices=["mhr", "mixamo", "unity"],
        help="Armature profile to use (default: mhr)"
    )
    parser.add_argument(
        "--tracking_mode",
        type=str,
        default="count",
        choices=["bbox", "count", "inference", "inference_bbox"],
        help="Tracking mode: bbox (use bbox file), count (N people), inference (tracklets), inference_bbox (bbox file + tracklets) (default: count)"
    )
    parser.add_argument(
        "--tracking_config",
        type=str,
        default=None,
        help="Path to tracking config JSON (used in inference tracking mode)"
    )
    parser.add_argument(
        "--num_people",
        type=int,
        default=1,
        help="Number of people to detect (default: 1, ignored if --bbox_file is provided)"
    )
    parser.add_argument(
        "--bbox_file",
        type=str,
        default=None,
        help="Path to bounding box file (CSV format)"
    )
    parser.add_argument(
        "--missing_bbox_behavior",
        type=str,
        default="Run Detection",
        choices=["Run Detection", "Skip Frame"],
        help="What to do when bbox data is missing for a frame: Run Detection (use num_people to detect) or Skip Frame (skip pose estimation for that frame) (default: Run Detection)"
    )
    parser.add_argument(
        "--frame_batch_size",
        type=int,
        default=1,
        help="Run pose model once per N frames (1=per-frame). 4 or 8 can speed up video. Higher uses more GPU memory (default: 1)"
    )
    parser.add_argument(
        "--detection_batch_size",
        type=int,
        default=1,
        help="When no bbox file: number of images per detector batch (1=per-frame). Higher can speed up detection (default: 1)"
    )

    # FOV options
    parser.add_argument(
        "--fov_method",
        type=str,
        default="Default",
        choices=["Default", "File", "Sample"],
        help="FOV estimation method (default: Default)"
    )
    parser.add_argument(
        "--fov_file",
        type=str,
        default=None,
        help="Path to camera intrinsics file (required if --fov_method is File)"
    )
    parser.add_argument(
        "--sample_number",
        type=int,
        default=1,
        help="Number of images to sample for FOV estimation (default: 1, used if --fov_method is Sample)"
    )

    # Precision and hand crop scale
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="Inference precision: fp32 (full), bf16 (fast + safer), fp16 (fastest)"
    )
    
    # Root motion
    parser.add_argument(
        "--use_root_motion",
        action="store_true",
        default=True,
        help="Apply root motion (default: True)"
    )
    parser.add_argument(
        "--no_root_motion",
        action="store_false",
        dest="use_root_motion",
        help="Disable root motion"
    )

    # Auto-floor
    parser.add_argument(
        "--auto_floor",
        action="store_true",
        default=True,
        help="Auto-floor: offset average pred_cam_t.y to 0 (default: True)"
    )
    parser.add_argument(
        "--no_auto_floor",
        action="store_false",
        dest="auto_floor",
        help="Disable auto-floor"
    )
    
    # Estimation JSON options
    parser.add_argument(
        "--save_estimation_json",
        type=str,
        default=None,
        help="Path to save estimation results JSON file"
    )
    parser.add_argument(
        "--load_estimation_json",
        type=str,
        default=None,
        help="Path to load estimation results JSON file (skips estimation step)"
    )

    # FBX-only options (used when --load_estimation_json is set)
    parser.add_argument(
        "--refinement_config",
        type=str,
        default=None,
        help="Path to refinement config JSON (FBX-from-JSON only)"
    )
    parser.add_argument(
        "--extrinsics_file",
        type=str,
        default=None,
        help="Path to camera extrinsics file, e.g. COLMAP images.txt (FBX-from-JSON only)"
    )
    parser.add_argument(
        "--extrinsics_sample_rate",
        type=int,
        default=0,
        help="Extrinsics downsampling rate; 0 = auto (FBX-from-JSON only)"
    )
    parser.add_argument(
        "--extrinsics_scale",
        type=float,
        default=0.0,
        help="Scale for extrinsics translation (FBX-from-JSON only)"
    )
    parser.add_argument(
        "--extrinsics_invert_quaternion",
        action="store_true",
        default=False,
        help="Treat qvec as camera→world (FBX-from-JSON only)"
    )
    parser.add_argument(
        "--extrinsics_invert_translation",
        action="store_true",
        default=False,
        help="Treat tvec as camera→world (FBX-from-JSON only)"
    )

    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Validate arguments
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

    # Delegate to FBX-only CLI when loading from estimation JSON
    if args.load_estimation_json:
        from fbxify.cli_fbx_generation import run as run_fbx_generation
        fbx_args = argparse.Namespace(
            estimation_json=args.load_estimation_json,
            output_dir=args.output_dir,
            profile=args.profile,
            use_root_motion=args.use_root_motion,
            auto_floor=args.auto_floor,
            model=args.model,
            detector_name=args.detector_name,
            detector_path=args.detector_path,
            fov_name=args.fov_name,
            fov_path=args.fov_path,
            precision=args.precision,
            refinement_config=getattr(args, "refinement_config", None),
            extrinsics_file=getattr(args, "extrinsics_file", None),
            extrinsics_sample_rate=getattr(args, "extrinsics_sample_rate", 0),
            extrinsics_scale=getattr(args, "extrinsics_scale", 0.0),
            extrinsics_invert_quaternion=getattr(args, "extrinsics_invert_quaternion", False),
            extrinsics_invert_translation=getattr(args, "extrinsics_invert_translation", False),
        )
        run_fbx_generation(fbx_args)
        return

    # Full pipeline: resolve checkpoint path and initialize managers
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
    
    # Process input file
    print(f"Processing: {args.input_file}")
    
    # Determine if input is video
    file_ext = os.path.splitext(args.input_file)[1].lower()
    is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    # Prepare video or image
    temp_dir = None
    try:
        fps = 30.0  # Default FPS for images
        if is_video:
            print("Extracting frames from video...")
            frame_paths, temp_dir, fps = manager.prepare_video(args.input_file)
            print(f"Video FPS: {fps}")
        else:
            frame_paths = [args.input_file]
        
        print(f"Processing {len(frame_paths)} frame(s)...")
        
        # Prepare bboxes / tracking mode
        tracking_mode = args.tracking_mode
        if tracking_mode != "bbox" and args.bbox_file:
            print("Info: bbox file provided, switching tracking_mode to bbox")
            tracking_mode = "bbox"

        bbox_dict = None
        num_people = args.num_people
        if tracking_mode == "bbox":
            if not args.bbox_file:
                print("Error: --bbox_file is required when --tracking_mode is bbox")
                sys.exit(1)
            print("Loading bounding boxes...")
            bbox_dict = manager.prepare_bboxes(args.bbox_file)
            # Count unique person IDs
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
            
            # Set camera intrinsics
            if args.fov_method != "Default":
                print(f"Setting camera intrinsics (method: {args.fov_method})...")
                fov_file_path = args.fov_file if args.fov_method == "File" else None
                manager.set_camera_intrinsics(
                    args.fov_method,
                    fov_file_path,
                    frame_paths,
                    args.sample_number
                )
            
            # Process frames (tqdm used internally for progress)
            process_result = manager.process_frames(
                frame_paths,
                args.profile,
                num_people,
                bbox_dict,
                args.use_root_motion,
                fps,
                None,
                save_estimation_json=args.save_estimation_json,
                missing_bbox_behavior=args.missing_bbox_behavior,
                frame_batch_size=args.frame_batch_size,
                detection_batch_size=args.detection_batch_size,
            )
            
            # Print estimation JSON path if saved
            if args.save_estimation_json:
                print(f"Estimation results saved to: {args.save_estimation_json}")
            
            # Export FBX files
            print("Exporting FBX files...")
            fbx_paths = manager.export_fbx_files(
                process_result.profile_name,
                process_result.joint_to_bone_mappings,
                process_result.root_motions,
                process_result.frame_paths,
                process_result.fps,
                None,
                height_offset=getattr(process_result, "height_offset", 0.0)
            )
            
            # Move files to output directory if specified
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                moved_paths = []
                for fbx_path in fbx_paths:
                    filename = os.path.basename(fbx_path)
                    dest_path = os.path.join(args.output_dir, filename)
                    shutil.copy2(fbx_path, dest_path)
                    moved_paths.append(dest_path)
                fbx_paths = moved_paths
            
            print(f"Exported {len(fbx_paths)} FBX file(s):")
            for fbx_path in fbx_paths:
                print(f"  - {fbx_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("Cleaned up temporary files")


if __name__ == "__main__":
    main()
