"""
Command-line interface for SAM 3D Body to FBX conversion.

This module provides a CLI interface that uses FbxifyManager for all processing.
"""
import os
import argparse
import sys
import shutil
from fbxify.pose_estimation_manager import PoseEstimationManager
from fbxify.fbx_data_prep_manager import FbxDataPrepManager
from fbxify.fbxify_manager import FbxifyManager

VITH_CHECKPOINT_PATH = "/workspace/checkpoints/sam-3d-body-vith"
DINOV3_CHECKPOINT_PATH = "/workspace/checkpoints/sam-3d-body-dinov3"


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
    
    # Determine checkpoint path
    if args.model == "vith":
        checkpoint_base_path = VITH_CHECKPOINT_PATH
    elif args.model == "dinov3":
        checkpoint_base_path = DINOV3_CHECKPOINT_PATH
    else:
        print(f"Error: Invalid model: {args.model}")
        sys.exit(1)
    
    checkpoint_path = os.path.join(checkpoint_base_path, "model.ckpt")
    mhr_path = os.path.join(checkpoint_base_path, "assets", "mhr_model.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    if not os.path.exists(mhr_path):
        print(f"Error: MHR model not found: {mhr_path}")
        sys.exit(1)
    
    # Get detector and FOV paths from args or environment
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", None)
    
    # Initialize managers
    estimation_manager = None
    if args.load_estimation_json is None:
        # Only initialize estimation manager if we're not loading from JSON
        print("Initializing SAM 3D Body estimator...")
        try:
            estimation_manager = PoseEstimationManager(
                checkpoint_path=checkpoint_path,
                mhr_path=mhr_path,
                detector_name=args.detector_name,
                detector_path=detector_path,
                fov_name=args.fov_name,
                fov_path=fov_path,
                precision=args.precision
            )
        except Exception as e:
            print(f"Error initializing estimator: {e}")
            sys.exit(1)
    else:
        # Still need estimation manager for faces, but can use minimal initialization
        # Actually, we need it for faces in export, so let's initialize it anyway
        print("Initializing SAM 3D Body estimator (for mesh export)...")
        try:
            estimation_manager = PoseEstimationManager(
                checkpoint_path=checkpoint_path,
                mhr_path=mhr_path,
                detector_name=args.detector_name,
                detector_path=detector_path,
                fov_name=args.fov_name,
                fov_path=fov_path,
                precision=args.precision
            )
        except Exception as e:
            print(f"Error initializing estimator: {e}")
            sys.exit(1)
    
    data_prep_manager = FbxDataPrepManager()
    
    # Create manager
    manager = FbxifyManager(estimation_manager, data_prep_manager)
    
    # Handle loading from estimation JSON
    if args.load_estimation_json:
        print(f"Loading from estimation JSON: {args.load_estimation_json}")
        try:
            process_result = manager.process_from_estimation_json(
                args.load_estimation_json,
                args.profile,
                args.use_root_motion,
                fps=30.0
            )
            
            # Export FBX files
            print("Exporting FBX files...")
            fbx_paths = manager.export_fbx_files(
                process_result.profile_name,
                process_result.joint_to_bone_mappings,
                process_result.root_motions,
                process_result.frame_paths,
                process_result.fps,
                progress_callback=lambda p, d: print(f"Progress: {p*100:.1f}% - {d}")
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
            print(f"Error processing from estimation JSON: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return
    
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
        
        # Prepare bboxes
        bbox_dict = None
        num_people = args.num_people
        if args.bbox_file:
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
        
        # Process frames
        def progress_callback(progress_value, description):
            print(f"Progress: {progress_value*100:.1f}% - {description}")
        
        process_result = manager.process_frames(
            frame_paths,
            args.profile,
            num_people,
            bbox_dict,
            args.use_root_motion,
            fps,
            progress_callback,
            save_estimation_json=args.save_estimation_json,
            missing_bbox_behavior=args.missing_bbox_behavior
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
            progress_callback
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
