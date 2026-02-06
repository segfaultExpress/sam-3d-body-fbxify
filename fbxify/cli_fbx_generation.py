"""
Command-line interface for FBX generation only.

Reads an existing estimation.json and exports FBX files.
Does not run pose estimation. Use cli_pose_estimation to create the estimation JSON.
"""
import os
import argparse
import sys
import shutil

from fbxify.cli_common import get_checkpoint_paths
from fbxify.pose_estimation_manager import PoseEstimationManager
from fbxify.fbx_data_prep_manager import FbxDataPrepManager
from fbxify.fbxify_manager import FbxifyManager
from fbxify.refinement.refinement_config import RefinementConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate FBX files from an existing estimation.json (no pose estimation)"
    )
    # Model (needed for mesh export / PoseEstimationManager init)
    parser.add_argument(
        "--model",
        type=str,
        default="vith",
        choices=["vith", "dinov3"],
        help="Model used for estimation (default: vith)",
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
        "estimation_json",
        type=str,
        help="Path to estimation results JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for FBX files (default: same directory as estimation JSON)",
    )
    # FBX options
    parser.add_argument(
        "--profile",
        type=str,
        default="mhr",
        choices=["mhr", "mixamo", "unity"],
        help="Armature profile (default: mhr)",
    )
    parser.add_argument(
        "--use_root_motion",
        action="store_true",
        default=True,
        help="Apply root motion (default: True)",
    )
    parser.add_argument(
        "--no_root_motion",
        action="store_false",
        dest="use_root_motion",
        help="Disable root motion",
    )
    parser.add_argument(
        "--auto_floor",
        action="store_true",
        default=True,
        help="Auto-floor: offset average pred_cam_t.y to 0 (default: True)",
    )
    parser.add_argument(
        "--no_auto_floor",
        action="store_false",
        dest="auto_floor",
        help="Disable auto-floor",
    )
    # Camera extrinsics (apply to estimation before FBX; COLMAP images.txt-style)
    parser.add_argument(
        "--extrinsics_file",
        type=str,
        default=None,
        help="Path to camera extrinsics file (COLMAP images.txt). Applied to estimation before export.",
    )
    parser.add_argument(
        "--extrinsics_sample_rate",
        type=int,
        default=0,
        help="Extrinsics downsampling rate; 0 = auto (default: 0)",
    )
    parser.add_argument(
        "--extrinsics_scale",
        type=float,
        default=0.0,
        help="Scale for extrinsics translation components (default: 0.0)",
    )
    parser.add_argument(
        "--extrinsics_invert_quaternion",
        action="store_true",
        default=False,
        help="Treat qvec as camera→world instead of world→camera",
    )
    parser.add_argument(
        "--extrinsics_invert_translation",
        action="store_true",
        default=False,
        help="Treat tvec as camera→world instead of world→camera",
    )
    # Refinement (smoothing / spike fix applied before joint mapping)
    parser.add_argument(
        "--refinement_config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to refinement config JSON file. Applied to estimation before FBX export.",
    )
    return parser.parse_args()


def run(args):
    """
    Run FBX generation from an estimation JSON (used by this CLI and by unified cli.py).
    args must have: estimation_json, output_dir, profile, use_root_motion, auto_floor,
    model, detector_name, detector_path, fov_name, fov_path, precision.
    """
    if not os.path.exists(args.estimation_json):
        print(f"Error: Estimation JSON not found: {args.estimation_json}")
        sys.exit(1)

    try:
        checkpoint_path, mhr_path = get_checkpoint_paths(args.model)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", None)

    print("Initializing SAM 3D Body estimator (for mesh export)...")
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

    refinement_config = None
    refinement_config_path = getattr(args, "refinement_config", None)
    if refinement_config_path:
        if not os.path.exists(refinement_config_path):
            print(f"Error: Refinement config file not found: {refinement_config_path}")
            sys.exit(1)
        try:
            refinement_config = RefinementConfig.from_json_file(refinement_config_path)
            print(f"Loaded refinement config from: {refinement_config_path}")
        except Exception as e:
            print(f"Error loading refinement config: {e}")
            sys.exit(1)

    print(f"Loading from estimation JSON: {args.estimation_json}")
    try:
        process_result = manager.process_from_estimation_json(
            args.estimation_json,
            args.profile,
            args.use_root_motion,
            fps=None,
            auto_floor=args.auto_floor,
            collect_refinement_logs=False,
            refinement_config=refinement_config,
            extrinsics_file=getattr(args, "extrinsics_file", None),
            extrinsics_sample_rate=getattr(args, "extrinsics_sample_rate", 0),
            extrinsics_scale=getattr(args, "extrinsics_scale", 0.0),
            extrinsics_invert_quaternion=getattr(args, "extrinsics_invert_quaternion", False),
            extrinsics_invert_translation=getattr(args, "extrinsics_invert_translation", False),
        )

        print("Exporting FBX files...")
        fbx_paths = manager.export_fbx_files(
            process_result.profile_name,
            process_result.joint_to_bone_mappings,
            process_result.root_motions,
            process_result.frame_paths,
            process_result.fps,
            progress_callback=None,
            height_offset=process_result.height_offset,
        )

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


def main():
    """Main entry point for FBX-generation-only CLI."""
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
