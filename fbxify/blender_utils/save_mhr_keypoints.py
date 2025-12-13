"""
Script to save MHR 3D keypoints from an image to JSON
Uses SAM 3D Body to predict keypoints and saves them in a format
that can be loaded by the Blender visualizer

example usage:
python fbxify/blender_utils/save_mhr_keypoints.py \ 
  --image /workspace/images/test.png \
  --output /workspace/output/keypoints.json \
  --mhr_path /workspace/checkpoints/sam-3d-body-vith/assets/mhr_model.pt \
  --checkpoint /workspace/checkpoints/sam-3d-body-vith/model.ckpt
"""

import argparse
import os
import json
import numpy as np
import torch
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator


def convert_numpy_to_list(obj):
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    else:
        return obj


def save_keypoints_to_json(image_path, output_path, checkpoint_path, mhr_path=None, device=None):
    """
    Process an image with SAM 3D Body and save keypoints to JSON
    
    Args:
        image_path: Path to input image
        output_path: Path to save JSON file
        checkpoint_path: Path to SAM 3D Body checkpoint
        mhr_path: Path to MHR model (optional, uses env var if not provided)
        device: Device to use (cuda/cpu, defaults to cuda if available)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use environment variable if mhr_path not provided
    if mhr_path is None:
        mhr_path = os.environ.get("SAM3D_MHR_PATH", "")
    
    print(f"Loading SAM 3D Body on {device}...")
    model, model_cfg = load_sam_3d_body(
        checkpoint_path, device=device, mhr_path=mhr_path
    )
    
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )
    
    print(f"Processing image: {image_path}")
    outputs_raw = estimator.process_one_image(image_path)
    
    # Pick the largest person if multiple detected
    if isinstance(outputs_raw, list):
        sizes = []
        for o in outputs_raw:
            verts = o["pred_vertices"]
            sizes.append((np.max(verts[:,0]) - np.min(verts[:,0])) +
                        (np.max(verts[:,1]) - np.min(verts[:,1])))
        outputs = outputs_raw[int(np.argmax(sizes))]
    else:
        outputs = outputs_raw
    
    # Extract keypoints (70 keypoints in MHR format)
    keypoints_3d = outputs["pred_keypoints_3d"]  # Shape: (70, 3)
    
    # Get keypoint names from metadata
    try:
        from sam_3d_body.metadata.mhr70 import pose_info
        keypoint_names = {}
        for kp_id, kp_data in pose_info['keypoint_info'].items():
            keypoint_names[kp_id] = kp_data['name']
    except ImportError:
        print("Warning: Could not import pose_info, using numeric IDs")
        keypoint_names = {i: f"keypoint_{i}" for i in range(70)}
    
    # Prepare data structure
    keypoints_data = {
        "image_path": image_path,
        "num_keypoints": len(keypoints_3d),
        "keypoints": []
    }
    
    # Add each keypoint with its name and 3D position
    for kp_id in range(len(keypoints_3d)):
        kp_name = keypoint_names.get(kp_id, f"keypoint_{kp_id}")
        x, y, z = keypoints_3d[kp_id]
        keypoints_data["keypoints"].append({
            "id": int(kp_id),
            "name": kp_name,
            "position": [float(x), float(y), float(z)]
        })
    
    # Also include raw keypoints array for easy loading
    keypoints_data["keypoints_3d_array"] = convert_numpy_to_list(keypoints_3d)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(keypoints_data, f, indent=2)
    
    print(f"Saved {len(keypoints_3d)} keypoints to {output_path}")
    return keypoints_data


def main():
    parser = argparse.ArgumentParser(description="Save MHR 3D keypoints from image to JSON")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM 3D Body checkpoint")
    parser.add_argument("--mhr_path", type=str, default=None, help="Path to MHR model (or use SAM3D_MHR_PATH env var)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    save_keypoints_to_json(
        image_path=args.image,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        mhr_path=args.mhr_path,
        device=args.device
    )


if __name__ == "__main__":
    main()
