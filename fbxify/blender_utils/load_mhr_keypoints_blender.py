"""
Blender script to load MHR keypoints from JSON and visualize them
Run this script in Blender's Scripting workspace
"""

import bpy
import json
import os
import sys

# Add the blender_utils directory to path
# Handle both when run as script and when run in Blender
blender_utils_dir = None

# Method 1: Try __file__ (works when script is executed directly)
try:
    if '__file__' in globals() and __file__:
        blender_utils_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Using __file__ method: {blender_utils_dir}")
except:
    pass

# Method 2: Try Blender's text block filepath (works in Blender text editor)
if not blender_utils_dir or not os.path.exists(blender_utils_dir):
    try:
        # Get the current text block's filepath
        text_block = bpy.context.space_data.text
        if text_block and text_block.filepath:
            blender_utils_dir = os.path.dirname(os.path.abspath(text_block.filepath))
            print(f"Using text block method: {blender_utils_dir}")
    except:
        pass

# Method 3: Try .blend file's directory
if not blender_utils_dir or not os.path.exists(blender_utils_dir):
    try:
        if bpy.data.filepath:
            blend_dir = os.path.dirname(bpy.data.filepath)
            # Look for blender_utils in the blend file's directory or parent
            potential_paths = [
                os.path.join(blend_dir, "blender_utils"),
                os.path.join(os.path.dirname(blend_dir), "fbxify", "blender_utils"),
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    blender_utils_dir = path
                    print(f"Using blend file method: {blender_utils_dir}")
                    break
    except:
        pass

# Method 4: Hardcoded fallback (update this to your actual path)
if not blender_utils_dir or not os.path.exists(blender_utils_dir):
    blender_utils_dir = r"F:\Projects\Project-Mimesis\sam-3d-body-main\fbxify\blender_utils"
    print(f"Using hardcoded path: {blender_utils_dir}")

# Find and import the visualizer module directly
import importlib.util

visualizer_file = None
if blender_utils_dir and os.path.exists(blender_utils_dir):
    visualizer_file = os.path.join(blender_utils_dir, "mhr_keypoints_visualizer.py")
    if not os.path.exists(visualizer_file):
        visualizer_file = None

# If still not found, try hardcoded path
if not visualizer_file or not os.path.exists(visualizer_file):
    hardcoded_path = r"F:\Projects\Project-Mimesis\sam-3d-body-main\fbxify\blender_utils\mhr_keypoints_visualizer.py"
    if os.path.exists(hardcoded_path):
        visualizer_file = hardcoded_path
        blender_utils_dir = os.path.dirname(hardcoded_path)

if not visualizer_file or not os.path.exists(visualizer_file):
    raise FileNotFoundError(
        f"Could not find mhr_keypoints_visualizer.py\n"
        f"Tried: {visualizer_file}\n"
        f"Please ensure the file exists in: {blender_utils_dir}"
    )

print(f"Loading visualizer from: {visualizer_file}")

# Load the module directly from file
spec = importlib.util.spec_from_file_location("mhr_keypoints_visualizer", visualizer_file)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not create spec for {visualizer_file}")

mhr_visualizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mhr_visualizer_module)

# Import the classes/functions we need
MHRKeypointsVisualizer = mhr_visualizer_module.MHRKeypointsVisualizer
apply_keypoints = mhr_visualizer_module.apply_keypoints
snapshot_frame = mhr_visualizer_module.snapshot_frame

print("Successfully imported mhr_keypoints_visualizer")


def load_keypoints_from_json(json_path):
    """
    Load keypoints from JSON file and visualize in Blender
    
    Args:
        json_path: Path to JSON file containing keypoints
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create visualizer
    visualizer = MHRKeypointsVisualizer()
    visualizer.create_skeleton()
    
    # Get keypoints array
    if "keypoints_3d_array" in data:
        keypoints_3d = data["keypoints_3d_array"]
    elif "keypoints" in data:
        # Reconstruct array from keypoints list
        keypoints_3d = []
        keypoints_list = sorted(data["keypoints"], key=lambda x: x["id"])
        for kp in keypoints_list:
            keypoints_3d.append(kp["position"])
    else:
        raise ValueError("JSON file must contain either 'keypoints_3d_array' or 'keypoints'")
    
    # Apply keypoints to skeleton
    print(f"Applying {len(keypoints_3d)} keypoints to skeleton...")
    apply_keypoints(visualizer, keypoints_3d)
    
    # Create keyframe at frame 1
    bpy.context.scene.frame_set(1)
    snapshot_frame(visualizer, 1)
    
    print(f"Successfully loaded and visualized keypoints from {json_path}")
    print(f"Image path: {data.get('image_path', 'N/A')}")
    print(f"Number of keypoints: {len(keypoints_3d)}")
    
    return visualizer

# Main execution
if __name__ == "__main__":
    # Example usage - update the path to your JSON file
    json_file_path = r"F:/Projects/Project-Mimesis/sam-3d-body-main/output/keypoints.json"
    
    # Alternative: uncomment to use interactive file selection
    # import bpy
    # json_file_path = bpy.path.abspath("//keypoints.json")  # Relative to .blend file
    
    if os.path.exists(json_file_path):
        load_keypoints_from_json(json_file_path)
    else:
        print(f"JSON file not found: {json_file_path}")
        print("\nTo use this script:")
        print("1. Run save_mhr_keypoints.py to generate a JSON file from an image")
        print("2. Update json_file_path in this script to point to your JSON file")
        print("3. Run this script in Blender's Scripting workspace")
        print("\nExample:")
        print("  python save_mhr_keypoints.py --image image.jpg --output keypoints.json --checkpoint checkpoint.pth")
