# Mapping Guide

This guide explains how to configure bone mappings for custom rigs in FBXify. The system supports remapping SAM 3D Body pose estimates to various rig formats including MHR, Mixamo, and Unity.

## Overview

FBXify uses a flexible mapping system that allows you to configure how MHR pose data is translated to your target rig. For standard Mixamo rigs, the [Rokoko Retargeting plugin](https://support.rokoko.com/hc/en-us/articles/4410463481489-Retarget-an-animation-in-Blender) can handle most remapping automatically. However, for custom or special rigs, you can configure detailed bone mappings using the methods described below.

## Adding a Custom Rig

### Step 1: Extract Skeleton Information

To add support for a custom rig, you need to extract its skeleton structure and rest pose:

1. **Extract bone structure:**
   - Run `extract_armature_bone_struct_for_mapping.py` in Blender
   - Make sure your armature is selected and in a **rest pose**

2. **Extract skeleton and rest pose:**
   - Run `extract_armature_skeleton_and_rest_pose.py` in Blender
   - Ensure the armature is still selected and in rest pose

3. **Save the configuration:**
   - Copy the contents from the Blender console output
   - Save it as a JSON file
   - Add it to `metadata.py`'s `PROFILES` dictionary

### Step 2: Configure Bone Mappings

Once you have the skeleton information, configure how each bone should be mapped from MHR pose data.

## Mapping Methods

The system supports three mapping methods, each suited for different rig types and use cases.

### Method 1: Direct Rotation

The `direct_rotation` method provides a direct mapping from MHR bone rotations to your target bone. This method is almost certainly only useful for MHR rigs due to the perpendicular nature of MHR bone rotations.

**Configuration:**

```json
{
  "mapping": {
    "method": "direct_rotation",
    "reference": "<joint_name>",
    "data": {
      "rotation": ["<pred_global_rotation per frame, 3x3 matrix>"]
    }
  }
}
```

**Parameters:**
- `method`: Set to `"direct_rotation"`
- `reference`: The MHR joint name to use as the rotation source
- `data.rotation`: Array of 3x3 rotation matrices (one per frame), generated automatically by the system

### Method 2: Keypoint with Global Rotation Roll

The `keypoint_with_global_rot_roll` method extends keypoint-based mapping by adding roll rotation. This method expands on [tori29umai0123's work](https://github.com/facebookresearch/sam-3d-body/issues/66) by incorporating roll information from `pred_global_rotation`, which is essential for MHR rig bones that typically run perpendicular to the rig's mesh.

**Configuration:**

```json
{
  "mapping": {
    "method": "keypoint_with_global_rot_roll",
    "point_reference": ["<from_keypoint>", "<to_keypoint>"],
    "roll_reference": "<joint_name>",
    "roll_additional_from_reference": <0-360>,
    "data": {
      "dir_vector": ["<keyframe_3d per frame, Vector3>"],
      "roll_vector": ["<pred_global_rotation per frame, 3x3 matrix>"]
    }
  }
}
```

**Parameters:**
- `method`: Set to `"keypoint_with_global_rot_roll"`
- `point_reference`: Array with two keypoint names: `[from_keypoint, to_keypoint]` - defines the bone direction
- `roll_reference`: MHR joint name to use for roll rotation
- `roll_additional_from_reference`: Additional roll angle in degrees (0-360) from the reference joint's "up" direction (clockwise from head to tail)
- `data.dir_vector`: Array of 3D vectors (one per frame), generated automatically
- `data.roll_vector`: Array of 3x3 rotation matrices (one per frame), generated automatically

### Method 3: Keypoint-Based Mapping

Standard keypoint-based mapping uses keypoint positions to determine bone orientation.

**Keypoint References:**

Keypoints can be any keypoint returned by:

```python
from sam_3d_body.metadata.mhr70 import pose_info
```

Additionally, helper keypoints may be added under `utils`.

**Roll Reference:**

The `roll_reference` parameter is based on the 127 joints returned by MHR under:
- `pred_global_rots`: Global rotation matrices
- `pred_global_coords`: Global joint coordinates

These rotations are used to "roll" the bone correctly. You may need to experiment with different roll references and additional roll angles unless you're experienced with vector mathematics.

## Tips and Best Practices

1. **Start with Mixamo:** For standard Mixamo rigs, try the Rokoko Retargeting plugin first before configuring custom mappings.

2. **Test incrementally:** Add mappings one bone at a time and test the results to understand how each mapping affects the final pose.

3. **Roll adjustments:** The roll rotation can be tricky to get right. Use the `roll_additional_from_reference` parameter to fine-tune bone orientation. You'll likely need to guess and check different values.

4. **Rest pose matters:** Always ensure your armature is in rest pose when extracting skeleton information.

5. **MHR compatibility:** The mapping system has been tested with MHR, Mixamo, and Unity rigs. Additional rig types should work but may require more configuration.

## Troubleshooting

- **Twisted bones:** If bones appear twisted, adjust the `roll_reference` or `roll_additional_from_reference` parameters
- **Incorrect orientation:** Try different keypoint pairs in `point_reference`
- **Missing bones:** Ensure all required bones are present in your rig's skeleton structure

## Next Steps

Once your mapping is configured, the system should automatically generate the appropriate bone rotations for each frame. If you encounter issues or need to add support for additional rig types, refer to the source code in `fbxify/metadata.py` and `fbxify/pose_estimator.py` for implementation details.

Good luck!
