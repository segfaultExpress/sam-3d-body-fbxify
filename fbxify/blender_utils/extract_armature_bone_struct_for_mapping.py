import bpy
import json
from difflib import SequenceMatcher

# ------------------------------------------------------------
# Optional: Simple string-matching helper
# "RightUpperLeg" -> "UpperRightLeg"
# ------------------------------------------------------------

def smart_reorder(name, target_patterns):
    """
    Try to reorder tokens in 'name' to match any target pattern.
    target_patterns is a list of patterns, e.g. ["UpperRightLeg"].
    Returns best matching target or "" if no good match found.
    """
    best = ("", 0.0)
    for target in target_patterns:
        ratio = SequenceMatcher(None, name.lower(), target.lower()).ratio()
        if ratio > best[1]:
            best = (target, ratio)

    # Only accept if similarity is decent
    return best[0] if best[1] > 0.6 else ""


# ------------------------------------------------------------
# RECURSIVE EXPORT
# ------------------------------------------------------------

def serialize_bone(bone):
    """
    bone: an object with:
       - bone.name
       - bone.children (list of bone objects)
    """
    if bone is None:
        return None

    print(bone)

    return {
        "name": bone.name,
        "method": "direct_rotation", # this is almost certainly wrong unless you're doing MHR -> MHR mapping (see remapping guide)
        "reference": bone.name, # <- again, only useful if the mapping is direct. Mixamo/Unity won't know what to do with this.
        "children": [serialize_bone(child) for child in bone.children if child is not None]
    }


# ------------------------------------------------------------
# EXAMPLE USAGE (fake simple hierarchy)
# Replace with your own armature structure.
# ------------------------------------------------------------

class Bone:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []


# Get the selected armature
obj = bpy.context.object
if obj is None or obj.type != 'ARMATURE':
    raise RuntimeError("Select an Armature object")

arm = obj.data
bones = arm.bones

# Find root bones (bones without parents) and serialize them
root_bones = [bone for bone in bones if bone.parent is None]

# Serialize all root bones
if len(root_bones) == 1:
    # Single root bone - serialize it directly
    data = serialize_bone(root_bones[0])
else:
    # Multiple root bones - serialize as a list
    data = [serialize_bone(bone) for bone in root_bones]

# Print JSON
print(json.dumps(data, indent=2))
