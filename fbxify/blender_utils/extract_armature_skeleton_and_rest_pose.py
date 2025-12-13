##
# Blender script to extract the selected armature (in rest pose)
# Schema:
# "<bone>": {
#   "parent": "<parent-or-null>",
#   "length": float,
#   "direction": [dx, dy, dz],   # normalized (tail - head) in armature space
#   "offset": [ox, oy, oz],      # head - parent_tail in armature space
#   "roll": float                # bone roll in radians
# }
##

import bpy
import json
from mathutils import Vector

# ---------- helpers ----------

def clean_float(x, ndigits=6, eps=1e-4):
    """Round small stuff to 0 and avoid e-XX in JSON."""
    if abs(x) < eps:
        return 0.0
    return round(x, ndigits)

def clean_vec(v, ndigits=6, eps=1e-4):
    return [clean_float(c, ndigits, eps) for c in v]

# ---------- main ----------

obj = bpy.context.object
if obj is None or obj.type != 'ARMATURE':
    raise RuntimeError("Select an Armature object")

# Switch to edit mode to access EditBone objects (which have .roll attribute)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')

arm = obj.data
edit_bones = arm.edit_bones  # EditBone objects in edit mode

export_data = {}

for ebone in edit_bones:
    head = Vector(ebone.head)  # armature space
    tail = Vector(ebone.tail)  # armature space

    vec = tail - head
    length = vec.length

    if length > 0.0:
        dir_norm = vec / length
    else:
        # Degenerate; arbitrary fallback
        dir_norm = Vector((0.0, 1.0, 0.0))

    if ebone.parent:
        parent_tail = Vector(ebone.parent.tail)
        offset_vec = head - parent_tail
    else:
        # Root: offset is absolute head position in armature space
        offset_vec = head

    export_data[ebone.name] = {
        "parent": ebone.parent.name if ebone.parent else None,
        "length": clean_float(length),
        "direction": clean_vec([dir_norm.x, dir_norm.y, dir_norm.z]),
        "offset": clean_vec([offset_vec.x, offset_vec.y, offset_vec.z]),
        "roll": clean_float(ebone.roll),
    }

# Dump to JSON string
json_str = json.dumps(export_data, indent=2)
print(json_str)

bpy.ops.object.mode_set(mode='OBJECT')