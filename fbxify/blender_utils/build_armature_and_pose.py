import bpy
import json
from mathutils import Vector, Matrix, Quaternion
import sys
import math

metadata_path = sys.argv[-6]
joint_mapping_path = sys.argv[-5]
root_motion_path = sys.argv[-4]
rest_pose_path = sys.argv[-3]
faces_path = sys.argv[-2]
fbx_path = sys.argv[-1]

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)["metadata"]
with open(joint_mapping_path, "r", encoding="utf-8") as f:
    joint_mapping = json.load(f)["joint_mapping"]
with open(root_motion_path, "r", encoding="utf-8") as f:
    root_motion = json.load(f)["root_motion"]
with open(rest_pose_path, "r", encoding="utf-8") as f:
    rest_pose = json.load(f)["rest_pose"]
with open(faces_path) as f:
    faces = json.load(f)["faces"]

# ------------------------------------------------------------------------
# METADATA EXTRACTION
# ------------------------------------------------------------------------

num_keyframes = metadata["num_keyframes"]
profile_name = metadata["profile_name"]
armature_name = f"sam3d_body_{profile_name}_armature"

# ------------------------------------------------------------------------
# RECONSTRUCT WORLD/ARMATURE-SPACE HEAD & TAIL
# ------------------------------------------------------------------------
computed = {}  # bone_name -> dict with head, tail, length, parent

def solve_bone(bone_name):
    """Compute head/tail in armature space for this bone (recursive)."""
    if bone_name in computed:
        return

    data = rest_pose[bone_name]
    parent_name = data["parent"]
    length = data["length"]
    direction = Vector(data["direction"])
    offset = Vector(data["offset"])

    if parent_name is None:
        # Root: offset is absolute head in armature/obj space
        head = offset
        tail = head + direction * length
    else:
        solve_bone(parent_name)
        parent = computed[parent_name]
        parent_tail = parent["tail"]

        # offset is relative to parent tail in armature/obj space
        head = parent_tail + offset
        tail = head + direction * length

    # Get roll if available, otherwise default to 0.0
    roll = data.get("roll", 0.0)
    
    computed[bone_name] = {
        "head": head,
        "tail": tail,
        "length": length,
        "parent": parent_name,
        "roll": roll,
    }

# Solve all bones
for name in rest_pose.keys():
    solve_bone(name)

# ------------------------------------------------------------------------
# RESET SCENE
# ------------------------------------------------------------------------

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# ------------------------------------------------------------------------
# CREATE NEW ARMATURE AND APPLY REST POSE
# ------------------------------------------------------------------------

# Create armature data + object
arm_data = bpy.data.armatures.new(armature_name)
arm_obj = bpy.data.objects.new(armature_name, arm_data)

# Link to current collection
bpy.context.collection.objects.link(arm_obj)
bpy.context.view_layer.objects.active = arm_obj

# Go to EDIT mode
bpy.ops.object.mode_set(mode='EDIT')
ebones = arm_data.edit_bones

# First pass: create all edit bones
edit_bones_map = {}
for name in rest_pose.keys():
    b = ebones.new(name)
    edit_bones_map[name] = b

# Second pass: set heads, tails, parents, and rolls
for name, info in computed.items():
    b = edit_bones_map[name]

    head = info["head"]
    tail = info["tail"]
    roll = info.get("roll", 0.0)

    b.head = head
    b.tail = tail
    b.roll = roll

    parent_name = info["parent"]
    if parent_name is not None:
        b.parent = edit_bones_map[parent_name]
        # Optionally connect if head ≈ parent tail
        # if (head - computed[parent_name]["tail"]).length < 1e-3:
        #     b.use_connect = True

# Back to OBJECT mode
bpy.ops.object.mode_set(mode='OBJECT')

# Set armature scale to 0.01 on all axes
arm_obj.scale = (0.01, 0.01, 0.01)

print("Rebuilt armature from JSON:", rest_pose_path)

# ------------------------------------------------------------------------
# POSE ARMATURE WITH MATRICES
# ------------------------------------------------------------------------

# switch to pose mode
bpy.ops.object.mode_set(mode='POSE')

# ------------------------------------------------------------------------
# CREATE ACTION AND SET SCENE FRAME RANGE
# ------------------------------------------------------------------------
# Create animation data if it doesn't exist
if not arm_obj.animation_data:
    arm_obj.animation_data_create()

# Create action for keyframes
action = bpy.data.actions.new(name="HumanoidAnimation")
arm_obj.animation_data.action = action

# Keyframe the armature scale at frame 0 (switch to OBJECT mode temporarily)
bpy.context.scene.frame_set(0)
bpy.ops.object.mode_set(mode='OBJECT')
arm_obj.keyframe_insert(data_path="scale", frame=0)
bpy.ops.object.mode_set(mode='POSE')

# Set scene frame range
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 1  # Will be updated after we know num_keyframes

# Update scene frame range
bpy.context.scene.frame_end = num_keyframes

# Get pose bones
blender_pose_bones = arm_obj.pose.bones

# ------------------------------------------------------------
# FUNCTION: APPLY POSE FROM GLOBAL ROTATION
# ------------------------------------------------------------
arm_world = arm_obj.matrix_world.copy()
arm_world_inv = arm_world.inverted()

def set_pose_from_global_rotation(pbone, R_global):
    """
    R_global = 3×3 rotation in WORLD SPACE.
    We must convert this to the BONE'S LOCAL POSE MATRIX.

    Correct math:
        M_global = [ R_global | current_global_translation ]
        M_local = arm_world_inv @ M_global
    """
    # 1. Current GLOBAL matrix of bone (to get correct translation)
    M_current_global = arm_world @ pbone.matrix

    # 2. Build NEW GLOBAL matrix (replace rotation only)
    M_new_global = Matrix.Identity(4)
    M_new_global.col[0].xyz = R_global.col[0]
    M_new_global.col[1].xyz = R_global.col[1]
    M_new_global.col[2].xyz = R_global.col[2]
    M_new_global.translation = M_current_global.translation

    # 3. Convert GLOBAL → ARMATURE SPACE
    M_new_armature = arm_world_inv @ M_new_global

    # 4. Assign FINAL LOCAL POSE MATRIX
    pbone.matrix = M_new_armature

# ------------------------------------------------------------------------
# PROCESS EACH KEYFRAME
# ------------------------------------------------------------------------

def list_to_matrix(lst):
    return Matrix(lst)

def reset_pose_bones(blender_pose_bones):
    for p in blender_pose_bones:
        p.rotation_mode = 'QUATERNION'
        p.location = Vector((0,0,0))
        p.rotation_quaternion = Quaternion((1,0,0,0))
        p.matrix_basis.identity()
    bpy.context.view_layer.update()

# set frame 0 as the tpose to preserve the rest pose
bpy.context.scene.frame_set(0)
reset_pose_bones(blender_pose_bones)
for p in blender_pose_bones:
    # p.keyframe_insert(data_path="location", frame=0)
    p.keyframe_insert(data_path="rotation_quaternion", frame=0)

arm_world = arm_obj.matrix_world.copy()
arm_world_inv = arm_world.inverted()

# ------------------------------------------------------------
# FUNCTIONS: GET DIR VECTOR FROM MAPPING METHOD
# ------------------------------------------------------------
def project_onto_plane(v, normal):
    """Project vector v onto plane perpendicular to normal."""
    return v - v.dot(normal) * normal

def signed_angle_on_axis(a, b, axis):
    """
    Signed angle from a -> b around 'axis'.
    a, b, axis are Vectors.
    """
    a_n = a.normalized()
    b_n = b.normalized()
    axis_n = axis.normalized()

    dot = max(min(a_n.dot(b_n), 1.0), -1.0)
    cross = a_n.cross(b_n)
    # atan2 of component along axis vs dot product
    angle = math.atan2(cross.dot(axis_n), dot)
    return angle  # radians

def calc_rotation_from_dir_and_mhr_roll(
    dir_vector,
    R_mhr,
    mhr_up_axis_local=Vector((0, 0, 1)),  # which local axis MHR treats as "up"
    extra_roll_degrees=0.0,
    world_up=Vector((0, 0, 1))
):
    """
    Build a WORLD-SPACE rotation such that:

      - Bone forward axis (Y) points along dir_vector.
      - Bone is twisted (rolled) so that its 'up' matches
        the MHR 'up' axis as much as possible, plus extra_roll_degrees.

    R_mhr: 3×3 global rotation matrix for the MHR joint.
    mhr_up_axis_local: the local axis of the MHR frame you consider "up".
                       (e.g. Vector((0,0,1)) or Vector((1,0,0)), etc.)
    """

    dir_vec = Vector(dir_vector)
    if dir_vec.length == 0:
        raise ValueError("dir_vector must be non-zero")

    # Forward axis for Mixamo bone
    forward = dir_vec.normalized()

    # --------------------------------------------------------
    # Step 1: Build zero-roll basis for this forward direction
    # --------------------------------------------------------
    up_ref = Vector(world_up)
    if abs(forward.dot(up_ref)) > 0.999:
        # Too parallel, pick another axis
        up_ref = Vector((0, 1, 0))

    # right0 = forward × up_ref
    right0 = forward.cross(up_ref)
    if right0.length < 1e-8:
        right0 = forward.orthogonal()
    right0.normalize()

    # up0 = right0 × forward
    up0 = right0.cross(forward).normalized()

    # --------------------------------------------------------
    # Step 2: Compute the MHR joint's "up" direction in world
    # --------------------------------------------------------
    # This says: "take this local axis of the MHR frame, rotate by R_mhr"
    up_mhr_world = (R_mhr @ mhr_up_axis_local).normalized()

    # --------------------------------------------------------
    # Step 3: Project both ups onto plane ⟂ forward
    # --------------------------------------------------------
    up0_plane = project_onto_plane(up0, forward)
    up_mhr_plane = project_onto_plane(up_mhr_world, forward)

    if up0_plane.length < 1e-8 or up_mhr_plane.length < 1e-8:
        # Degenerate case: fallback to zero roll + extra offset
        base_roll_rad = 0.0
    else:
        up0_plane.normalize()
        up_mhr_plane.normalize()
        # Angle from zero-roll up -> MHR up around the forward axis
        base_roll_rad = signed_angle_on_axis(up0_plane, up_mhr_plane, forward)

    # Extra roll from user (degrees)
    extra_roll_rad = math.radians(extra_roll_degrees)

    total_roll = base_roll_rad + extra_roll_rad

    # --------------------------------------------------------
    # Step 4: Apply total roll around the forward axis
    # --------------------------------------------------------
    if abs(total_roll) > 1e-8:
        R_roll = Matrix.Rotation(total_roll, 3, forward)
        right = R_roll @ right0
        up = R_roll @ up0
    else:
        right = right0
        up = up0

    # Construct rotation matrix with columns: X=right, Y=forward, Z=up
    R = Matrix.Identity(3)
    R.col[0] = right
    R.col[1] = forward
    R.col[2] = up
    return R

def get_rotation_direct_mapping(bone_dict, keyframe_index):
    if 'data' not in bone_dict or not isinstance(bone_dict['data'], dict):
        print(f"  WARNING: [{bone_dict['name']}] has 'direct mapping' but no data found")
        return None # no data

    if 'rotation' not in bone_dict['data'] or not isinstance(bone_dict['data']['rotation'], list):
        print(f"  WARNING: [{bone_dict['name']}] has 'direct mapping' but no rotation found")
        return None # no rotation

    if len(bone_dict['data']['rotation']) <= keyframe_index or bone_dict['data']['rotation'][keyframe_index] is None or \
        not isinstance(bone_dict['data']['rotation'][keyframe_index], list):
        print(f"  WARNING: [{bone_dict['name']}] has 'direct mapping' but no rotation found for frame {keyframe_index + 1}")
        return None # no rotation

    return list_to_matrix(bone_dict['data']['rotation'][keyframe_index])

def get_rotation_keypoint_with_global_rot_roll(bone_dict, keyframe_index):
    if 'data' not in bone_dict or not isinstance(bone_dict['data'], dict):
        print(f"  WARNING: [{bone_dict['name']}] has 'keypoint with global rot roll' but no data found")
        return None # no data

    if 'dir_vector' not in bone_dict['data'] or not isinstance(bone_dict['data']['dir_vector'], list):
        print(f"  WARNING: [{bone_dict['name']}] has 'keypoint with global rot roll' but no dir vector found")
        return None # no dir vector

    if 'roll_additional_from_reference' not in bone_dict or not isinstance(bone_dict['roll_additional_from_reference'], int):
        print(f"  WARNING: [{bone_dict['name']}] has 'keypoint with global rot roll' but no roll pred global rotation reference found")
        return None # no roll pred global rotation reference
        
    if len(bone_dict['data']['dir_vector']) <= keyframe_index or bone_dict['data']['dir_vector'][keyframe_index] is None or \
        not isinstance(bone_dict['data']['dir_vector'][keyframe_index], list):
        print(f"  WARNING: [{bone_dict['name']}] has 'keypoint with global rot roll' but no dir vector found for frame {keyframe_index + 1}")
        return None # no dir vector

    if 'roll_vector' not in bone_dict['data'] or not isinstance(bone_dict['data']['roll_vector'], list):
        print(f"  WARNING: [{bone_dict['name']}] has 'keypoint with global rot roll' but no roll vector found")
        return None # no roll vector

    if len(bone_dict['data']['roll_vector']) <= keyframe_index or bone_dict['data']['roll_vector'][keyframe_index] is None or \
        not isinstance(bone_dict['data']['roll_vector'][keyframe_index], list):
        print(f"  WARNING: [{bone_dict['name']}] has 'keypoint with global rot roll' but no roll vector found for frame {keyframe_index + 1}")
        return None # no roll vector

    dir_vector = bone_dict['data']['dir_vector'][keyframe_index]
    roll_matrix = list_to_matrix(bone_dict['data']['roll_vector'][keyframe_index])
    roll_degrees = bone_dict['roll_additional_from_reference']

    return calc_rotation_from_dir_and_mhr_roll(dir_vector, roll_matrix, extra_roll_degrees=roll_degrees)

def get_rotation_from_mapping_method(bone_dict, keyframe_index):
    # Depending on the mapping, we output a global 3x3 rotation matrix that can be used to set the bone
    # 3x3 rotation allows preservation of the roll during the "point"
    # Direct mapping allows us to just return data.rotation[keyframe_index]!
    # Otherwise we need to do some complicated math
    if 'data' not in bone_dict or not isinstance(bone_dict['data'], dict):
        return None # no data

    if bone_dict['method'] == 'keypoint_with_global_rot_roll':
        return get_rotation_keypoint_with_global_rot_roll(bone_dict, keyframe_index)
    if bone_dict['method'] == 'direct_rotation':
        return get_rotation_direct_mapping(bone_dict, keyframe_index)

    return None


def breadth_first_pose_application(joint_mapping, frame_idx):
    # bpy.context.view_layer.update() is expensive - but needs to run after every layer
    # We can do an O(log n) traversal of the bone hierarchy by going breadth-first

    # TODO: It would be amazing if we could do additive global rotation mapping, so
    # child_rotation = R_global @ parent_R_global @ parent_R_global ... @ root_R_global
    # But I haven't been able to crack it, maybe someone with more vector math knowledge could do it?

    current_layer = [joint_mapping] # hips

    applied_bones = set()
    while current_layer:
        next_layer = []
        for bone_dict in current_layer:
            if bone_dict['name'] not in blender_pose_bones:
                print(f"  WARNING: [{bone_dict['name']}] Bone not found in armature")
                continue

            pbone = blender_pose_bones[bone_dict['name']]

            R_global = get_rotation_from_mapping_method(bone_dict, frame_idx)

            if (R_global is None):
                print(f"  WARNING: [{bone_dict['name']}] has no rotation found for frame {frame_idx + 1}")
            else:
                set_pose_from_global_rotation(pbone, R_global)
                applied_bones.add(bone_dict['name'])
                bpy.context.view_layer.update()

            for child in bone_dict.get("children", []):
                next_layer.append(child)

        bpy.context.view_layer.update() # expensive update
        current_layer = next_layer

    for bone in blender_pose_bones:
        bone.keyframe_insert("rotation_quaternion", frame=frame_idx + 1)

    print(f"Applied pose to {len(applied_bones)} bones on keyframe {frame_idx + 1}")
    reset_pose_bones(blender_pose_bones)

for frame_idx in range(num_keyframes):
    breadth_first_pose_application(joint_mapping, frame_idx)

bpy.ops.object.mode_set(mode="OBJECT")

# ------------------------------------------------------------------------
# APPLY ROOT MOTION
# ------------------------------------------------------------------------
"""
bpy.ops.object.mode_set(mode="OBJECT")

if len(root_motion) > 0: # root motion can be passed empty, if the user doesn't want root motion
    # In object mode, use root_motion, which is a list of global rotation euler angles and camera translation vectors
    # apply keyframes to the armature, not any bone
    arm_obj.rotation_mode = 'QUATERNION'
    
    for frame_idx, root_motion_entry in enumerate(root_motion, start=1):
        # Use camera translation as-is (the base 90° rotation at frame 0 handles coord system)
        cam_translation = root_motion_entry["pred_cam_t"]
        arm_obj.location = Vector((cam_translation[0], -cam_translation[1], -cam_translation[2])) # Vector((cam_translation[0], -cam_translation[2], cam_translation[1]))

        # Use rotation as-is, convert Euler to Quaternion to avoid gimbal lock
        euler = root_motion_entry["global_rot"]
        arm_obj.rotation_euler = euler

        arm_obj.keyframe_insert(data_path="location", frame=frame_idx)
        arm_obj.keyframe_insert(data_path="rotation_euler", frame=frame_idx)

bpy.ops.object.mode_set(mode="OBJECT")
"""

bpy.ops.object.mode_set(mode="OBJECT")

# TODO: This is basically a hardcoded value for my default extracted armature's height to hip bone, in its rest pose
# We should use extracted values from MHR to get height and reinforce armature scaling, height, weight, face values, etc.
try:
    rest_pose_hips_offset = rest_pose[list(rest_pose.keys())[0]]["offset"][1] / 100.0 # cm -> m
except:
    print("  WARNING: No rest pose hips offset (y value from floor) found, using default value of 0.0")
    rest_pose_hips_offset = 0.0

if len(root_motion) > 0: # root motion can be passed empty, if the user doesn't want root motion
    # In object mode, use root_motion, which is a list of global rotation euler angles and camera translation vectors
    # apply keyframes to the armature, not any bone
    arm_obj.rotation_mode = 'XYZ'
    
    for frame_idx, root_motion_entry in enumerate(root_motion, start=1):
        # Use camera translation as-is (the base 90° rotation at frame 0 handles coord system)
        cam_translation = root_motion_entry["pred_cam_t"]

        arm_obj.location = Vector((cam_translation[0], cam_translation[2], cam_translation[1] - rest_pose_hips_offset))

        # As far as I can tell, this value of global rotation is already passed to the root bone (hips), but if you'd rather 
        # have rotation be a "root" motion, apply it here and skip in pose application
        # euler = root_motion_entry["global_rot"]
        
        # Each keyframe does need to rotate by math.pi/2 x rotation to "stand up" in blender
        arm_obj.rotation_euler = (math.pi/2, 0, 0)

        arm_obj.keyframe_insert(data_path="location", frame=frame_idx)
        arm_obj.keyframe_insert(data_path="rotation_euler", frame=frame_idx)

bpy.ops.object.mode_set(mode="OBJECT")

# -----------------------------------------------------------------------------
# FBXエクスポート
# -----------------------------------------------------------------------------

# Rigging cleanup before exporting
bpy.ops.object.mode_set(mode="POSE")
bpy.ops.pose.transforms_clear()
bpy.ops.object.mode_set(mode="OBJECT")

bpy.context.scene.frame_set(0)

# After we've set all the rotations, we should rotate the entire armature to 90,0,0 (degrees) so that the rig "stands up" in blender
arm_obj.location = Vector((0, 0, 0))
# Convert 90° X rotation to quaternion
arm_obj.rotation_euler = (math.pi/2, 0, 0)
arm_obj.keyframe_insert(data_path="location", frame=0)
arm_obj.keyframe_insert(data_path="rotation_euler", frame=0)

bpy.ops.object.select_all(action='DESELECT')
arm_obj.select_set(True)
bpy.context.view_layer.objects.active = arm_obj

bpy.context.scene.frame_set(0)

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end   = num_keyframes

bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=True,
    bake_anim=True,
    bake_anim_use_nla_strips=True,
    bake_anim_simplify_factor=0.0,  # keep keys exact
    apply_scale_options='FBX_SCALE_NONE',
    apply_unit_scale=True,
    axis_forward='-Z',
    axis_up='Y',
    primary_bone_axis='Y',
    secondary_bone_axis='X',
    use_armature_deform_only=True,
    add_leaf_bones=False,
)

print(f"✓ FBXエクスポート完了: {fbx_path}")
print("✓ 1フレームアニメーション: HumanoidAnimation")