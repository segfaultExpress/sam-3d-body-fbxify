import bpy
import json
from mathutils import Vector, Matrix, Quaternion
from bpy_extras import anim_utils
import sys
import math
import os

metadata_path = sys.argv[-8]
joint_mapping_path = sys.argv[-7]
root_motion_path = sys.argv[-6]
rest_pose_path = sys.argv[-5]
faces_path = sys.argv[-4]
fbx_path = sys.argv[-3]
lod_fbx_path = sys.argv[-2] if sys.argv[-2] else None
mesh_obj_path = sys.argv[-1] if sys.argv[-1] else None

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

root_motion_by_frame = {
    entry.get("frame_index"): entry
    for entry in (root_motion or [])
    if isinstance(entry, dict) and entry.get("frame_index") is not None
}

# ------------------------------------------------------------------------
# DEV TOOLS FOR DEBUGGING/FAST APPLY
# ------------------------------------------------------------------------
APPLY_ROOT_MOTION_OVERRIDE = True
APPLY_POSE_OVERRIDE = True
APPLY_BLENDER_AXIS_ROTATION = True

# ------------------------------------------------------------------------
# METADATA EXTRACTION
# ------------------------------------------------------------------------

num_keyframes = metadata["num_keyframes"]
profile_name = metadata["profile_name"]
height_offset = metadata.get("height_offset", 0.0)
fps = metadata.get("fps", 30.0)  # Frame rate, default to 30.0 if not present
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
# HELPER FUNCTIONS FOR ARMATURE CREATION AND MESH RESKINNING
# ------------------------------------------------------------------------

def force_rebuild_via_mode_toggle(obj):
    """
    Mimics the exact 'Edit Mode -> Object Mode' nuke that makes Blender
    commit mesh/shapekey data and refresh depsgraph for exporters.
    """
    if obj is None:
        return

    view_layer = bpy.context.view_layer
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    view_layer.objects.active = obj

    # Must be in OBJECT mode before switching
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # The hammer
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Extra flush
    if obj.data:
        obj.data.update()
    obj.update_tag()
    view_layer.update()
    bpy.context.evaluated_depsgraph_get().update()

def create_armature_from_rest_pose(armature_name, rest_pose, computed):
    """Create a new armature from rest_pose JSON data."""
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
    return arm_obj

def check_mesh_compatibility(me_a, me_b):
    """Check if two meshes are compatible for reskinning."""
    if len(me_a.vertices) != len(me_b.vertices):
        return False, f"Vertex count differs: {len(me_a.vertices)} vs {len(me_b.vertices)}"
    if len(me_a.polygons) != len(me_b.polygons):
        return False, f"Polygon count differs: {len(me_a.polygons)} vs {len(me_b.polygons)}"
    
    # Check face connectivity
    sig_a = [tuple(p.vertices) for p in me_a.polygons]
    sig_b = [tuple(p.vertices) for p in me_b.polygons]
    if sig_a != sig_b:
        return False, "Face connectivity/order differs"
    
    return True, "Meshes are compatible"

def get_vert_coords(obj):
    """Return vertex coordinates, handling shape keys if present."""
    me = obj.data
    if me.shape_keys and me.shape_keys.key_blocks:
        kb = me.shape_keys.key_blocks.get("Basis") or me.shape_keys.key_blocks[0]
        return kb.data
    return me.vertices

def set_dst_coords(obj, i, co):
    """Write coordinates into destination, handling shape keys if present."""
    me = obj.data
    if me.shape_keys and me.shape_keys.key_blocks:
        kb = me.shape_keys.key_blocks.get("Basis")
        if kb is None:
            raise RuntimeError(f"Destination '{obj.name}' has shape keys but no 'Basis' key.")
        kb.data[i].co = co
    else:
        me.vertices[i].co = co

def import_armature_from_fbx(lod_fbx_path, armature_name):
    """Import armature from LOD FBX file and return the armature object."""
    bpy.ops.import_scene.fbx(filepath=lod_fbx_path)
    imported_objects = bpy.context.selected_objects
    armature_obj = None
    for obj in imported_objects:
        if obj.type == 'ARMATURE':
            # Rename to match expected name
            obj.name = armature_name
            obj.data.name = armature_name
            # Set armature scale to 0.01 on all axes (matching rest_pose creation)
            obj.scale = (0.01, 0.01, 0.01)
            armature_obj = obj
            break
    
    # Ensure armature is selected and active after import
    if armature_obj:
        # Reset rotation to match generated armature (rotation will be applied later before export)
        armature_obj.rotation_euler = (0, 0, 0)
        bpy.ops.object.select_all(action='DESELECT')
        armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj
        print(f"Imported armature from FBX: {lod_fbx_path}")
        return armature_obj
    return None

def import_mesh_obj(mesh_obj_path):
    """Import OBJ mesh and return the mesh object."""
    # Blender 5.0+ uses wm.obj_import instead of import_scene.obj
    try:
        bpy.ops.wm.obj_import(filepath=mesh_obj_path)
    except AttributeError:
        # Fallback for older Blender versions
        bpy.ops.import_scene.obj(filepath=mesh_obj_path)
    imported_objects = bpy.context.selected_objects
    for obj in imported_objects:
        if obj.type == 'MESH':
            return obj
    return None

def find_lod_mesh(lod_armature_obj, exclude_mesh_obj):
    """Find the mesh object parented to the LOD armature."""
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj != exclude_mesh_obj:
            if obj.parent == lod_armature_obj:
                return obj
    return None

def reskin_mesh(source_mesh_obj, dst_mesh_obj):
    """Reskin the destination mesh with coordinates from source mesh."""
    # Ensure destination mesh is active and in OBJECT mode (critical for headless mode)
    bpy.context.view_layer.objects.active = dst_mesh_obj
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    src_coords = get_vert_coords(source_mesh_obj)
    
    # Axis/scale fix (from reskin_mhr_rig.py)
    AXIS_SCALE = Vector((1, -1, -1))
    dst_me = dst_mesh_obj.data
    src_me = source_mesh_obj.data

    if len(dst_me.vertices) != len(src_me.vertices):
        raise RuntimeError(f"Vertex count mismatch: dst={len(dst_me.vertices)} src={len(src_me.vertices)}")

    src_mw = source_mesh_obj.matrix_world.copy()
    dst_mw_inv = dst_mesh_obj.matrix_world.inverted_safe()

    for i in range(len(dst_me.vertices)):
        v = src_coords[i].co.copy()
        
        # Apply axis/scale fix in source local space
        v.x *= AXIS_SCALE.x
        v.y *= AXIS_SCALE.y
        v.z *= AXIS_SCALE.z
        
        # Convert: src local -> world -> dst local
        v_world = src_mw @ v
        v_dst_local = dst_mw_inv @ v_world
        
        set_dst_coords(dst_mesh_obj, i, v_dst_local)
    
    # Force updates (vertex groups are preserved automatically - they're stored per-vertex, not per-position)
    dst_me.update()
    # Update the object to ensure changes are reflected
    dst_mesh_obj.update_tag()
    # Force view layer update in headless mode
    bpy.context.view_layer.update()
    # Force dependency graph update to ensure changes are visible
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()

def parent_mesh_to_armature(mesh_obj, arm_obj):
    """Parent mesh to armature and set up armature modifier."""
    mesh_obj.parent = arm_obj
    mesh_obj.parent_type = 'OBJECT'
    
    # Copy armature modifier if it exists, or create one
    if mesh_obj.modifiers:
        for mod in mesh_obj.modifiers:
            if mod.type == 'ARMATURE':
                mod.object = arm_obj
                return
    else:
        arm_mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
        arm_mod.object = arm_obj

def load_and_reskin_mesh(arm_obj, mesh_obj_path):
    """Main function to load and reskin mesh. Returns reskinned mesh object or None."""
    # Check if files are valid
    if not arm_obj or not mesh_obj_path:
        return None
    
    print("Loading mesh files for reskinning...")
    
    source_mesh_obj = None
    lod_mesh_obj = None

    print(f"Loading mesh files for reskinning... mesh_obj_path: {mesh_obj_path}")
    
    try:
        # Import OBJ mesh (source mesh with new shape)
        source_mesh_obj = import_mesh_obj(mesh_obj_path)

        # The mesh comes in upside down but weirdly (90,0,0) rotated. So we need to set the axis scale to (1,-1,-1) and need to rotate it back to (0,0,0)
        source_mesh_obj.rotation_euler = (0,0,0)

        if source_mesh_obj is None:
            print("Warning: No mesh found in OBJ file, skipping mesh reskinning")
            return None
        
        print(f"Found mesh in OBJ: {source_mesh_obj.name}")
        
        # Find LOD mesh parented to the imported armature (destination mesh to reskin)
        lod_mesh_obj = find_lod_mesh(arm_obj, source_mesh_obj)
        if lod_mesh_obj is None:
            print("Warning: No mesh found parented to LOD armature, skipping mesh reskinning")
            # Clean up imported source mesh
            if source_mesh_obj:
                try:
                    bpy.data.objects.remove(source_mesh_obj, do_unlink=True)
                except:
                    pass
            # Ensure armature is selected
            bpy.ops.object.select_all(action='DESELECT')
            arm_obj.select_set(True)
            bpy.context.view_layer.objects.active = arm_obj
            return None
        
        # Log vertex groups and weights before reskinning
        lod_me = lod_mesh_obj.data
        
        # Check compatibility
        new_me = source_mesh_obj.data
        compatible, msg = check_mesh_compatibility(lod_me, new_me)
        
        if not compatible:
            print(f"[Warning]: Meshes are not compatible: {msg}")
            print("Skipping mesh reskinning, will export armature only")
            # Clean up imported objects before returning
            if source_mesh_obj:
                try:
                    bpy.data.objects.remove(source_mesh_obj, do_unlink=True)
                except:
                    pass
            # Ensure armature is selected before returning
            bpy.ops.object.select_all(action='DESELECT')
            arm_obj.select_set(True)
            bpy.context.view_layer.objects.active = arm_obj
            return None
        
        # Ensure mesh is in OBJECT mode before reskinning (important for vertex group preservation)
        # In headless mode, we need to be more explicit about context
        bpy.ops.object.select_all(action='DESELECT')
        lod_mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = lod_mesh_obj
        # Ensure we're in OBJECT mode (this is critical for vertex group preservation)
        if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # Reskin mesh (this only changes vertex positions, not weights)
        print("Reskinning mesh (copying vertex positions)...")
        reskin_mesh(source_mesh_obj, lod_mesh_obj)
        print("✓ Mesh reskinning complete")

        # I swear to god sometimes Blender just puts something like this in just to mess with us?
        # Why do meshes have visible changes that are not reflected until a hard mode toggle?
        # Why are they visible for all intents and purposes until a headless export/import removes them?
        force_rebuild_via_mode_toggle(lod_mesh_obj)
        
        # Ensure mesh data is updated after reskinning
        # Force a view layer update in headless mode to ensure changes are reflected
        lod_me.update()
        lod_mesh_obj.update_tag()
        bpy.context.view_layer.update()
        # Force dependency graph update to ensure changes are visible
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        
        # Parent to new armature (vertex groups should be preserved)
        parent_mesh_to_armature(lod_mesh_obj, arm_obj)
        print(f"✓ Mesh parented to new armature")
        
        # Log armature modifier setup
        arm_mod = None
        for mod in lod_mesh_obj.modifiers:
            if mod.type == 'ARMATURE':
                arm_mod = mod
                break
        if arm_mod:
            print(f"  Armature modifier found: object='{arm_mod.object.name if arm_mod.object else 'None'}'")
            print(f"  Armature modifier use_deform_preserve_volume: {arm_mod.use_deform_preserve_volume}")
        else:
            print("  WARNING: No armature modifier found after parenting!")
        
        # Check if vertex group names match bone names
        bone_names = set(b.name for b in arm_obj.data.bones)
        vg_names = set(vg.name for vg in lod_mesh_obj.vertex_groups)
        matching = bone_names & vg_names
        missing_in_armature = vg_names - bone_names
        missing_in_mesh = bone_names - vg_names
        print(f"  Vertex group / bone name matching:")
        print(f"    Matching: {len(matching)} groups")
        if len(missing_in_armature) > 0:
            print(f"    WARNING: {len(missing_in_armature)} vertex groups not found in armature: {list(missing_in_armature)[:5]}")
        if len(missing_in_mesh) > 0:
            print(f"    INFO: {len(missing_in_mesh)} bones have no vertex groups (may be normal for end bones)")
        
        # Clean up temporary source mesh object
        bpy.data.objects.remove(source_mesh_obj, do_unlink=True)
        
        # Ensure armature is selected after successful reskinning
        bpy.ops.object.select_all(action='DESELECT')
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj
        
        return lod_mesh_obj
        
    except Exception as e:
        print(f"Error during mesh loading/reskinning: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with armature-only export...")
        
        # Clean up any partially imported objects
        if source_mesh_obj:
            try:
                bpy.data.objects.remove(source_mesh_obj, do_unlink=True)
            except:
                pass
        
        # Ensure armature is selected after error
        bpy.ops.object.select_all(action='DESELECT')
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj
        
        return None

# ------------------------------------------------------------------------
# ARMATURE CREATION: FROM FBX OR FROM REST_POSE
# ------------------------------------------------------------------------

mesh_obj = None

if lod_fbx_path and os.path.exists(lod_fbx_path) and os.path.getsize(lod_fbx_path) > 0:
    # LOD FBX provided: Import armature from FBX and reskin mesh
    print("LOD FBX provided: Importing armature from FBX and reskinning mesh")
    
    arm_obj = import_armature_from_fbx(lod_fbx_path, armature_name)
    if arm_obj is None:
        print("Warning: Failed to import armature from FBX, falling back to rest_pose creation")
        arm_obj = create_armature_from_rest_pose(armature_name, rest_pose, computed)
    else:
        # Reskin mesh if mesh OBJ is also provided
        mesh_obj = load_and_reskin_mesh(arm_obj, mesh_obj_path)
        # Ensure armature is selected after mesh operations (in case selection changed or reskinning failed)
        # This is important because load_and_reskin_mesh may have imported objects that changed selection
        bpy.ops.object.select_all(action='DESELECT')
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj
else:
    # No LOD FBX: Build armature from scratch using rest_pose
    print("No LOD FBX provided: Building armature from rest_pose JSON")
    arm_obj = create_armature_from_rest_pose(armature_name, rest_pose, computed)

# ------------------------------------------------------------------------
# POSE ARMATURE WITH MATRICES
# ------------------------------------------------------------------------

# Ensure armature is selected and active before switching to pose mode
bpy.ops.object.select_all(action='DESELECT')
arm_obj.select_set(True)
bpy.context.view_layer.objects.active = arm_obj

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
if APPLY_POSE_OVERRIDE:
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
        return None # no data

    if 'rotation' not in bone_dict['data'] or not isinstance(bone_dict['data']['rotation'], list):
        return None # no rotation

    if len(bone_dict['data']['rotation']) <= keyframe_index or bone_dict['data']['rotation'][keyframe_index] is None or \
        not isinstance(bone_dict['data']['rotation'][keyframe_index], list):
        return None # no rotation

    return list_to_matrix(bone_dict['data']['rotation'][keyframe_index])

def get_rotation_keypoint_with_global_rot_roll(bone_dict, keyframe_index):
    if 'data' not in bone_dict or not isinstance(bone_dict['data'], dict):
        return None # no data

    if 'dir_vector' not in bone_dict['data'] or not isinstance(bone_dict['data']['dir_vector'], list):
        return None # no dir vector

    if 'roll_additional_from_reference' not in bone_dict or not isinstance(bone_dict['roll_additional_from_reference'], int):
        return None # no roll pred global rotation reference
        
    if len(bone_dict['data']['dir_vector']) <= keyframe_index or bone_dict['data']['dir_vector'][keyframe_index] is None or \
        not isinstance(bone_dict['data']['dir_vector'][keyframe_index], list):
        return None # no dir vector

    if 'roll_vector' not in bone_dict['data'] or not isinstance(bone_dict['data']['roll_vector'], list):
        return None # no roll vector

    if len(bone_dict['data']['roll_vector']) <= keyframe_index or bone_dict['data']['roll_vector'][keyframe_index] is None or \
        not isinstance(bone_dict['data']['roll_vector'][keyframe_index], list):
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

def ensure_action(arm_obj, action_name="ImportedAction"):
    ad = arm_obj.animation_data
    if ad is None:
        arm_obj.animation_data_create()
        ad = arm_obj.animation_data

    act = ad.action
    if act is None:
        act = bpy.data.actions.new(action_name)
        ad.action = act
    
    return act, ad

def ensure_quat_fcurves(action, anim_data, bone_name):
    """
    Returns list of 4 fcurves for rotation_quaternion (w,x,y,z).
    Compatible with Blender 5.0+ channel bag API.
    """
    data_path = f'pose.bones["{bone_name}"].rotation_quaternion'
    fcs = []
    
    # Check if we're using Blender 5.0+ (action.fcurves doesn't exist)
    is_blender_5_plus = not hasattr(action, 'fcurves')
    
    if is_blender_5_plus:
        # Blender 5.0+ requires using channel bags via anim_utils
        if anim_data is None:
            raise RuntimeError("Animation data is required for Blender 5.0+")
        
        # Get action slot - try multiple ways to access it
        slot = None
        # Method 1: Try action_slot (singular)
        if hasattr(anim_data, 'action_slot') and anim_data.action_slot:
            slot = anim_data.action_slot
        # Method 2: Try action_slots (plural) - first slot
        elif hasattr(anim_data, 'action_slots') and len(anim_data.action_slots) > 0:
            slot = anim_data.action_slots[0]
        # Method 3: Try to get slot from action itself
        elif hasattr(action, 'slots') and len(action.slots) > 0:
            slot = action.slots[0]
        
        # If no slot found, try to create one
        if slot is None:
            try:
                if hasattr(anim_data, 'action_slots'):
                    slot = anim_data.action_slots.new(name=action.name)
            except (AttributeError, TypeError):
                pass
        
        if slot is None:
            raise RuntimeError("Cannot access fcurves: action slot not available in Blender 5.0+. "
                             "Please ensure the action is properly assigned to the armature.")
        
        try:
            channelbag = anim_utils.action_ensure_channelbag_for_slot(action, slot)
            
            for i in range(4):
                # Try to find existing fcurve
                fc = None
                for existing_fc in channelbag.fcurves:
                    if existing_fc.data_path == data_path and existing_fc.array_index == i:
                        fc = existing_fc
                        break
                
                # Create new fcurve if not found
                if fc is None:
                    fc = channelbag.fcurves.new(data_path=data_path, index=i)
                fcs.append(fc)
        except (AttributeError, TypeError, ValueError) as e:
            raise RuntimeError(f"Cannot access fcurves via channel bag in Blender 5.0+: {e}")
    else:
        # Legacy API (Blender < 5.0)
        for i in range(4):
            fc = action.fcurves.find(data_path=data_path, index=i)
            if fc is None:
                fc = action.fcurves.new(data_path=data_path, index=i)
            fcs.append(fc)
    
    return fcs

def set_curve_defaults(fc, interpolation='BEZIER', handle_left='AUTO', handle_right='AUTO'):
    # These are defaults for new points; we’ll also set per-point below.
    fc.auto_smoothing = 'NONE'  # optional; keeps things predictable
    # Note: interpolation/handles are per keyframe_point, not per fcurve.

def insert_quat_key(fcs4, frame, quat, interpolation='BEZIER', handle_type='AUTO'):
    """
    fcs4: [fc_w, fc_x, fc_y, fc_z]
    quat: mathutils.Quaternion or (w,x,y,z)
    """
    w, x, y, z = quat
    vals = (w, x, y, z)

    for fc, v in zip(fcs4, vals):
        kp = fc.keyframe_points.insert(frame, v, options={'FAST'})
        # Make interpolation/handles explicit to match your current behavior.
        kp.interpolation = interpolation
        kp.handle_left_type = handle_type
        kp.handle_right_type = handle_type

def build_fcurve_cache(arm_obj, pose_bones, action_name="ImportedAction"):
    action, anim_data = ensure_action(arm_obj, action_name=action_name)

    cache = {}
    for pbone in pose_bones:
        # Ensure quaternion rotation mode for safety / identicalness
        pbone.rotation_mode = 'QUATERNION'

        fcs4 = ensure_quat_fcurves(action, anim_data, pbone.name)
        for fc in fcs4:
            set_curve_defaults(fc)
        cache[pbone.name] = fcs4

    return action, cache

def breadth_first_pose_application(joint_mapping, frame_idx, suppress_all_missing_warning=False):
    # bpy.context.view_layer.update() is expensive - but needs to run after every layer
    # We can do an O(log n) traversal of the bone hierarchy by going breadth-first

    # TODO: It would be amazing if we could do additive global rotation mapping, so
    # child_rotation = R_global @ parent_R_global @ parent_R_global ... @ root_R_global
    # But I haven't been able to crack it, maybe someone with more vector math knowledge could do it?

    current_layer = [joint_mapping] # hips

    applied_bones = set()
    missing_rotation_bones = []  # Collect bones with missing rotations
    while current_layer:
        next_layer = []
        for bone_dict in current_layer:
            if bone_dict['name'] not in blender_pose_bones:
                print(f"  WARNING: [{bone_dict['name']}] Bone not found in armature")
                continue

            pbone = blender_pose_bones[bone_dict['name']]

            R_global = get_rotation_from_mapping_method(bone_dict, frame_idx)

            if (R_global is None):
                missing_rotation_bones.append(bone_dict['name'])
            else:
                set_pose_from_global_rotation(pbone, R_global)
                applied_bones.add(bone_dict['name'])

            for child in bone_dict.get("children", []):
                next_layer.append(child)

        bpy.context.view_layer.update() # expensive update
        current_layer = next_layer

    # Print grouped warning if there are missing rotations
    # Return True if all rotations are missing (for range consolidation)
    all_missing = len(applied_bones) == 0
    if missing_rotation_bones:
        if all_missing:
            # Only suppress if requested (for range consolidation)
            if not suppress_all_missing_warning:
                print(f"  WARNING: No rotations found for frame {frame_idx + 1}")
        else:
            # Partial missing - always print
            bones_list = ", ".join(missing_rotation_bones)
            print(f"  WARNING: No rotation found for frame {frame_idx + 1} on the following bones: [{bones_list}]")

    # Insert keyframes for bone rotations if pose override is enabled
    if APPLY_POSE_OVERRIDE:
        # inefficient
        # for bone in blender_pose_bones:
        #    bone.keyframe_insert("rotation_quaternion", frame=frame_idx + 1)
        for pbone in blender_pose_bones:
            fcs4 = fcurve_cache[pbone.name]
            insert_quat_key(
                fcs4,
                frame_idx + 1,
                pbone.rotation_quaternion,
                interpolation='BEZIER',
                handle_type='AUTO'
            )

    # Print progress in parseable format for parent process
    print(f"PROGRESS: {frame_idx + 1}/{num_keyframes}", flush=True)
    reset_pose_bones(blender_pose_bones)
    
    return all_missing

# Build fcurve cache if pose override is enabled
if APPLY_POSE_OVERRIDE:
    action, fcurve_cache = build_fcurve_cache(
        arm_obj,
        blender_pose_bones,
        action_name="ImportedAction"
    )
else:
    # Create dummy cache to avoid errors when pose is disabled
    fcurve_cache = {}

# Track consecutive frames with missing rotations for consolidated warnings
missing_rotation_range_start = None
missing_rotation_range_end = None

def flush_missing_rotation_warning():
    """Print consolidated warning for accumulated missing rotation range."""
    global missing_rotation_range_start, missing_rotation_range_end
    if missing_rotation_range_start is not None:
        if missing_rotation_range_start == missing_rotation_range_end:
            print(f"  WARNING: No rotations found for frame {missing_rotation_range_start}")
        else:
            print(f"  WARNING: No rotations found for frame {missing_rotation_range_start}-{missing_rotation_range_end}")
        missing_rotation_range_start = None
        missing_rotation_range_end = None

for frame_idx in range(num_keyframes):
    # Check if all rotations are missing (suppress individual warning, we'll consolidate)
    all_missing = breadth_first_pose_application(joint_mapping, frame_idx, suppress_all_missing_warning=True)
    
    if all_missing:
        # Add to current range or start new range
        frame_num = frame_idx + 1
        if missing_rotation_range_start is None:
            # Start new range
            missing_rotation_range_start = frame_num
            missing_rotation_range_end = frame_num
        elif frame_num == missing_rotation_range_end + 1:
            # Extend current range
            missing_rotation_range_end = frame_num
        else:
            # Gap detected, flush previous range and start new one
            flush_missing_rotation_warning()
            missing_rotation_range_start = frame_num
            missing_rotation_range_end = frame_num
    else:
        # Frame has rotations, flush any accumulated range
        flush_missing_rotation_warning()

# Flush any remaining range at the end
flush_missing_rotation_warning()

bpy.ops.object.mode_set(mode="OBJECT")

# ------------------------------------------------------------------------
# APPLY ROOT MOTION
# ------------------------------------------------------------------------
bpy.ops.object.mode_set(mode="OBJECT")

if APPLY_ROOT_MOTION_OVERRIDE and root_motion is not None and len(root_motion) > 0: # root motion can be passed empty, if the user doesn't want root motion
    # In object mode, use root_motion, which is a list of global rotation euler angles and camera translation vectors
    # apply keyframes to the armature, not any bone
    arm_obj.rotation_mode = 'XYZ'
        
    for root_motion_entry in root_motion:
        # Get frame_index from entry (0-based, matching joint_mapping)
        # Convert to 1-based for Blender keyframes (frame 0 is rest pose)
        frame_index_0based = root_motion_entry.get("frame_index")
        if frame_index_0based is None:
            print(f"  WARNING: root_motion entry missing 'frame_index', skipping entry")
            continue
        
        frame_idx = frame_index_0based + 1  # Convert to 1-based for Blender
        
        cam_translation = root_motion_entry.get("pred_cam_t")
        if cam_translation is None:
            print("  WARNING: root_motion entry missing 'pred_cam_t', skipping entry")
            continue
        cam_translation = list(cam_translation)
        # height is cam_translation[1], use the metadata height_offset to adjust it
        cam_translation[1] += height_offset
        if APPLY_BLENDER_AXIS_ROTATION:
            # armature space has z as backward, y as up. So in the 90 degree X rotation, we swap y and z, and negate z
            final_location = Vector((cam_translation[0], - cam_translation[2], cam_translation[1]))
        else:
            final_location = Vector((cam_translation[0], cam_translation[1], cam_translation[2]))
        arm_obj.location = final_location

        # As far as I can tell, this value of global rotation is already passed to the root bone (hips), but if you'd rather 
        # have rotation be a "root" motion, apply it here and skip in pose application
        # euler = root_motion_entry["global_rot"]
        
        # Each keyframe does need to rotate by math.pi/2 x rotation to "stand up" in blender
        if APPLY_BLENDER_AXIS_ROTATION:
            arm_obj.rotation_euler = (math.pi/2, 0, 0)
        else:
            arm_obj.rotation_euler = (0, 0, 0)

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
if APPLY_BLENDER_AXIS_ROTATION:
    arm_obj.rotation_euler = (math.pi/2, 0, 0)
else:
    arm_obj.rotation_euler = (0, 0, 0)
arm_obj.keyframe_insert(data_path="location", frame=0)
arm_obj.keyframe_insert(data_path="rotation_euler", frame=0)

bpy.ops.object.select_all(action='DESELECT')
arm_obj.select_set(True)

# Find and select mesh if it exists (could be from reskinning or already parented)
export_mesh_obj = None
if mesh_obj:
    # Use the mesh_obj returned from load_and_reskin_mesh
    export_mesh_obj = mesh_obj
else:
    # Try to find mesh parented to armature (in case mesh_obj wasn't tracked)
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.parent == arm_obj:
            export_mesh_obj = obj
            break

# Select mesh for export if found
if export_mesh_obj:
    export_mesh_obj.select_set(True)
    print(f"✓ Mesh '{export_mesh_obj.name}' selected for export")
else:
    print("No mesh found for export (armature-only)")

bpy.context.view_layer.objects.active = arm_obj

bpy.context.scene.frame_set(0)

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end   = num_keyframes

# Ensure mesh is properly evaluated before export (critical for reskinned meshes)
if export_mesh_obj:
    export_mesh_obj.update_tag()
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()

print("=== EXPORTING FBX ===")
bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=True,
    use_mesh_modifiers=True,  # Bake mesh modifiers (including armature) into geometry
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