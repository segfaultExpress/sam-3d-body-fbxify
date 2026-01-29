import bpy
import os
import sys
import json
import math
import shutil
import tempfile
from mathutils import Matrix, Vector

CAMERA_OBJECT_NAME = "Video_Cam"
VIDEO_PLANE_OBJECT_NAME = "Video"
MATERIAL_NAME = "Video_Mat"


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_obj(name: str):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise ValueError(f"Object '{name}' not found.")
    return obj


def create_camera_scene():
    """
    I don't think I will ever truly understand blender's coordinate system, and how it can't seem to even
    interact with itself? I tried to save this as a scene, but it imported wonky no matter what I did.
    So instead, we're just hardcoding all the camera values. Weeeee.
    """
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = CAMERA_OBJECT_NAME
    cam.rotation_mode = 'XYZ'
    cam.location = (0.0, 0.0, 0.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    cam.scale = (1.0, 1.0, 1.0)

    bpy.ops.object.empty_add(type='PLAIN_AXES')
    scaling_offset = bpy.context.active_object
    scaling_offset.name = "Scaling_Offset"
    scaling_offset.parent = cam
    scaling_offset.rotation_mode = 'XYZ'
    scaling_offset.location = (0.0, 0.0, -500.0)

    scaling_offset.rotation_euler = (0, 0, 0)
    scaling_offset.scale = (0.180, 0.180, 1.0)

    bpy.ops.mesh.primitive_plane_add()
    plane = bpy.context.active_object
    plane.name = VIDEO_PLANE_OBJECT_NAME
    plane.parent = scaling_offset
    plane.rotation_mode = 'XYZ'
    plane.location = (0.0, 0.0, 0.0)
    plane.rotation_euler = (0.0, 0.0, 0.0)
    plane.scale = (1000.0, 1000.0, 1.0)
    return cam, scaling_offset, plane


def ensure_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    return mat


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".exr"}


def _is_video_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTENSIONS


def _list_sequence_frames(media_dir: str) -> list:
    if not os.path.isdir(media_dir):
        return []
    frame_paths = []
    for name in os.listdir(media_dir):
        path = os.path.join(media_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            frame_paths.append(path)
    return sorted(frame_paths, key=lambda p: os.path.basename(p).lower())


def load_media_image(media_path: str) -> bpy.types.Image:
    print(f"[OK] Loading media image from {media_path}...")
    if not os.path.exists(media_path):
        raise FileNotFoundError(f"Media file not found: {media_path}")
    if os.path.isdir(media_path):
        frame_paths = _list_sequence_frames(media_path)
        if not frame_paths:
            raise FileNotFoundError(f"No image frames found in: {media_path}")
        img = bpy.data.images.load(frame_paths[0], check_existing=False)
        img.source = 'SEQUENCE'
        try:
            img.filepath = frame_paths[0]
            img.filepath_raw = frame_paths[0]
            img.reload()
        except Exception:
            pass
        try:
            user = img.image_user
            user.frame_start = bpy.context.scene.frame_start
            user.frame_offset = 0
            user.frame_duration = len(frame_paths)
            user.use_auto_refresh = True
        except Exception:
            pass
        return img
    img = bpy.data.images.load(media_path, check_existing=False)
    img.source = 'MOVIE' if _is_video_path(media_path) else 'FILE'
    return img


def set_scene_resolution(width: int, height: int):
    scene = bpy.context.scene
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100


def scale_video_object_to_aspect(video_obj_name: str, width: int, height: int):
    obj = bpy.data.objects.get(video_obj_name)
    if obj is None:
        raise ValueError(f"Video object '{video_obj_name}' not found.")

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid resolution {width}x{height}")

    # The larger dimension always needs to stay 1000, the smaller dimension will be scaled to maintain the aspect ratio
    # i.e. 1920x1080 -> 1000x562.5
    # i.e. 1080x1920 -> 562.5x1000

    base_x = obj.scale.x
    base_y = obj.scale.y
    if width > height:
        obj.scale.x = base_x
        obj.scale.y = (height / width) * base_y
    else:
        obj.scale.x = (width / height) * base_x
        obj.scale.y = base_y
    return obj


def build_video_material(
    mat: bpy.types.Material,
    movie_img: bpy.types.Image,
    frame_count_override: int | None = None,
):
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (250, 0)

    tex = nodes.new("ShaderNodeTexImage")
    tex.location = (0, 0)
    tex.image = movie_img
    tex.interpolation = 'Linear'

    base_color = bsdf.inputs.get("Base Color")
    if base_color:
        links.new(tex.outputs["Color"], base_color)

    emission_color = (bsdf.inputs.get("Emission") or bsdf.inputs.get("Emission Color"))
    if emission_color:
        links.new(tex.outputs["Color"], emission_color)

    emission_strength = bsdf.inputs.get("Emission Strength")
    if emission_strength and hasattr(emission_strength, "default_value"):
        emission_strength.default_value = 1.0

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    user = tex.image_user
    user.use_cyclic = False
    user.frame_start = bpy.context.scene.frame_start
    user.frame_offset = 0

    user.use_auto_refresh = False
    user.frame_duration = 1
    return 1


def assign_material_to_object(obj: bpy.types.Object, mat: bpy.types.Material):
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


def put_media_on_plane(
    media_path: str,
    plane_obj_name: str = VIDEO_PLANE_OBJECT_NAME,
    material_name: str = MATERIAL_NAME,
    frame_count_override: int | None = None,
):
    plane = get_obj(plane_obj_name)
    mat = ensure_material(material_name)
    movie = load_media_image(media_path)
    frame_count = build_video_material(
        mat,
        movie,
        frame_count_override=frame_count_override,
    )
    assign_material_to_object(plane, mat)

    if hasattr(movie, "size") and len(movie.size) >= 2:
        width, height = movie.size[0], movie.size[1]
        if width and height:
            set_scene_resolution(width, height)
            scale_video_object_to_aspect(plane_obj_name, width, height)

    print(f"[OK] Assigned media '{media_path}' to '{plane_obj_name}'.")
    return frame_count, movie


def set_camera_keyframe(cam: bpy.types.Object, frame: int):
    cam.keyframe_insert(data_path="location", frame=frame)
    cam.keyframe_insert(data_path="rotation_quaternion", frame=frame)

def apply_camera_offsets(cam: bpy.types.Object, height_offset: float, camera_zoom: float):
    world_height = Vector((0.0, 0.0, float(height_offset)))
    local_zoom = Vector((0.0, 0.0, float(camera_zoom))) # Local Z will usually be different from world Z, so no worries with mixing with world height
    cam.location = Vector(cam.location) + world_height
    cam.location = Vector(cam.location) - (cam.matrix_world.to_3x3() @ local_zoom)


def apply_default_keyframes(cam: bpy.types.Object, height_offset: float, camera_zoom: float):
    print(f"[OK] Applying default keyframes...")
    cam.rotation_mode = 'QUATERNION'
    cam.location = (0, 0, 0)

    cam.rotation_quaternion = (1, 0, 0, 0)
    set_camera_keyframe(cam, 1)

    cam.location = (0, 0, 0)
    cam.matrix_world = Matrix.Rotation(math.radians(90.0), 4, 'X')
    apply_camera_offsets(cam, height_offset, camera_zoom)
    set_camera_keyframe(cam, 2)


def apply_extrinsics_keyframes(cam: bpy.types.Object,
                               frame_extrinsics: list,
                               height_offset: float,
                               camera_zoom: float,
                               keyframe_stride: int):
    print(f"[OK] Applying extrinsics keyframes...")
    cam.rotation_mode = 'QUATERNION'
    cam.location = (0, 0, 0)
    cam.rotation_quaternion = (1, 0, 0, 0)
    set_camera_keyframe(cam, 1)

    stride = max(int(keyframe_stride), 1)
    basis = Matrix.Rotation(math.radians(90.0), 4, 'X')

    for frame_idx, frame_entry in enumerate(frame_extrinsics):
        if frame_idx % stride != 0:
            continue
        T_wc = frame_entry.get("T_wc")
        if T_wc is None:
            continue

        frame = frame_idx + 2
        bpy.context.scene.frame_set(frame)

        T_wc_armature = Matrix(T_wc)
        T_wc_blender = basis @ T_wc_armature

        cam.matrix_world = T_wc_blender
        apply_camera_offsets(cam, height_offset, camera_zoom)
        set_camera_keyframe(cam, frame)

def export_scene(output_path: str):
    output_path = os.path.abspath(output_path)
    bpy.ops.wm.save_as_mainfile(filepath=output_path)

def main():
    print("Building camera scene...")
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    print(f"[OK] Parsing arguments...")
    output_path = args[0] if len(args) > 0 else None
    metadata_path = args[1] if len(args) > 1 else None
    camera_scene_path = args[2] if len(args) > 2 else None
    camera_zoom = float(args[3]) if len(args) > 3 and args[3] else 0.0
    extrinsics_json_path = args[4] if len(args) > 4 and args[4] else None

    if output_path is None and camera_scene_path:
        output_path = os.path.join(os.path.dirname(camera_scene_path), "camera_scene.blend")
    elif output_path is None:
        output_path = "camera_scene.blend"

    if metadata_path is None:
        height_offset = 5.0
        fps = 30.0
        num_keyframes = 0
        keyframe_stride = 1
    else:
        metadata_payload = load_json(metadata_path)
        metadata = metadata_payload.get("metadata", metadata_payload)
        height_offset = float(metadata.get("height_offset", 0.0))
        fps = metadata.get("fps")
        num_keyframes = int(metadata.get("num_keyframes", 0))
        keyframe_stride = int(metadata.get("extrinsics_sample_rate", 0)) or 1

    if fps:
        bpy.context.scene.render.fps = int(fps)


    # ------------------------------------------------------------------------
    # RESET SCENE
    # ------------------------------------------------------------------------

    print(f"[OK] Creating camera scene...")
    bpy.ops.wm.read_factory_settings(use_empty=True)

    create_camera_scene()

    media_img = None
    if camera_scene_path:
        try:
            frame_count_override = None
            if metadata_path:
                frame_count_override = int(metadata.get("num_keyframes", 0) or 0)
            frame_count, media_img = put_media_on_plane(
                camera_scene_path,
                frame_count_override=frame_count_override,
            )
            if num_keyframes <= 0:
                num_keyframes = frame_count
            
            bpy.context.view_layer.update()

            # pack the media
            bpy.ops.file.pack_all()
            bpy.ops.file.make_paths_absolute()

        except Exception as e:
            print(f"[WARN] Failed to load camera scene media: {e}")

    cam = get_obj(CAMERA_OBJECT_NAME)

    if extrinsics_json_path and os.path.exists(extrinsics_json_path):
        extrinsics_payload = load_json(extrinsics_json_path)
        frame_extrinsics = extrinsics_payload.get("frame_extrinsics", [])
        apply_extrinsics_keyframes(
            cam,
            frame_extrinsics,
            height_offset=height_offset,
            camera_zoom=camera_zoom,
            keyframe_stride=keyframe_stride,
        )
    else:
        apply_default_keyframes(
            cam,
            height_offset=height_offset,
            camera_zoom=camera_zoom,
        )

    if num_keyframes > 0:
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = int(num_keyframes)

    export_scene(output_path)
    print(f"[OK] Camera scene exported: {output_path}")


if __name__ == "__main__":
    main()
