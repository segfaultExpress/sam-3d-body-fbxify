import os
import sys

import bpy
import numpy as np


def _parse_args():
    if "--" not in sys.argv:
        raise RuntimeError("Missing '--' separator for script args.")
    idx = sys.argv.index("--") + 1
    args = sys.argv[idx:]
    if len(args) < 2:
        raise RuntimeError("Usage: extract_fbx_faces.py -- <fbx_path> <out_npy>")
    return args[0], args[1]


def _find_mesh_objects():
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not meshes:
        return []
    meshes.sort(key=lambda o: len(o.data.vertices), reverse=True)
    return meshes


def _faces_from_mesh(mesh_obj):
    me = mesh_obj.data
    faces = []
    for poly in me.polygons:
        verts = list(poly.vertices)
        if len(verts) == 3:
            faces.append(verts)
        elif len(verts) > 3:
            # Fan triangulation
            for i in range(1, len(verts) - 1):
                faces.append([verts[0], verts[i], verts[i + 1]])
    return np.array(faces, dtype=np.int64)


def main():
    fbx_path, out_path = _parse_args()
    if not os.path.exists(fbx_path):
        raise RuntimeError(f"FBX not found: {fbx_path}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    meshes = _find_mesh_objects()
    if not meshes:
        raise RuntimeError("No mesh objects found in FBX.")

    # Use largest mesh; this matches the body mesh in LOD FBXs.
    mesh_obj = meshes[0]
    faces = _faces_from_mesh(mesh_obj)
    if faces.size == 0:
        raise RuntimeError("No faces found on mesh.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, faces)
    print(f"Saved faces: {faces.shape} -> {out_path}")


if __name__ == "__main__":
    main()
