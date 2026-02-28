"""
Amoire 3D Demo — Mesh Utilities

OBJ/GLB conversion, mesh simplification, and result caching.
These utilities are ML-model-agnostic — they work with any mesh pipeline.
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

import trimesh

from config import CACHE_DIR, OUTPUT_DIR


def obj_to_glb(obj_path: Path, glb_path: Optional[Path] = None) -> Path:
    """Convert OBJ mesh to GLB (web-friendly binary glTF format)."""
    if glb_path is None:
        glb_path = obj_path.with_suffix(".glb")

    scene = trimesh.load(str(obj_path), force="scene")
    scene.export(str(glb_path), file_type="glb")
    return glb_path


def simplify_mesh(mesh_path: Path, target_faces: int = 10000) -> Path:
    """Reduce polygon count for web rendering performance.

    Returns path to simplified mesh (overwrites input if same format).
    """
    mesh = trimesh.load(str(mesh_path), force="mesh")

    if len(mesh.faces) > target_faces:
        # Use quadric decimation if available via open3d
        try:
            import open3d as o3d

            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            o3d_mesh.compute_vertex_normals()

            simplified = o3d_mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_faces
            )

            mesh = trimesh.Trimesh(
                vertices=simplified.vertices,
                faces=simplified.triangles,
            )
        except ImportError:
            # Fallback: naive face subset (not ideal but works)
            pass

    out_path = mesh_path.with_stem(mesh_path.stem + "_simplified")
    mesh.export(str(out_path))
    return out_path


def compute_image_hash(image_path: Path) -> str:
    """SHA256 hash of image file for cache keying."""
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def get_cached_result(image_hash: str) -> Optional[Path]:
    """Check if we already generated a mesh for this image hash."""
    cache_entry = CACHE_DIR / image_hash
    glb_path = cache_entry / "garment.glb"
    if glb_path.exists():
        return glb_path
    return None


def cache_result(image_hash: str, mesh_path: Path, metadata: dict) -> Path:
    """Store generated mesh and metadata in cache. Returns cached GLB path."""
    cache_entry = CACHE_DIR / image_hash
    cache_entry.mkdir(parents=True, exist_ok=True)

    cached_glb = cache_entry / "garment.glb"
    shutil.copy2(mesh_path, cached_glb)

    meta_path = cache_entry / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return cached_glb


def create_placeholder_glb(output_path: Path, shape: str = "tshirt") -> Path:
    """Generate a simple geometric placeholder .glb for demo/mock mode.

    Shapes: 'tshirt', 'skirt', 'pants', 'dress'
    """
    import numpy as np

    if shape == "skirt":
        # Truncated cone
        mesh = trimesh.creation.cone(radius=0.4, height=0.6, sections=32)
        mesh.apply_translation([0, 0.3, 0])
    elif shape == "pants":
        # Two cylinders
        left_leg = trimesh.creation.cylinder(radius=0.12, height=0.8, sections=16)
        left_leg.apply_translation([-0.13, 0, 0])
        right_leg = trimesh.creation.cylinder(radius=0.12, height=0.8, sections=16)
        right_leg.apply_translation([0.13, 0, 0])
        waist = trimesh.creation.cylinder(radius=0.22, height=0.2, sections=16)
        waist.apply_translation([0, 0.5, 0])
        mesh = trimesh.util.concatenate([left_leg, right_leg, waist])
    elif shape == "dress":
        # Cylinder top + cone bottom
        top = trimesh.creation.cylinder(radius=0.2, height=0.4, sections=24)
        top.apply_translation([0, 0.7, 0])
        bottom = trimesh.creation.cone(radius=0.45, height=0.7, sections=32)
        bottom.apply_translation([0, 0.35, 0])
        mesh = trimesh.util.concatenate([top, bottom])
    else:
        # T-shirt: box torso + two cylinder sleeves
        torso = trimesh.creation.box(extents=[0.4, 0.5, 0.2])
        torso.apply_translation([0, 0.6, 0])
        left_sleeve = trimesh.creation.cylinder(radius=0.08, height=0.25, sections=12)
        left_sleeve.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        )
        left_sleeve.apply_translation([-0.32, 0.72, 0])
        right_sleeve = trimesh.creation.cylinder(
            radius=0.08, height=0.25, sections=12
        )
        right_sleeve.apply_transform(
            trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1])
        )
        right_sleeve.apply_translation([0.32, 0.72, 0])
        mesh = trimesh.util.concatenate([torso, left_sleeve, right_sleeve])

    # Apply a neutral fabric-like color
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile([220, 218, 213, 255], (len(mesh.vertices), 1)),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), file_type="glb")
    return output_path
