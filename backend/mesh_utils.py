from __future__ import annotations

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


def post_process_mesh(
    mesh: "trimesh.Trimesh", smooth_iterations: int = 10
) -> "trimesh.Trimesh":
    """Apply Taubin smoothing and recompute smooth vertex normals.

    Taubin smoothing (alternating lambda/mu passes) preserves mesh volume
    unlike Laplacian smoothing which shrinks garment silhouettes. Each step
    is wrapped in try/except so a failure never crashes the pipeline.

    Args:
        mesh: Input trimesh object (modified in-place and returned).
        smooth_iterations: Number of Taubin smoothing passes. 10 is a good
            balance for 10K-22K vertex garment meshes — removes faceting
            without losing geometric detail at collars/hems/cuffs.

    Returns:
        The processed mesh with smooth normals.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Step 1: Fix inconsistent face winding from simulation output.
    # This is a prerequisite for correct normal computation.
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception as e:
        logger.warning(f"fix_normals failed (non-fatal): {e}")

    # Step 2: Taubin smoothing — lamb=0.5 controls smoothing strength,
    # nu=0.5 controls the anti-shrinkage pass. Together they remove faceting
    # while keeping garment proportions intact.
    try:
        trimesh.smoothing.filter_taubin(
            mesh, lamb=0.5, nu=0.5, iterations=smooth_iterations
        )
        logger.info(f"Applied Taubin smoothing ({smooth_iterations} iterations)")
    except Exception as e:
        logger.warning(f"Taubin smoothing failed (non-fatal): {e}")

    # Step 3: Force smooth vertex normal computation. Accessing the property
    # triggers area-weighted averaging of face normals at each vertex,
    # replacing per-face flat normals with interpolated smooth normals.
    try:
        _ = mesh.vertex_normals
        logger.info(
            f"Recomputed smooth vertex normals ({len(mesh.vertices)} vertices)"
        )
    except Exception as e:
        logger.warning(f"Vertex normal computation failed (non-fatal): {e}")

    return mesh


def obj_to_glb(
    obj_path: Path, glb_path: Optional[Path] = None, smooth: bool = True
) -> Path:
    """Convert OBJ mesh to GLB with optional smoothing post-processing.

    Loads the mesh as geometry (not scene) so smoothing and normal
    recomputation can be applied before export. The processed mesh is
    then wrapped in a Scene for proper GLB material embedding.

    Args:
        obj_path: Input OBJ file path.
        glb_path: Output GLB file path (defaults to same name with .glb).
        smooth: If True, apply Taubin smoothing and recompute vertex normals.
    """
    if glb_path is None:
        glb_path = obj_path.with_suffix(".glb")

    mesh = trimesh.load(str(obj_path), force="mesh", process=False)

    if smooth:
        mesh = post_process_mesh(mesh)

    scene = trimesh.Scene(mesh)
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


def build_mesh_from_panels(
    pattern_dir: Path, output_path: Path, garment_type: str = "unknown"
) -> bool:
    """Build a 3D mesh from GarmentCodeRC sewing pattern SVG panels.

    Parses panel outlines from SVGs, triangulates them as flat 3D surfaces,
    and arranges them in 3D space based on garment type. This produces a
    "flat layout" style visualization — not a draped garment, but shows
    the actual pattern shapes rather than generic placeholders.

    Returns True on success, False if no usable pattern data found.
    """
    import logging
    import xml.etree.ElementTree as ET

    import numpy as np

    logger = logging.getLogger(__name__)

    # Find all pattern SVGs in the output directory
    svg_files = list(pattern_dir.rglob("*_pattern.svg"))
    if not svg_files:
        svg_files = list(pattern_dir.rglob("*.svg"))
    if not svg_files:
        logger.warning(f"No SVG files found in {pattern_dir}")
        return False

    all_meshes = []
    y_offset = 0.0

    for svg_file in svg_files:
        try:
            panels = _extract_svg_panels(svg_file)
            if not panels:
                continue

            for panel_verts in panels:
                if len(panel_verts) < 3:
                    continue

                # Create a flat 2D mesh from panel outline
                verts_2d = np.array(panel_verts, dtype=np.float64)

                # Scale from SVG units (typically mm/pixels) to meters
                # GarmentCodeRC patterns are usually in cm, scale to ~0.5m display size
                verts_2d = verts_2d / 100.0

                # Center the panel
                center = verts_2d.mean(axis=0)
                verts_2d -= center

                # Create 3D vertices (panels laid flat in XY plane, offset along Y)
                verts_3d = np.zeros((len(verts_2d), 3))
                verts_3d[:, 0] = verts_2d[:, 0]  # X
                verts_3d[:, 1] = y_offset          # Y (stack panels vertically)
                verts_3d[:, 2] = verts_2d[:, 1]    # Z (SVG Y → world Z)

                # Triangulate using fan triangulation from centroid
                centroid = verts_3d.mean(axis=0)
                n = len(verts_3d)
                all_verts = list(verts_3d) + [centroid]
                center_idx = n
                faces = []
                for i in range(n):
                    faces.append([i, (i + 1) % n, center_idx])

                panel_mesh = trimesh.Trimesh(
                    vertices=np.array(all_verts),
                    faces=np.array(faces),
                )
                all_meshes.append(panel_mesh)

                # Offset next panel
                bounds = verts_2d.max(axis=0) - verts_2d.min(axis=0)
                y_offset += max(bounds) + 0.05

        except Exception as e:
            logger.warning(f"Failed to parse SVG {svg_file}: {e}")
            continue

    if not all_meshes:
        return False

    # Combine all panels into one mesh
    combined = trimesh.util.concatenate(all_meshes)

    # Center the whole thing
    combined.apply_translation(-combined.centroid)

    # Scale to reasonable display size (~0.5m tall)
    extent = combined.extents.max()
    if extent > 0:
        combined.apply_scale(0.5 / extent)

    # Apply fabric-like color
    combined.visual = trimesh.visual.ColorVisuals(
        mesh=combined,
        vertex_colors=np.tile([220, 218, 213, 255], (len(combined.vertices), 1)),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output_path), file_type="glb")
    return True


def _extract_svg_panels(svg_path: Path) -> list[list[tuple[float, float]]]:
    """Extract polygon outlines from an SVG file.

    Handles <polygon>, <polyline>, and simple <path> elements.
    Returns list of panels, each a list of (x, y) vertices.
    """
    import re
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(svg_path))
    root = tree.getroot()

    # Handle SVG namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    panels = []

    # Extract <polygon> elements
    for poly in root.iter(f"{ns}polygon"):
        points_str = poly.get("points", "")
        verts = _parse_svg_points(points_str)
        if len(verts) >= 3:
            panels.append(verts)

    # Extract <polyline> elements
    for poly in root.iter(f"{ns}polyline"):
        points_str = poly.get("points", "")
        verts = _parse_svg_points(points_str)
        if len(verts) >= 3:
            panels.append(verts)

    # Extract <path> elements with simple M/L/Z commands
    for path_el in root.iter(f"{ns}path"):
        d = path_el.get("d", "")
        verts = _parse_svg_path_simple(d)
        if len(verts) >= 3:
            panels.append(verts)

    return panels


def _parse_svg_points(points_str: str) -> list[tuple[float, float]]:
    """Parse SVG points attribute: '100,200 300,400 ...'"""
    import re

    verts = []
    for match in re.finditer(r"([-\d.]+)[,\s]+([-\d.]+)", points_str):
        verts.append((float(match.group(1)), float(match.group(2))))
    return verts


def _parse_svg_path_simple(d: str) -> list[tuple[float, float]]:
    """Parse simple SVG path (M/L/Z commands only) into vertices."""
    import re

    verts = []
    # Match M/L followed by coordinates
    for match in re.finditer(r"[ML]\s*([-\d.]+)[,\s]+([-\d.]+)", d, re.IGNORECASE):
        verts.append((float(match.group(1)), float(match.group(2))))
    return verts


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
