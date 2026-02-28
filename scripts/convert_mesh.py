#!/usr/bin/env python3
"""Batch OBJ → GLB converter.

Converts all .obj files in a directory to .glb format.

Usage:
    python scripts/convert_mesh.py data/outputs/
    python scripts/convert_mesh.py path/to/file.obj
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from mesh_utils import obj_to_glb, simplify_mesh


def main():
    parser = argparse.ArgumentParser(description="Convert OBJ meshes to GLB")
    parser.add_argument("path", type=str, help="OBJ file or directory to convert")
    parser.add_argument("--simplify", type=int, default=0, help="Target face count (0 = no simplification)")
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = list(target.rglob("*.obj"))
    else:
        print(f"Path not found: {target}")
        sys.exit(1)

    if not files:
        print("No .obj files found")
        sys.exit(0)

    print(f"Converting {len(files)} mesh(es)...\n")

    for obj_path in files:
        try:
            if args.simplify > 0:
                simplified = simplify_mesh(obj_path, target_faces=args.simplify)
                glb = obj_to_glb(simplified)
            else:
                glb = obj_to_glb(obj_path)

            size_kb = glb.stat().st_size / 1024
            print(f"  {obj_path.name} → {glb.name} ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"  {obj_path.name} → FAILED: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
