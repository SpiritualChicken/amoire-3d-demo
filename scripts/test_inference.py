#!/usr/bin/env python3
"""Standalone test: image → 3D mesh.

Tests the full inference pipeline with a sample image.

Usage:
    python scripts/test_inference.py --image path/to/garment.jpg
    python scripts/test_inference.py --mock  # test with mock inference
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


async def main():
    parser = argparse.ArgumentParser(description="Test Amoire inference pipeline")
    parser.add_argument("--image", type=str, help="Path to a garment image")
    parser.add_argument("--mock", action="store_true", help="Use mock inference")
    args = parser.parse_args()

    if args.mock:
        import config
        config.MOCK_MODE = True

    from inference import generate_3d_garment, load_model

    print("Loading model...")
    load_model()

    if args.image:
        image_path = Path(args.image)
    else:
        # Use a placeholder
        from PIL import Image
        import config
        image_path = config.UPLOAD_DIR / "test_input.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (336, 336), (200, 180, 160))
        img.save(image_path)
        print(f"Created test image: {image_path}")

    print(f"Running inference on: {image_path}")
    result = await generate_3d_garment(image_path, job_id="test_001")

    print("\n=== Results ===")
    print(f"  Mesh:       {result['mesh_path']}")
    print(f"  Type:       {result['garment_type']}")
    print(f"  Time:       {result['processing_time_seconds']:.1f}s")
    print(f"  JSON keys:  {list(result['garment_json'].keys())}")

    if result["mesh_path"].exists():
        size_kb = result["mesh_path"].stat().st_size / 1024
        print(f"  File size:  {size_kb:.1f} KB")
        print("\nSuccess!")
    else:
        print("\nERROR: Mesh file was not created")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
