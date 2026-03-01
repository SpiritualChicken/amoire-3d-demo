from __future__ import annotations

"""
Amoire 3D Demo — Inference Pipeline

Wraps ChatGarment's ML pipeline into a clean async interface.
This is the ONLY file that needs to change when swapping ML models.

Pipeline: Image → VLM (LLaVA + LoRA) → GarmentCode JSON → Sewing Patterns → 3D Mesh

For mock mode, this returns placeholder meshes without any ML dependencies.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from config import (
    CHATGARMENT_DIR,
    CONTOURCRAFT_DIR,
    DEVICE,
    GARMENTCODE_DIR,
    LLAVA_MODEL_PATH,
    LORA_WEIGHTS_PATH,
    MOCK_MODE,
    OUTPUT_DIR,
    USE_4BIT_QUANTIZATION,
)

logger = logging.getLogger(__name__)

# Global model state
_model = None
_tokenizer = None
_image_processor = None
model_loaded = False


# ---------------------------------------------------------------------------
# Mock inference (no GPU required)
# ---------------------------------------------------------------------------

async def mock_generate_3d_garment(image_path: Path, job_id: str) -> dict:
    """Return a placeholder mesh. Used when --mock flag is set or no GPU available."""
    from mesh_utils import create_placeholder_glb

    import random

    shape = random.choice(["tshirt", "skirt", "pants", "dress"])
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    glb_path = output_dir / "garment.glb"
    create_placeholder_glb(glb_path, shape=shape)

    # Simulate processing time
    await asyncio.sleep(3)

    return {
        "mesh_path": glb_path,
        "sewing_pattern_svg": None,
        "garment_json": {"type": shape, "mock": True},
        "processing_time_seconds": 3.0,
        "garment_type": shape,
    }


# ---------------------------------------------------------------------------
# Real ML inference
# ---------------------------------------------------------------------------

def load_model():
    """Load LLaVA base model + ChatGarment LoRA weights into GPU memory.

    Call once at server startup. Keeps model resident for fast inference.
    """
    global _model, _tokenizer, _image_processor, model_loaded

    if MOCK_MODE:
        logger.info("Mock mode enabled — skipping model loading")
        model_loaded = True
        return

    logger.info("Loading ChatGarment model...")

    # Add ChatGarment to Python path
    chatgarment_path = str(CHATGARMENT_DIR)
    if chatgarment_path not in sys.path:
        sys.path.insert(0, chatgarment_path)

    try:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        model_name = "llava-v1.5-7b-lora"

        # Load with 4-bit quantization to reduce VRAM (~14GB → ~6GB)
        load_kwargs = {}
        if USE_4BIT_QUANTIZATION:
            load_kwargs["load_4bit"] = True
            logger.info("Using 4-bit quantization")

        _tokenizer, _model, _image_processor, _ = load_pretrained_model(
            model_path=str(LORA_WEIGHTS_PATH),
            model_base=str(LLAVA_MODEL_PATH),
            model_name=model_name,
            device=DEVICE,
            **load_kwargs,
        )

        model_loaded = True
        logger.info("Model loaded successfully")

    except ImportError as e:
        logger.error(
            f"Failed to import ChatGarment modules: {e}. "
            "Make sure ml/ChatGarment is cloned and dependencies installed."
        )
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def _run_vlm_inference(image_path: Path) -> tuple[str, dict]:
    """Two-step Chain-of-Thought inference.

    Step 1: Image → text description of garment(s)
    Step 2: Image + description → GarmentCode JSON

    Returns (description_text, garmentcode_json_dict).
    """
    from PIL import Image as PILImage

    sys.path.insert(0, str(CHATGARMENT_DIR))
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    import torch

    image = PILImage.open(image_path).convert("RGB")
    image_tensor = process_images([image], _image_processor, _model.config)
    if isinstance(image_tensor, list):
        image_tensor = [t.to(DEVICE, dtype=torch.float16) for t in image_tensor]
    else:
        image_tensor = image_tensor.to(DEVICE, dtype=torch.float16)

    # Step 1: Generate text description
    conv = conv_templates["v1"].copy()
    prompt_step1 = (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        "Describe the garment(s) in this image in detail, including type, "
        "shape, length, fit, neckline, sleeves, and any distinctive features."
    )
    conv.append_message(conv.roles[0], prompt_step1)
    conv.append_message(conv.roles[1], None)

    input_ids = tokenizer_image_token(
        conv.get_prompt(), _tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output_ids = _model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=512,
            use_cache=True,
        )

    description = _tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    logger.info(f"Step 1 description: {description[:200]}...")

    # Step 2: Generate GarmentCode JSON
    conv2 = conv_templates["v1"].copy()
    prompt_step2 = (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        f"The garment in this image is described as: {description}\n\n"
        "Based on this image and description, generate the GarmentCode JSON "
        "specification for this garment. Output ONLY valid JSON."
    )
    conv2.append_message(conv2.roles[0], prompt_step2)
    conv2.append_message(conv2.roles[1], None)

    input_ids2 = tokenizer_image_token(
        conv2.get_prompt(), _tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output_ids2 = _model.generate(
            input_ids2,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=2048,
            use_cache=True,
            repetition_penalty=1.2,
        )

    json_text = _tokenizer.batch_decode(output_ids2, skip_special_tokens=True)[0].strip()

    # Extract JSON from response (may have markdown fences)
    if "```json" in json_text:
        json_text = json_text.split("```json")[1].split("```")[0].strip()
    elif "```" in json_text:
        json_text = json_text.split("```")[1].split("```")[0].strip()

    # Clean up common model output issues before parsing
    def _truncate_to_valid(text: str) -> str:
        """Truncate text to last complete JSON object by balancing braces."""
        start = text.find("{")
        if start < 0:
            return text
        depth = 0
        last_valid_end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    last_valid_end = i + 1
                    break
        if last_valid_end > start:
            return text[start:last_valid_end]
        return text

    def _remove_duplicate_keys(text: str) -> str:
        """Remove repeated key-value pairs caused by model repetition loops."""
        import re
        # Match repeated key-value patterns
        text = re.sub(r"((['\"][^'\"]+['\"])\s*:\s*[^,}]+,?\s*)\1+", r"\1", text)
        return text

    def _clean_json_text(text: str) -> str:
        """Fix Python-style dict output to valid JSON."""
        import ast
        text = _remove_duplicate_keys(text)
        text = _truncate_to_valid(text)
        # Try Python literal eval first (handles single quotes, True/False/None)
        try:
            py_dict = ast.literal_eval(text)
            return json.dumps(py_dict)
        except (ValueError, SyntaxError):
            pass
        # Manual fixes
        text = text.replace("'", '"')
        text = text.replace("True", "true").replace("False", "false").replace("None", "null")
        return text

    try:
        garment_json = json.loads(json_text)
    except json.JSONDecodeError:
        logger.warning("First JSON parse failed, retrying with cleanup...")
        cleaned = _clean_json_text(json_text)
        try:
            garment_json = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    garment_json = json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    raise ValueError(f"Could not parse GarmentCode JSON from model output: {json_text[:500]}")
            else:
                raise ValueError(f"Could not parse GarmentCode JSON from model output: {json_text[:500]}")

    return description, garment_json


def _run_garmentcode_simulation(garment_json: dict, output_dir: Path) -> Path:
    """Run GarmentCodeRC to generate 2D sewing patterns from GarmentCode JSON.

    Returns path to the sewing pattern data directory.
    """
    sys.path.insert(0, str(GARMENTCODE_DIR))

    json_path = output_dir / "garment_spec.json"
    with open(json_path, "w") as f:
        json.dump(garment_json, f, indent=2)

    pattern_dir = output_dir / "sewing_patterns"
    pattern_dir.mkdir(exist_ok=True)

    # Shell out to GarmentCodeRC's simulation script
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            str(GARMENTCODE_DIR / "pattern_gen.py"),
            "--config", str(json_path),
            "--output", str(pattern_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(GARMENTCODE_DIR),
    )

    if result.returncode != 0:
        logger.error(f"GarmentCodeRC failed: {result.stderr}")
        raise RuntimeError(f"Sewing pattern generation failed: {result.stderr[:500]}")

    return pattern_dir


def _run_contourcraft_draping(pattern_dir: Path, output_dir: Path) -> Path:
    """Run ContourCraft-CG to drape sewing patterns onto a body mesh.

    Returns path to the output OBJ mesh.
    """
    obj_path = output_dir / "garment.obj"

    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            str(CONTOURCRAFT_DIR / "drape.py"),
            "--patterns", str(pattern_dir),
            "--output", str(obj_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(CONTOURCRAFT_DIR),
    )

    if result.returncode != 0:
        logger.error(f"ContourCraft failed: {result.stderr}")
        raise RuntimeError(f"3D draping simulation failed: {result.stderr[:500]}")

    return obj_path


async def generate_3d_garment(
    image_path: Path,
    job_id: str,
    progress_callback: Optional[callable] = None,
) -> dict:
    """Full pipeline: image → 3D mesh (.glb).

    Args:
        image_path: Path to the uploaded image file.
        job_id: Unique identifier for this generation job.
        progress_callback: Optional async callback(step: str, progress: int)

    Returns dict with:
        - mesh_path: Path to .glb file
        - sewing_pattern_svg: Path to SVG visualization (if available)
        - garment_json: the GarmentCode JSON dict
        - processing_time_seconds: float
        - garment_type: detected garment type string
    """
    if MOCK_MODE:
        return await mock_generate_3d_garment(image_path, job_id)

    if not model_loaded:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    from mesh_utils import obj_to_glb, simplify_mesh

    start_time = time.time()
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 & 2: VLM inference (CoT)
    if progress_callback:
        await progress_callback("analyzing", 10)
    logger.info(f"[{job_id}] Running VLM inference...")
    description, garment_json = await asyncio.to_thread(
        _run_vlm_inference, image_path
    )

    # Save intermediate results
    with open(output_dir / "description.txt", "w") as f:
        f.write(description)
    with open(output_dir / "garment_spec.json", "w") as f:
        json.dump(garment_json, f, indent=2)

    # Step 3: Generate sewing patterns
    if progress_callback:
        await progress_callback("sewing_patterns", 40)
    logger.info(f"[{job_id}] Generating sewing patterns...")
    pattern_dir = await asyncio.to_thread(
        _run_garmentcode_simulation, garment_json, output_dir
    )

    # Step 4: 3D draping simulation
    if progress_callback:
        await progress_callback("draping", 65)
    logger.info(f"[{job_id}] Running 3D draping simulation...")
    obj_path = await asyncio.to_thread(
        _run_contourcraft_draping, pattern_dir, output_dir
    )

    # Step 5: Simplify + convert to GLB
    if progress_callback:
        await progress_callback("converting", 90)
    logger.info(f"[{job_id}] Converting mesh to GLB...")
    simplified = await asyncio.to_thread(simplify_mesh, obj_path)
    glb_path = output_dir / "garment.glb"
    await asyncio.to_thread(obj_to_glb, simplified, glb_path)

    processing_time = time.time() - start_time
    logger.info(f"[{job_id}] Complete in {processing_time:.1f}s")

    # Try to detect garment type from the JSON
    garment_type = garment_json.get("type", garment_json.get("garment_type", "unknown"))

    return {
        "mesh_path": glb_path,
        "sewing_pattern_svg": pattern_dir / "pattern.svg" if (pattern_dir / "pattern.svg").exists() else None,
        "garment_json": garment_json,
        "processing_time_seconds": processing_time,
        "garment_type": garment_type,
    }
