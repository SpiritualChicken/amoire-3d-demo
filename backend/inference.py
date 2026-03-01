from __future__ import annotations

"""
Amoire 3D Demo — Inference Pipeline

Wraps ChatGarment's GarmentGPTFloat50 model into a clean async interface.
This is the ONLY file that needs to change when swapping ML models.

Pipeline: Image → GarmentGPTFloat50 (2-step CoT) → GarmentCode JSON + Float Predictions
          → Sewing Pattern Spec Files → (GarmentCodeRC → ContourCraft → 3D Mesh)

The model loading follows ChatGarment's own inference procedure:
  1. Load base LLaVA model as GarmentGPTFloat50ForCausalLM
  2. Wrap with PEFT/LoRA (r=128, alpha=256)
  3. Load full PEFT checkpoint (pytorch_model.bin) via load_state_dict
  This ensures float_layer, LoRA weights, mm_projector all load correctly.

For mock mode, this returns placeholder meshes without any ML dependencies.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
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
_seg_token_idx = None
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
# Real ML inference — GarmentGPTFloat50
# ---------------------------------------------------------------------------

def _find_lora_targets(model):
    """Find Linear layers to apply LoRA to (q_proj, v_proj only).

    Excludes mm_projector, vision_tower, vision_resampler, and float_layer
    to match ChatGarment's training configuration.
    """
    import torch

    skip = ["mm_projector", "vision_tower", "vision_resampler", "float_layer"]
    targets = ["q_proj", "v_proj"]
    names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Linear)
            and all(x not in name for x in skip)
            and any(x in name for x in targets)
        ):
            names.add(name)
    return sorted(names)


def load_model():
    """Load ChatGarment GarmentGPTFloat50 model into GPU memory.

    Follows ChatGarment's own loading procedure:
      1. Tokenizer from base model + add [SEG] token
      2. GarmentGPTFloat50ForCausalLM.from_pretrained(base_model)
      3. Initialize CLIP vision tower
      4. Wrap with PEFT LoRA (r=128, alpha=256, targets=[q_proj, v_proj])
      5. Load full PEFT checkpoint (pytorch_model.bin) via load_state_dict

    Requires ~20GB VRAM in bf16. Call once at server startup.
    """
    global _model, _tokenizer, _image_processor, _seg_token_idx, model_loaded

    if MOCK_MODE:
        logger.info("Mock mode enabled — skipping model loading")
        model_loaded = True
        return

    import torch
    import transformers

    # Add ChatGarment to Python path
    chatgarment_path = str(CHATGARMENT_DIR)
    if chatgarment_path not in sys.path:
        sys.path.insert(0, chatgarment_path)

    try:
        # Import the CORRECT model class (not LlavaLlamaForCausalLM!)
        from llava.model.language_model.llava_garment_float50 import (
            GarmentGPTFloat50ForCausalLM,
        )
        from llava import conversation as conversation_lib
        from peft import LoraConfig, get_peft_model

        # Choose dtype: bf16 if supported, else fp16
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            logger.info("Using bfloat16 precision")
        else:
            dtype = torch.float16
            logger.info("GPU doesn't support bf16, using float16")

        # --- 1. Tokenizer ---
        logger.info("Loading tokenizer from base model...")
        _tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(LLAVA_MODEL_PATH),
            model_max_length=2048,
            padding_side="right",
            use_fast=False,
        )
        _tokenizer.pad_token = _tokenizer.unk_token
        _tokenizer.add_tokens("[SEG]")
        _seg_token_idx = _tokenizer(
            "[SEG]", add_special_tokens=False
        ).input_ids[-1]
        logger.info(f"[SEG] token index: {_seg_token_idx}")

        # --- 2. Base model as GarmentGPTFloat50ForCausalLM ---
        # ignore_mismatched_sizes=True because base model has vocab_size=32000
        # but config says 32001 (for [SEG] token). The full PEFT checkpoint
        # loaded in step 5 will overwrite embed_tokens/lm_head with correct weights.
        logger.info(f"Loading GarmentGPTFloat50 from {LLAVA_MODEL_PATH}...")
        model = GarmentGPTFloat50ForCausalLM.from_pretrained(
            str(LLAVA_MODEL_PATH),
            torch_dtype=dtype,
            seg_token_idx=_seg_token_idx,
            ignore_mismatched_sizes=True,
        )
        model.config.eos_token_id = _tokenizer.eos_token_id
        model.config.bos_token_id = _tokenizer.bos_token_id
        model.config.pad_token_id = _tokenizer.pad_token_id

        # --- 3. Vision tower (CLIP) ---
        logger.info("Initializing vision tower (CLIP ViT-L/14@336)...")
        vision_cfg = SimpleNamespace(
            vision_tower="openai/clip-vit-large-patch14-336",
            mm_vision_select_layer=-2,
            mm_vision_select_feature="patch",
            mm_use_im_start_end=False,
            mm_use_im_patch_token=False,
            mm_projector_type="mlp2x_gelu",
            pretrain_mm_mlp_adapter=None,
        )
        model.get_model().initialize_vision_modules(
            model_args=vision_cfg, fsdp=None
        )
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=dtype)
        for p in vision_tower.parameters():
            p.requires_grad = False
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
        _image_processor = vision_tower.image_processor

        # --- 4. LoRA adapters (must match training config) ---
        lora_targets = _find_lora_targets(model)
        logger.info(f"Applying LoRA (r=128, alpha=256) to {len(lora_targets)} layers")
        lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=lora_targets,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.resize_token_embeddings(len(_tokenizer))

        # --- 5. Load full PEFT checkpoint ---
        ckpt_path = str(LORA_WEIGHTS_PATH / "pytorch_model.bin")
        logger.info(f"Loading full PEFT checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        logger.info(f"Checkpoint has {len(state_dict)} keys")
        model.load_state_dict(state_dict, strict=True)
        logger.info("Checkpoint loaded successfully (strict=True)")

        # --- 6. Move to GPU and configure ---
        _model = model.to(dtype=dtype).cuda()

        # Config needed by process_images and generation
        _model.config.image_aspect_ratio = "pad"
        _model.config.tokenizer_padding_side = _tokenizer.padding_side
        _model.config.tokenizer_model_max_length = _tokenizer.model_max_length

        # Set conversation template
        conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]

        model_loaded = True
        logger.info("GarmentGPTFloat50 loaded successfully!")

    except ImportError as e:
        logger.error(
            f"Failed to import required modules: {e}. "
            "Make sure ChatGarment is cloned and peft is installed."
        )
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def _run_vlm_inference(image_path: Path) -> tuple[str, dict, object]:
    """Two-step Chain-of-Thought inference using GarmentGPTFloat50.

    Step 1: Image → JSON geometry description
    Step 2: Image + description → sewing pattern code + float predictions

    Returns (description_text, garment_json_dict, float_preds_tensor).
    float_preds is a tensor of shape [N_seg_tokens, 76] or None.
    """
    from PIL import Image as PILImage
    import torch

    if chatgarment_path := str(CHATGARMENT_DIR):
        if chatgarment_path not in sys.path:
            sys.path.insert(0, chatgarment_path)

    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    # Process image
    image = PILImage.open(image_path).convert("RGB")
    image_tensor = process_images([image], _image_processor, _model.config)
    if isinstance(image_tensor, list):
        image_tensor = [t.to(DEVICE, dtype=_model.dtype) for t in image_tensor]
    else:
        image_tensor = image_tensor.to(DEVICE, dtype=_model.dtype)

    # ---- Step 1: Geometry description ----
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        "Can you describe the geometry features of the garments "
        "worn by the model in the Json format?"
    ))
    conv.append_message(conv.roles[1], None)

    input_ids = tokenizer_image_token(
        conv.get_prompt(), _tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(DEVICE)

    logger.info("Step 1: Generating geometry description...")
    with torch.no_grad():
        output_ids, _, _ = _model.evaluate(
            image_tensor, image_tensor, input_ids,
            max_new_tokens=2048, tokenizer=_tokenizer,
        )

    # Decode (skip BOS token at index 0, keep special tokens for cleaning)
    description = _tokenizer.decode(
        output_ids[0, 1:], skip_special_tokens=False
    ).strip().replace("</s>", "")
    for tok in ["[STARTS]", "[SEG]", "[ENDS]"]:
        description = description.replace(tok, "")
    description = description.strip()
    logger.info(f"Step 1 output: {description[:300]}...")

    # ---- Step 2: Sewing pattern code + float predictions ----
    conv2 = conv_templates["v1"].copy()
    desc_for_prompt = description.replace(
        "upper_garment", "upperbody"
    ).replace(
        "lower_garment", "lowerbody"
    )
    conv2.append_message(conv2.roles[0], (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        "Can you estimate the sewing pattern code based on the image "
        f"and Json format garment geometry description?\n{desc_for_prompt}"
    ))
    conv2.append_message(conv2.roles[1], None)

    input_ids2 = tokenizer_image_token(
        conv2.get_prompt(), _tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(DEVICE)

    logger.info("Step 2: Generating sewing pattern code...")
    with torch.no_grad():
        output_ids2, float_preds, seg_mask = _model.evaluate(
            image_tensor, image_tensor, input_ids2,
            max_new_tokens=2048, tokenizer=_tokenizer,
        )

    json_text = _tokenizer.decode(
        output_ids2[0, 1:], skip_special_tokens=False
    ).strip().replace("</s>", "")
    for tok in ["[STARTS]", "[SEG]", "[ENDS]"]:
        json_text = json_text.replace(tok, "")
    json_text = json_text.strip()
    logger.info(f"Step 2 output: {json_text[:500]}...")

    if float_preds is not None:
        logger.info(f"Float predictions shape: {float_preds.shape}")
    else:
        logger.warning("No [SEG] tokens found — float_preds is None")

    # Parse JSON from text output
    garment_json = _parse_model_json(json_text)
    logger.info(f"Parsed garment JSON keys: {list(garment_json.keys())}")

    return description, garment_json, float_preds


def _parse_model_json(text: str) -> dict:
    """Parse JSON from model text output with multiple fallback strategies."""
    # Strategy 1: ChatGarment's json_fixer (handles common LLM JSON errors)
    try:
        sys.path.insert(0, str(CHATGARMENT_DIR))
        from llava.json_fixer import repair_json
        result = repair_json(text, return_objects=True)
        if isinstance(result, dict):
            return result
    except (ImportError, Exception):
        pass

    # Strategy 2: json_repair library
    try:
        from json_repair import repair_json
        result = repair_json(text, return_objects=True)
        if isinstance(result, dict):
            return result
    except (ImportError, Exception):
        pass

    # Strategy 3: Extract from markdown fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Strategy 4: Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 5: Find balanced braces + ast.literal_eval
    import ast
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        return ast.literal_eval(chunk)
                    except (ValueError, SyntaxError):
                        chunk = (chunk.replace("'", '"')
                                .replace("True", "true")
                                .replace("False", "false")
                                .replace("None", "null"))
                        try:
                            return json.loads(chunk)
                        except json.JSONDecodeError:
                            pass
                    break

    raise ValueError(f"Could not parse GarmentCode JSON from model output: {text[:500]}")


def _run_garmentcode_parser(
    garment_json: dict, float_preds, output_dir: Path
) -> list[Path]:
    """Use ChatGarment's parser to generate GarmentCode specification files.

    Combines the text JSON structure with numerical float predictions to produce
    complete GarmentCode specification files for sewing pattern generation.

    Returns list of paths to generated specification JSON files.
    """
    sys.path.insert(0, str(CHATGARMENT_DIR))
    from llava.garment_utils_v2 import run_garmentcode_parser_float50

    spec_files = run_garmentcode_parser_float50(
        [], garment_json, float_preds, str(output_dir)
    )
    return [Path(f) for f in spec_files]


def _run_garmentcode_simulation(spec_files: list[Path], output_dir: Path) -> Path:
    """Run GarmentCodeRC to generate 2D sewing patterns from spec files.

    Returns path to the sewing pattern data directory.
    """
    sys.path.insert(0, str(GARMENTCODE_DIR))

    pattern_dir = output_dir / "sewing_patterns"
    pattern_dir.mkdir(exist_ok=True)

    import subprocess

    for spec_file in spec_files:
        result = subprocess.run(
            [
                sys.executable,
                str(GARMENTCODE_DIR / "test_garmentcode.py"),
                "--config", str(spec_file),
                "--output", str(pattern_dir),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(GARMENTCODE_DIR),
        )

        if result.returncode != 0:
            logger.error(f"GarmentCodeRC failed for {spec_file}: {result.stderr}")
            raise RuntimeError(
                f"Sewing pattern generation failed: {result.stderr[:500]}"
            )

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

    from mesh_utils import create_placeholder_glb

    start_time = time.time()
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 & 2: VLM inference (two-step CoT)
    if progress_callback:
        await progress_callback("analyzing", 10)
    logger.info(f"[{job_id}] Running VLM inference...")
    description, garment_json, float_preds = await asyncio.to_thread(
        _run_vlm_inference, image_path
    )

    # Save intermediate results
    with open(output_dir / "description.txt", "w") as f:
        f.write(description)
    with open(output_dir / "garment_spec.json", "w") as f:
        json.dump(garment_json, f, indent=2)

    # Step 3: Parse float predictions → GarmentCode specification files
    spec_files = []
    if float_preds is not None:
        if progress_callback:
            await progress_callback("parsing", 30)
        logger.info(f"[{job_id}] Running GarmentCode parser with float predictions...")
        try:
            spec_files = await asyncio.to_thread(
                _run_garmentcode_parser, garment_json, float_preds, output_dir
            )
            logger.info(f"[{job_id}] Generated {len(spec_files)} spec files")
        except Exception as e:
            logger.warning(f"[{job_id}] GarmentCode parser failed: {e}")

    # Step 4: Generate sewing patterns from spec files
    pattern_dir = None
    if spec_files:
        if progress_callback:
            await progress_callback("sewing_patterns", 50)
        logger.info(f"[{job_id}] Generating sewing patterns...")
        try:
            pattern_dir = await asyncio.to_thread(
                _run_garmentcode_simulation, spec_files, output_dir
            )
        except Exception as e:
            logger.warning(f"[{job_id}] Sewing pattern generation failed: {e}")

    # Step 5: 3D draping simulation
    obj_path = None
    if pattern_dir:
        if progress_callback:
            await progress_callback("draping", 70)
        logger.info(f"[{job_id}] Running 3D draping simulation...")
        try:
            obj_path = await asyncio.to_thread(
                _run_contourcraft_draping, pattern_dir, output_dir
            )
        except Exception as e:
            logger.warning(f"[{job_id}] 3D draping failed: {e}")

    # Step 6: Convert to GLB
    glb_path = output_dir / "garment.glb"
    if obj_path and obj_path.exists():
        if progress_callback:
            await progress_callback("converting", 90)
        logger.info(f"[{job_id}] Converting mesh to GLB...")
        try:
            from mesh_utils import obj_to_glb, simplify_mesh
            simplified = await asyncio.to_thread(simplify_mesh, obj_path)
            await asyncio.to_thread(obj_to_glb, simplified, glb_path)
        except Exception as e:
            logger.warning(f"[{job_id}] GLB conversion failed: {e}")
            glb_path = None
    else:
        # Downstream pipeline not available — create a placeholder mesh
        logger.info(f"[{job_id}] Creating placeholder mesh (downstream pipeline not ready)")
        create_placeholder_glb(glb_path, shape="tshirt")

    processing_time = time.time() - start_time
    logger.info(f"[{job_id}] Complete in {processing_time:.1f}s")

    # Detect garment type from JSON
    garment_type = "unknown"
    if "upperbody_garment" in garment_json:
        garment_type = "outfit"
    elif "wholebody_garment" in garment_json:
        garment_type = "dress"
    else:
        garment_type = garment_json.get(
            "type", garment_json.get("garment_type", "unknown")
        )

    svg_path = None
    if pattern_dir and (pattern_dir / "pattern.svg").exists():
        svg_path = pattern_dir / "pattern.svg"

    return {
        "mesh_path": glb_path,
        "sewing_pattern_svg": svg_path,
        "garment_json": garment_json,
        "processing_time_seconds": processing_time,
        "garment_type": garment_type,
    }
