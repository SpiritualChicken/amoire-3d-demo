from __future__ import annotations

"""
Amoire 3D Demo — Inference Pipeline

Wraps ChatGarment's GarmentGPTFloat50 model into a clean async interface.
This is the ONLY file that needs to change when swapping ML models.

Pipeline: Image → GarmentGPTFloat50 (2-step CoT) → GarmentCode JSON + Float Predictions
          → Sewing Pattern Spec Files → BoxMesh → Warp Simulation (draping) → 3D Mesh

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
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from config import (
    CHATGARMENT_DIR,
    DEVICE,
    DRAPING_ENABLED,
    GARMENTCODE_DIR,
    LLAVA_MODEL_PATH,
    LORA_WEIGHTS_PATH,
    MOCK_MODE,
    OUTPUT_DIR,
    USE_4BIT_QUANTIZATION,
    UV_TEXTURES_ENABLED,
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
            mm_patch_merge_type="flat",
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
        "upper_garment", "upperbody_garment"
    ).replace(
        "lower_garment", "lowerbody_garment"
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
        # Log per-[SEG]-token statistics to diagnose model collapse
        fp_cpu = float_preds.detach().cpu().float()
        for seg_idx in range(fp_cpu.shape[0]):
            vals = fp_cpu[seg_idx]
            logger.info(
                f"  [SEG] token {seg_idx}: "
                f"min={vals.min().item():.4f}, max={vals.max().item():.4f}, "
                f"mean={vals.mean().item():.4f}, std={vals.std().item():.4f}, "
                f"first_5={[round(v, 4) for v in vals[:5].tolist()]}"
            )
    else:
        logger.warning("No [SEG] tokens found — float_preds is None")

    # Parse JSON from text output
    garment_json = _parse_model_json(json_text)
    logger.info(f"Parsed garment JSON keys: {list(garment_json.keys())}")

    return description, garment_json, float_preds, json_text


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
    # Ensure both ChatGarment and GarmentCodeRC (pygarment) are on sys.path
    for p in [str(CHATGARMENT_DIR), str(GARMENTCODE_DIR)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Must chdir BEFORE import — garment_utils_v2 loads docs/all_float_paths.json
    # at module level with a relative path
    old_cwd = os.getcwd()
    os.chdir(str(CHATGARMENT_DIR))
    try:
        from llava.garment_utils_v2 import run_garmentcode_parser_float50

        spec_files = run_garmentcode_parser_float50(
            [], garment_json, float_preds, str(output_dir)
        )
    finally:
        os.chdir(old_cwd)
    return [Path(f) for f in spec_files]


def _generate_garment_mesh(spec_files: list[Path], output_dir: Path) -> Path:
    """Generate 3D garment mesh from specification files using GarmentCodeRC BoxMesh.

    Creates a triangulated 3D mesh from sewing pattern specifications.
    Each garment piece (upper, lower) is loaded as a BoxMesh with proper
    3D panel positioning, then combined into a single OBJ mesh.

    Returns path to the output OBJ file.
    """
    import numpy as np
    import trimesh

    for p in [str(GARMENTCODE_DIR)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from pygarment.meshgen.boxmeshgen import BoxMesh
    import pygarment.data_config as data_config

    props = data_config.Properties(
        str(GARMENTCODE_DIR / "assets" / "Sim_props" / "default_sim_props.yaml")
    )
    resolution = props["sim"]["config"]["resolution_scale"]

    all_meshes = []
    for spec_file in spec_files:
        logger.info(f"Generating BoxMesh from {spec_file.name}...")
        bm = BoxMesh(str(spec_file), resolution)
        bm.load()

        mesh = trimesh.Trimesh(
            vertices=np.array(bm.vertices),
            faces=np.array(bm.faces),
        )
        logger.info(
            f"  {spec_file.stem}: {len(mesh.vertices)} verts, "
            f"{len(mesh.faces)} faces, {len(bm.panels)} panels"
        )
        all_meshes.append(mesh)

    if not all_meshes:
        raise RuntimeError("No meshes generated from spec files")

    # Combine all garment pieces into one mesh
    if len(all_meshes) > 1:
        combined = trimesh.util.concatenate(all_meshes)
    else:
        combined = all_meshes[0]

    # Apply a neutral fabric color
    combined.visual = trimesh.visual.ColorVisuals(
        mesh=combined,
        vertex_colors=np.tile([220, 218, 213, 255], (len(combined.vertices), 1)),
    )

    obj_path = output_dir / "garment.obj"
    combined.export(str(obj_path))
    logger.info(
        f"Combined garment mesh: {len(combined.vertices)} verts, "
        f"{len(combined.faces)} faces → {obj_path}"
    )

    return obj_path


def _run_warp_simulation(spec_files: list[Path], output_dir: Path) -> Path:
    """Run Warp physics simulation to drape garment panels onto a body mesh.

    Uses GarmentCodeRC's built-in FEM cloth simulation (NVIDIA Warp) to drape
    flat sewing panels onto a neutral body model. This produces realistic 3D
    garment shapes instead of flat spread-out panels.

    For each spec file: BoxMesh.load() → BoxMesh.serialize() → run_sim()
    The simulation output is a draped OBJ mesh at {garment_name}_sim.obj.

    Returns path to the combined output OBJ file.
    """
    import numpy as np
    import trimesh

    # pygarment.meshgen.simulation imports pyrender which needs OpenGL.
    # On headless servers, use OSMesa for off-screen rendering.
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    # Restore deprecated numpy type aliases removed in numpy 1.24+.
    # Older pygarment code (UV unwrapping, mesh operations) uses np.float,
    # np.int, np.bool which were removed from numpy.
    import numpy as _np
    if not hasattr(_np, "float"):
        _np.float = _np.float64
    if not hasattr(_np, "int"):
        _np.int = _np.int64
    if not hasattr(_np, "bool"):
        _np.bool = _np.bool_

    for p in [str(GARMENTCODE_DIR), str(CHATGARMENT_DIR)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from pygarment.meshgen.boxmeshgen import BoxMesh
    from pygarment.meshgen.simulation import run_sim
    import pygarment.data_config as data_config
    from pygarment.meshgen.sim_config import PathCofig

    sim_config_path = str(GARMENTCODE_DIR / "assets" / "Sim_props" / "default_sim_props.yaml")
    system_json_path = str(GARMENTCODE_DIR / "system.json")

    # Must chdir to GarmentCodeRC — system.json uses relative paths like
    # "./assets/bodies" that resolve from GarmentCodeRC root
    old_cwd = os.getcwd()
    os.chdir(str(GARMENTCODE_DIR))

    draped_meshes = []
    try:
        for spec_file in spec_files:
            # Parse garment name: "valid_garment_upper_specification" → "valid_garment_upper"
            garment_name, _, _ = spec_file.stem.rpartition("_")
            logger.info(f"Running Warp simulation for {garment_name}...")

            # Load sim properties fresh for each garment
            props = data_config.Properties(sim_config_path)
            props.set_section_stats(
                "sim", fails={}, sim_time={}, spf={},
                fin_frame={}, body_collisions={}, self_collisions={},
            )
            props.set_section_stats("render", render_time={})

            paths = PathCofig(
                in_element_path=spec_file.parent,
                out_path=str(output_dir),
                in_name=garment_name,
                body_name="mean_all",
                smpl_body=False,
                add_timestamp=False,
                system_path=system_json_path,
            )

            # Generate box mesh and serialize (creates _boxmesh.obj, segmentation,
            # edge lengths, vertex labels needed by the simulator)
            resolution = props["sim"]["config"]["resolution_scale"]
            garment_box_mesh = BoxMesh(paths.in_g_spec, resolution)
            garment_box_mesh.load()

            # UV texture generation — conditionally enabled via config flag.
            # Pass uv_config=None to skip UV generation entirely (patched in boxmeshgen).
            uv_cfg = props["render"]["config"]["uv_texture"] if UV_TEXTURES_ENABLED else None
            try:
                garment_box_mesh.serialize(paths, store_panels=False, uv_config=uv_cfg)
            except Exception as e:
                if uv_cfg is not None:
                    logger.warning(f"UV texture serialization failed, retrying without UVs: {e}")
                    garment_box_mesh.serialize(paths, store_panels=False, uv_config=None)
                else:
                    raise
            props.serialize(paths.element_sim_props)

            # Run FEM cloth simulation (drapes panels onto body)
            run_sim(
                garment_box_mesh.name,
                props,
                paths,
                save_v_norms=True,
                store_usd=False,
                optimize_storage=False,
                verbose=False,
            )
            props.serialize(paths.element_sim_props)

            # Load the draped result. process=True merges duplicate vertices
            # at panel seams and unifies face winding — prerequisites for
            # correct smooth normal computation during GLB export.
            sim_obj_path = Path(paths.g_sim)
            if sim_obj_path.exists():
                logger.info(f"  Simulation complete: {sim_obj_path}")
                mesh = trimesh.load(str(sim_obj_path), process=True)
                draped_meshes.append(mesh)
            else:
                logger.warning(f"  Simulation output not found at {sim_obj_path}")
    finally:
        os.chdir(old_cwd)

    if not draped_meshes:
        raise RuntimeError("Warp simulation produced no output meshes")

    # Combine all draped garment pieces
    if len(draped_meshes) > 1:
        combined = trimesh.util.concatenate(draped_meshes)
    else:
        combined = draped_meshes[0]

    # Apply a neutral fabric color
    combined.visual = trimesh.visual.ColorVisuals(
        mesh=combined,
        vertex_colors=np.tile([220, 218, 213, 255], (len(combined.vertices), 1)),
    )

    obj_path = output_dir / "garment_draped.obj"
    combined.export(str(obj_path))
    logger.info(
        f"Draped garment mesh: {len(combined.vertices)} verts, "
        f"{len(combined.faces)} faces → {obj_path}"
    )

    return obj_path


def _detect_garment_type(garment_json: dict) -> str:
    """Detect garment type from the GarmentCode JSON output."""
    if "upperbody_garment" in garment_json and "lowerbody_garment" in garment_json:
        return "outfit"
    if "upperbody_garment" in garment_json:
        return "top"
    if "lowerbody_garment" in garment_json:
        return "bottom"
    if "wholebody_garment" in garment_json:
        return "dress"
    return garment_json.get("type", garment_json.get("garment_type", "unknown"))


def _garment_type_to_shape(garment_type: str) -> str:
    """Map detected garment type to placeholder shape name."""
    mapping = {
        "outfit": "tshirt",
        "top": "tshirt",
        "bottom": "pants",
        "dress": "dress",
        "skirt": "skirt",
        "pants": "pants",
    }
    return mapping.get(garment_type, "tshirt")


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
    description, garment_json, float_preds, raw_step2_text = await asyncio.to_thread(
        _run_vlm_inference, image_path
    )

    # Save intermediate results for diagnostics
    with open(output_dir / "description.txt", "w") as f:
        f.write(description)
    with open(output_dir / "garment_spec.json", "w") as f:
        json.dump(garment_json, f, indent=2)
    with open(output_dir / "step2_raw_output.txt", "w") as f:
        f.write(raw_step2_text)
    if float_preds is not None:
        import torch as _torch
        _torch.save(float_preds.detach().cpu(), output_dir / "float_predictions.pt")
        fp_list = float_preds.detach().cpu().float().tolist()
        with open(output_dir / "float_predictions.json", "w") as f:
            json.dump(fp_list, f, indent=2)

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
            for sf in spec_files:
                logger.info(f"[{job_id}]   Spec: {sf.name} ({sf.stat().st_size} bytes)")
                try:
                    with open(sf) as _f:
                        spec_data = json.load(_f)
                    logger.info(f"[{job_id}]     Keys: {list(spec_data.keys())}")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[{job_id}] GarmentCode parser failed: {e}", exc_info=True)

    # Detect garment type from JSON (used for placeholder fallback)
    garment_type = _detect_garment_type(garment_json)

    # Step 4: Generate 3D garment mesh from spec files
    obj_path = None
    if spec_files and DRAPING_ENABLED:
        # Warp simulation: BoxMesh → serialize → physics draping onto body
        if progress_callback:
            await progress_callback("simulating", 50)
        logger.info(f"[{job_id}] Running Warp physics simulation (draping onto body)...")
        try:
            obj_path = await asyncio.to_thread(
                _run_warp_simulation, spec_files, output_dir
            )
        except Exception as e:
            logger.warning(
                f"[{job_id}] Warp simulation failed, falling back to BoxMesh: {e}",
                exc_info=True,
            )

    if spec_files and obj_path is None:
        # Fallback: flat BoxMesh panels (no draping)
        if progress_callback:
            await progress_callback("generating_mesh", 50)
        logger.info(f"[{job_id}] Generating flat BoxMesh (no draping)...")
        try:
            obj_path = await asyncio.to_thread(
                _generate_garment_mesh, spec_files, output_dir
            )
        except Exception as e:
            logger.warning(f"[{job_id}] Mesh generation failed: {e}", exc_info=True)

    # Step 6: Convert to GLB
    glb_path = output_dir / "garment.glb"
    if obj_path and obj_path.exists():
        if progress_callback:
            await progress_callback("converting", 90)
        logger.info(f"[{job_id}] Converting mesh to GLB...")
        try:
            from mesh_utils import obj_to_glb
            await asyncio.to_thread(obj_to_glb, obj_path, glb_path)
        except Exception as e:
            logger.warning(f"[{job_id}] GLB conversion failed: {e}")
            create_placeholder_glb(glb_path, shape=_garment_type_to_shape(garment_type))
    else:
        logger.info(f"[{job_id}] No mesh generated — using placeholder")
        create_placeholder_glb(glb_path, shape=_garment_type_to_shape(garment_type))

    processing_time = time.time() - start_time
    logger.info(f"[{job_id}] Complete in {processing_time:.1f}s")

    svg_path = None
    svgs = list(output_dir.rglob("*_pattern.svg"))
    if svgs:
        svg_path = svgs[0]

    return {
        "mesh_path": glb_path,
        "sewing_pattern_svg": svg_path,
        "garment_json": garment_json,
        "processing_time_seconds": processing_time,
        "garment_type": garment_type,
    }
