"""
Amoire 3D Demo — Configuration

All paths and settings for the backend server and ML inference pipeline.
This is the single source of truth for project configuration.
"""

from pathlib import Path
import os

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
BACKEND_DIR = Path(__file__).parent.resolve()
ML_DIR = PROJECT_ROOT / "ml"
DATA_DIR = PROJECT_ROOT / "data"

# ML model paths
CHATGARMENT_DIR = ML_DIR / "ChatGarment"
GARMENTCODE_DIR = ML_DIR / "GarmentCodeRC"
CONTOURCRAFT_DIR = ML_DIR / "ContourCraft-CG"

# Model weights
LLAVA_MODEL_PATH = CHATGARMENT_DIR / "checkpoints" / "llava-v1.5-7b"
LORA_WEIGHTS_PATH = CHATGARMENT_DIR / "checkpoints" / "try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final"

# Data directories
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
CACHE_DIR = DATA_DIR / "cache"

# Server config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MAX_UPLOAD_SIZE_MB = 20
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
INFERENCE_TIMEOUT_SECONDS = 300
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# Inference config
USE_4BIT_QUANTIZATION = os.getenv("USE_4BIT", "true").lower() == "true"
DEVICE = os.getenv("DEVICE", "cuda")

# Mock mode — skip ML inference, return placeholder meshes
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# Ensure data directories exist
for d in [UPLOAD_DIR, OUTPUT_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
