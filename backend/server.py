from __future__ import annotations

"""
Amoire 3D Demo — FastAPI Server

Endpoints:
  POST /api/generate-3d   — Upload image, start 3D generation job
  GET  /api/status/{id}   — Poll job status
  GET  /api/mesh/{id}     — Download generated .glb mesh
  GET  /api/gallery       — List previously generated garments
  GET  /api/health        — Server health check
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import (
    ALLOWED_ORIGINS,
    HOST,
    MAX_UPLOAD_SIZE_BYTES,
    MOCK_MODE,
    OUTPUT_DIR,
    PORT,
    UPLOAD_DIR,
)
from inference import generate_3d_garment, load_model, model_loaded
from mesh_utils import cache_result, compute_image_hash, get_cached_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# In-memory job tracking
jobs: dict[str, dict] = {}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load ML model. Shutdown: cleanup."""
    logger.info(f"Starting Amoire 3D backend (mock={MOCK_MODE})")
    try:
        load_model()
    except Exception as e:
        logger.error(f"Model loading failed: {e}. Server will run in degraded mode.")
    yield
    logger.info("Shutting down Amoire 3D backend")


app = FastAPI(
    title="Amoire 3D Canvas API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images as static files for thumbnail display
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


async def _run_job(job_id: str, image_path: Path, image_hash: str):
    """Background task: run the full inference pipeline for a job."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["step"] = "analyzing"
        jobs[job_id]["progress"] = 5

        async def progress_callback(step: str, progress: int):
            jobs[job_id]["step"] = step
            jobs[job_id]["progress"] = progress

        result = await generate_3d_garment(image_path, job_id, progress_callback)

        # Cache the result
        cache_result(image_hash, result["mesh_path"], {
            "job_id": job_id,
            "garment_type": result.get("garment_type", "unknown"),
            "processing_time": result["processing_time_seconds"],
            "created_at": datetime.utcnow().isoformat(),
        })

        jobs[job_id].update({
            "status": "complete",
            "progress": 100,
            "step": "done",
            "mesh_path": str(result["mesh_path"]),
            "garment_type": result.get("garment_type", "unknown"),
            "processing_time": result["processing_time_seconds"],
            "completed_at": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "step": "error",
        })


@app.post("/api/generate-3d")
async def generate_3d(image: UploadFile = File(...)):
    """Upload an image and start 3D garment generation."""
    # Validate file type
    ext = Path(image.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Invalid file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    # Read and validate size
    content = await image.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(400, f"File too large. Max {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB")

    # Save upload
    job_id = uuid.uuid4().hex[:12]
    image_path = UPLOAD_DIR / f"{job_id}{ext}"
    with open(image_path, "wb") as f:
        f.write(content)

    # Check cache
    image_hash = compute_image_hash(image_path)
    cached = get_cached_result(image_hash)
    if cached:
        logger.info(f"Cache hit for {image_hash}")
        jobs[job_id] = {
            "status": "complete",
            "progress": 100,
            "step": "done",
            "mesh_path": str(cached),
            "image_path": str(image_path),
            "image_hash": image_hash,
            "cache_hit": True,
            "created_at": datetime.utcnow().isoformat(),
        }
        return JSONResponse(
            {"job_id": job_id, "status": "complete", "cache_hit": True},
            headers={"X-Cache-Hit": "true"},
        )

    # Create job and start background processing
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "step": "queued",
        "image_path": str(image_path),
        "image_hash": image_hash,
        "cache_hit": False,
        "created_at": datetime.utcnow().isoformat(),
    }

    asyncio.create_task(_run_job(job_id, image_path, image_hash))

    return JSONResponse(
        {"job_id": job_id, "status": "queued"},
        status_code=202,
    )


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Poll generation job status."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "step": job.get("step", "unknown"),
        "error": job.get("error"),
        "processing_time": job.get("processing_time"),
        "garment_type": job.get("garment_type"),
    }


@app.get("/api/mesh/{job_id}")
async def get_mesh(job_id: str):
    """Download the generated .glb mesh for a completed job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "complete":
        raise HTTPException(400, f"Job not complete (status: {job['status']})")

    mesh_path = Path(job["mesh_path"])
    if not mesh_path.exists():
        raise HTTPException(500, "Mesh file not found on disk")

    return FileResponse(
        mesh_path,
        media_type="model/gltf-binary",
        filename=f"garment-{job_id}.glb",
        headers={
            "X-Processing-Time": str(job.get("processing_time", "")),
            "X-Cache-Hit": str(job.get("cache_hit", False)).lower(),
        },
    )


@app.get("/api/gallery")
async def get_gallery():
    """List previously generated garments."""
    items = []
    for job_id, job in jobs.items():
        if job["status"] != "complete":
            continue

        image_path = job.get("image_path", "")
        # Build thumbnail URL from the upload path
        thumbnail_url = None
        if image_path:
            p = Path(image_path)
            if p.exists():
                thumbnail_url = f"/uploads/{p.name}"

        items.append({
            "id": job_id,
            "thumbnail_url": thumbnail_url,
            "mesh_url": f"/api/mesh/{job_id}",
            "created_at": job.get("created_at"),
            "garment_type": job.get("garment_type", "unknown"),
            "processing_time": job.get("processing_time"),
        })

    # Sort by creation time, newest first
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"items": items}


@app.get("/api/health")
async def health():
    """Server health check."""
    gpu_info = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_total_mb": torch.cuda.get_device_properties(0).total_mem // (1024 * 1024),
                "memory_used_mb": torch.cuda.memory_allocated(0) // (1024 * 1024),
            }
    except ImportError:
        pass

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "mock_mode": MOCK_MODE,
        "gpu": gpu_info,
        "active_jobs": sum(1 for j in jobs.values() if j["status"] in ("queued", "processing")),
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Amoire 3D Backend")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no ML inference)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", type=str, default=HOST)
    args = parser.parse_args()

    if args.mock:
        os.environ["MOCK_MODE"] = "true"
        import config
        config.MOCK_MODE = True
        import inference
        inference.MOCK_MODE = True

    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.mock)
