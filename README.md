# Amoire 3D Canvas Demo

Transform 2D clothing images into interactive 3D garments. This demo takes a flat garment photo and generates a draped 3D mesh using [ChatGarment](https://github.com/biansy000/ChatGarment) (CVPR 2025) — a vision-language model that predicts sewing patterns from images, which are then physically simulated into 3D. Built for [Amoire](https://amoire.com), a style discovery platform.

## Architecture

```
User uploads image
       │
       ▼
┌─────────────────────────────────────────┐
│  Next.js Frontend (localhost:3000)       │
│  - Drag-and-drop image upload           │
│  - Babylon.js 3D viewer (orbit/zoom)    │
│  - Loading states + error handling      │
│  - Gallery of previously generated      │
└──────────────┬──────────────────────────┘
               │ POST /api/generate-3d
               │ Returns: .glb binary
               ▼
┌──────────────────────────────────────────┐
│  FastAPI Backend (localhost:8000)         │
│  1. Save uploaded image                  │
│  2. ChatGarment VLM → GarmentCode JSON  │
│  3. GarmentCodeRC → 2D sewing patterns  │
│  4. ContourCraft-CG → 3D draped mesh    │
│  5. Convert to .glb → return            │
└──────────────────────────────────────────┘
```

## Prerequisites

**Frontend only (demo mode):**
- Node.js 18+
- npm

**Full pipeline (with ML inference):**
- NVIDIA GPU with 8GB+ VRAM (16GB recommended)
- CUDA 12.1
- Python 3.10
- conda
- ~20GB disk for model weights

## Quick Start

### Demo Mode (no GPU needed)

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

The frontend runs standalone with pre-generated sample meshes. Upload any image and it will load a random demo garment after a short delay.

### Mock Backend (no GPU, full API)

```bash
cd backend
pip install -r requirements.txt
python server.py --mock
# Backend runs at http://localhost:8000

# In another terminal:
cd frontend
npm run dev
```

### Full Pipeline (GPU required)

```bash
# 1. Run ML setup (clones repos, installs deps, downloads weights)
./scripts/setup_ml.sh

# 2. Activate environment and start backend
conda activate amoire-3d
cd backend
python server.py

# 3. Start frontend
cd frontend
npm install
npm run dev
```

### Docker

```bash
docker compose up
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

Requires NVIDIA Container Toolkit for GPU access.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate-3d` | Upload image, start 3D generation. Returns `{ job_id, status }` |
| `GET` | `/api/status/{job_id}` | Poll job progress. Returns `{ status, progress, step }` |
| `GET` | `/api/mesh/{job_id}` | Download generated `.glb` mesh |
| `GET` | `/api/gallery` | List previously generated garments |
| `GET` | `/api/health` | Server health, GPU info, model status |

## Project Structure

```
amoire-3d-demo/
├── backend/
│   ├── server.py          # FastAPI app with job queue
│   ├── inference.py       # ML pipeline (swap this for different models)
│   ├── mesh_utils.py      # OBJ/GLB conversion, caching
│   └── config.py          # All paths and settings
├── frontend/
│   ├── src/components/
│   │   ├── GarmentViewer.tsx    # Babylon.js 3D viewer
│   │   ├── ImageUpload.tsx      # Drag-drop upload
│   │   ├── GenerationStatus.tsx # Progress display
│   │   └── GarmentGallery.tsx   # Sample/history gallery
│   └── src/lib/api.ts          # API client
├── ml/                    # Cloned ML repos (gitignored)
├── scripts/
│   ├── setup_ml.sh        # One-command ML setup
│   ├── generate_samples.py # Create placeholder meshes
│   └── test_inference.py  # Test the pipeline
└── docker-compose.yml
```

## Development Notes

- **Swapping ML models:** Only `inference.py` needs to change. The API, mesh conversion, and caching layers are model-agnostic. When DressWild releases code, create a new inference backend implementing the same `generate_3d_garment()` interface.

- **Frontend standalone:** The frontend detects backend availability on load. If the health check fails, it enters demo mode with pre-generated samples. No code changes needed.

- **Caching:** Results are cached by image SHA256 hash. Uploading the same image twice returns the cached mesh instantly.

- **Mock mode:** `python server.py --mock` runs the full API but returns placeholder meshes instead of running ML inference. Useful for frontend development.

## Known Limitations

- Single garment per image (multi-garment support is partial)
- No texture/color transfer from the source image to the 3D mesh yet
- Inference takes 2-5 minutes per garment on a single GPU
- The placeholder meshes are simple geometric shapes, not realistic garments
- No persistent storage for the gallery (in-memory only, resets on server restart)

## Roadmap

- [ ] DressWild integration (in-the-wild images, when code releases)
- [ ] Texture transfer from source image to 3D mesh
- [ ] Multi-garment composition (layer top + bottom)
- [ ] Body shape customization (SMPL parameters)
- [ ] Amoire taste profile integration (recommend garments based on style preferences)
- [ ] Persistent gallery with database backend
- [ ] WebSocket for real-time progress (replace polling)
