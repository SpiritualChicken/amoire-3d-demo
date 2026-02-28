#!/usr/bin/env bash
# =============================================================================
# Amoire 3D Demo — ML Setup Script
#
# Clones repos, installs dependencies, downloads model weights.
# Safe to re-run (idempotent).
#
# Prerequisites: conda, git, ~20GB disk, NVIDIA GPU with CUDA 12.1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ML_DIR="$PROJECT_ROOT/ml"
DATA_DIR="$PROJECT_ROOT/data"
CONDA_ENV="amoire-3d"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()    { echo -e "${RED}[FAIL]${NC} $1"; }
info()    { echo -e "     $1"; }

errors=()

# ─── Step 1: Create directories ─────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " Amoire 3D Demo — ML Setup"
echo "═══════════════════════════════════════════════════════"
echo ""

echo "Creating directories..."
mkdir -p "$ML_DIR" "$DATA_DIR/uploads" "$DATA_DIR/outputs" "$DATA_DIR/cache"
success "Directories created"

# ─── Step 2: Clone repositories ─────────────────────────────────────────────
echo ""
echo "Cloning ML repositories..."

clone_repo() {
    local url="$1"
    local dest="$2"
    local name="$(basename "$dest")"

    if [ -d "$dest/.git" ]; then
        info "$name already cloned, pulling latest..."
        cd "$dest" && git pull --ff-only 2>/dev/null || true
        cd "$PROJECT_ROOT"
        success "$name up to date"
    else
        info "Cloning $name..."
        if git clone "$url" "$dest" 2>&1; then
            success "$name cloned"
        else
            fail "Failed to clone $name"
            errors+=("Clone $name failed")
        fi
    fi
}

clone_repo "https://github.com/biansy000/ChatGarment.git" "$ML_DIR/ChatGarment"
clone_repo "https://github.com/biansy000/GarmentCodeRC.git" "$ML_DIR/GarmentCodeRC"
clone_repo "https://github.com/biansy000/ContourCraft-CG.git" "$ML_DIR/ContourCraft-CG"

# ─── Step 3: Conda environment ──────────────────────────────────────────────
echo ""
echo "Setting up conda environment..."

if ! command -v conda &>/dev/null; then
    fail "conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    errors+=("conda not found")
else
    # Initialize conda for this shell
    eval "$(conda shell.bash hook)"

    if conda env list | grep -q "^${CONDA_ENV} "; then
        info "Environment '$CONDA_ENV' already exists"
        conda activate "$CONDA_ENV"
        success "Activated $CONDA_ENV"
    else
        info "Creating environment '$CONDA_ENV' with Python 3.10..."
        conda create -n "$CONDA_ENV" python=3.10 -y
        conda activate "$CONDA_ENV"
        success "Created and activated $CONDA_ENV"
    fi

    # ─── Step 4: Install PyTorch with CUDA ───────────────────────────────
    echo ""
    echo "Installing PyTorch with CUDA 12.1..."
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        info "PyTorch with CUDA already installed"
        success "PyTorch OK"
    else
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
        if python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
            success "PyTorch installed with CUDA"
        else
            warn "PyTorch installed but CUDA not detected (may work on CPU only)"
        fi
    fi

    # ─── Step 5: Install ChatGarment dependencies ────────────────────────
    echo ""
    echo "Installing ChatGarment dependencies..."
    if [ -d "$ML_DIR/ChatGarment" ]; then
        cd "$ML_DIR/ChatGarment"
        pip install -e . 2>&1 | tail -3 || { fail "ChatGarment install failed"; errors+=("ChatGarment deps"); }
        cd "$PROJECT_ROOT"
        success "ChatGarment deps installed"
    else
        warn "ChatGarment not cloned, skipping"
    fi

    # ─── Step 6: Install GarmentCodeRC dependencies ──────────────────────
    echo ""
    echo "Installing GarmentCodeRC dependencies..."
    if [ -d "$ML_DIR/GarmentCodeRC" ]; then
        cd "$ML_DIR/GarmentCodeRC"
        pip install -e . 2>&1 | tail -3 || { fail "GarmentCodeRC install failed"; errors+=("GarmentCodeRC deps"); }
        cd "$PROJECT_ROOT"
        success "GarmentCodeRC deps installed"
    else
        warn "GarmentCodeRC not cloned, skipping"
    fi

    # ─── Step 7: Install ContourCraft-CG dependencies ────────────────────
    echo ""
    echo "Installing ContourCraft-CG dependencies..."
    if [ -d "$ML_DIR/ContourCraft-CG" ]; then
        cd "$ML_DIR/ContourCraft-CG"
        pip install -e . 2>&1 | tail -3 || { fail "ContourCraft-CG install failed"; errors+=("ContourCraft-CG deps"); }
        cd "$PROJECT_ROOT"
        success "ContourCraft-CG deps installed"
    else
        warn "ContourCraft-CG not cloned, skipping"
    fi

    # ─── Step 8: Install additional dependencies ─────────────────────────
    echo ""
    echo "Installing additional Python packages..."
    pip install \
        transformers \
        accelerate \
        peft \
        bitsandbytes \
        trimesh \
        open3d \
        fastapi \
        uvicorn[standard] \
        python-multipart \
        aiofiles \
        Pillow \
        2>&1 | tail -3
    success "Additional packages installed"
fi

# ─── Step 9: Download model weights ─────────────────────────────────────────
echo ""
echo "Downloading model weights from HuggingFace..."

if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

CHECKPOINTS_DIR="$ML_DIR/ChatGarment/checkpoints"
mkdir -p "$CHECKPOINTS_DIR"

# Download LLaVA base model
if [ -d "$CHECKPOINTS_DIR/llava-v1.5-7b" ] && [ "$(ls -A "$CHECKPOINTS_DIR/llava-v1.5-7b" 2>/dev/null)" ]; then
    info "LLaVA base model already downloaded"
    success "LLaVA model OK"
else
    info "Downloading LLaVA v1.5 7B base model (this may take a while)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('liuhaotian/llava-v1.5-7b', local_dir='$CHECKPOINTS_DIR/llava-v1.5-7b')
print('Download complete')
" 2>&1 || { fail "LLaVA download failed"; errors+=("LLaVA download"); }
    success "LLaVA base model downloaded"
fi

# Download ChatGarment dataset + LoRA weights
if [ -d "$CHECKPOINTS_DIR/chatgarment-lora" ] && [ "$(ls -A "$CHECKPOINTS_DIR/chatgarment-lora" 2>/dev/null)" ]; then
    info "ChatGarment LoRA weights already downloaded"
    success "LoRA weights OK"
else
    info "Downloading ChatGarment LoRA weights..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('sy000/ChatGarmentDataset', local_dir='$CHECKPOINTS_DIR/chatgarment-lora')
print('Download complete')
" 2>&1 || { fail "ChatGarment weights download failed"; errors+=("ChatGarment weights"); }
    success "ChatGarment LoRA weights downloaded"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " Setup Summary"
echo "═══════════════════════════════════════════════════════"
echo ""
echo " ML repos:     $ML_DIR/"
echo " Data dirs:    $DATA_DIR/"
echo " Conda env:    $CONDA_ENV"
echo ""

if [ ${#errors[@]} -eq 0 ]; then
    success "All steps completed successfully!"
    echo ""
    echo " Next steps:"
    echo "   conda activate $CONDA_ENV"
    echo "   cd $PROJECT_ROOT/backend"
    echo "   python server.py              # Full ML inference"
    echo "   python server.py --mock       # Mock mode (no GPU needed)"
else
    echo ""
    fail "Completed with ${#errors[@]} error(s):"
    for err in "${errors[@]}"; do
        echo "   - $err"
    done
fi
echo ""
