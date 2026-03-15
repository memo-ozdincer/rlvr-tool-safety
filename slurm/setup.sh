#!/bin/bash
# =============================================================================
# One-time cluster setup for RLVR tool-call safety training.
# Works on both Toronto AIS and Trillium (CC) clusters.
#
# Usage (run from login node):
#   bash slurm/setup.sh
# =============================================================================

set -euo pipefail

# =============================================================================
# Auto-detect cluster
# =============================================================================
detect_cluster() {
    if [[ -d "/mfs1/u/memo" ]]; then
        echo "toronto"
    elif [[ -d "/project/def-zhijing" ]]; then
        echo "trillium"
    else
        echo "unknown"
    fi
}

CLUSTER=$(detect_cluster)
echo "Detected cluster: $CLUSTER"

case "$CLUSTER" in
    toronto)
        SCRATCH="/mfs1/u/memo"
        REPO_DIR="$SCRATCH/rlvr-tool-safety"
        VENV_DIR="$SCRATCH/rlvr_env"
        ;;
    trillium)
        PROJECT_DIR="/project/def-zhijing/memoozd"
        SCRATCH="/scratch/memoozd/cb-scratch"
        REPO_DIR="$PROJECT_DIR/rlvr-tool-safety"
        VENV_DIR="$PROJECT_DIR/.venvs/rlvr_env"
        ;;
    *)
        echo "ERROR: Unknown cluster. Set SCRATCH and REPO_DIR manually."
        exit 1
        ;;
esac

CACHE_DIR="$SCRATCH/cache"

# =============================================================================
# Create directories
# =============================================================================
echo "Creating directories..."
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,triton}
mkdir -p "$SCRATCH"/{outputs,logs}
mkdir -p "$REPO_DIR/data/traces"

# =============================================================================
# Python environment
# =============================================================================
echo "Setting up Python environment at $VENV_DIR..."

if [[ "$CLUSTER" == "trillium" ]]; then
    module --force purge || true
    module load StdEnv/2023
    module load cuda/12.6
    module load python/3.11.5
fi

if command -v uv &>/dev/null; then
    echo "Using uv..."
    if [[ ! -d "$VENV_DIR" ]]; then
        uv venv "$VENV_DIR" --python 3.11
    fi
    source "$VENV_DIR/bin/activate"
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    uv pip install "trl>=0.15.0" transformers accelerate peft bitsandbytes
    uv pip install vllm datasets huggingface-hub
    uv pip install wandb numpy pandas tqdm
    # Try unsloth — optional, not in all envs
    uv pip install unsloth 2>/dev/null || echo "WARNING: unsloth not available"
else
    echo "Using pip..."
    if [[ ! -d "$VENV_DIR" ]]; then
        python3.11 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install "trl>=0.15.0" transformers accelerate peft bitsandbytes
    pip install vllm datasets huggingface-hub
    pip install wandb numpy pandas tqdm
    pip install unsloth 2>/dev/null || echo "WARNING: unsloth not available"
fi

# =============================================================================
# Environment variables
# =============================================================================
ENV_BLOCK="
# RLVR tool-safety env vars
export SCRATCH=$SCRATCH
export HF_HOME=$CACHE_DIR/hf
export HF_HUB_CACHE=$CACHE_DIR/hf/hub
export HF_DATASETS_CACHE=$CACHE_DIR/hf/datasets
export TORCH_HOME=$CACHE_DIR/torch
export TRITON_CACHE_DIR=$CACHE_DIR/triton
export XDG_CACHE_HOME=$CACHE_DIR
"

SHELLRC="$HOME/.bashrc"
if ! grep -q "RLVR tool-safety" "$SHELLRC" 2>/dev/null; then
    echo "$ENV_BLOCK" >> "$SHELLRC"
    echo "Added env vars to $SHELLRC"
else
    echo "Env vars already in $SHELLRC"
fi

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo "Cluster:    $CLUSTER"
echo "Repo:       $REPO_DIR"
echo "Venv:       $VENV_DIR"
echo "Cache:      $CACHE_DIR"
echo ""
echo "Next steps:"
echo "  1. Cache model:  sbatch slurm/cache_models.sbatch"
echo "  2. Copy traces:  cp /path/to/traces data/traces/"
echo "  3. Train:        sbatch slurm/train.sbatch"
