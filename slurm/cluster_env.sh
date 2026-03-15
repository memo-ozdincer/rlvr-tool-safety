#!/bin/bash
# =============================================================================
# Shared cluster environment detection and setup.
# Source this file at the top of every sbatch script:
#   source "$(dirname "$0")/cluster_env.sh"
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
        # Load modules on Trillium
        module --force purge || true
        module load StdEnv/2023
        module load cuda/12.6
        module load python/3.11.5
        # Trillium uses SLURM output dirs set in sbatch headers;
        # override LOG_DIR so scripts find the right place
        LOG_DIR="$SCRATCH/logs"
        ;;
    *)
        echo "ERROR: Unknown cluster"
        exit 1
        ;;
esac

CACHE_DIR="$SCRATCH/cache"

# Activate venv
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

# Cache env vars
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export XDG_CACHE_HOME="$CACHE_DIR"
mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true

# Offline mode (disable if caching)
if [[ "${ONLINE_MODE:-0}" != "1" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

export TQDM_DISABLE="${TQDM_DISABLE:-0}"

# Data paths — Trillium uses rlvr_traces subdir (symlinked from rrfa data)
if [[ "$CLUSTER" == "trillium" ]]; then
    DATA_DIR="$SCRATCH/data/rlvr_traces"
else
    DATA_DIR="$SCRATCH/data/traces"
fi
OUTPUT_BASE="$SCRATCH/outputs"
LOG_DIR="$SCRATCH/logs"
mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

# Model resolution
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_CACHE_NAME=$(echo "$MODEL_ID" | sed 's/\//-/g')
SNAPSHOT_ROOT="$CACHE_DIR/hf/hub/models--${MODEL_CACHE_NAME}/snapshots"
MODEL_PATH=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)

if [[ -z "$MODEL_PATH" && "${ONLINE_MODE:-0}" != "1" ]]; then
    echo "WARNING: Model not cached at $SNAPSHOT_ROOT"
    echo "  Run: ONLINE_MODE=1 sbatch slurm/cache_models.sbatch"
    MODEL_PATH="$MODEL_ID"  # Fall back to HF ID
fi

echo "========================================"
echo "Cluster:  $CLUSTER"
echo "Repo:     $REPO_DIR"
echo "Scratch:  $SCRATCH"
echo "Model:    ${MODEL_PATH:-$MODEL_ID}"
echo "========================================"
