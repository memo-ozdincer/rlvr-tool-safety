#!/bin/bash
# =============================================================================
# Trillium-specific setup for RLVR tool-call safety.
#
# Run from Trillium login node:
#   ssh trillium
#   cd /project/def-zhijing/memoozd/rlvr-tool-safety
#   bash slurm/setup_trillium.sh
#
# This reuses the existing HF cache from rrfa (model already downloaded).
# Creates a fresh venv with RLVR-specific deps (trl, unsloth, vllm).
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
REPO_DIR="$PROJECT_DIR/rlvr-tool-safety"
VENV_DIR="$PROJECT_DIR/.venvs/rlvr_env"
CACHE_DIR="$CB_SCRATCH/cache"

echo "========================================"
echo "TRILLIUM SETUP — RLVR Tool-Call Safety"
echo "========================================"

# -------------------------------------------------------------------------
# 1. Verify existing cache (model should be there from rrfa)
# -------------------------------------------------------------------------
MODEL_CACHE="$CACHE_DIR/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct"
if [[ -d "$MODEL_CACHE" ]]; then
    echo "[OK] Model cached at $MODEL_CACHE"
    SNAPSHOT=$(ls -1td "$MODEL_CACHE/snapshots"/* 2>/dev/null | head -n 1)
    echo "     Snapshot: $SNAPSHOT"
else
    echo "[WARN] Model not cached. Run cache_models.sh after setup."
fi

# -------------------------------------------------------------------------
# 2. Create directories
# -------------------------------------------------------------------------
mkdir -p "$CB_SCRATCH"/{outputs/rlvr,logs}
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,triton}
echo "[OK] Directories ready"

# -------------------------------------------------------------------------
# 3. Copy trace data (reuse from rrfa scratch)
# -------------------------------------------------------------------------
RRFA_TRACES="$CB_SCRATCH/data/traces"
RLVR_TRACES="$CB_SCRATCH/data/rlvr_traces"
mkdir -p "$RLVR_TRACES"

for f in agentdojo_augmented.jsonl contrastive_pairs.jsonl; do
    if [[ -f "$RRFA_TRACES/$f" && ! -f "$RLVR_TRACES/$f" ]]; then
        echo "  Linking $f from rrfa traces..."
        ln -sf "$RRFA_TRACES/$f" "$RLVR_TRACES/$f"
    elif [[ -f "$RLVR_TRACES/$f" ]]; then
        echo "  [OK] $f already in place"
    else
        echo "  [WARN] $f not found in rrfa traces"
    fi
done

# Also link Fujitsu data if available
SHARED="$CB_SCRATCH/sweeps/shared_data_cb_full_sequence_ts"
for f in fujitsu_b4_ds.jsonl fujitsu_b4_dr.jsonl; do
    src="$SHARED/traces/$f"
    if [[ -f "$src" && ! -f "$RLVR_TRACES/$f" ]]; then
        ln -sf "$src" "$RLVR_TRACES/$f"
        echo "  Linked $f"
    fi
done

echo "[OK] Trace data ready at $RLVR_TRACES"

# -------------------------------------------------------------------------
# 4. Python environment
# -------------------------------------------------------------------------
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ -d "$VENV_DIR" ]]; then
    echo "[OK] Venv exists at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo "Creating venv at $VENV_DIR..."
    python3.11 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

# Install / upgrade deps
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel 2>&1 | tail -1

# Core ML
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1

# TRL + GRPO
pip install "trl>=0.15.0" transformers accelerate peft bitsandbytes 2>&1 | tail -1

# Inference
pip install vllm 2>&1 | tail -1

# Data + logging
pip install datasets huggingface-hub wandb numpy pandas tqdm 2>&1 | tail -1

# Unsloth (optional — may not build on all systems)
pip install unsloth 2>&1 || echo "[WARN] unsloth failed to install — will use standard HF+PEFT"

echo "[OK] Venv ready"

# -------------------------------------------------------------------------
# 5. Verify installation
# -------------------------------------------------------------------------
echo ""
echo "Verifying imports..."
python -c "
import torch
import transformers
import trl
import peft
print(f'torch:          {torch.__version__}')
print(f'transformers:   {transformers.__version__}')
print(f'trl:            {trl.__version__}')
print(f'peft:           {peft.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
try:
    import unsloth
    print(f'unsloth:        {unsloth.__version__}')
except ImportError:
    print('unsloth:        NOT INSTALLED (will use standard PEFT)')
try:
    import vllm
    print(f'vllm:           {vllm.__version__}')
except ImportError:
    print('vllm:           NOT INSTALLED')
"

echo ""
echo "========================================"
echo "SETUP COMPLETE"
echo "========================================"
echo ""
echo "Paths:"
echo "  Repo:     $REPO_DIR"
echo "  Venv:     $VENV_DIR"
echo "  Cache:    $CACHE_DIR"
echo "  Traces:   $RLVR_TRACES"
echo "  Outputs:  $CB_SCRATCH/outputs/rlvr"
echo ""
echo "Quick start:"
echo "  # Validate data"
echo "  python scripts/prepare_data.py \\"
echo "    --traces $RLVR_TRACES/agentdojo_augmented.jsonl \\"
echo "    --tool-schema configs/tool_schemas/agentdojo_v1.json \\"
echo "    --validate"
echo ""
echo "  # Train (ignore policy)"
echo "  POLICY=ignore TRACES=$RLVR_TRACES/agentdojo_augmented.jsonl \\"
echo "    sbatch --account=def-zhijing slurm/train.sbatch"
