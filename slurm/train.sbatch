#!/bin/bash
# =============================================================================
# GRPO Training for Tool-Call Safety
# =============================================================================
# Works on both Toronto AIS (H100) and Trillium (H100) clusters.
#
# Usage:
#   # Toronto (ignore policy):
#   POLICY=ignore sbatch -p ml -A ml -q ml --gres=gpu:1 slurm/train.sbatch
#
#   # Toronto (reject policy):
#   POLICY=reject sbatch -p ml -A ml -q ml --gres=gpu:1 slurm/train.sbatch
#
#   # Trillium:
#   POLICY=ignore sbatch --account=def-zhijing slurm/train.sbatch
#
#   # Override traces:
#   POLICY=ignore TRACES=data/traces/fujitsu_b4_ds.jsonl sbatch ...
# =============================================================================
#SBATCH --job-name=grpo_train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err

set -euo pipefail
source "$(dirname "$0")/cluster_env.sh"

# =============================================================================
# Configuration (override via env vars)
# =============================================================================
POLICY="${POLICY:-ignore}"
TRACES="${TRACES:-$DATA_DIR/agentdojo_augmented.jsonl}"
TOOL_SCHEMA="${TOOL_SCHEMA:-$REPO_DIR/configs/tool_schemas/agentdojo_v1.json}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RUN_NAME="${RUN_NAME:-grpo_${POLICY}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="$OUTPUT_BASE/$RUN_NAME"

# GRPO hyperparams
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_COMPLETION="${MAX_COMPLETION:-512}"
BETA="${BETA:-0.04}"
LOSS_TYPE="${LOSS_TYPE:-grpo}"

# LoRA
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"

# Wandb
WANDB_PROJECT="${WANDB_PROJECT:-rlvr-tool-safety}"
NO_WANDB="${NO_WANDB:-}"

# vLLM
USE_VLLM="${USE_VLLM:-1}"

mkdir -p "$OUTPUT_DIR"

# Verify data exists
if [[ ! -f "$TRACES" ]]; then
    echo "ERROR: Traces file not found: $TRACES"
    echo "  Copy traces: cp /path/to/agentdojo_augmented.jsonl $DATA_DIR/"
    exit 1
fi

echo "========================================"
echo "GRPO TRAINING"
echo "========================================"
echo "Policy:       $POLICY"
echo "Traces:       $TRACES"
echo "Model:        ${MODEL_PATH:-$MODEL_ID}"
echo "Output:       $OUTPUT_DIR"
echo "Run name:     $RUN_NAME"
echo "Epochs:       $NUM_EPOCHS"
echo "LR:           $LR"
echo "Batch:        ${BATCH_SIZE}x${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM)) effective"
echo "Generations:  $NUM_GENERATIONS"
echo "Beta (KL):    $BETA"
echo "Loss:         $LOSS_TYPE"
echo "LoRA:         r=$LORA_R, alpha=$LORA_ALPHA"
echo "========================================"

# Build command
CMD="python src/training/train.py \
    --traces $TRACES \
    --tool-schema $TOOL_SCHEMA \
    --policy $POLICY \
    --model ${MODEL_PATH:-$MODEL_ID} \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --num-generations $NUM_GENERATIONS \
    --max-completion-length $MAX_COMPLETION \
    --beta $BETA \
    --loss-type $LOSS_TYPE \
    --lr $LR \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --grad-accum $GRAD_ACCUM \
    --output-dir $OUTPUT_DIR \
    --wandb-project $WANDB_PROJECT \
    --wandb-run $RUN_NAME \
    --logging-steps 10 \
    --save-steps 500 \
    --log-completions-every 50"

if [[ -n "$MAX_SAMPLES" ]]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

if [[ -n "$NO_WANDB" ]]; then
    CMD="$CMD --no-wandb"
fi

if [[ "$USE_VLLM" == "0" ]]; then
    CMD="$CMD --no-vllm"
fi

echo ""
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "========================================"
echo "TRAINING COMPLETE"
echo "========================================"
echo "Output: $OUTPUT_DIR"
echo "Date:   $(date)"
