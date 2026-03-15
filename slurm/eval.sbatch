#!/bin/bash
# =============================================================================
# Evaluation for RLVR tool-call safety models.
# =============================================================================
# Usage:
#   # Toronto:
#   MODEL_DIR=/mfs1/u/memo/outputs/grpo_ignore_XXXX/final/merged \
#     sbatch -p ml -A ml -q ml --gres=gpu:1 slurm/eval.sbatch
#
#   # Trillium:
#   MODEL_DIR=/scratch/memoozd/cb-scratch/outputs/grpo_ignore_XXXX/final/merged \
#     sbatch --account=def-zhijing slurm/eval.sbatch
#
#   # Eval baseline:
#   MODEL_DIR=meta-llama/Llama-3.1-8B-Instruct \
#     sbatch -p ml -A ml -q ml --gres=gpu:1 slurm/eval.sbatch
# =============================================================================
#SBATCH --job-name=grpo_eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err

set -euo pipefail
source "$(dirname "$0")/cluster_env.sh"

MODEL_DIR="${MODEL_DIR:?ERROR: Set MODEL_DIR to model path or HF ID}"
EVAL_DATA="${EVAL_DATA:-$DATA_DIR/agentdojo_augmented.jsonl}"
TOOL_SCHEMA="${TOOL_SCHEMA:-$REPO_DIR/configs/tool_schemas/agentdojo_v1.json}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
EVAL_NAME="${EVAL_NAME:-eval_$(basename $MODEL_DIR)_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="$OUTPUT_BASE/$EVAL_NAME.json"

echo "========================================"
echo "EVALUATION"
echo "========================================"
echo "Model:    $MODEL_DIR"
echo "Data:     $EVAL_DATA"
echo "Samples:  $MAX_SAMPLES"
echo "Output:   $OUTPUT"
echo "========================================"

python src/evaluation/eval_tool.py \
    --model "$MODEL_DIR" \
    --eval-data "$EVAL_DATA" \
    --tool-schema "$TOOL_SCHEMA" \
    --max-samples "$MAX_SAMPLES" \
    --output "$OUTPUT"

echo ""
echo "Evaluation complete. Results: $OUTPUT"
