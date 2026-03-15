#!/bin/bash
# =============================================================================
# Cache models for offline SLURM usage.
# Auto-detects cluster and uses correct SLURM flags.
#
# Usage:
#   # Toronto:  sbatch -p ml -A ml -q ml slurm/cache_models.sbatch
#   # Trillium: sbatch --account=def-zhijing slurm/cache_models.sbatch
#   # Or just run from login node if it has internet:
#   #   bash slurm/cache_models.sbatch
# =============================================================================
#SBATCH --job-name=cache_models
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=16G

set -euo pipefail

# Enable online mode for caching
export ONLINE_MODE=1
source "$(dirname "$0")/cluster_env.sh"

echo "Caching models to $CACHE_DIR/hf/hub ..."

python -c "
from huggingface_hub import snapshot_download
import os

cache_dir = os.environ.get('HF_HUB_CACHE', '$CACHE_DIR/hf/hub')
model_id = '$MODEL_ID'

print(f'Downloading {model_id} to {cache_dir}...')
path = snapshot_download(model_id, cache_dir=cache_dir)
print(f'Cached at: {path}')
print('Done!')
"

echo "Model cached successfully."
echo "You can now run training with HF_HUB_OFFLINE=1"
