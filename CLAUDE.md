# RLVR Tool-Call Safety

RLVR (Reinforcement Learning with Verifiable Rewards) for LLM tool-call safety.
Trains Llama 3.1 8B Instruct to resist prompt injection attacks in tool-calling scenarios.

## Architecture

- **Training**: TRL GRPOTrainer + Unsloth LoRA + vLLM generation
- **Reward**: Custom verifiers that parse tool calls and check correctness
- **Data**: Canonical traces from AgentDojo + Fujitsu B4 (ported from rrfa repo)
- **Eval**: next_tool_prediction + tool-flip ASR

## Two Policies

1. **Ignore**: Call the correct tool despite injection (reward=1.0 for correct tool)
2. **Reject**: No tool call on injection, correct tool on benign (reward=1.0 for correct behavior)

## Key Commands

```bash
# Setup cluster (run once)
bash slurm/setup.sh

# Cache model
sbatch -p ml -A ml -q ml slurm/cache_models.sbatch          # Toronto
sbatch --account=def-zhijing slurm/cache_models.sbatch       # Trillium

# Train
POLICY=ignore sbatch -p ml -A ml -q ml --gres=gpu:1 slurm/train.sbatch
POLICY=reject sbatch -p ml -A ml -q ml --gres=gpu:1 slurm/train.sbatch

# Evaluate
MODEL_DIR=path/to/checkpoint sbatch -p ml -A ml -q ml --gres=gpu:1 slurm/eval.sbatch
```

## Directory Layout

```
src/training/       # GRPO training: data adapter, rewards, train loop, callbacks
src/evaluation/     # Eval: tool call parsing, next_tool_prediction, ASR
src/etl/            # Data ingestion (ported from rrfa, run if re-processing needed)
configs/            # Tool schemas, GRPO configs
slurm/              # SLURM scripts (dual-cluster: Toronto AIS + Trillium CC)
scripts/            # CLI utilities
data/traces/        # Trace JSONL files (gitignored, copy manually)
```

## Conventions

- Traces use canonical format from rrfa repo (id, messages, labels, signal_hints)
- Tool calls parsed from Llama 3.1 `<|python_tag|>` format with JSON/function-style fallbacks
- Effective batch = per_device_batch_size * gradient_accumulation_steps (default: 4*8=32)
- SLURM scripts auto-detect cluster via `slurm/cluster_env.sh`
