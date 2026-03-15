# Starter Prompt for Claude on Cluster

Copy-paste this into your first Claude session on Trillium.

---

## PROMPT

You're working on the RLVR tool-call safety project. Read the CLAUDE.md file first — it has full context.

Here's what needs to happen, in order:

### Phase 0: Environment check
1. Verify the venv works: `source /project/def-zhijing/memoozd/.venvs/rlvr_env/bin/activate && python -c "import torch; import trl; import peft; print('OK', torch.cuda.is_available())"`
2. If venv doesn't exist, run `bash slurm/setup_trillium.sh`
3. Verify trace data is symlinked: `ls -la /scratch/memoozd/cb-scratch/data/rlvr_traces/`
4. Verify model is cached: `ls /scratch/memoozd/cb-scratch/cache/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/`

### Phase 1: Data diagnostics (NO GPU needed — run on login node)
Do all of these and show me the output:

1. **Print 3 sample traces** — one benign, one harmful (attack_succeeded=True), one resisted — from `agentdojo_augmented.jsonl`. For each, show:
   - trace ID, category, attack_succeeded, attack_present
   - Full message structure (role, tool_calls, content preview)
   - signal_hints (especially injection_char_span)
   - The actual injection text if present (extract from the tool response at injection_char_span)

2. **Print 2 sample Fujitsu traces** — one harmful, one benign — from `fujitsu_b4_traces.jsonl`. Show same info plus `expected_tool_name` and `observed_tool_name` from signal_hints.

3. **Contrastive pair validation**: Load `contrastive_pairs.jsonl` and `agentdojo_augmented.jsonl`. For 5 random harmful traces that have contrastive pairs:
   - Show the harmful trace's last tool call (the WRONG tool the model called)
   - Show the benign pair's last tool call (the CORRECT tool)
   - Confirm they differ (true tool-flip) or are the same (arg-level attack)

4. **Dataset stats**:
   - Total traces by category (harmful/benign/resisted) for both agentdojo and fujitsu
   - How many harmful AgentDojo traces have contrastive pairs? How many of those are true tool-flips (different tool)?
   - How many unique tool names appear across all traces?
   - Average message count per trace

### Phase 2: Fix data.py — expected_tool derivation

`src/training/data.py` currently has `_extract_expected_tool()` that only checks `labels.expected_tool`, `tool_attack.expected_tool`, and `signal_hints.expected_tool_name`. But:
- AgentDojo traces have NONE of these fields (all 1225 harmful traces)
- Fujitsu traces DO have `signal_hints.expected_tool_name`

Fix `build_grpo_dataset()` to:
1. Accept an optional `contrastive_pairs_path` parameter
2. Load contrastive pairs as `{harmful_id: benign_id}` mapping
3. In `_extract_expected_tool()`, after the existing lookups fail, try:
   - Look up the harmful trace's benign pair via contrastive pairs
   - Load that benign trace's last tool call as the expected tool
4. This means `build_grpo_dataset` needs the full trace index (all traces by ID) available, not just the current trace

Also update `scripts/prepare_data.py` to accept `--contrastive-pairs` and pass it through.

### Phase 3: Validate GRPO dataset build
After fixing data.py:
```bash
python scripts/prepare_data.py \
    --traces /scratch/memoozd/cb-scratch/data/rlvr_traces/agentdojo_augmented.jsonl \
    --tool-schema configs/tool_schemas/agentdojo_v1.json \
    --contrastive-pairs /scratch/memoozd/cb-scratch/data/rlvr_traces/contrastive_pairs.jsonl \
    --validate
```
Show me the output. I want to see:
- How many prompts were built
- How many were skipped and why
- Category breakdown (harmful with injection vs benign)
- Sample prompt (the chat messages that will go to the model)

### Phase 4: Baseline eval (needs GPU — submit as SLURM job)
Run eval on the BASE model (no training) to establish baselines:
```bash
# Get an interactive GPU session
srun --account=def-zhijing --gres=gpu:1 --cpus-per-task=8 --mem=40G --time=02:00:00 --pty bash

# Run eval
python src/evaluation/eval_tool.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --eval-data /scratch/memoozd/cb-scratch/data/rlvr_traces/agentdojo_augmented.jsonl \
    --tool-schema configs/tool_schemas/agentdojo_v1.json \
    --max-samples 100 \
    --output /scratch/memoozd/cb-scratch/outputs/baseline_eval.json
```

This tells us how bad the base model is at resisting injections — the number we're trying to improve.

### Phase 5: First training run (submit as SLURM)
Once data and eval are validated:
```bash
POLICY=ignore \
TRACES=/scratch/memoozd/cb-scratch/data/rlvr_traces/agentdojo_augmented.jsonl \
MAX_SAMPLES=500 \
NUM_EPOCHS=1 \
sbatch --account=def-zhijing slurm/train.sbatch
```

Start small (500 samples, 1 epoch) to validate the training loop works end-to-end before committing to a full run.

---

## Key things to remember
- AgentDojo expected_tool comes from contrastive pairs, NOT from the trace itself
- Fujitsu expected_tool is in `signal_hints.expected_tool_name`
- Tool calls use Llama 3.1 `<|python_tag|>` format
- All SLURM jobs need `--account=def-zhijing` on Trillium
- Model is cached at `/scratch/memoozd/cb-scratch/cache/hf/hub/`
- Never commit under Claude's name — use git config user settings
