"""
Wandb callback for logging trace outputs during GRPO training.

Logs a table of sample completions every N steps so you can see
exactly what the model is generating and how it's being scored.
"""

import logging
from typing import Optional

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class WandbTraceLogger(TrainerCallback):
    """Log sample completions to a wandb.Table every `log_every` steps.

    Columns logged:
    - step: training step
    - trace_id: source trace identifier
    - category: harmful/benign
    - is_injection: whether prompt contained injection
    - expected_tool: correct tool name
    - completion: model's raw completion text
    - observed_tool: parsed tool name from completion (or None)
    - reward: reward assigned to this completion
    - correct: whether observed == expected
    """

    def __init__(self, log_every: int = 50, max_samples: int = 20):
        self.log_every = log_every
        self.max_samples = max_samples
        self._wandb = None
        self._table_columns = [
            "step", "trace_id", "category", "is_injection",
            "expected_tool", "completion", "observed_tool",
            "reward", "correct",
        ]

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import wandb
            if wandb.run is not None:
                self._wandb = wandb
        except ImportError:
            logger.warning("wandb not installed — trace logging disabled")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._wandb is None or self._wandb.run is None:
            return
        if state.global_step % self.log_every != 0:
            return

        # GRPOTrainer stores completions in the most recent batch
        model = kwargs.get("model")
        if model is None:
            return

        # Log whatever completion data is available in logs
        self._log_from_logs(state.global_step, logs or {})

    def _log_from_logs(self, step: int, logs: dict):
        """Extract and log completion info from trainer logs."""
        if self._wandb is None:
            return

        # TRL's GRPOTrainer logs reward stats — log them as summary
        reward_keys = [k for k in logs if "reward" in k.lower()]
        if reward_keys:
            self._wandb.log(
                {f"rewards/{k}": logs[k] for k in reward_keys},
                step=step,
            )

    def log_completions(
        self,
        step: int,
        completions: list[str],
        rewards: list[float],
        metadata: list[dict],
    ):
        """Manually log a batch of completions. Call from training loop."""
        if self._wandb is None or self._wandb.run is None:
            return

        from src.training.rewards import parse_tool_call

        rows = []
        for i, (comp, rew, meta) in enumerate(
            zip(completions, rewards, metadata)
        ):
            if i >= self.max_samples:
                break
            observed = parse_tool_call(comp)
            expected = meta.get("expected_tool", "")
            rows.append([
                step,
                meta.get("trace_id", ""),
                meta.get("category", ""),
                meta.get("is_injection", False),
                expected,
                comp[:500],  # Truncate long completions
                observed or "(none)",
                rew,
                observed == expected if observed else False,
            ])

        if rows:
            table = self._wandb.Table(
                columns=self._table_columns, data=rows
            )
            self._wandb.log({"completions": table}, step=step)
            logger.info("Logged %d completions at step %d", len(rows), step)


class CompletionLoggingGRPOCallback(TrainerCallback):
    """Hooks into GRPOTrainer to capture and log completions.

    GRPOTrainer stores generation data internally. This callback
    captures it and logs to wandb tables.
    """

    def __init__(self, log_every: int = 50, max_samples: int = 32):
        self.log_every = log_every
        self.max_samples = max_samples
        self._wandb: Optional["wandb"] = None  # noqa: F821
        self._step_data: dict = {}

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import wandb
            if wandb.run is not None:
                self._wandb = wandb
                # Create a persistent table artifact
                logger.info("Wandb trace logging enabled (every %d steps)", self.log_every)
        except ImportError:
            pass

    def on_step_end(self, args, state, control, **kwargs):
        if self._wandb is None or self._wandb.run is None:
            return
        if state.global_step % self.log_every != 0:
            return

        # Access trainer's internal completion buffer if available
        trainer = kwargs.get("trainer") or kwargs.get("model")
        if trainer is None:
            return

        # GRPOTrainer stores _last_completions after each step
        completions = getattr(trainer, "_last_completions", None)
        rewards = getattr(trainer, "_last_rewards", None)
        prompts = getattr(trainer, "_last_prompts", None)

        if completions is None:
            return

        from src.training.rewards import parse_tool_call

        rows = []
        n = min(len(completions), self.max_samples)
        for i in range(n):
            comp = completions[i] if isinstance(completions[i], str) else str(completions[i])
            rew = float(rewards[i]) if rewards is not None and i < len(rewards) else 0.0
            observed = parse_tool_call(comp)

            rows.append([
                state.global_step,
                comp[:500],
                observed or "(none)",
                rew,
            ])

        if rows:
            table = self._wandb.Table(
                columns=["step", "completion", "observed_tool", "reward"],
                data=rows,
            )
            self._wandb.log(
                {"trace_outputs": table},
                step=state.global_step,
            )
