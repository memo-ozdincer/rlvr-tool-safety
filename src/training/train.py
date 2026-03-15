#!/usr/bin/env python3
"""
GRPO training for tool-call safety.

Usage:
    # Ignore policy (call correct tool despite injection)
    python src/training/train.py \
        --traces data/traces/agentdojo_augmented.jsonl \
        --tool-schema configs/tool_schemas/agentdojo_v1.json \
        --policy ignore \
        --output-dir outputs/grpo_ignore

    # Reject policy (refuse tool call on injection)
    python src/training/train.py \
        --traces data/traces/agentdojo_augmented.jsonl \
        --tool-schema configs/tool_schemas/agentdojo_v1.json \
        --policy reject \
        --output-dir outputs/grpo_reject
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="GRPO training for tool-call safety")

    # Data
    p.add_argument("--traces", type=Path, required=True,
                    help="Path to JSONL traces file")
    p.add_argument("--tool-schema", type=Path, required=True,
                    help="Path to tool schema JSON")
    p.add_argument("--max-samples", type=int, default=None,
                    help="Limit number of training samples")

    # Policy
    p.add_argument("--policy", choices=["ignore", "reject"], required=True,
                    help="Training policy: ignore (call correct tool) or reject (refuse on injection)")

    # Model
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                    help="Base model ID or path")
    p.add_argument("--load-in-4bit", action="store_true",
                    help="Use 4-bit quantization (QLoRA)")

    # LoRA
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # GRPO
    p.add_argument("--num-generations", type=int, default=8,
                    help="G: completions per prompt")
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--beta", type=float, default=0.04,
                    help="KL penalty coefficient")
    p.add_argument("--loss-type", type=str, default="grpo",
                    choices=["grpo", "dr_grpo", "dapo"],
                    help="GRPO loss variant")

    # Training
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--output-dir", type=Path, required=True)

    # Logging
    p.add_argument("--wandb-project", type=str, default="rlvr-tool-safety")
    p.add_argument("--wandb-run", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--log-completions-every", type=int, default=50,
                    help="Log sample completions to wandb every N steps")

    # vLLM
    p.add_argument("--use-vllm", action="store_true", default=True,
                    help="Use vLLM for fast generation (default: True)")
    p.add_argument("--no-vllm", dest="use_vllm", action="store_false")

    return p.parse_args()


def main():
    args = parse_args()

    # =========================================================================
    # Load model with Unsloth
    # =========================================================================
    try:
        from unsloth import FastLanguageModel, PatchFastRL
        PatchFastRL("unsloth", FastLanguageModel)
        USE_UNSLOTH = True
        logger.info("Using Unsloth for efficient LoRA + GRPO")
    except ImportError:
        USE_UNSLOTH = False
        logger.info("Unsloth not available — using standard HF + PEFT")

    if USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_prompt_length + args.max_completion_length,
            load_in_4bit=args.load_in_4bit,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            use_rslora=True,
            use_gradient_checkpointing="unsloth",
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    # =========================================================================
    # Build dataset
    # =========================================================================
    from src.training.data import build_grpo_dataset

    dataset = build_grpo_dataset(
        traces_path=args.traces,
        tool_schema_path=args.tool_schema,
        max_samples=args.max_samples,
    )
    logger.info("Dataset: %d prompts", len(dataset))

    # =========================================================================
    # Select reward function
    # =========================================================================
    from src.training.rewards import reward_ignore, reward_reject, reward_format

    if args.policy == "ignore":
        reward_funcs = [reward_ignore, reward_format]
        reward_weights = [1.0, 0.1]
    else:
        reward_funcs = [reward_reject, reward_format]
        reward_weights = [1.0, 0.1]

    # =========================================================================
    # Configure GRPO
    # =========================================================================
    from trl import GRPOTrainer, GRPOConfig

    report_to = "none" if args.no_wandb else "wandb"

    grpo_config = GRPOConfig(
        output_dir=str(args.output_dir),
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        beta=args.beta,
        loss_type=args.loss_type,
        max_grad_norm=args.max_grad_norm,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=report_to,
        run_name=args.wandb_run or f"grpo_{args.policy}",
        scale_rewards=True,
        use_vllm=args.use_vllm,
    )

    # =========================================================================
    # Callbacks
    # =========================================================================
    callbacks = []
    if not args.no_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"grpo_{args.policy}",
            config=vars(args),
        )
        from src.training.callbacks import CompletionLoggingGRPOCallback
        callbacks.append(
            CompletionLoggingGRPOCallback(
                log_every=args.log_completions_every,
                max_samples=32,
            )
        )

    # =========================================================================
    # Train
    # =========================================================================
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        config=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info("Starting GRPO training: policy=%s", args.policy)
    trainer.train()

    # =========================================================================
    # Save
    # =========================================================================
    save_dir = args.output_dir / "final"
    if USE_UNSLOTH:
        model.save_pretrained_merged(
            str(save_dir / "merged"),
            tokenizer,
            save_method="merged_16bit",
        )
        model.save_pretrained(str(save_dir / "lora"))
    else:
        model.save_pretrained(str(save_dir / "lora"))
        tokenizer.save_pretrained(str(save_dir / "lora"))

    logger.info("Model saved to %s", save_dir)


if __name__ == "__main__":
    main()
