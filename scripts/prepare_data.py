#!/usr/bin/env python3
"""
Prepare and validate trace data for GRPO training.

This script:
1. Loads traces from the specified JSONL file
2. Validates trace format and required fields
3. Reports dataset statistics (harmful/benign/resisted split)
4. Optionally builds and validates the GRPO dataset

Usage:
    python scripts/prepare_data.py \
        --traces data/traces/agentdojo_augmented.jsonl \
        --tool-schema configs/tool_schemas/agentdojo_v1.json \
        --validate
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate trace data")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--tool-schema", type=Path, default=None)
    parser.add_argument("--validate", action="store_true",
                        help="Build GRPO dataset and validate")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--contrastive-pairs", type=Path, default=None,
                        help="Path to contrastive_pairs.jsonl for AgentDojo expected_tool derivation")
    args = parser.parse_args()

    # Load and analyze traces
    traces = []
    with open(args.traces) as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

    print(f"\nLoaded {len(traces)} traces from {args.traces}")

    # Category breakdown
    categories = Counter()
    sources = Counter()
    has_tool_calls = 0
    has_expected_tool = 0

    for t in traces:
        labels = t.get("labels", {})
        categories[labels.get("category", "unknown")] += 1
        sources[t.get("source", {}).get("dataset", "unknown")] += 1

        # Check for tool calls
        for msg in t.get("messages", []):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                has_tool_calls += 1
                break

        # Check for expected_tool
        for path in [
            ("labels", "expected_tool"),
            ("tool_attack", "expected_tool"),
            ("signal_hints", "expected_tool_name"),
        ]:
            obj = t
            for key in path:
                obj = obj.get(key, {}) if isinstance(obj, dict) else None
                if obj is None:
                    break
            if isinstance(obj, str) and obj:
                has_expected_tool += 1
                break

    print(f"\nCategories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    print(f"\nSources:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")

    print(f"\nHas tool calls: {has_tool_calls}/{len(traces)}")
    print(f"Has expected_tool metadata: {has_expected_tool}/{len(traces)}")

    # Count harmful with attack_succeeded but no expected_tool
    harmful_no_expected = 0
    for t in traces:
        labels = t.get("labels", {})
        if labels.get("category") == "harmful" and labels.get("attack_succeeded"):
            found = False
            for path in [
                ("labels", "expected_tool"),
                ("tool_attack", "expected_tool"),
                ("signal_hints", "expected_tool_name"),
            ]:
                obj = t
                for key in path:
                    obj = obj.get(key, {}) if isinstance(obj, dict) else None
                    if obj is None:
                        break
                if isinstance(obj, str) and obj:
                    found = True
                    break
            if not found:
                harmful_no_expected += 1

    if harmful_no_expected:
        print(f"\nWARNING: {harmful_no_expected} harmful (attack_succeeded) traces "
              f"missing expected_tool metadata in trace fields")
        if args.contrastive_pairs:
            print(f"  → Will derive expected_tool from contrastive pairs")
        else:
            print(f"  → These will be skipped unless --contrastive-pairs is provided")

    # Validate GRPO dataset build
    if args.validate and args.tool_schema:
        print("\n" + "=" * 50)
        print("GRPO DATASET VALIDATION")
        print("=" * 50)

        from src.training.data import build_grpo_dataset

        try:
            dataset = build_grpo_dataset(
                traces_path=args.traces,
                tool_schema_path=args.tool_schema,
                max_samples=args.max_samples,
                contrastive_pairs_path=args.contrastive_pairs,
            )
            print(f"\nDataset built: {len(dataset)} prompts")

            # Analyze dataset
            inj_count = sum(1 for r in dataset if r["is_injection"])
            benign_count = sum(1 for r in dataset if not r["is_injection"])
            cats = Counter(r["category"] for r in dataset)

            print(f"  Injection prompts: {inj_count}")
            print(f"  Benign prompts:    {benign_count}")
            print(f"  Categories: {dict(cats)}")

            # Check expected tools
            tool_counts = Counter(r["expected_tool"] for r in dataset)
            print(f"  Unique expected tools: {len(tool_counts)}")
            print(f"  Top 5 tools: {tool_counts.most_common(5)}")

            # Truncation stats
            trunc_counts = Counter(r.get("truncation_type", "unknown") for r in dataset)
            print(f"\n  Truncation types:")
            for k, v in trunc_counts.most_common():
                print(f"    {k}: {v}")

            # Sample prompt
            sample = dataset[0]
            print(f"\n  Sample prompt ({len(sample['prompt'])} messages):")
            for msg in sample["prompt"][:3]:
                content_preview = msg["content"][:100].replace("\n", " ")
                print(f"    [{msg['role']}]: {content_preview}...")
            print(f"  Expected tool: {sample['expected_tool']}")
            print(f"  Is injection: {sample['is_injection']}")
            print(f"  Truncation: {sample.get('truncation_type', 'unknown')}")

            print("\nValidation PASSED")

        except Exception as e:
            print(f"\nValidation FAILED: {e}")
            raise


if __name__ == "__main__":
    main()
