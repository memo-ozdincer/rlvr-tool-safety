#!/usr/bin/env python3
"""
Validate that truncation points are correct for GRPO training.

For each harmful trace, confirms:
1. The truncation point is at the first tool call AFTER the injection
2. The tool called at the truncation point is the WRONG tool (attack succeeded)
3. The expected_tool (from contrastive pair) is DIFFERENT from the wrong tool
4. The injection text is visible in the prompt (appears before truncation)

Usage:
    python scripts/validate_truncation.py \
        --traces /path/to/agentdojo_augmented.jsonl \
        --contrastive-pairs /path/to/contrastive_pairs.jsonl \
        --tool-schema configs/tool_schemas/agentdojo_v1.json \
        --show-examples 5
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.data import (
    _find_truncation_point,
    _extract_expected_tool,
    _build_task_index,
    load_traces,
)


def extract_injection_text(trace: dict) -> str | None:
    """Extract the actual injection text from a trace using injection_char_span."""
    hint = trace.get("signal_hints", {}).get("injection_char_span", {})
    msg_idx = hint.get("message_index")
    char_start = hint.get("char_start")
    char_end = hint.get("char_end")

    if msg_idx is None or char_start is None or char_end is None:
        return None

    messages = trace.get("messages", [])
    if msg_idx >= len(messages):
        return None

    content = messages[msg_idx].get("content", "") or ""
    if char_end <= len(content):
        return content[char_start:char_end]
    return None


def main():
    parser = argparse.ArgumentParser(description="Validate truncation points")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--tool-schema", type=Path, default=None)
    parser.add_argument("--show-examples", type=int, default=5)
    parser.add_argument("--max-traces", type=int, default=None)
    args = parser.parse_args()

    traces = load_traces(args.traces, args.max_traces)
    task_index = _build_task_index(traces)

    print(f"Loaded {len(traces)} traces, {len(task_index)} original benign tasks\n")

    # =========================================================================
    # Validate harmful traces
    # =========================================================================
    harmful = [t for t in traces if t["labels"].get("attack_succeeded")]
    print(f"Harmful traces (attack_succeeded=True): {len(harmful)}")

    stats = Counter()
    examples_shown = 0
    issues = []

    for trace in harmful:
        tid = trace["id"]
        messages = trace["messages"]
        trunc_idx, trunc_type = _find_truncation_point(trace)

        if trunc_idx is None:
            stats["no_truncation_point"] += 1
            continue

        stats[f"trunc_{trunc_type}"] += 1

        # What tool does the model call at the truncation point?
        wrong_tool = None
        if trunc_idx < len(messages):
            trunc_msg = messages[trunc_idx]
            if trunc_msg.get("tool_calls"):
                wrong_tool = trunc_msg["tool_calls"][0].get("function", {}).get("name")
        else:
            trunc_msg = None  # full_trace_after_injection: no decision msg in trace

        # What should it have called?
        expected_tool = _extract_expected_tool(trace, trunc_msg, task_index)

        # Injection location
        hint = trace.get("signal_hints", {}).get("injection_char_span", {})
        inj_msg_idx = hint.get("message_index")

        # All tool call indices
        tc_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]

        # Validate expected vs wrong tool
        if expected_tool and wrong_tool:
            if expected_tool != wrong_tool:
                stats["true_tool_flip"] += 1
            else:
                stats["same_tool_attack"] += 1
        elif expected_tool and trunc_type == "full_trace_after_injection":
            stats["full_trace_with_expected"] += 1
        elif expected_tool is None:
            stats["no_expected_tool"] += 1

        # Check injection is visible in prompt
        if inj_msg_idx is not None and inj_msg_idx < trunc_idx:
            stats["injection_in_prompt"] += 1
        elif inj_msg_idx is not None and inj_msg_idx >= trunc_idx:
            stats["INJECTION_AFTER_TRUNCATION"] += 1
            issues.append(f"  {tid}: injection at msg[{inj_msg_idx}] but truncation at msg[{trunc_idx}]")

        # Show examples
        if examples_shown < args.show_examples:
            print(f"\n{'='*70}")
            print(f"TRACE: {tid}")
            print(f"  Messages: {len(messages)}")
            print(f"  Injection msg: {inj_msg_idx}")
            full = " (FULL TRACE)" if trunc_idx == len(messages) else ""
            print(f"  Truncation msg: {trunc_idx}{full} (model sees msgs[:{trunc_idx}])")
            print(f"  Truncation type: {trunc_type}")
            print(f"  Tool calls at: {tc_indices}")

            # Show message summary
            for i, m in enumerate(messages):
                role = m["role"]
                tc = m.get("tool_calls")
                marker = ""
                if i == trunc_idx:
                    marker = " ◄◄ TRUNCATION (model generates this)"
                if i == inj_msg_idx:
                    marker = " ◄◄ INJECTION HERE"
                content_preview = (m.get("content", "") or "")[:80].replace("\n", " ")
                tc_info = ""
                if tc:
                    tc_info = f" → {tc[0]['function']['name']}()"
                print(f"    [{i}] {role}{tc_info}: {content_preview}...{marker}")

            print(f"\n  Wrong tool (in trace):   {wrong_tool}")
            print(f"  Expected tool (correct): {expected_tool}")
            if expected_tool and wrong_tool:
                print(f"  Tool flip: {'YES' if expected_tool != wrong_tool else 'NO (same tool, different args?)'}")

            inj_text = extract_injection_text(trace)
            if inj_text:
                print(f"  Injection text: {inj_text[:150]}...")

            examples_shown += 1

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("TRUNCATION VALIDATION SUMMARY")
    print(f"{'='*70}")
    for k, v in sorted(stats.items()):
        pct = f" ({100*v/len(harmful):.1f}%)" if harmful else ""
        print(f"  {k}: {v}{pct}")

    if issues:
        print(f"\nISSUES ({len(issues)}):")
        for issue in issues[:10]:
            print(issue)

    # =========================================================================
    # Benign traces sanity check
    # =========================================================================
    benign = [t for t in traces if t["labels"].get("category") == "benign"]
    print(f"\nBenign traces: {len(benign)}")
    benign_with_tc = sum(1 for t in benign if _find_truncation_point(t) is not None)
    print(f"  With tool calls (usable): {benign_with_tc}")

    # Show one benign example
    for t in benign[:1]:
        trunc = _find_truncation_point(t)
        if trunc is not None:
            msg = t["messages"][trunc]
            tool = msg["tool_calls"][0]["function"]["name"] if msg.get("tool_calls") else "?"
            print(f"  Example: {t['id'][:40]} → expected tool: {tool}")

    print(f"\nValidation complete.")


if __name__ == "__main__":
    main()
