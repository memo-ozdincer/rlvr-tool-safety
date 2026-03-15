"""
Convert canonical traces to GRPO-compatible prompt datasets.

Each trace becomes a prompt = conversation up to (but not including)
the last assistant tool-call turn. The model must generate the tool call.
Tool definitions are embedded in the system message so GRPOTrainer's
chat template works without needing a `tools` parameter.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset

logger = logging.getLogger(__name__)


def load_traces(path: Path, max_samples: Optional[int] = None) -> list[dict]:
    """Load traces from JSONL file."""
    traces = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            traces.append(json.loads(line))
            if max_samples and len(traces) >= max_samples:
                break
    return traces


def load_tool_schema(path: Path) -> list[dict]:
    """Load tool definitions from schema file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("tools", data if isinstance(data, list) else [])


def _format_tools_for_system(tools: list[dict]) -> str:
    """Format tool definitions as text for embedding in system message."""
    formatted = []
    for tool in tools:
        fn = tool.get("function", tool)
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        param_lines = []
        for pname, pinfo in props.items():
            req = " (required)" if pname in required else ""
            param_lines.append(f"    - {pname}: {pinfo.get('type', 'any')}{req} — {pinfo.get('description', '')}")

        params_text = "\n".join(param_lines) if param_lines else "    (no parameters)"
        formatted.append(f"  {name}: {desc}\n{params_text}")

    return "Available tools:\n" + "\n".join(formatted)


def _extract_expected_tool(trace: dict, last_tc_msg: dict) -> Optional[str]:
    """Extract the expected (correct) tool from a trace.

    For benign traces: the tool in the last assistant turn IS the correct tool.
    For harmful traces: we need the expected_tool from labels/signal_hints.
    """
    labels = trace.get("labels", {})
    attack_succeeded = labels.get("attack_succeeded", False)

    # If attack succeeded, the tool in the trace is WRONG — get the expected one
    if attack_succeeded:
        # Check various locations for expected_tool
        for field_path in [
            ("labels", "expected_tool"),
            ("tool_attack", "expected_tool"),
            ("signal_hints", "expected_tool_name"),
        ]:
            obj = trace
            for key in field_path:
                obj = obj.get(key, {}) if isinstance(obj, dict) else None
                if obj is None:
                    break
            if isinstance(obj, str) and obj:
                return obj
        # Fallback: can't determine expected tool for this harmful trace
        return None

    # Benign/resisted: the tool in the trace is correct
    tcs = last_tc_msg.get("tool_calls", [])
    if tcs:
        fn = tcs[0].get("function", {})
        return fn.get("name")
    return None


def _convert_messages_for_chat(
    messages: list[dict], tools_text: str
) -> list[dict[str, str]]:
    """Convert trace messages to chat format suitable for apply_chat_template.

    - Embeds tool definitions in the system message
    - Converts 'tool' role to 'user' role (GRPOTrainer doesn't handle tool role)
    - Formats assistant tool_calls as text in content
    """
    converted = []
    system_seen = False

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "system":
            content = f"{content}\n\n{tools_text}"
            converted.append({"role": "system", "content": content})
            system_seen = True

        elif role == "tool":
            # Convert tool response to user message
            name = msg.get("name", "tool")
            converted.append({
                "role": "user",
                "content": f"[Tool Result from {name}]:\n{content}",
            })

        elif role == "assistant":
            parts = [content] if content.strip() else []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    parts.append(f"{name}({args})")
                else:
                    parts.append(f"{name}({json.dumps(args)})")
            converted.append({"role": "assistant", "content": "\n".join(parts)})

        elif role == "user":
            converted.append({"role": "user", "content": content})

    # Ensure system message exists
    if not system_seen:
        converted.insert(0, {
            "role": "system",
            "content": f"You are a helpful assistant.\n\n{tools_text}",
        })

    return converted


def build_grpo_dataset(
    traces_path: Path,
    tool_schema_path: Path,
    mode: str = "ignore",
    max_samples: Optional[int] = None,
) -> Dataset:
    """Build a GRPO-compatible dataset from canonical traces.

    Args:
        traces_path: Path to JSONL traces file.
        tool_schema_path: Path to tool schema JSON.
        mode: "ignore" or "reject" — determines which traces to include.
        max_samples: Limit number of samples.

    Returns:
        HuggingFace Dataset with columns:
        - prompt: list of message dicts (chat format)
        - expected_tool: str — correct tool name
        - is_injection: bool — whether prompt contains injection
        - trace_id: str — source trace ID
        - category: str — harmful/benign/resisted
    """
    traces = load_traces(traces_path, max_samples)
    tools = load_tool_schema(tool_schema_path)
    tools_text = _format_tools_for_system(tools)

    logger.info("Loaded %d traces from %s", len(traces), traces_path)

    records = []
    skipped = {"no_tool_call": 0, "no_expected_tool": 0, "too_short": 0}

    for trace in traces:
        messages = trace.get("messages", [])
        labels = trace.get("labels", {})

        # Find last assistant message with tool_calls
        last_tc_idx = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                last_tc_idx = i
                break

        if last_tc_idx is None:
            skipped["no_tool_call"] += 1
            continue

        # Extract expected tool
        expected_tool = _extract_expected_tool(trace, messages[last_tc_idx])
        if not expected_tool:
            skipped["no_expected_tool"] += 1
            continue

        # Truncate to before this turn
        prompt_messages = messages[:last_tc_idx]

        # Need at least system + user
        if len(prompt_messages) < 2:
            skipped["too_short"] += 1
            continue

        # Convert to chat format
        chat_messages = _convert_messages_for_chat(prompt_messages, tools_text)

        is_injection = labels.get("attack_present", False)
        category = labels.get("category", "unknown")

        records.append({
            "prompt": chat_messages,
            "expected_tool": expected_tool,
            "is_injection": is_injection,
            "trace_id": trace.get("id", ""),
            "category": category,
        })

    logger.info(
        "Built %d prompts (skipped: %s)", len(records), skipped
    )

    if not records:
        raise ValueError(
            f"No valid prompts built from {traces_path}. "
            f"Skipped: {skipped}"
        )

    return Dataset.from_list(records)
