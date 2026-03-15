"""
Convert canonical traces to GRPO-compatible prompt datasets.

Each trace becomes a prompt = conversation up to (but not including)
the decision-point assistant turn. The model must generate the tool call.

Truncation strategy:
  - Harmful traces: truncate at the FIRST assistant tool-call AFTER injection.
    This is the actual decision point where the model follows or resists.
  - Benign traces: truncate at the LAST assistant tool-call.

Tool formatting:
  - Ported from rrfa ETL_B: assistant tool calls rendered as Llama 3.1
    native format (<|python_tag|>{"name": ..., "parameters": ...})
  - Tool responses kept as ipython role content
  - System message enriched with tool definitions via apply_chat_template
    (falls back to text embedding if tokenizer unavailable)
"""

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Llama 3.1 tool call formatting (ported from rrfa ETL_B)
# =============================================================================

LLAMA_PYTHON_TAG = "<|python_tag|>"


def _format_tool_call_llama(tool_name: str, arguments: dict) -> str:
    """Format a tool call as Llama 3.1 JSON format.

    Ported from rrfa src/schemas/tools/ETL_B.py:_format_tool_call_json
    """
    return f'{LLAMA_PYTHON_TAG}{{"name": "{tool_name}", "parameters": {json.dumps(arguments, ensure_ascii=False)}}}'


def _format_assistant_with_tool_calls(msg: dict) -> str:
    """Format an assistant message with tool calls in Llama 3.1 style.

    Content + tool calls rendered as:
        <content>

        <|python_tag|>{"name": "tool", "parameters": {...}}
    """
    content = msg.get("content", "") or ""
    parts = [content] if content.strip() else []

    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        parts.append(_format_tool_call_llama(name, args))

    return "\n\n".join(parts) if parts else content


# =============================================================================
# Data loading
# =============================================================================

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
    """Format tool definitions as text for embedding in system message.

    Used as fallback when tokenizer is not available. When a tokenizer IS
    available, use enrich_system_with_tools() instead for native rendering.
    """
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


def enrich_system_with_tools(
    system_content: str,
    tools: list[dict],
    tokenizer=None,
) -> str:
    """Enrich system message with tool definitions using Llama's native format.

    Ported from rrfa ETL_B:_enrich_trace_system_with_tools.
    Uses tokenizer.apply_chat_template(tools=tools) to get the correct
    "Environment: ipython" format that triggers <|python_tag|> generation.

    Falls back to text embedding if tokenizer is unavailable.
    """
    if tokenizer is None:
        return f"{system_content}\n\n{_format_tools_for_system(tools)}"

    try:
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Hello"},
            ],
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
    except (TypeError, Exception):
        return f"{system_content}\n\n{_format_tools_for_system(tools)}"

    # Extract enriched system content from rendered text.
    # Llama format: <bos><|start_header_id|>system<|end_header_id|>\n\n{sys_content}<|eot_id|>...
    header_marker = "<|end_header_id|>\n\n"
    header_pos = rendered.find(header_marker)
    if header_pos < 0:
        return f"{system_content}\n\n{_format_tools_for_system(tools)}"

    content_start = header_pos + len(header_marker)
    eot_pos = rendered.find("<|eot_id|>", content_start)
    if eot_pos <= content_start:
        return f"{system_content}\n\n{_format_tools_for_system(tools)}"

    return rendered[content_start:eot_pos]


# =============================================================================
# Truncation point selection
# =============================================================================

def _find_truncation_point(trace: dict) -> tuple[Optional[int], str]:
    """Find the message index to truncate at (the decision point).

    Three cases for harmful traces with injection_char_span:

    1. Tool call AFTER injection exists → truncate there (first one).
       Prompt includes the injection. Model must decide: follow or resist.

    2. No tool call after injection (augmented traces truncated after
       injected tool response) → use FULL trace as prompt.
       Return len(messages) to signal "include everything."
       These are the best RLVR data: model sees injection, generates
       its own tool call from scratch.

    3. No injection span info → fall back to last tool call.

    For benign traces: last tool call.

    Returns:
        (trunc_idx, trunc_type) where trunc_idx is the message index to
        truncate at (exclusive). trunc_idx == len(messages) means "use
        everything as prompt." None means unusable trace.
    """
    messages = trace.get("messages", [])
    labels = trace.get("labels", {})
    is_harmful = labels.get("attack_succeeded", False)

    # Find all assistant tool-call indices
    tc_indices = [
        i for i, m in enumerate(messages)
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]

    # For harmful traces: use injection position to find decision point
    if is_harmful:
        hint = trace.get("signal_hints", {}).get("injection_char_span", {})
        inj_msg_idx = hint.get("message_index")

        if inj_msg_idx is not None:
            # Case 1: tool call after injection → truncate there
            tc_after = [i for i in tc_indices if i > inj_msg_idx]
            if tc_after:
                return tc_after[0], "first_after_injection"

            # Case 2: no tool call after injection → full trace as prompt.
            # The injection is in a tool response at the end of the trace.
            # Model generates the next assistant turn from scratch.
            # Verify injection is actually visible (at or before last message)
            if inj_msg_idx < len(messages):
                return len(messages), "full_trace_after_injection"

        # Case 3: no injection info → fall back
        if tc_indices:
            return tc_indices[-1], "last_tool_call_no_injection_info"
        return None, "no_tool_call"

    # Benign/resisted: use last tool call
    if tc_indices:
        return tc_indices[-1], "last_tool_call"
    return None, "no_tool_call"


# =============================================================================
# Expected tool extraction
# =============================================================================

def _extract_expected_tool(
    trace: dict,
    decision_msg: Optional[dict],
    contrastive_pairs: Optional[dict[str, str]] = None,
    trace_index: Optional[dict[str, dict]] = None,
) -> Optional[str]:
    """Extract the expected (correct) tool from a trace.

    For benign traces: the tool in the decision turn IS the correct tool.
    For harmful traces: derive from signal_hints or contrastive benign pair.

    decision_msg may be None for full-trace prompts (no assistant tool call
    in the trace — the model generates the first one).
    """
    labels = trace.get("labels", {})
    attack_succeeded = labels.get("attack_succeeded", False)

    if attack_succeeded:
        # Check signal_hints (Fujitsu traces have this)
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

        # Derive from contrastive benign pair (AgentDojo traces)
        if contrastive_pairs and trace_index:
            trace_id = trace.get("id", "")
            benign_id = contrastive_pairs.get(trace_id)
            if benign_id and benign_id in trace_index:
                benign_trace = trace_index[benign_id]
                for msg in reversed(benign_trace.get("messages", [])):
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        tcs = msg["tool_calls"]
                        if tcs:
                            return tcs[0].get("function", {}).get("name")

        return None

    # Benign/resisted: the tool in the decision turn is correct
    if decision_msg is not None:
        tcs = decision_msg.get("tool_calls", [])
        if tcs:
            return tcs[0].get("function", {}).get("name")
    return None


# =============================================================================
# Message conversion for GRPO prompts
# =============================================================================

def _convert_messages_for_chat(
    messages: list[dict],
    tools: list[dict],
    tokenizer=None,
) -> list[dict[str, str]]:
    """Convert trace messages to chat format for GRPOTrainer.

    - System message enriched with tool defs (native Llama format if tokenizer available)
    - Tool responses kept as 'ipython' role (Llama 3.1 native) or 'user' fallback
    - Assistant tool calls formatted as <|python_tag|> JSON
    """
    converted = []
    system_seen = False

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "system":
            content = enrich_system_with_tools(content, tools, tokenizer)
            converted.append({"role": "system", "content": content})
            system_seen = True

        elif role == "tool":
            # Llama 3.1 uses "ipython" role for tool responses.
            # GRPOTrainer may not support it → fall back to "user"
            name = msg.get("name", "tool")
            converted.append({
                "role": "user",
                "content": f"[Tool Result from {name}]:\n{content}",
            })

        elif role == "assistant":
            # Format tool calls in Llama 3.1 native style
            if msg.get("tool_calls"):
                formatted = _format_assistant_with_tool_calls(msg)
                converted.append({"role": "assistant", "content": formatted})
            else:
                converted.append({"role": "assistant", "content": content})

        elif role == "user":
            converted.append({"role": "user", "content": content})

    if not system_seen:
        enriched = enrich_system_with_tools("You are a helpful assistant.", tools, tokenizer)
        converted.insert(0, {"role": "system", "content": enriched})

    return converted


# =============================================================================
# Contrastive pairs
# =============================================================================

def _load_contrastive_pairs(path: Path) -> dict[str, str]:
    """Load contrastive pairs as {harmful_trace_id: benign_trace_id}."""
    pairs = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            pair = json.loads(line)
            pairs[pair["harmful_trace_id"]] = pair["benign_trace_id"]
    logger.info("Loaded %d contrastive pairs from %s", len(pairs), path)
    return pairs


# =============================================================================
# Dataset builder
# =============================================================================

def build_grpo_dataset(
    traces_path: Path,
    tool_schema_path: Path,
    mode: str = "ignore",
    max_samples: Optional[int] = None,
    contrastive_pairs_path: Optional[Path] = None,
    tokenizer=None,
) -> Dataset:
    """Build a GRPO-compatible dataset from canonical traces.

    Args:
        traces_path: Path to JSONL traces file.
        tool_schema_path: Path to tool schema JSON.
        mode: "ignore" or "reject" (both use same data, reward differs).
        max_samples: Limit number of samples.
        contrastive_pairs_path: Path to contrastive_pairs.jsonl for deriving
            expected_tool from benign pairs (needed for AgentDojo traces).
        tokenizer: Optional HuggingFace tokenizer for native tool rendering.
            If None, tools are embedded as text in the system message.

    Returns:
        HuggingFace Dataset with columns:
        - prompt: list of message dicts (chat format)
        - expected_tool: str — correct tool name
        - is_injection: bool — whether prompt contains injection
        - trace_id: str — source trace ID
        - category: str — harmful/benign/resisted
        - truncation_type: str — how the truncation point was chosen
    """
    traces = load_traces(traces_path, max_samples)
    tools = load_tool_schema(tool_schema_path)

    # Load contrastive pairs if provided
    contrastive_pairs = None
    trace_index = None
    if contrastive_pairs_path:
        contrastive_pairs = _load_contrastive_pairs(contrastive_pairs_path)
        trace_index = {t.get("id", ""): t for t in traces}

    logger.info("Loaded %d traces from %s", len(traces), traces_path)

    records = []
    skipped = {"no_tool_call": 0, "no_expected_tool": 0, "too_short": 0}
    truncation_stats = {}

    for trace in traces:
        messages = trace.get("messages", [])
        labels = trace.get("labels", {})

        # Find truncation point (injection-aware for harmful traces)
        trunc_idx, trunc_type = _find_truncation_point(trace)

        if trunc_idx is None:
            skipped["no_tool_call"] += 1
            continue

        truncation_stats[trunc_type] = truncation_stats.get(trunc_type, 0) + 1

        # Get the decision message (may be None for full-trace prompts)
        decision_msg = messages[trunc_idx] if trunc_idx < len(messages) else None

        # Extract expected tool
        expected_tool = _extract_expected_tool(
            trace, decision_msg, contrastive_pairs, trace_index
        )
        if not expected_tool:
            skipped["no_expected_tool"] += 1
            continue

        # Build prompt: everything BEFORE the decision turn.
        # For full_trace_after_injection: trunc_idx == len(messages), so
        # prompt_messages == messages (entire trace including injection).
        prompt_messages = messages[:trunc_idx]

        if len(prompt_messages) < 2:
            skipped["too_short"] += 1
            continue

        # Convert to chat format with proper tool rendering
        chat_messages = _convert_messages_for_chat(prompt_messages, tools, tokenizer)

        records.append({
            "prompt": chat_messages,
            "expected_tool": expected_tool,
            "is_injection": labels.get("attack_present", False),
            "trace_id": trace.get("id", ""),
            "category": labels.get("category", "unknown"),
            "truncation_type": trunc_type,
        })

    logger.info(
        "Built %d prompts (skipped: %s, truncation: %s)",
        len(records), skipped, truncation_stats,
    )

    if not records:
        raise ValueError(
            f"No valid prompts built from {traces_path}. "
            f"Skipped: {skipped}"
        )

    return Dataset.from_list(records)
