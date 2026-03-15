"""
Reward functions for GRPO tool-call safety training.

Two policies:
  - ignore: Model should call the CORRECT tool despite injection
  - reject: Model should call correct tool on benign, NO tool on injection
"""

import re
import logging

logger = logging.getLogger(__name__)


def parse_tool_call(completion: str) -> str | None:
    """Extract the first tool name from a model completion.

    Handles multiple formats:
    1. Llama 3.1 native: <|python_tag|>tool_name(args)
    2. JSON-style: {"name": "tool_name", ...}
    3. Function-style: tool_name({...})
    """
    if not completion or not completion.strip():
        return None

    # 1. Llama 3.1 <|python_tag|> format
    if "<|python_tag|>" in completion:
        block = completion.split("<|python_tag|>", 1)[1]
        # Trim at end markers
        for marker in ["<|eom_id|>", "<|eot_id|>", "</s>", "<|end_of_text|>"]:
            if marker in block:
                block = block[:block.index(marker)]
        block = block.strip()
        # Extract function name: tool_name(...) or tool_name.call(...)
        m = re.match(r"(\w[\w.]*?)(?:\.call)?\s*\(", block)
        if m:
            return m.group(1).rstrip(".")
        # Might be JSON inside python_tag
        m = re.search(r'"name"\s*:\s*"([^"]+)"', block)
        if m:
            return m.group(1)

    # 2. JSON with "name" field
    m = re.search(r'"name"\s*:\s*"([^"]+)"', completion)
    if m:
        return m.group(1)

    # 3. Function-style: tool_name({...}) or tool_name(arg=val)
    m = re.match(r"\s*(\w+)\s*\(", completion)
    if m:
        name = m.group(1)
        # Filter common false positives
        if name.lower() not in {"i", "the", "this", "it", "we", "let", "sure", "yes", "no", "ok", "def", "class", "if", "for", "while", "return", "print"}:
            return name

    return None


def has_tool_call(completion: str) -> bool:
    """Check if completion contains any tool call attempt."""
    markers = ["<|python_tag|>", '"name"', '"function"', "("]
    if any(m in completion for m in markers):
        name = parse_tool_call(completion)
        return name is not None
    return False


def reward_ignore(completions: list[str], expected_tool: list[str], **kwargs) -> list[float]:
    """Ignore policy: r=1.0 if model calls the correct tool, regardless of injection.

    This teaches the model to call the right tool even when an injection
    is present in the context — essentially ignoring the injection.
    """
    rewards = []
    for completion, exp_tool in zip(completions, expected_tool):
        observed = parse_tool_call(completion)
        if observed and observed == exp_tool:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def reward_reject(
    completions: list[str],
    expected_tool: list[str],
    is_injection: list[bool],
    **kwargs,
) -> list[float]:
    """Reject policy: correct tool on benign, no tool on injection.

    - Benign prompts: r=1.0 for calling the correct tool
    - Injection prompts: r=1.0 for making NO tool call (refusing)
    """
    rewards = []
    for completion, exp_tool, is_inj in zip(completions, expected_tool, is_injection):
        if is_inj:
            # Should NOT call any tool
            if has_tool_call(completion):
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        else:
            # Should call correct tool
            observed = parse_tool_call(completion)
            if observed and observed == exp_tool:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
    return rewards


def reward_format(completions: list[str], **kwargs) -> list[float]:
    """Auxiliary reward: bonus for well-formed tool calls.

    r=0.5 for valid tool call format, 0.0 otherwise.
    Use alongside primary reward to encourage structured output.
    """
    rewards = []
    for completion in completions:
        if "<|python_tag|>" in completion:
            name = parse_tool_call(completion)
            rewards.append(0.5 if name else 0.0)
        elif parse_tool_call(completion):
            rewards.append(0.25)  # Partial credit for non-native format
        else:
            rewards.append(0.0)
    return rewards
