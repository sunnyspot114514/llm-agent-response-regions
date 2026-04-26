"""Strict JSON output protocol helpers for agent actions."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Optional, Sequence

from llama_cpp import LlamaGrammar

from schemas.action import Action

PROMPT_ECHO_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|begin_of_thought|>",
    "<|end_of_thought|>",
    "assistant:",
    "user:",
)

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)
_THINK_RE = re.compile(
    r"<think>[\s\n]*(.*?)[\s\n]*</think>",
    flags=re.DOTALL | re.IGNORECASE,
)
_BEGIN_THOUGHT_RE = re.compile(
    r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>",
    flags=re.DOTALL,
)
_DEEPSEEK_BEGIN_RE = re.compile(r"<\\begin(.*?)(?=\{|$)", flags=re.DOTALL)


def extract_think_content(output: str) -> Optional[str]:
    """Extract optional reasoning traces when a model emits tagged thoughts."""

    for pattern in (_THINK_RE, _BEGIN_THOUGHT_RE, _DEEPSEEK_BEGIN_RE):
        match = pattern.search(output)
        if match:
            return match.group(1).strip()
    return None


def _strip_known_wrappers(output: str) -> str:
    """Remove common wrappers around a JSON object without guessing an action."""

    stripped = output.strip()
    stripped = _THINK_RE.sub("", stripped).strip()
    stripped = _BEGIN_THOUGHT_RE.sub("", stripped).strip()
    stripped = _DEEPSEEK_BEGIN_RE.sub("", stripped).strip()

    fence_match = _CODE_FENCE_RE.match(stripped)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def _find_json_object(output: str) -> tuple[Optional[dict], bool]:
    """Return the first valid JSON object plus whether extra text surrounded it."""

    decoder = json.JSONDecoder()
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            parsed, end_index = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        prefix = output[:index].strip()
        suffix = output[index + end_index :].strip()
        return parsed, bool(prefix or suffix)
    return None, False


def parse_action_output(output: str, allowed_actions: Sequence[str]) -> tuple[Optional[Action], dict]:
    """Parse a model response without falling back to free-text action guessing."""

    cleaned_output = _strip_known_wrappers(output)
    parsed_json, has_extra_text = _find_json_object(cleaned_output)
    think_content = extract_think_content(output)

    metadata = {
        "parser_status": "parse_failed",
        "prompt_echo_detected": any(marker in output for marker in PROMPT_ECHO_MARKERS),
        "output_has_extra_text": has_extra_text,
        "think_content": think_content,
    }

    if parsed_json is None:
        return None, metadata

    action_type = str(parsed_json.get("action", "")).strip().lower()
    if action_type not in allowed_actions:
        metadata["parser_status"] = "invalid_action"
        return None, metadata

    reason = parsed_json.get("reason")
    if reason is not None and not isinstance(reason, str):
        reason = json.dumps(reason, ensure_ascii=False)

    metadata["parser_status"] = "json_with_extra_text" if has_extra_text else "valid_json"
    return Action(action_type=action_type, reason=reason), metadata


@lru_cache(maxsize=16)
def get_action_output_grammar(action_choices: tuple[str, ...]) -> LlamaGrammar:
    """Create a JSON grammar that forces the model to emit one valid action object."""

    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(action_choices),
            },
            "reason": {
                "type": "string",
            },
        },
        "required": ["action", "reason"],
        "additionalProperties": False,
    }
    return LlamaGrammar.from_json_schema(
        json.dumps(schema, ensure_ascii=False),
        verbose=False,
    )
