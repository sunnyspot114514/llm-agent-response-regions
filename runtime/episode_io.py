"""Shared helpers for reading normalized episode artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def normalize_episode_artifact(data: Any) -> dict:
    """Normalize legacy and current episode payloads to a single dict shape."""

    if isinstance(data, dict) and "rounds" in data:
        normalized = dict(data)
        normalized.setdefault("schema_version", 2)
        normalized.setdefault("failures", [])
        normalized.setdefault("social_exposure_config", None)
        normalized.setdefault("observation_config", None)
        normalized.setdefault("prompt_config", None)
        normalized.setdefault("task_variant", None)
        return normalized

    if isinstance(data, list):
        return {
            "schema_version": 1,
            "rounds": data,
            "failures": [],
            "social_exposure_config": None,
            "observation_config": None,
            "prompt_config": None,
            "task_variant": None,
        }

    raise ValueError("Unsupported episode payload shape")


def load_episode_artifact(filepath: Path | str) -> dict:
    """Load one episode JSON file and normalize it."""

    path = Path(filepath)
    return normalize_episode_artifact(json.loads(path.read_text(encoding="utf-8")))


def get_rounds(artifact: dict) -> list[dict]:
    return artifact.get("rounds", [])


def get_episode_agent_ids(artifact: dict) -> list[str]:
    agent_ids = artifact.get("agent_ids")
    if agent_ids:
        return list(agent_ids)

    rounds = get_rounds(artifact)
    if not rounds:
        return []

    actions = rounds[0].get("actions", {})
    if isinstance(actions, dict):
        return list(actions.keys())
    return [
        action_data.get("agent_id")
        for action_data in actions
        if action_data.get("agent_id") is not None
    ]


def iter_round_action_rows(artifact: dict) -> Iterable[dict]:
    """Yield per-agent action rows across legacy and current schemas."""

    for round_data in get_rounds(artifact):
        round_id = round_data.get("round_id")
        actions = round_data.get("actions", {})

        if isinstance(actions, dict):
            for agent_id, action_data in actions.items():
                yield {
                    "round_id": round_id,
                    "agent_id": agent_id,
                    "action": str(action_data.get("action", "")).lower(),
                    "reason": action_data.get("reason"),
                    "was_forbidden": bool(action_data.get("was_forbidden", False)),
                    "parser_status": action_data.get("parser_status"),
                    "raw_output": action_data.get("raw_output"),
                }
            continue

        for action_data in actions:
            action_obj = action_data.get("action", {})
            if isinstance(action_obj, dict):
                action = str(action_obj.get("action_type", "")).lower()
                reason = action_obj.get("reason")
            else:
                action = str(action_obj).lower()
                reason = None
            yield {
                "round_id": round_id,
                "agent_id": action_data.get("agent_id"),
                "action": action,
                "reason": reason,
                "was_forbidden": bool(action_data.get("was_forbidden", False)),
                "parser_status": action_data.get("parser_status"),
                "raw_output": action_data.get("raw_output"),
            }


def get_agent_action_sequence(artifact: dict, agent_id: str) -> list[str]:
    """Return the ordered action sequence for one agent."""

    return [
        row["action"]
        for row in iter_round_action_rows(artifact)
        if row["agent_id"] == agent_id and row["action"]
    ]


def get_agent_forbidden_count(artifact: dict, agent_id: str) -> int:
    return sum(
        1
        for row in iter_round_action_rows(artifact)
        if row["agent_id"] == agent_id and row["was_forbidden"]
    )
