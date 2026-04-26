"""Helpers for durable batch progress tracking."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

_EPISODE_FILE_RE = re.compile(r"episode_(\d+)\.json$")


def list_completed_episode_ids(log_dir: Path) -> set[int]:
    completed = set()
    for path in log_dir.glob("episode_*.json"):
        match = _EPISODE_FILE_RE.match(path.name)
        if match:
            completed.add(int(match.group(1)))
    return completed


def get_pending_episode_ids(log_dir: Path, num_episodes: int) -> list[int]:
    completed = list_completed_episode_ids(log_dir)
    return [episode_id for episode_id in range(num_episodes) if episode_id not in completed]


def load_existing_results(log_dir: Path) -> list[dict]:
    summary_path = log_dir / "batch_summary.json"
    if not summary_path.exists():
        return []
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    results = data.get("results", [])
    return list(results) if isinstance(results, list) else []


def save_batch_summary(log_dir: Path, experiment_name: str, new_results: list[dict], requested_episodes: int):
    """Merge new results with the existing summary and write atomically."""

    merged = {
        result["episode_id"]: result
        for result in load_existing_results(log_dir)
        if "episode_id" in result
    }
    for result in new_results:
        merged[result["episode_id"]] = result

    completed_ids = list_completed_episode_ids(log_dir)
    summary = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "episodes_completed": len(completed_ids),
        "requested_episodes": requested_episodes,
        "resumed_from": min(
            (result["episode_id"] for result in new_results if "episode_id" in result),
            default=0,
        ),
        "results": [merged[episode_id] for episode_id in sorted(merged)],
    }

    filepath = log_dir / "batch_summary.json"
    temp = filepath.with_suffix(".tmp")
    temp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(filepath)
