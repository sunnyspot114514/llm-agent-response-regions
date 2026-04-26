"""Shared memory bus for multi-agent communication."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from schemas.action import Action
from schemas.episode import Episode
from schemas.world_state import WorldState


class MemoryBus:
    """Agent 闂村叡浜姸鎬佹€荤嚎"""

    def __init__(self, log_dir: Optional[Path] = None):
        self.world_state = WorldState()
        self.action_history: list[dict] = []
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def record_round(
        self,
        round_id: int,
        actions: dict[str, Action],
        forbidden_triggers: dict[str, bool],
        metadata: Optional[dict[str, dict]] = None,
    ):
        """璁板綍涓€杞殑鎵€鏈夎涓恒€?"""

        metadata = metadata or {}
        record = {
            "round_id": round_id,
            "actions": {
                agent_id: {
                    "action": action.action_type,
                    "reason": action.reason,
                    "was_forbidden": forbidden_triggers.get(agent_id, False),
                    "parser_status": metadata.get(agent_id, {}).get("parser_status"),
                }
                for agent_id, action in actions.items()
            },
        }
        self.action_history.append(record)

    def get_last_round_actions(self) -> Optional[dict[str, Action]]:
        """鑾峰彇涓婁竴杞殑琛屼负"""
        if not self.action_history:
            return None

        last = self.action_history[-1]
        return {
            agent_id: Action(action_type=data["action"], reason=data.get("reason"))
            for agent_id, data in last["actions"].items()
        }

    def save_episode(self, episode: Episode):
        """淇濆瓨 episode 鏁版嵁."""
        filepath = self.log_dir / f"episode_{episode.episode_id:04d}.json"
        temp_file = filepath.with_suffix(".tmp")
        temp_file.write_text(
            episode.model_dump_json(indent=2),
            encoding="utf-8",
        )
        temp_file.replace(filepath)
        logger.info(f"Episode {episode.episode_id} saved to {filepath}")

    def reset(self):
        """閲嶇疆鐘舵€?"""
        self.world_state = WorldState()
        self.action_history = []
