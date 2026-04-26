"""Helpers for fixed transcript-based social exposure controls."""

from __future__ import annotations

from typing import Optional

from agents.base_agent import BaseAgent
from schemas.action import Action, ActionSpace, ActionType
from schemas.world_state import WorldState


class ScriptedPeer(BaseAgent):
    """Observer peer used only to expose a stable social context in prompts."""

    def __init__(self, agent_id: str, action_type: ActionType):
        super().__init__(agent_id, ActionSpace())
        self.default_action = action_type
        self.state.model_name = "scripted_peer"

    def decide(
        self,
        world_state: WorldState,
        other_agents: list[BaseAgent],
        last_round_actions: Optional[dict[str, Action]] = None,
    ) -> Action:
        return Action(action_type=self.default_action, reason="scripted_social_transcript")


def build_scripted_peers(config: Optional[dict]) -> list[ScriptedPeer]:
    if not config or not config.get("enabled"):
        return []
    peers: list[ScriptedPeer] = []
    seen: dict[str, ActionType] = {}

    if config.get("pattern"):
        for snapshot in config.get("pattern", []):
            for peer in snapshot.get("peers", []):
                seen.setdefault(peer["id"], peer["action"])
    else:
        for peer in config.get("peers", []):
            seen.setdefault(peer["id"], peer["action"])

    for peer_id, action_type in seen.items():
        peers.append(ScriptedPeer(agent_id=peer_id, action_type=action_type))
    return peers


def make_social_transcript_provider(config: Optional[dict]):
    if not config or not config.get("enabled"):
        return None

    start_round = int(config.get("start_round", 1))
    peers = list(config.get("peers", []))
    pattern = list(config.get("pattern", []))
    cycle = bool(config.get("cycle", True))

    def provider(round_id: int) -> Optional[dict[str, Action]]:
        if round_id < start_round:
            return None
        active_peers = peers
        if pattern:
            offset = round_id - start_round
            if not cycle and offset >= len(pattern):
                active_peers = pattern[-1].get("peers", [])
            else:
                active_peers = pattern[offset % len(pattern)].get("peers", [])
        return {
            peer["id"]: Action(
                action_type=peer["action"],
                reason="scripted_social_transcript",
            )
            for peer in active_peers
        }

    return provider
