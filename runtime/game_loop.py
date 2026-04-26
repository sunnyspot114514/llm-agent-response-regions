"""Main game loop for multi-agent experiments."""

from __future__ import annotations

import time
from collections import Counter
from typing import Callable, Optional

from loguru import logger

from agents.base_agent import BaseAgent
from schemas.action import Action, ActionSpace
from schemas.episode import AgentAction, Episode, Round
from schemas.failure_record import FailureRecord
from .memory_bus import MemoryBus


class GameLoop:
    """Runs one episode at a time and persists the full artifact."""

    def __init__(
        self,
        agents: list[BaseAgent],
        action_space: ActionSpace,
        max_rounds: int = 50,
        memory_bus: Optional[MemoryBus] = None,
        observer_agents: Optional[list[BaseAgent]] = None,
        social_transcript_provider: Optional[Callable[[int], Optional[dict[str, Action]]]] = None,
        social_exposure_config: Optional[dict] = None,
        observation_config: Optional[dict] = None,
    ):
        self.agents = agents
        self.action_space = action_space
        self.max_rounds = max_rounds
        self.memory_bus = memory_bus or MemoryBus()
        self.observer_agents = observer_agents or []
        self.social_transcript_provider = social_transcript_provider
        self.social_exposure_config = social_exposure_config
        self.observation_config = observation_config or {"visibility_mode": "full"}

        for agent in self.agents:
            agent.action_space = action_space

        self.failures: list[FailureRecord] = []

    def run_episode(self, episode_id: int, seed: int = 42) -> Episode:
        logger.info(f"Starting episode {episode_id} with seed {seed}")

        self.failures = []
        self.memory_bus.reset()
        for agent in self.agents:
            agent.reset()

        episode = Episode(
            episode_id=episode_id,
            seed=seed,
            experiment_name=self.memory_bus.log_dir.name,
            agent_ids=[agent.agent_id for agent in self.agents],
            agent_models={agent.agent_id: agent.state.model_name for agent in self.agents},
            norm_config={
                "forbidden": self.action_space.forbidden,
                "mode": self.action_space.norm_mode,
                "rules": self.action_space.norm_rules,
            },
            social_exposure_config=self.social_exposure_config,
            observation_config=self.observation_config,
            prompt_config=getattr(self.agents[0], "prompt_config", {}) if self.agents else {},
            task_variant=(getattr(self.agents[0], "prompt_config", {}) or {}).get("task_variant"),
        )

        for round_id in range(self.max_rounds):
            round_data = self._run_round(episode_id, round_id)
            episode.add_round(round_data)
            self.memory_bus.world_state.step()

        episode.failures = list(self.failures)
        episode.finalize()
        self.memory_bus.save_episode(episode)

        logger.info(
            f"Episode {episode_id} finished: "
            f"{episode.total_rounds} rounds, "
            f"{episode.total_forbidden_triggers} forbidden triggers, "
            f"{episode.parser_failure_count} parser failures"
        )
        return episode

    def _run_round(self, episode_id: int, round_id: int) -> Round:
        world_state = self.memory_bus.world_state
        last_actions = self.memory_bus.get_last_round_actions()
        if self.social_transcript_provider:
            transcript_actions = self.social_transcript_provider(round_id) or {}
            if transcript_actions:
                merged = dict(last_actions or {})
                merged.update(transcript_actions)
                last_actions = merged

        round_actions: dict[str, Action] = {}
        forbidden_triggers: dict[str, bool] = {}
        agent_actions: list[AgentAction] = []
        agent_metadata: dict[str, dict] = {}

        for agent in self.agents:
            other_agents, visible_last_actions, observation_context = self._build_observation_inputs(
                agent, last_actions
            )
            start_time = time.time()

            try:
                result = agent.decide(
                    world_state,
                    other_agents,
                    visible_last_actions,
                    observation_context=observation_context,
                )
                if isinstance(result, tuple):
                    action, metadata = result
                else:
                    action, metadata = result, {}
            except Exception as exc:
                logger.error(f"Agent {agent.agent_id} error: {exc}")
                action = Action(action_type="abstain", reason="error")
                metadata = {
                    "error": str(exc),
                    "parser_status": "model_error",
                    "raw_output": None,
                    "think_content": None,
                    "prompt_echo_detected": False,
                    "output_has_extra_text": False,
                    "load_time_ms": 0,
                    "infer_time_ms": 0,
                }
                self.failures.append(
                    FailureRecord(
                        episode_id=episode_id,
                        round_id=round_id,
                        agent_id=agent.agent_id,
                        failure_type="model_error",
                        error_message=str(exc),
                    )
                )

            inference_time = int((time.time() - start_time) * 1000)
            parser_status = metadata.get("parser_status", "unknown")

            if parser_status in {"parse_failed", "invalid_action"}:
                self.failures.append(
                    FailureRecord(
                        episode_id=episode_id,
                        round_id=round_id,
                        agent_id=agent.agent_id,
                        failure_type="parse_error",
                        action_attempted=action.action_type,
                        raw_output=metadata.get("raw_output"),
                        error_message=parser_status,
                        context={
                            "prompt_echo_detected": metadata.get("prompt_echo_detected", False),
                            "output_has_extra_text": metadata.get("output_has_extra_text", False),
                        },
                    )
                )

            was_forbidden = self.action_space.is_forbidden(action.action_type)
            if was_forbidden:
                logger.warning(f"Agent {agent.agent_id} triggered forbidden: {action.action_type}")
                self.failures.append(
                    FailureRecord(
                        episode_id=episode_id,
                        round_id=round_id,
                        agent_id=agent.agent_id,
                        failure_type="norm_violation",
                        action_attempted=action.action_type,
                        raw_output=metadata.get("raw_output"),
                        context={"parser_status": parser_status},
                    )
                )
                agent.state.violation_count += 1

            round_actions[agent.agent_id] = action
            forbidden_triggers[agent.agent_id] = was_forbidden
            agent_metadata[agent.agent_id] = metadata

            agent_actions.append(
                AgentAction(
                    agent_id=agent.agent_id,
                    action=action,
                    was_forbidden=was_forbidden,
                    inference_time_ms=inference_time,
                    load_time_ms=metadata.get("load_time_ms", 0),
                    parser_status=parser_status,
                    raw_output=metadata.get("raw_output"),
                    think_content=metadata.get("think_content"),
                    prompt_echo_detected=metadata.get("prompt_echo_detected", False),
                    output_has_extra_text=metadata.get("output_has_extra_text", False),
                    error_message=metadata.get("error"),
                )
            )
            agent.update_state(action, {})
            self._apply_norm_effects(agent, was_forbidden)

        self.memory_bus.record_round(round_id, round_actions, forbidden_triggers, agent_metadata)
        return Round(
            round_id=round_id,
            actions=agent_actions,
            world_state_snapshot=world_state.model_dump(),
        )

    def _apply_norm_effects(self, agent: BaseAgent, was_forbidden: bool):
        """The strong norm adds an actual state penalty on violation."""

        if self.action_space.norm_mode != "strong" or not was_forbidden:
            return

        agent.state.resources = max(0.0, agent.state.resources - 2.0)
        agent.state.reputation = max(0.0, agent.state.reputation - 0.2)

    def _build_observation_inputs(
        self,
        agent: BaseAgent,
        last_actions: Optional[dict[str, Action]],
    ) -> tuple[list[BaseAgent], Optional[dict[str, Action]], dict]:
        mode = (self.observation_config or {}).get("visibility_mode", "full")
        observation_context = {"visibility_mode": mode}

        if mode == "full":
            other_agents = [other for other in self.agents if other.agent_id != agent.agent_id]
            other_agents.extend(self.observer_agents)
            return other_agents, last_actions, observation_context

        own_history = None
        if last_actions and agent.agent_id in last_actions:
            own_history = {agent.agent_id: last_actions[agent.agent_id]}

        if mode == "blind":
            hidden_peers = [
                other.agent_id for other in self.agents if other.agent_id != agent.agent_id
            ] + [observer.agent_id for observer in self.observer_agents]
            observation_context["hidden_peer_ids"] = hidden_peers
            return [], own_history, observation_context

        if mode == "aggregate":
            peer_counts = Counter()
            for actor_id, action in (last_actions or {}).items():
                if actor_id != agent.agent_id:
                    peer_counts[action.action_type] += 1
            observation_context["aggregate_last_actions"] = {
                action: peer_counts.get(action, 0) for action in self.action_space.allowed
            }
            observation_context["hidden_peer_ids"] = [
                other.agent_id for other in self.agents if other.agent_id != agent.agent_id
            ] + [observer.agent_id for observer in self.observer_agents]
            return [], own_history, observation_context

        other_agents = [other for other in self.agents if other.agent_id != agent.agent_id]
        other_agents.extend(self.observer_agents)
        return other_agents, last_actions, observation_context
