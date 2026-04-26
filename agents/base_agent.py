"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from schemas.action import Action, ActionSpace
from schemas.agent_state import AgentState
from schemas.world_state import WorldState


class BaseAgent(ABC):
    """Agent 鍩虹被"""

    def __init__(
        self,
        agent_id: str,
        action_space: Optional[ActionSpace] = None,
        prompt_config: Optional[dict] = None,
    ):
        self.agent_id = agent_id
        self.action_space = action_space or ActionSpace()
        self.prompt_config = prompt_config or {}
        self.state = AgentState(agent_id=agent_id, model_name="base")

    @abstractmethod
    def decide(
        self,
        world_state: WorldState,
        other_agents: list["BaseAgent"],
        last_round_actions: Optional[dict[str, Action]] = None,
        observation_context: Optional[dict] = None,
    ) -> Action | tuple[Action, dict]:
        """
        鍐崇瓥锛氭牴鎹綋鍓嶇姸鎬侀€夋嫨琛屼负

        Args:
            world_state: 褰撳墠涓栫晫鐘舵€?
            other_agents: 鍏朵粬 agent 鍒楄〃
            last_round_actions: 涓婁竴杞悇 agent 鐨勮涓?

        Returns:
            閫夋嫨鐨勮涓?
        """

    def update_state(self, action: Action, result: dict):
        """鏇存柊 agent 鐘舵€?"""
        self.state.record_action(action.action_type)

    def reset(self):
        """閲嶇疆鐘舵€?"""
        self.state = AgentState(
            agent_id=self.agent_id,
            model_name=self.state.model_name,
        )
