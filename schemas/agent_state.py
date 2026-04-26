"""Agent state schema."""

from typing import Optional

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Agent 鐘舵€?"""

    agent_id: str
    model_name: str

    # 璧勬簮鐘舵€?
    resources: float = 10.0

    # 绀句細鐘舵€?
    reputation: float = 0.5  # 0-1
    trust_scores: dict[str, float] = Field(default_factory=dict)  # 瀵瑰叾浠?agent 鐨勪俊浠诲害

    # 鍘嗗彶
    action_history: list[str] = Field(default_factory=list)  # 鏈€杩?k 姝ヨ涓?

    # 瑙勮寖鐘舵€?
    violation_count: int = 0

    def record_action(self, action: str, max_history: int = 10):
        """璁板綍琛屼负鍒板巻鍙?"""
        self.action_history.append(action)
        if len(self.action_history) > max_history:
            self.action_history = self.action_history[-max_history:]
