"""Action definitions for the multi-agent environment."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal["cooperate", "defect", "defend", "negotiate", "abstain"]
NormMode = Literal["soft", "strong", "hard_mask", "multi_rule"]


class Action(BaseModel):
    """One agent action."""

    action_type: ActionType
    target: Optional[str] = None
    reason: Optional[str] = None


class ActionSpace(BaseModel):
    """Allowed actions plus norm-binding metadata."""

    allowed: list[ActionType] = Field(
        default_factory=lambda: ["cooperate", "defect", "defend", "negotiate", "abstain"]
    )
    forbidden: list[ActionType] = Field(default_factory=list)
    norm_mode: NormMode = "soft"
    norm_rules: list[str] = Field(default_factory=list)

    def effective_allowed(self) -> list[ActionType]:
        """Actions that the decoder may emit under the current norm mode."""

        if self.norm_mode != "hard_mask":
            return list(self.allowed)
        forbidden = set(self.forbidden)
        return [action for action in self.allowed if action not in forbidden]

    def is_valid(self, action: ActionType) -> bool:
        return action in self.effective_allowed()

    def is_forbidden(self, action: ActionType) -> bool:
        return action in self.forbidden
