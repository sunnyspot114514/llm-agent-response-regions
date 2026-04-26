"""World state schema - minimal state machine."""

from typing import Literal

from pydantic import BaseModel, Field

ResourceLevel = Literal["scarce", "normal", "abundant"]
RiskLevel = Literal["safe", "risky", "dangerous"]


class WorldState(BaseModel):
    """Minimal world state passed into each agent turn."""

    resource_level: ResourceLevel = "normal"
    risk_level: RiskLevel = "safe"
    current_round: int = 0
    active_events: list[str] = Field(default_factory=list)

    def step(self):
        """Advance the environment by one round."""
        self.current_round += 1

    def to_prompt(self) -> str:
        """Convert the state into a stable text prompt."""
        events = ", ".join(self.active_events) if self.active_events else "none"
        return (
            "Current world state:\n"
            f"- resources: {self.resource_level}\n"
            f"- risk: {self.risk_level}\n"
            f"- round: {self.current_round}\n"
            f"- events: {events}"
        )
