"""Episode and Round schemas."""

from collections import Counter
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .action import Action
from .failure_record import FailureRecord


class AgentAction(BaseModel):
    """鍗曚釜 agent 鍦ㄦ煇鍥炲悎鐨勮涓?"""

    agent_id: str
    action: Action
    was_forbidden: bool = False
    inference_time_ms: int = 0
    load_time_ms: int = 0
    parser_status: str = "unknown"
    raw_output: Optional[str] = None
    think_content: Optional[str] = None
    prompt_echo_detected: bool = False
    output_has_extra_text: bool = False
    error_message: Optional[str] = None


class Round(BaseModel):
    """鍗曞洖鍚堟暟鎹?"""

    round_id: int
    actions: list[AgentAction] = Field(default_factory=list)
    world_state_snapshot: Optional[dict] = None


class Episode(BaseModel):
    """鍗曟瀹為獙 episode"""

    schema_version: int = 2
    episode_id: int
    seed: int

    # 閰嶇疆
    experiment_name: Optional[str] = None
    agent_ids: list[str]
    agent_models: dict[str, str] = Field(default_factory=dict)
    norm_config: Optional[dict] = None
    social_exposure_config: Optional[dict] = None
    observation_config: Optional[dict] = None
    prompt_config: Optional[dict] = None
    task_variant: Optional[str] = None

    # 鏁版嵁
    rounds: list[Round] = Field(default_factory=list)
    failures: list[FailureRecord] = Field(default_factory=list)

    # 鍏冩暟鎹?
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_rounds: int = 0

    # 缁熻
    total_forbidden_triggers: int = 0
    parser_failure_count: int = 0
    parser_status_counts: dict[str, int] = Field(default_factory=dict)

    def add_round(self, round_data: Round):
        self.rounds.append(round_data)
        self.total_rounds = len(self.rounds)

    def finalize(self):
        self.end_time = datetime.now()
        self.total_forbidden_triggers = sum(
            1 for round_data in self.rounds for action in round_data.actions if action.was_forbidden
        )
        parser_status_counts = Counter(
            action.parser_status for round_data in self.rounds for action in round_data.actions
        )
        self.parser_status_counts = dict(parser_status_counts)
        self.parser_failure_count = (
            parser_status_counts.get("parse_failed", 0)
            + parser_status_counts.get("invalid_action", 0)
        )
