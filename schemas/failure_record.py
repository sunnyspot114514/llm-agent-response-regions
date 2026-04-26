"""Failure record schema."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

FailureType = Literal[
    "norm_violation",   # 瑙﹀彂 forbidden action
    "invalid_action",   # 杈撳嚭涓嶅湪 action space
    "timeout",          # 鎺ㄧ悊瓒呮椂
    "parse_error",      # 杈撳嚭鏍煎紡閿欒
    "model_error",      # 妯″瀷鍔犺浇/鎺ㄧ悊閿欒
]


class FailureRecord(BaseModel):
    """澶辫触璁板綍 - 鐮旂┒鏁版嵁"""

    # 瀹氫綅
    episode_id: int
    round_id: int
    agent_id: str

    # 澶辫触淇℃伅
    failure_type: FailureType
    action_attempted: Optional[str] = None
    raw_output: Optional[str] = None
    error_message: Optional[str] = None

    # 涓婁笅鏂?
    context: dict = Field(default_factory=dict)

    # 鍏冩暟鎹?
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return self.model_dump()
