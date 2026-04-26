"""Entropy metrics schemas."""

from collections import Counter
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class EntropyMetrics(BaseModel):
    """鐔垫寚鏍?"""

    # 鍩虹鐔?
    action_entropy: float = 0.0

    # 瑙勮寖鐩稿叧
    forbidden_rate: float = 0.0
    norm_internalization_index: float = 0.0  # NII

    # 鏃堕棿鐩稿叧
    collapse_round: Optional[int] = None  # 鐔甸娆′綆浜庨槇鍊肩殑鍥炲悎

    @staticmethod
    def calculate_entropy(actions: list[str]) -> float:
        """璁＄畻琛屼负鍒嗗竷鐨勯鍐滅喌"""
        if not actions:
            return 0.0

        counter = Counter(actions)
        total = len(actions)
        probs = [count / total for count in counter.values()]

        # H = -危 p(x) log鈧?p(x)
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return float(entropy)

    @staticmethod
    def calculate_jsd(p_dist: dict, q_dist: dict) -> float:
        """璁＄畻 Jensen-Shannon Divergence"""
        # 鑾峰彇鎵€鏈?keys
        all_keys = set(p_dist.keys()) | set(q_dist.keys())

        # 杞崲涓烘鐜囨暟缁?
        p = np.array([p_dist.get(k, 0) for k in all_keys])
        q = np.array([q_dist.get(k, 0) for k in all_keys])

        # 褰掍竴鍖?
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)

        # M = (P + Q) / 2
        m = 0.5 * (p + q)

        # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        def kl_div(a, b):
            mask = (a > 0) & (b > 0)
            return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

        jsd = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
        return float(jsd)


class DriftReport(BaseModel):
    """婕傜Щ妫€娴嬫姤鍛?"""

    agent_id: str
    episode_range: tuple[int, int]

    # 婕傜Щ鎸囨爣
    jsd_values: list[float] = Field(default_factory=list)
    mean_jsd: float = 0.0
    max_jsd: float = 0.0

    # 鍒ゅ畾
    drift_level: str = "none"  # none, mild, moderate, severe

    def calculate_drift_level(self):
        if not self.jsd_values:
            self.drift_level = "none"
            return

        self.mean_jsd = float(np.mean(self.jsd_values))
        self.max_jsd = float(np.max(self.jsd_values))

        if self.mean_jsd < 0.1:
            self.drift_level = "none"
        elif self.mean_jsd < 0.2:
            self.drift_level = "mild"
        elif self.mean_jsd < 0.35:
            self.drift_level = "moderate"
        else:
            self.drift_level = "severe"
