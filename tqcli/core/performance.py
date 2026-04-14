"""Performance monitoring — tracks tokens/second and triggers alerts."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from tqcli.config import PerformanceConfig


@dataclass
class PerfSample:
    timestamp: float
    tokens: int
    elapsed_s: float

    @property
    def tps(self) -> float:
        return self.tokens / self.elapsed_s if self.elapsed_s > 0 else 0.0


class PerformanceMonitor:
    """Tracks token generation performance and detects degradation."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self._samples: deque[PerfSample] = deque(maxlen=100)
        self._session_tokens: int = 0
        self._session_start: float = time.time()
        self._slow_count: int = 0
        self._total_inferences: int = 0

    def record(self, tokens: int, elapsed_s: float) -> PerfSample:
        sample = PerfSample(timestamp=time.time(), tokens=tokens, elapsed_s=elapsed_s)
        self._samples.append(sample)
        self._session_tokens += tokens
        self._total_inferences += 1
        if sample.tps < self.config.min_tokens_per_second:
            self._slow_count += 1
        return sample

    @property
    def current_tps(self) -> float:
        if not self._samples:
            return 0.0
        return self._samples[-1].tps

    @property
    def average_tps(self) -> float:
        if not self._samples:
            return 0.0
        total_tokens = sum(s.tokens for s in self._samples)
        total_time = sum(s.elapsed_s for s in self._samples)
        return total_tokens / total_time if total_time > 0 else 0.0

    @property
    def rolling_tps(self) -> float:
        """Average TPS over the last 5 samples."""
        recent = list(self._samples)[-5:]
        if not recent:
            return 0.0
        total_tokens = sum(s.tokens for s in recent)
        total_time = sum(s.elapsed_s for s in recent)
        return total_tokens / total_time if total_time > 0 else 0.0

    @property
    def is_below_threshold(self) -> bool:
        return self.rolling_tps < self.config.min_tokens_per_second and len(self._samples) >= 3

    @property
    def is_warning(self) -> bool:
        return (
            self.config.min_tokens_per_second
            <= self.rolling_tps
            < self.config.warning_tokens_per_second
            and len(self._samples) >= 3
        )

    @property
    def slow_ratio(self) -> float:
        if self._total_inferences == 0:
            return 0.0
        return self._slow_count / self._total_inferences

    @property
    def should_handoff(self) -> bool:
        return self.is_below_threshold and self.slow_ratio > 0.5

    def get_stats_display(self) -> dict:
        return {
            "current_tps": round(self.current_tps, 1),
            "average_tps": round(self.average_tps, 1),
            "rolling_tps": round(self.rolling_tps, 1),
            "session_tokens": self._session_tokens,
            "total_inferences": self._total_inferences,
            "slow_ratio": round(self.slow_ratio * 100, 1),
            "threshold_tps": self.config.min_tokens_per_second,
            "session_duration_s": round(time.time() - self._session_start, 1),
        }
