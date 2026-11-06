"""Exponential backoff strategy for reliable reconnection and retry logic."""

from __future__ import annotations


class ExponentialBackoff:
    """Calculates exponential backoff delays with monotonic growth up to max_s."""

    def __init__(
        self,
        initial_s: float = 1.0,
        max_s: float = 10.0,
        factor: float = 2.0,
    ) -> None:
        self.initial_s = max(0.01, float(initial_s))
        self.max_s = max(self.initial_s, float(max_s))
        self.factor = max(1.0, float(factor))
        self._current_s = self.initial_s

    def next_delay(self) -> float:
        """Return the current backoff delay and advance state exponentially."""
        delay = self._current_s
        self._current_s = min(self.max_s, self._current_s * self.factor)
        return delay

    def reset(self) -> None:
        """Reset delay back to initial_s."""
        self._current_s = self.initial_s
