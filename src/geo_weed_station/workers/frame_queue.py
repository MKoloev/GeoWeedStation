"""Thread-safe bounded frame queue with drop-oldest overflow policy."""

from __future__ import annotations

import threading
from collections import deque
from typing import Any


class FrameQueue:
    """Bounded FIFO queue; on overflow the oldest item is dropped (memory stays bounded)."""

    def __init__(self, maxlen: int = 4) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self.maxlen = maxlen
        self._items: deque[Any] = deque()
        self._cond = threading.Condition()
        self._dropped = 0

    def put(self, item: Any) -> bool:
        """Append item. Returns True if an older item had to be dropped."""
        with self._cond:
            dropped = False
            if len(self._items) >= self.maxlen:
                self._items.popleft()
                self._dropped += 1
                dropped = True
            self._items.append(item)
            self._cond.notify()
            return dropped

    def get(self, timeout: float | None = None) -> Any | None:
        """Pop the oldest item, waiting up to timeout seconds; None if empty."""
        with self._cond:
            if not self._items:
                self._cond.wait(timeout)
            return self._items.popleft() if self._items else None

    def clear(self) -> None:
        with self._cond:
            self._items.clear()

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)

    @property
    def is_full(self) -> bool:
        return len(self) >= self.maxlen

    @property
    def dropped(self) -> int:
        with self._cond:
            return self._dropped
