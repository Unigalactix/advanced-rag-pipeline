"""An LRU cache for query results, with hit/miss statistics."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class QueryCache:
    """Bounded least-recently-used cache.

    Keys are normalised (lower-cased, whitespace-collapsed) by the caller or
    via :meth:`make_key`, so trivially re-typed repeat questions still hit.
    """

    def __init__(self, maxsize: int = 128) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._store: "OrderedDict[str, Any]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(question: str, top_k: int = 5) -> str:
        normalised = " ".join(question.strip().lower().split())
        return f"{top_k}::{normalised}"

    def get(self, key: str) -> Any | None:
        if key in self._store:
            self.hits += 1
            self._store.move_to_end(key)
            return self._store[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)  # evict least-recently-used

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hit_rate, 4), "size": len(self)}
