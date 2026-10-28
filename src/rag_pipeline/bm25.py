"""Okapi BM25, implemented from scratch with the standard library only."""

from __future__ import annotations

import math
from collections import Counter

from .chunking import Chunk
from .embeddings import tokenize


class BM25Index:
    """Sparse retrieval over a fixed corpus of chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._doc_freqs: list[Counter] = []
        self._idf: dict[str, float] = {}
        self._doc_lengths: list[int] = []
        self._avgdl: float = 0.0

    def __len__(self) -> int:
        return len(self.chunks)

    def build(self, chunks: list[Chunk]) -> "BM25Index":
        self.chunks = list(chunks)
        self._doc_freqs = []
        self._doc_lengths = []
        df: Counter = Counter()
        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            counts = Counter(tokens)
            self._doc_freqs.append(counts)
            self._doc_lengths.append(len(tokens))
            for term in counts:
                df[term] += 1
        n = max(len(self.chunks), 1)
        self._avgdl = (sum(self._doc_lengths) / n) if self.chunks else 0.0
        # Robertson / Spark-Jones IDF, floored at 0 via the +0.5 smoothing form.
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }
        return self

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        counts = self._doc_freqs[doc_index]
        dl = self._doc_lengths[doc_index]
        denom_norm = self.k1 * (1 - self.b + self.b * (dl / self._avgdl if self._avgdl else 1.0))
        total = 0.0
        for term in set(query_tokens):
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf.get(term, 0.0)
            total += idf * (tf * (self.k1 + 1)) / (tf + denom_norm)
        return total

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the *top_k* ``(chunk, bm25_score)`` pairs, best first."""
        if not self.chunks or top_k <= 0:
            return []
        tokens = tokenize(query)
        scored = [(i, self.score(tokens, i)) for i in range(len(self.chunks))]
        scored.sort(key=lambda x: -x[1])
        return [(self.chunks[i], s) for i, s in scored[:top_k] if s > 0]
