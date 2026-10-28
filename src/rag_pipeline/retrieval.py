"""Hybrid retrieval (BM25 + dense fusion) and MMR re-ranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import BM25Index
from .chunking import Chunk
from .embeddings import Embedder
from .vector_store import InMemoryVectorStore


@dataclass
class RetrievalResult:
    """One retrieved chunk with its component and fused scores."""

    chunk: Chunk
    score: float  # fused (and, after re-ranking, the ordering score)
    dense_score: float = 0.0
    bm25_score: float = 0.0


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    """Normalise a score dict to [0, 1]; a single/constant list maps to 1.0."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def mmr_rerank(
    candidates: list[RetrievalResult],
    store: InMemoryVectorStore,
    top_k: int,
    lambda_param: float = 0.7,
) -> list[RetrievalResult]:
    """Re-order *candidates* with Maximal Marginal Relevance.

    Greedily picks the chunk maximising
    ``lambda * relevance - (1 - lambda) * max_similarity_to_selected``,
    trading a little relevance for a much less redundant context set.
    """
    if top_k <= 0 or not candidates:
        return []
    index_of = {c.chunk_id: i for i, c in enumerate(store.chunks)}
    remaining = list(candidates)
    selected: list[RetrievalResult] = []
    selected_vecs: list[np.ndarray] = []
    while remaining and len(selected) < top_k:
        best, best_value = None, -np.inf
        for cand in remaining:
            vec = store.vectors[index_of[cand.chunk.chunk_id]]
            redundancy = 0.0
            if selected_vecs:
                redundancy = max(float(vec @ s) for s in selected_vecs)
            value = lambda_param * cand.score - (1 - lambda_param) * redundancy
            if value > best_value:
                best, best_value = cand, value
        assert best is not None
        selected.append(best)
        selected_vecs.append(store.vectors[index_of[best.chunk.chunk_id]])
        remaining.remove(best)
    return selected


class HybridRetriever:
    """Fuses dense (vector) and sparse (BM25) rankings into one result list."""

    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        bm25_index: BM25Index,
        embedder: Embedder,
        fusion_alpha: float = 0.5,
        candidate_k: int = 20,
        mmr_lambda: float = 0.7,
    ) -> None:
        if not 0.0 <= fusion_alpha <= 1.0:
            raise ValueError("fusion_alpha must be between 0 and 1")
        self.store = vector_store
        self.bm25 = bm25_index
        self.embedder = embedder
        self.alpha = fusion_alpha
        self.candidate_k = candidate_k
        self.mmr_lambda = mmr_lambda

    def retrieve(self, query: str, top_k: int = 5, rerank: bool = True) -> list[RetrievalResult]:
        """Retrieve, fuse, and (optionally) MMR re-rank chunks for *query*."""
        if len(self.store) == 0 or top_k <= 0:
            return []
        query_vec = self.embedder.embed(query)
        dense = dict()  # chunk_id -> (chunk, score)
        for chunk, score in self.store.search(query_vec, top_k=self.candidate_k):
            dense[chunk.chunk_id] = (chunk, score)
        sparse = {}
        for chunk, score in self.bm25.search(query, top_k=self.candidate_k):
            sparse[chunk.chunk_id] = (chunk, score)

        dense_norm = _minmax({cid: s for cid, (_, s) in dense.items()})
        sparse_norm = _minmax({cid: s for cid, (_, s) in sparse.items()})

        results: list[RetrievalResult] = []
        for cid in set(dense) | set(sparse):
            chunk = (dense.get(cid) or sparse.get(cid))[0]
            d_raw = dense.get(cid, (None, 0.0))[1]
            b_raw = sparse.get(cid, (None, 0.0))[1]
            fused = self.alpha * dense_norm.get(cid, 0.0) + (1 - self.alpha) * sparse_norm.get(cid, 0.0)
            results.append(RetrievalResult(chunk=chunk, score=fused, dense_score=d_raw, bm25_score=b_raw))
        results.sort(key=lambda r: -r.score)
        if rerank:
            return mmr_rerank(results[: max(top_k * 2, top_k)], self.store, top_k, self.mmr_lambda)
        return results[:top_k]
