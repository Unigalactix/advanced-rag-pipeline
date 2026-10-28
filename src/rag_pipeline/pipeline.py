"""RAGPipeline: ingest -> index -> ask, tying every component together."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bm25 import BM25Index
from .cache import QueryCache
from .chunking import Chunk, chunk_documents
from .embeddings import Embedder, HashingEmbedder
from .ingestion import Document, load_documents
from .llm import LLM, ExtractiveLLM
from .retrieval import HybridRetriever, RetrievalResult
from .vector_store import InMemoryVectorStore


@dataclass
class RAGResponse:
    """The result of :meth:`RAGPipeline.ask`."""

    question: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    results: list[RetrievalResult] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def cache_hit(self) -> bool:
        return bool(self.stats.get("cache_hit"))


class RAGPipeline:
    """End-to-end retrieval-augmented generation over a local knowledge base.

    Defaults are fully offline: :class:`HashingEmbedder` for vectors and
    :class:`ExtractiveLLM` for generation. Swap in ``OpenAIEmbedder`` /
    ``OpenAILLM`` (or any object satisfying those protocols) to go live.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        fusion_alpha: float = 0.5,
        cache_size: int = 128,
    ) -> None:
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.llm: LLM = llm or ExtractiveLLM()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.fusion_alpha = fusion_alpha
        self.cache = QueryCache(maxsize=cache_size)
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.store: InMemoryVectorStore | None = None
        self.bm25: BM25Index | None = None
        self.retriever: HybridRetriever | None = None

    # -- indexing ---------------------------------------------------------
    def ingest(self, directory: str | Path) -> int:
        """Load documents from *directory*, chunk them, and build both indexes.

        Returns the number of chunks indexed.
        """
        documents = load_documents(directory)
        return self.add_documents(documents)

    def add_documents(self, documents: list[Document]) -> int:
        self.documents.extend(documents)
        new_chunks = chunk_documents(documents, self.chunk_size, self.chunk_overlap)
        return self._index(new_chunks)

    def _index(self, new_chunks: list[Chunk]) -> int:
        if self.store is None:
            self.store = InMemoryVectorStore(dim=self.embedder.dim)
        self.chunks.extend(new_chunks)
        if new_chunks:
            embeddings = self.embedder.embed_batch([c.text for c in new_chunks])
            self.store.add(new_chunks, embeddings)
        # BM25 is rebuilt over the full corpus so IDF stays correct.
        self.bm25 = BM25Index().build(self.chunks)
        assert self.store is not None and self.bm25 is not None
        self.retriever = HybridRetriever(
            self.store, self.bm25, self.embedder, fusion_alpha=self.fusion_alpha
        )
        self.cache.clear()
        return len(new_chunks)

    # -- querying ---------------------------------------------------------
    def ask(self, question: str, top_k: int = 5, use_cache: bool = True) -> RAGResponse:
        """Answer *question* grounded in retrieved chunks, with sources and stats."""
        if self.retriever is None:
            raise ValueError("No documents indexed yet — call ingest() first.")
        key = QueryCache.make_key(question, top_k)
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                # Return a copy so the stored response keeps cache_hit=False.
                return RAGResponse(
                    question=cached.question,
                    answer=cached.answer,
                    sources=cached.sources,
                    results=cached.results,
                    stats={**cached.stats, "cache_hit": True, "cache": self.cache.stats()},
                )
        started = time.perf_counter()
        results = self.retriever.retrieve(question, top_k=top_k)
        answer = self.llm.answer(question, results)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        sources = [
            {
                "rank": i + 1,
                "source": r.chunk.source,
                "chunk_id": r.chunk.chunk_id,
                "score": round(r.score, 4),
                "dense_score": round(r.dense_score, 4),
                "bm25_score": round(r.bm25_score, 4),
                "text": r.chunk.text,
            }
            for i, r in enumerate(results)
        ]
        response = RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            results=results,
            stats={
                "cache_hit": False,
                "retrieved": len(results),
                "latency_ms": round(elapsed_ms, 2),
                "cache": self.cache.stats(),
            },
        )
        if use_cache:
            self.cache.put(key, response)
        return response
