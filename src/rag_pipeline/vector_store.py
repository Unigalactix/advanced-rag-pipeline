"""In-memory NumPy vector store with cosine search and disk persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .chunking import Chunk


class InMemoryVectorStore:
    """Stores chunk embeddings in one NumPy matrix and searches by cosine.

    All stored vectors are expected to be L2-normalised (the embedders in
    this package guarantee that), so cosine similarity is a dot product.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.chunks: list[Chunk] = []
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2 or arr.shape != (len(chunks), self.dim):
            raise ValueError(f"embeddings must have shape ({len(chunks)}, {self.dim}), got {arr.shape}")
        self.chunks.extend(chunks)
        self._vectors = arr if len(self._vectors) == 0 else np.vstack([self._vectors, arr])

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the *top_k* ``(chunk, cosine_score)`` pairs, best first."""
        if len(self.chunks) == 0 or top_k <= 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        scores = self._vectors @ q
        order = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def save(self, directory: str | Path) -> Path:
        """Persist vectors (``index.npz``) and chunk metadata (``chunks.json``)."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "index.npz", vectors=self._vectors, dim=np.array([self.dim]))
        payload = [
            {
                "text": c.text, "source": c.source, "doc_id": c.doc_id,
                "chunk_index": c.chunk_index, "chunk_id": c.chunk_id, "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        (out / "chunks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, directory: str | Path) -> "InMemoryVectorStore":
        """Load a store previously written by :meth:`save`."""
        root = Path(directory)
        data = np.load(root / "index.npz")
        store = cls(dim=int(data["dim"][0]))
        store._vectors = data["vectors"].astype(np.float32)
        raw = json.loads((root / "chunks.json").read_text(encoding="utf-8"))
        store.chunks = [Chunk(**item) for item in raw]
        if len(store.chunks) != store._vectors.shape[0]:
            raise ValueError("Corrupt store: chunk count does not match vector count")
        return store
