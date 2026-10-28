"""Embedders: a protocol, an offline hashing embedder, and an OpenAI stub."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokenisation shared across the package."""
    return _TOKEN_RE.findall(text.lower())


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into a dense vector."""

    dim: int

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic, offline embedder based on token hashing (the hashing trick).

    Each token is hashed to a bucket in a ``dim``-dimensional space with a
    signed contribution, tokens are counted, and the vector is L2-normalised.
    It needs no model files or network access, is identical across runs and
    machines, and gives sensible cosine similarity for keyword-overlapping
    text — enough to run and test the whole pipeline offline.
    """

    def __init__(self, dim: int = 384, use_bigrams: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.use_bigrams = use_bigrams

    def _add_token(self, vec: np.ndarray, token: str, weight: float = 1.0) -> None:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign * weight

    def embed(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        vec = np.zeros(self.dim, dtype=np.float64)
        for token in tokens:
            self._add_token(vec, token)
        if self.use_bigrams:
            for a, b in zip(tokens, tokens[1:]):
                self._add_token(vec, f"{a}_{b}", weight=0.5)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])


class OpenAIEmbedder:
    """OpenAI embeddings behind a lazy import (optional extra).

    Install with ``pip install "rag-pipeline[openai]"`` and set
    ``OPENAI_API_KEY``. Raises a clear :class:`ImportError` otherwise.
    """

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536,
                 api_key: str | None = None) -> None:
        try:
            import openai  # noqa: F401  (lazy, optional dependency)
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "OpenAIEmbedder requires the 'openai' package. "
                'Install it with: pip install "rag-pipeline[openai]"'
            ) from exc
        if not (api_key or os.environ.get("OPENAI_API_KEY")):  # pragma: no cover
            raise ValueError("Set OPENAI_API_KEY or pass api_key= to OpenAIEmbedder.")
        self.model = model
        self.dim = dim
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]

    def embed(self, text: str) -> np.ndarray:  # pragma: no cover - needs network/key
        import openai

        client = openai.OpenAI(api_key=self._api_key)
        resp = client.embeddings.create(model=self.model, input=[text])
        return np.asarray(resp.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        return np.vstack([self.embed(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)
