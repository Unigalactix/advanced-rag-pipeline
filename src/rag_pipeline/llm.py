"""LLMs: a protocol, an offline extractive fallback, and an OpenAI stub."""

from __future__ import annotations

import os
import re
from typing import Protocol, Sequence, runtime_checkable

from .embeddings import tokenize
from .retrieval import RetrievalResult

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who", "how",
    "does", "do", "did", "in", "on", "of", "to", "for", "and", "or", "it", "its",
    "that", "this", "with", "by", "as", "at", "from", "be", "can", "used", "use",
}


@runtime_checkable
class LLM(Protocol):
    """Anything that can answer a question from retrieved results."""

    def answer(self, question: str, contexts: Sequence[RetrievalResult]) -> str: ...


def _sentences(text: str) -> list[str]:
    # Drop Markdown heading lines — they are titles, not answer content.
    body = re.sub(r"(?m)^#{1,6}\s+.*$", "", text)
    return [s.strip() for s in _SENTENCE_RE.split(body.replace("\n", " ")) if s.strip()]


class ExtractiveLLM:
    """Offline fallback LLM: composes an answer strictly from retrieved chunks.

    Sentences from the retrieved chunks are scored by keyword overlap with
    the question; the best sentences are quoted back with citation markers.
    Because every sentence comes verbatim from a retrieved chunk, the answer
    is grounded by construction — it cannot introduce outside facts.
    """

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def answer(self, question: str, contexts: Sequence[RetrievalResult]) -> str:
        if not contexts:
            return "I don't have enough information in the knowledge base to answer that question."
        q_tokens = {t for t in tokenize(question) if t not in STOPWORDS}
        scored: list[tuple[float, int, str]] = []
        for rank, result in enumerate(contexts):
            for sentence in _sentences(result.chunk.text):
                s_tokens = set(tokenize(sentence))
                overlap = len(q_tokens & s_tokens)
                # Exact-phrase bonus: a sentence containing a question bigram is a strong hit.
                score = overlap + 0.25 * result.score
                if overlap > 0:
                    scored.append((score, rank, sentence))
        if not scored:
            # No keyword overlap: fall back to the lead sentence of the top chunk.
            lead = _sentences(contexts[0].chunk.text)
            if not lead:
                return "I don't have enough information in the knowledge base to answer that question."
            return f"{lead[0]} [1]"
        scored.sort(key=lambda x: (-x[0], x[1]))
        chosen: list[str] = []
        seen: set[str] = set()
        for _, rank, sentence in scored:
            if sentence in seen:
                continue
            seen.add(sentence)
            chosen.append(f"{sentence} [{rank + 1}]")
            if len(chosen) >= self.max_sentences:
                break
        return " ".join(chosen)


class OpenAILLM:
    """OpenAI chat model behind a lazy import (optional extra).

    Install with ``pip install "rag-pipeline[openai]"`` and set
    ``OPENAI_API_KEY``. Raises a clear :class:`ImportError` otherwise.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "OpenAILLM requires the 'openai' package. "
                'Install it with: pip install "rag-pipeline[openai]"'
            ) from exc
        if not (api_key or os.environ.get("OPENAI_API_KEY")):  # pragma: no cover
            raise ValueError("Set OPENAI_API_KEY or pass api_key= to OpenAILLM.")
        self.model = model
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]

    def answer(self, question: str, contexts: Sequence[RetrievalResult]) -> str:  # pragma: no cover
        import openai

        context_text = "\n\n".join(
            f"[{i + 1}] ({r.chunk.source}) {r.chunk.text}" for i, r in enumerate(contexts)
        )
        prompt = (
            "Answer the question using ONLY the context below. Cite sources as [n]. "
            "If the context does not contain the answer, say you don't know.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}"
        )
        client = openai.OpenAI(api_key=self._api_key)
        resp = client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content or ""
