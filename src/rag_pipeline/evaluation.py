"""Retrieval evaluation: recall@k, MRR, and context precision."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from .retrieval import RetrievalResult


class _Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]: ...


@dataclass
class QuestionResult:
    question: str
    first_relevant_rank: int | None  # 1-based; None if no relevant chunk in top-k
    relevant_retrieved: int
    retrieved: int


@dataclass
class EvaluationResult:
    k: int
    num_questions: int
    recall_at_k: float
    mrr: float
    context_precision: float
    per_question: list[QuestionResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "num_questions": self.num_questions,
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "context_precision": round(self.context_precision, 4),
        }

    def __str__(self) -> str:
        d = self.as_dict()
        return (
            f"Evaluation (k={d['k']}, n={d['num_questions']}): "
            f"recall@{d['k']}={d['recall_at_k']}, MRR={d['mrr']}, "
            f"context_precision={d['context_precision']}"
        )


def load_qa_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load a QA dataset: a JSON list of ``{question, relevant_sources?, answer_contains?}``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("QA dataset must be a JSON list")
    return data


def _is_relevant(result: RetrievalResult, item: dict[str, Any]) -> bool:
    sources = item.get("relevant_sources") or ([item["source"]] if item.get("source") else [])
    if sources and result.chunk.source in sources:
        return True
    needle = (item.get("answer_contains") or "").lower()
    if needle and needle in result.chunk.text.lower():
        return True
    return False


def evaluate_retrieval(
    retriever: _Retriever,
    dataset: Sequence[dict[str, Any]],
    k: int = 5,
) -> EvaluationResult:
    """Score *retriever* over *dataset* with recall@k, MRR, context precision."""
    per_question: list[QuestionResult] = []
    hits = 0
    reciprocal_ranks = 0.0
    precisions: list[float] = []
    for item in dataset:
        results = retriever.retrieve(item["question"], top_k=k)
        relevant_flags = [_is_relevant(r, item) for r in results]
        first_rank = next((i + 1 for i, flag in enumerate(relevant_flags) if flag), None)
        if first_rank is not None:
            hits += 1
            reciprocal_ranks += 1.0 / first_rank
        n_relevant = sum(relevant_flags)
        precisions.append(n_relevant / len(results) if results else 0.0)
        per_question.append(
            QuestionResult(
                question=item["question"],
                first_relevant_rank=first_rank,
                relevant_retrieved=n_relevant,
                retrieved=len(results),
            )
        )
    n = len(dataset) or 1
    return EvaluationResult(
        k=k,
        num_questions=len(dataset),
        recall_at_k=hits / n if dataset else 0.0,
        mrr=reciprocal_ranks / n if dataset else 0.0,
        context_precision=sum(precisions) / n if dataset else 0.0,
        per_question=per_question,
    )


def evaluate_pipeline(pipeline: Any, dataset: Sequence[dict[str, Any]], k: int = 5) -> EvaluationResult:
    """Convenience wrapper: evaluate a :class:`RAGPipeline`'s retriever."""
    if pipeline.retriever is None:
        raise ValueError("Pipeline has no index yet — call ingest() first.")
    return evaluate_retrieval(pipeline.retriever, dataset, k=k)
