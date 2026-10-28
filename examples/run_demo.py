#!/usr/bin/env python3
"""End-to-end demo: index the sample docs and answer two questions, offline.

Run from the project root:

    python examples/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_pipeline import RAGPipeline  # noqa: E402

QUESTIONS = [
    "What does Retrieval-Augmented Generation ground answers in?",
    "What does a vector database store?",
]


def main() -> None:
    docs_dir = Path(__file__).resolve().parent / "sample_docs"
    pipeline = RAGPipeline()
    n_chunks = pipeline.ingest(docs_dir)
    print(f"Indexed {len(pipeline.documents)} documents into {n_chunks} chunks.\n")

    for question in QUESTIONS:
        result = pipeline.ask(question, top_k=3)
        print(f"Q: {question}")
        print(f"A: {result.answer}")
        print("Sources:")
        for source in result.sources:
            print(f"  [{source['rank']}] {source['source']} (score={source['score']})")
        print(f"Stats: cache_hit={result.stats['cache_hit']}, retrieved={result.stats['retrieved']}, "
              f"latency_ms={result.stats['latency_ms']}\n")

    repeat = pipeline.ask(QUESTIONS[0], top_k=3)
    print(f"Repeat question served from cache: {repeat.stats['cache_hit']}")
    print(f"Cache stats: {pipeline.cache.stats()}")


if __name__ == "__main__":
    main()
