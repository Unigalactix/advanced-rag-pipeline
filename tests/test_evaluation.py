import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_pipeline import RAGPipeline  # noqa: E402
from rag_pipeline.chunking import Chunk  # noqa: E402
from rag_pipeline.evaluation import evaluate_retrieval, load_qa_dataset  # noqa: E402
from rag_pipeline.retrieval import RetrievalResult  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = ROOT / "examples" / "sample_docs"


class FakeRetriever:
    """Returns a fixed ranking: relevant doc at rank 2 for every question."""

    def retrieve(self, query, top_k=5):
        chunks = [
            Chunk(text="unrelated filler text", source="other.md", doc_id="o"),
            Chunk(text="the target fact lives here", source="target.md", doc_id="t"),
            Chunk(text="more unrelated filler", source="other2.md", doc_id="o2"),
        ]
        return [RetrievalResult(chunk=c, score=1.0 - 0.1 * i) for i, c in enumerate(chunks[:top_k])]


class TestEvaluationMetrics(unittest.TestCase):
    def test_known_ranking_gives_known_metrics(self):
        dataset = [
            {"question": "q1", "relevant_sources": ["target.md"]},
            {"question": "q2", "relevant_sources": ["target.md"]},
        ]
        result = evaluate_retrieval(FakeRetriever(), dataset, k=3)
        self.assertEqual(result.num_questions, 2)
        self.assertAlmostEqual(result.recall_at_k, 1.0)       # always found within k
        self.assertAlmostEqual(result.mrr, 0.5)               # always at rank 2 -> 1/2
        self.assertAlmostEqual(result.context_precision, 1 / 3)  # 1 of 3 chunks relevant

    def test_missing_relevant_doc_gives_zero_recall(self):
        dataset = [{"question": "q", "relevant_sources": ["absent.md"]}]
        result = evaluate_retrieval(FakeRetriever(), dataset, k=3)
        self.assertEqual(result.recall_at_k, 0.0)
        self.assertEqual(result.mrr, 0.0)

    def test_sample_dataset_loads_and_pipeline_scores_well(self):
        dataset = load_qa_dataset(ROOT / "evaluation" / "sample_qa.json")
        self.assertGreaterEqual(len(dataset), 3)
        self.assertTrue(all("question" in item for item in dataset))
        pipeline = RAGPipeline()
        pipeline.ingest(SAMPLE_DOCS)
        from rag_pipeline.evaluation import evaluate_pipeline

        result = evaluate_pipeline(pipeline, dataset, k=3)
        self.assertEqual(result.num_questions, len(dataset))
        self.assertGreaterEqual(result.recall_at_k, 0.75)
        self.assertGreater(result.mrr, 0.0)
        self.assertGreater(result.context_precision, 0.0)


if __name__ == "__main__":
    unittest.main()
