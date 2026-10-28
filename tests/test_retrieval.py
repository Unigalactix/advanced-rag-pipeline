import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_pipeline import RAGPipeline  # noqa: E402

SAMPLE_DOCS = Path(__file__).resolve().parents[1] / "examples" / "sample_docs"


class TestHybridRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = RAGPipeline()
        cls.pipeline.ingest(SAMPLE_DOCS)

    def test_retriever_finds_vector_database_doc(self):
        results = self.pipeline.retriever.retrieve("What does a vector database store?", top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.source, "vector_databases.md")
        self.assertGreater(results[0].score, 0)

    def test_retriever_finds_azure_doc(self):
        results = self.pipeline.retriever.retrieve(
            "Which Azure service gives access to large language models?", top_k=3
        )
        sources = [r.chunk.source for r in results]
        self.assertIn("azure_overview.md", sources)

    def test_retriever_finds_rag_doc(self):
        results = self.pipeline.retriever.retrieve(
            "How does retrieval augmented generation ground answers?", top_k=3
        )
        self.assertEqual(results[0].chunk.source, "rag_explained.md")

    def test_results_carry_both_component_scores(self):
        results = self.pipeline.retriever.retrieve("cosine similarity embeddings", top_k=2)
        for r in results:
            self.assertGreaterEqual(r.dense_score, -1.0)
            self.assertGreaterEqual(r.bm25_score, 0.0)


if __name__ == "__main__":
    unittest.main()
