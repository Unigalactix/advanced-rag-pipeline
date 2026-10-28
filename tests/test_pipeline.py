import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_pipeline import RAGPipeline  # noqa: E402
from rag_pipeline.ingestion import load_documents  # noqa: E402

SAMPLE_DOCS = Path(__file__).resolve().parents[1] / "examples" / "sample_docs"


class TestIngestion(unittest.TestCase):
    def test_loads_sample_docs_with_metadata(self):
        docs = load_documents(SAMPLE_DOCS)
        self.assertGreaterEqual(len(docs), 3)
        names = {d.source for d in docs}
        self.assertIn("rag_explained.md", names)
        self.assertTrue(all(d.text for d in docs))
        self.assertTrue(all(d.doc_id for d in docs))


class TestPipelineEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = RAGPipeline()
        cls.n_chunks = cls.pipeline.ingest(SAMPLE_DOCS)

    def test_ingest_indexes_chunks(self):
        self.assertGreater(self.n_chunks, 0)
        self.assertEqual(len(self.pipeline.chunks), self.n_chunks)
        self.assertIsNotNone(self.pipeline.retriever)

    def test_ask_returns_grounded_answer_with_source(self):
        result = self.pipeline.ask(
            "What does Retrieval-Augmented Generation ground answers in?", top_k=3
        )
        self.assertIn("retrieved documents", result.answer.lower())
        self.assertIn("[1]", result.answer)  # citation marker
        self.assertEqual(result.sources[0]["source"], "rag_explained.md")
        self.assertFalse(result.cache_hit)

    def test_ask_vector_database_question(self):
        result = self.pipeline.ask("What does a vector database store?", top_k=3)
        self.assertIn("embedding", result.answer.lower())
        self.assertEqual(result.sources[0]["source"], "vector_databases.md")

    def test_second_identical_ask_is_a_cache_hit(self):
        pipeline = RAGPipeline()
        pipeline.ingest(SAMPLE_DOCS)
        first = pipeline.ask("What does a vector database store?", top_k=3)
        second = pipeline.ask("What does a vector database store?", top_k=3)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.answer, second.answer)
        self.assertEqual(pipeline.cache.stats()["hits"], 1)

    def test_answer_only_uses_retrieved_context(self):
        # A question with no support in the corpus must not invent an answer
        # containing facts from outside the knowledge base.
        result = self.pipeline.ask("What is the capital of France?", top_k=3)
        self.assertNotIn("Paris", result.answer)

    def test_ask_before_ingest_raises(self):
        with self.assertRaises(ValueError):
            RAGPipeline().ask("anything")


if __name__ == "__main__":
    unittest.main()
