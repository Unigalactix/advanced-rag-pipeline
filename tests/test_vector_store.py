import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_pipeline.chunking import Chunk  # noqa: E402
from rag_pipeline.embeddings import HashingEmbedder  # noqa: E402
from rag_pipeline.vector_store import InMemoryVectorStore  # noqa: E402


class TestEmbeddings(unittest.TestCase):
    def test_hashing_embedder_is_deterministic_and_normalised(self):
        emb = HashingEmbedder(dim=128)
        a = emb.embed("vector databases store embeddings")
        b = emb.embed("vector databases store embeddings")
        self.assertEqual(a.shape, (128,))
        np.testing.assert_allclose(a, b)
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)

    def test_similar_text_is_closer_than_unrelated_text(self):
        emb = HashingEmbedder(dim=256)
        q = emb.embed("vector database embeddings search")
        near = emb.embed("embeddings search in a vector database")
        far = emb.embed("cooking recipes for chocolate cake")
        self.assertGreater(float(q @ near), float(q @ far))


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.emb = HashingEmbedder(dim=256)
        self.chunks = [
            Chunk(text="Azure provides cloud virtual machines and Kubernetes", source="azure.md", doc_id="a"),
            Chunk(text="RAG retrieves documents to ground language model answers", source="rag.md", doc_id="r"),
            Chunk(text="A vector database stores embeddings for similarity search", source="vec.md", doc_id="v"),
        ]
        self.store = InMemoryVectorStore(dim=self.emb.dim)
        self.store.add(self.chunks, self.emb.embed_batch([c.text for c in self.chunks]))

    def test_search_returns_most_similar_chunk_first(self):
        results = self.store.search(self.emb.embed("embeddings similarity search database"), top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0].source, "vec.md")
        self.assertGreaterEqual(results[0][1], results[1][1])

    def test_empty_store_returns_nothing(self):
        empty = InMemoryVectorStore(dim=64)
        self.assertEqual(empty.search(np.zeros(64, dtype=np.float32)), [])

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.store.save(tmp)
            loaded = InMemoryVectorStore.load(tmp)
        self.assertEqual(len(loaded), len(self.store))
        self.assertEqual([c.text for c in loaded.chunks], [c.text for c in self.chunks])
        results = loaded.search(self.emb.embed("cloud virtual machines Kubernetes"), top_k=1)
        self.assertEqual(results[0][0].source, "azure.md")

    def test_add_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            self.store.add(self.chunks, np.zeros((2, self.emb.dim), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
