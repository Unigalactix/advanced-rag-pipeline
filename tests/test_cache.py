import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_pipeline.cache import QueryCache  # noqa: E402


class TestQueryCache(unittest.TestCase):
    def test_miss_then_hit_and_stats(self):
        cache = QueryCache(maxsize=4)
        key = QueryCache.make_key("What is RAG?", top_k=3)
        self.assertIsNone(cache.get(key))
        cache.put(key, "answer")
        self.assertEqual(cache.get(key), "answer")
        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)

    def test_key_normalisation(self):
        a = QueryCache.make_key("  What   is   RAG? ", top_k=3)
        b = QueryCache.make_key("what is rag?", top_k=3)
        self.assertEqual(a, b)
        self.assertNotEqual(a, QueryCache.make_key("what is rag?", top_k=5))

    def test_lru_eviction(self):
        cache = QueryCache(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # 'a' is now most-recently-used
        cache.put("c", 3)  # evicts 'b'
        self.assertIn("a", cache)
        self.assertIn("c", cache)
        self.assertNotIn("b", cache)
        self.assertEqual(len(cache), 2)

    def test_clear_resets_stats(self):
        cache = QueryCache(maxsize=2)
        cache.put("x", 1)
        cache.get("x")
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.stats()["hits"], 0)


if __name__ == "__main__":
    unittest.main()
