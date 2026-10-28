import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_pipeline.bm25 import BM25Index  # noqa: E402
from rag_pipeline.chunking import Chunk  # noqa: E402


def _chunk(text, source):
    return Chunk(text=text, source=source, doc_id=source)


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            _chunk("The cat sat on the mat and the cat slept all day", "cats.txt"),
            _chunk("Dogs bark loudly and dogs enjoy long walks in the park", "dogs.txt"),
            _chunk("Quantum computers use qubits and superposition for computation", "quantum.txt"),
        ]
        self.index = BM25Index().build(self.chunks)

    def test_relevant_document_ranks_first(self):
        results = self.index.search("cat slept on the mat", top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0][0].source, "cats.txt")
        self.assertGreater(results[0][1], 0)

    def test_irrelevant_query_scores_zero_and_is_filtered(self):
        results = self.index.search("photosynthesis chlorophyll sunlight", top_k=3)
        self.assertEqual(results, [])

    def test_scores_are_descending(self):
        results = self.index.search("dogs walks park", top_k=3)
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(results[0][0].source, "dogs.txt")

    def test_rare_term_beats_common_term_document(self):
        # "qubits" appears in only one doc and should dominate its ranking.
        results = self.index.search("qubits superposition", top_k=1)
        self.assertEqual(results[0][0].source, "quantum.txt")


if __name__ == "__main__":
    unittest.main()
