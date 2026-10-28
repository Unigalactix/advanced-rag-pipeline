import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_pipeline.chunking import chunk_documents, split_text  # noqa: E402
from rag_pipeline.ingestion import Document  # noqa: E402


class TestChunking(unittest.TestCase):
    def test_chunks_respect_size_limit(self):
        text = ("Sentence number one is here. " * 30) + "\n\n" + ("Another paragraph follows now. " * 30)
        chunks = split_text(text, chunk_size=200, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)

    def test_overlap_prefixes_next_chunk_with_previous_tail(self):
        # Two paragraphs forced into separate pieces; the second must start
        # with the tail of the first when overlap is enabled.
        text = ("alpha " * 25).strip() + "\n\n" + ("beta " * 25).strip()
        chunks = split_text(text, chunk_size=170, chunk_overlap=30)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("alpha", chunks[1][:35])

    def test_no_overlap_when_disabled(self):
        text = ("gamma " * 40).strip()
        chunks = split_text(text, chunk_size=100, chunk_overlap=0)
        self.assertTrue(all("  " not in c for c in chunks))
        self.assertGreater(len(chunks), 1)

    def test_empty_text_gives_no_chunks(self):
        self.assertEqual(split_text("   \n  ", chunk_size=100, chunk_overlap=10), [])

    def test_invalid_overlap_rejected(self):
        with self.assertRaises(ValueError):
            split_text("hello world", chunk_size=50, chunk_overlap=50)

    def test_chunk_documents_preserves_source(self):
        docs = [Document(text="word " * 200, source="notes.md", doc_id="d1")]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.source == "notes.md" for c in chunks))
        self.assertTrue(all(c.doc_id == "d1" for c in chunks))
        self.assertEqual([c.chunk_index for c in chunks], list(range(len(chunks))))
        self.assertEqual(len({c.chunk_id for c in chunks}), len(chunks))


if __name__ == "__main__":
    unittest.main()
