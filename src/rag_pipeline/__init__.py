"""Advanced Retrieval-Augmented Generation (RAG) Pipeline.

Offline-first RAG: hybrid BM25 + dense retrieval, MMR re-ranking, LRU
query caching, grounded extractive answers with citations, and retrieval
evaluation (recall@k, MRR, context precision).
"""

from .bm25 import BM25Index
from .cache import QueryCache
from .chunking import Chunk, chunk_documents, split_text
from .embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from .evaluation import EvaluationResult, evaluate_pipeline, evaluate_retrieval, load_qa_dataset
from .ingestion import Document, load_document, load_documents
from .llm import LLM, ExtractiveLLM, OpenAILLM
from .pipeline import RAGPipeline, RAGResponse
from .retrieval import HybridRetriever, RetrievalResult, mmr_rerank
from .vector_store import InMemoryVectorStore

__version__ = "0.1.0"
__author__ = "Rajesh Kodaganti"

__all__ = [
    "BM25Index",
    "Chunk",
    "Document",
    "Embedder",
    "EvaluationResult",
    "ExtractiveLLM",
    "HashingEmbedder",
    "HybridRetriever",
    "InMemoryVectorStore",
    "LLM",
    "OpenAIEmbedder",
    "OpenAILLM",
    "QueryCache",
    "RAGPipeline",
    "RAGResponse",
    "RetrievalResult",
    "chunk_documents",
    "evaluate_pipeline",
    "evaluate_retrieval",
    "load_document",
    "load_documents",
    "load_qa_dataset",
    "mmr_rerank",
    "split_text",
]
