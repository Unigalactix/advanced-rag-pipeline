# Advanced Retrieval-Augmented Generation (RAG) Pipeline

A production-shaped, fully runnable RAG system built by **Rajesh Kodaganti** (AI Software Engineer). It ingests a knowledge base of documents, chunks and indexes it with **both** sparse and dense retrieval, re-ranks the results, caches repeat queries, and answers questions **grounded only in the retrieved context** — with citations on every answer.

No API keys, no cloud services, and no heavy frameworks are required to run it: the core is Python standard library + NumPy, with a deterministic local embedder and an extractive offline LLM. OpenAI and managed vector databases are supported behind clean interfaces as optional extras.

---

## What it does

1. **Ingest** `.txt` / `.md` files from a directory, with source metadata.
2. **Chunk** them with a recursive text splitter (configurable size / overlap).
3. **Index** every chunk twice: as a dense embedding vector *and* in a from-scratch BM25 index.
4. **Retrieve** with hybrid fusion (BM25 + dense cosine), then **re-rank** with Maximal Marginal Relevance (MMR) to balance relevance and diversity.
5. **Answer** questions using only the retrieved chunks, citing sources (`[1]`, `[2]`, …).
6. **Cache** repeat queries in an LRU cache and **evaluate** retrieval quality with recall@k, MRR, and context precision.

## Architecture

```
                        ┌──────────────────── OFFLINE / INGESTION ────────────────────┐
  documents (.txt/.md)  │                                                             │
  ───────────────────►  │  ingestion.py ─► chunking.py ─► embeddings.py               │
                        │        Document       Chunk        HashingEmbedder          │
                        │                                     │              │          │
                        │                                     ▼              ▼          │
                        │                              vector_store.py   bm25.py      │
                        │                              (NumPy cosine)  (Okapi BM25)   │
                        └──────────────────────────────────────┬────────────────────────┘
                                                               │  two indexes, one corpus
                        ┌──────────────────── ONLINE / QUERY ─▼───────────────────────┐
  question ───────────► │  cache.py ──hit──► cached answer                          │
                        │    │ miss                                                     │
                        │    ▼                                                          │
                        │  retrieval.py  HybridRetriever                            │
                        │    fused score = α · dense + (1 − α) · BM25             │
                        │    └─► re-ranking (MMR: relevance − redundancy)           │
                        │    ▼                                                          │
                        │  llm.py  ExtractiveLLM (offline) │ OpenAILLM (optional)   │
                        │    └─► grounded answer + citations                        │
                        └───────────────────────────────────────────────────────────────┘
  evaluation.py ─► recall@k · MRR · context precision  over a QA benchmark set
```

## The four key challenges — and how each is implemented

### 1. Hybrid retrieval
Keyword search misses paraphrases; pure vector search misses exact terms, codes, and names. `bm25.py` is a real Okapi BM25 implementation written from scratch (term frequencies, IDF, length normalisation), and `vector_store.py` does dense cosine search over NumPy embeddings. `retrieval.HybridRetriever` runs both, min-max normalises each score list, and fuses them: `score = α · dense + (1 − α) · sparse` (α is configurable, default 0.5).

### 2. Re-ranking
The first-stage retriever optimises for recall, not for what actually goes into the prompt. The re-ranker in `retrieval.py` applies **Maximal Marginal Relevance**: it greedily selects chunks that maximise `λ · relevance − (1 − λ) · max_similarity_to_already_selected`, so the final context is relevant *and* non-redundant — the top chunks no longer all say the same thing.

### 3. Intelligent caching
LLM/retrieval calls dominate latency and cost. `cache.py` is an LRU cache keyed on the normalised query (+ `top_k`), with bounded size, hit/miss counters, and a hit-rate stat. The pipeline checks it before retrieving and populates it after answering, so repeat (and near-repeat, after normalisation) questions are served without re-running retrieval or generation.

### 4. Evaluation benchmark
"You can't improve what you don't measure." `evaluation.py` scores retrieval against a QA dataset (`evaluation/sample_qa.json`) with:

| Metric | Meaning |
|---|---|
| **Recall@k** | Was a relevant chunk in the top *k* results? (averaged over questions) |
| **MRR** | Mean reciprocal rank of the first relevant chunk — rewards ranking it early |
| **Context precision** | Fraction of retrieved chunks that are actually relevant — the noise level of the prompt context |

## Quickstart

```bash
git clone https://github.com/Unigalactix/advanced-rag-pipeline.git
cd advanced-rag-pipeline
pip install -r requirements.txt        # numpy only
pip install -e .                       # optional: install the package

# Run the end-to-end demo (offline, on the sample docs)
python examples/run_demo.py

# Run the test suite
python -m unittest discover -s tests -v

# Run the retrieval evaluation benchmark
python -c "from rag_pipeline.evaluation import load_qa_dataset, evaluate_pipeline; \
from rag_pipeline import RAGPipeline; \
p = RAGPipeline(); p.ingest('examples/sample_docs'); \
print(evaluate_pipeline(p, load_qa_dataset('evaluation/sample_qa.json'), k=3))"
```

> Run scripts from the project root. If you skip `pip install -e .`, set `PYTHONPATH=src`.

## Usage example

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline(chunk_size=800, chunk_overlap=120, fusion_alpha=0.5)
pipeline.ingest("examples/sample_docs")          # load → chunk → index

result = pipeline.ask("What does a vector database store?", top_k=3)

print(result.answer)       # grounded answer, with [1] [2] citations
for source in result.sources:
    print(source["source"], round(source["score"], 3))
print(result.stats)        # {'cache_hit': False, 'retrieved': 3, ...}

again = pipeline.ask("What does a vector database store?", top_k=3)
print(again.stats["cache_hit"])   # True — served from the LRU cache
```

### Optional extras

```bash
pip install -e ".[openai]"     # OpenAI embeddings + LLM
pip install -e ".[api]"        # FastAPI serving layer (bring your own app)
```

```python
from rag_pipeline.embeddings import OpenAIEmbedder   # raises a clear error
from rag_pipeline.llm import OpenAILLM               # if `openai` isn't installed
```

Both classes import `openai` lazily, so the core package never depends on it. Pinecone / Chroma can be added the same way: implement the small vector-store interface used by `InMemoryVectorStore` (`add` / `search` / `save` / `load`).

## Project structure

```
advanced-rag-pipeline/
├── README.md
├── pyproject.toml
├── requirements.txt            # numpy only (core)
├── LICENSE                     # MIT
├── src/rag_pipeline/
│   ├── __init__.py             # public API
│   ├── ingestion.py            # directory → Document dataclasses
│   ├── chunking.py             # recursive splitter, Chunk dataclass
│   ├── embeddings.py           # Embedder protocol, HashingEmbedder, OpenAIEmbedder
│   ├── vector_store.py         # NumPy cosine store, save/load (.npz + JSON)
│   ├── bm25.py                 # Okapi BM25, from scratch
│   ├── retrieval.py            # hybrid fusion + MMR re-ranking
│   ├── cache.py                # LRU query cache with hit/miss stats
│   ├── llm.py                  # LLM protocol, ExtractiveLLM, OpenAILLM
│   ├── pipeline.py             # RAGPipeline: ingest → index → ask
│   └── evaluation.py           # recall@k, MRR, context precision
├── examples/
│   ├── sample_docs/            # Azure, RAG, vector DBs, embeddings primers
│   └── run_demo.py             # end-to-end offline demo
├── evaluation/
│   └── sample_qa.json          # benchmark QA set for the sample docs
└── tests/                      # stdlib unittest suite
    ├── test_chunking.py
    ├── test_bm25.py
    ├── test_vector_store.py
    ├── test_retrieval.py
    ├── test_cache.py
    ├── test_evaluation.py
    └── test_pipeline.py
```

## Design decisions

- **Offline-first core.** A hashing embedder (token hashing → L2-normalised vectors) makes embeddings deterministic and testable with no model downloads — the trade-off vs. a transformer embedder is semantic quality, and the `Embedder` protocol is exactly where a real one plugs in.
- **Grounding by construction.** The offline LLM is extractive: an answer is assembled only from sentences in retrieved chunks, each tagged with its citation. It cannot hallucinate content that retrieval didn't supply, and it says so when no context is found.
- **Interfaces over frameworks.** Embedder, LLM, and vector-store are small protocols/dataclasses, so each layer can be swapped (OpenAI, Pinecone, Chroma, a local transformer) without touching the pipeline.

## Roadmap

- [ ] Transformer embedder (sentence-transformers) behind the `Embedder` protocol
- [ ] Cross-encoder re-ranker as an alternative to MMR
- [ ] Chroma / Pinecone vector-store adapters
- [ ] FastAPI service + streaming answers
- [ ] PDF / HTML ingestion and per-chunk metadata filters
- [ ] Query rewriting / HyDE for hard questions
- [ ] Larger benchmark with answer-level faithfulness scoring

## License

MIT © 2026 Rajesh Kodaganti — see [LICENSE](LICENSE).
