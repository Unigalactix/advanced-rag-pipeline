# Retrieval-Augmented Generation Explained

Retrieval-Augmented Generation, usually shortened to RAG, is a technique that grounds a large language model in external documents. Instead of relying only on what the model memorised during training, a RAG system first retrieves relevant passages from a knowledge base and then gives those passages to the model as context.

Retrieval-Augmented Generation grounds answers in retrieved documents. The retriever searches an index for the chunks most relevant to the question, and the generator composes an answer using only that retrieved context. Because the answer is grounded in retrieved documents, a well-built RAG system can cite its sources and is less likely to hallucinate facts.

A typical RAG pipeline has four stages: ingestion loads documents, chunking splits documents into smaller passages, indexing stores those passages for search, and generation answers questions from the retrieved passages. Hybrid retrieval combines keyword search with vector search so the system finds both exact terms and paraphrased matches.
