# Embeddings and Chunking

An embedding model converts text into a dense vector of numbers. Semantically similar texts produce vectors that point in similar directions, which is why embedding vectors can be compared with cosine similarity to find related passages.

Chunking is the process of splitting a long document into smaller passages before embedding. Chunk size matters: chunks that are too large dilute the meaning of each vector, while chunks that are too small lose surrounding context. A common strategy is a chunk size of several hundred tokens with an overlap between consecutive chunks, so a sentence near a boundary still appears with its neighbours.

Embeddings are used in search, recommendation, clustering, and classification. In retrieval-augmented generation, both the document chunks and the user question are embedded with the same model, and the question vector is matched against the chunk vectors.
