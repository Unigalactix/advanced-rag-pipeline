# Vector Databases

A vector database stores embeddings, which are high-dimensional numeric vectors that represent the meaning of text, images, or other data. A vector database stores embeddings so that similar items can be found quickly by comparing the distance or angle between their vectors.

The most common similarity measure in a vector database is cosine similarity, which compares the angle between two vectors. Nearest neighbour search finds the stored vectors closest to a query vector, often using approximate nearest neighbour indexes to stay fast at large scale.

Popular vector databases include Pinecone, which is a managed cloud service, and Chroma, which is an open-source embedding database that can run locally. FAISS is a library for efficient similarity search that is often used to build a vector index inside an application. Choosing between them depends on scale, hosting, filtering needs, and operational effort.
