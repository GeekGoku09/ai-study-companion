import os

import faiss
import numpy as np
from openai import OpenAI

from .schemas import DocumentChunk


class Retriever:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        timeout: float = 60.0,
        batch_size: int = 100,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Retriever")
        
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[DocumentChunk] = []
        self.dimension = 1536  # Default for text-embedding-3-small

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for a list of texts in batches using OpenAI."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
            )
            all_embeddings.extend([data.embedding for data in response.data])

        return all_embeddings

    def build_index(self, chunks: list[DocumentChunk]):
        """Builds a FAISS index from the provided document chunks with L2 normalization."""
        if not chunks:
            self.chunks = []
            self.index = None
            return

        self.chunks = list(chunks)
        texts = [chunk.content for chunk in chunks]
        embeddings = self._get_embeddings(texts)
        
        embeddings_np = np.array(embeddings).astype("float32")
        self.dimension = embeddings_np.shape[1]
        
        # Normalize vectors for cosine similarity via inner product
        faiss.normalize_L2(embeddings_np)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_np)

    def retrieve(self, query: str, k: int = 5) -> list[DocumentChunk]:
        """Retrieves top-k most relevant chunks for a given query."""
        if self.index is None or not self.chunks:
            raise ValueError("Index has not been built. Call build_index first.")
        
        if k <= 0:
            raise ValueError("k must be a positive integer greater than 0.")

        actual_k = min(k, len(self.chunks))
        query_embedding = self._get_embeddings([query])[0]
        query_embedding_np = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_embedding_np)
        
        _distances, indices = self.index.search(query_embedding_np, actual_k)
        
        retrieved_chunks = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])
        
        return retrieved_chunks
