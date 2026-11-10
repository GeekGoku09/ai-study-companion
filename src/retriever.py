import os

import faiss
import numpy as np
from openai import OpenAI

from .schemas import DocumentChunk


class Retriever:
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small", timeout: float | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Retriever")

        self.client = OpenAI(api_key=self.api_key, timeout=timeout)
        self.model = model
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: list[DocumentChunk] = []
        self.dimension: int = 1536  # Default for text-embedding-3-small

    def _get_embeddings(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Generates embeddings for a list of texts using OpenAI in batches to respect API limits."""
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
            )
            # Maintain original order of response data
            embeddings.extend([data.embedding for data in response.data])

        return embeddings

    def build_index(self, chunks: list[DocumentChunk]):
        """Builds a FAISS index from the provided document chunks."""
        if not chunks:
            self.chunks = []
            self.index = None
            return

        self.chunks = list(chunks)
        texts = [chunk.content for chunk in chunks]
        embeddings = self._get_embeddings(texts)

        embeddings_np = np.array(embeddings).astype("float32")
        self.dimension = embeddings_np.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_np)

    def retrieve(self, query: str, k: int = 5) -> list[DocumentChunk]:
        """Retrieves top-k most relevant chunks for a given query."""
        if not query or not query.strip():
            return []

        if self.index is None or not self.chunks:
            raise ValueError("Index has not been built or is empty. Call build_index first.")

        # Limit k to total number of indexed chunks
        effective_k = min(k, len(self.chunks))
        if effective_k <= 0:
            return []

        query_embedding = self._get_embeddings([query])[0]
        query_embedding_np = np.array([query_embedding]).astype("float32")

        _distances, indices = self.index.search(query_embedding_np, effective_k)

        retrieved_chunks = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])

        return retrieved_chunks
