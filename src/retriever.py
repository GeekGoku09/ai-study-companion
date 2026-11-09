import os
from typing import List, Optional
import faiss
import numpy as np
from openai import OpenAI
from .schemas import DocumentChunk


class Retriever:
    """Handles vector embeddings and similarity retrieval using FAISS."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Retriever. Set OPENAI_API_KEY environment variable.")

        self.timeout = timeout
        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        self.model = model
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[DocumentChunk] = []
        self.dimension = 1536  # Default for text-embedding-3-small

    def _get_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generates embeddings for a list of texts using OpenAI with batching.
        Handles large chunk collections without hitting API batch limits.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
            )
            # Ensure order matches batch
            batch_embeddings = [data.embedding for data in sorted(response.data, key=lambda x: x.index)]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def build_index(self, chunks: List[DocumentChunk], batch_size: int = 100):
        """Builds a FAISS index from the provided document chunks."""
        if not chunks:
            self.chunks = []
            self.index = None
            return

        self.chunks = chunks
        texts = [chunk.content for chunk in chunks]
        embeddings = self._get_embeddings(texts, batch_size=batch_size)

        embeddings_np = np.array(embeddings).astype("float32")
        self.dimension = embeddings_np.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_np)

    def retrieve(self, query: str, k: int = 5) -> List[DocumentChunk]:
        """
        Retrieves top-k most relevant chunks for a given query.
        
        Raises:
            ValueError: If the index has not been built or query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        if self.index is None or len(self.chunks) == 0:
            raise ValueError("Index has not been built or contains no chunks. Call build_index first.")

        # Ensure k is positive and bounded
        k = max(1, min(k, len(self.chunks)))

        query_embedding = self._get_embeddings([query.strip()])[0]
        query_embedding_np = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding_np, k)

        retrieved_chunks = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])

        return retrieved_chunks
