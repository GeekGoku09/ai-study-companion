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
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Retriever")

        self.model = model
        self.timeout = timeout
        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        self.index = None
        self.chunks: list[DocumentChunk] = []
        self.dimension = 1536  # Default for text-embedding-3-small

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for a list of texts using OpenAI."""
        if not texts:
            return []
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [data.embedding for data in response.data]

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        """Builds a FAISS index from the provided document chunks."""
        if not chunks:
            self.chunks = []
            self.index = None
            return

        self.chunks = chunks
        texts = [chunk.content for chunk in chunks]
        embeddings = self._get_embeddings(texts)

        embeddings_np = np.array(embeddings).astype("float32")
        self.dimension = embeddings_np.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_np)

    def retrieve(self, query: str, k: int = 5) -> list[DocumentChunk]:
        """Retrieves top-k most relevant chunks for a given query."""
        if not self.chunks or self.index is None:
            return []

        if k <= 0:
            return []

        effective_k = min(k, len(self.chunks))
        query_embeddings = self._get_embeddings([query])
        if not query_embeddings:
            return []

        query_embedding_np = np.array([query_embeddings[0]]).astype("float32")
        _distances, indices = self.index.search(query_embedding_np, effective_k)

        retrieved_chunks = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])

        return retrieved_chunks
