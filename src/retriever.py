import os
import logging
from typing import List, Optional
import faiss
import numpy as np
from openai import OpenAI
from .schemas import DocumentChunk

logger = logging.getLogger(__name__)


class Retriever:
    """
    Handles embedding generation, vector indexing with FAISS, and similarity search.
    Supports batching embeddings to avoid API payload limits and supports index persistence.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        dimension: int = 1536,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Retriever")

        self.client = OpenAI(api_key=self.api_key, timeout=timeout)
        self.model = model
        self.batch_size = max(1, batch_size)
        self.index: Optional[faiss.Index] = None
        self.chunks: List[DocumentChunk] = []
        self.dimension = dimension

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts using OpenAI with batching.
        Handles empty texts list gracefully.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            # Ensure embeddings are stored in order
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Builds a FAISS index from the provided document chunks."""
        if not chunks:
            logger.warning("No chunks provided to build_index. Index will remain empty.")
            self.chunks = []
            self.index = faiss.IndexFlatL2(self.dimension)
            return

        self.chunks = list(chunks)
        texts = [chunk.content for chunk in self.chunks]
        embeddings = self._get_embeddings(texts)

        embeddings_np = np.array(embeddings).astype("float32")
        self.dimension = embeddings_np.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_np)
        logger.info("Successfully indexed %d chunks (dimension=%d)", len(self.chunks), self.dimension)

    def retrieve(self, query: str, k: int = 5) -> List[DocumentChunk]:
        """
        Retrieves top-k most relevant chunks for a given query.
        Returns empty list if index is not built or empty.
        """
        if self.index is None:
            raise ValueError("Index has not been built. Call build_index first.")

        if not self.chunks or self.index.ntotal == 0:
            return []

        # Bound k to available chunks
        k_actual = min(max(1, k), len(self.chunks))

        query_embedding = self._get_embeddings([query])[0]
        query_embedding_np = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding_np, k_actual)

        retrieved_chunks: List[DocumentChunk] = []
        for idx in indices[0]:
            if idx != -1 and 0 <= idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])

        return retrieved_chunks

    def save_index(self, index_path: str) -> None:
        """Persists the FAISS index to disk."""
        if self.index is None:
            raise ValueError("No index to save.")
        dir_path = os.path.dirname(index_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, index_path)

    def load_index(self, index_path: str, chunks: List[DocumentChunk]) -> None:
        """Loads a persisted FAISS index and associates it with document chunks."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found at: {index_path}")
        self.index = faiss.read_index(index_path)
        self.chunks = list(chunks)
        self.dimension = self.index.d
