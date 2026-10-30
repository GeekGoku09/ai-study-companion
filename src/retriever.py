import faiss
import numpy as np
import json
import os
from typing import List, Optional
from openai import OpenAI
from .schemas import DocumentChunk
from .utils import setup_logger
from .exceptions import VectorIndexError

logger = setup_logger(__name__)

class Retriever:
    """FAISS-backed vector retriever supporting batch embeddings and index persistence."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: int = 64
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Retriever")

        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.batch_size = max(1, batch_size)
        self.client = OpenAI(api_key=self.api_key)
        self.index: Optional[faiss.Index] = None
        self.chunks: List[DocumentChunk] = []
        self.dimension: int = 1536

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for texts in batches to prevent API limits from being exceeded."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i // self.batch_size + 1}: {e}")
                raise VectorIndexError(f"Embedding generation failed: {e}") from e

        return all_embeddings

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Builds a FAISS index from the provided document chunks."""
        if not chunks:
            logger.warning("No chunks provided to build_index. Index remains empty.")
            self.chunks = []
            self.index = None
            return

        self.chunks = list(chunks)
        texts = [chunk.content for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunk(s)...")
        embeddings = self._get_embeddings(texts)

        embeddings_np = np.array(embeddings, dtype="float32")
        self.dimension = embeddings_np.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_np)
        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors (dim={self.dimension}).")

    def retrieve(self, query: str, k: int = 5) -> List[DocumentChunk]:
        """Retrieves top-k most relevant chunks for a given query."""
        if not query or not query.strip():
            logger.warning("Empty query passed to retrieve. Returning empty list.")
            return []

        if self.index is None or self.index.ntotal == 0 or not self.chunks:
            raise VectorIndexError("Index has not been built or contains no vectors. Call build_index first.")

        k = max(1, min(k, len(self.chunks)))
        query_embedding = self._get_embeddings([query.strip()])[0]
        query_embedding_np = np.array([query_embedding], dtype="float32")

        distances, indices = self.index.search(query_embedding_np, k)

        retrieved_chunks = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])

        return retrieved_chunks

    def save_index(self, directory_path: str) -> None:
        """Saves the FAISS index and chunk metadata to a directory."""
        if self.index is None or not self.chunks:
            raise VectorIndexError("Cannot save empty or uninitialized index.")

        os.makedirs(directory_path, exist_ok=True)
        index_file = os.path.join(directory_path, "index.faiss")
        chunks_file = os.path.join(directory_path, "chunks.json")

        try:
            faiss.write_index(self.index, index_file)
            chunks_data = [chunk.model_dump() for chunk in self.chunks]
            with open(chunks_file, "w", encoding="utf-8") as f:
                json.dump({"model": self.model, "dimension": self.dimension, "chunks": chunks_data}, f, indent=2)
            logger.info(f"Vector index and metadata saved to {directory_path}")
        except Exception as e:
            raise VectorIndexError(f"Failed to save index to {directory_path}: {e}") from e

    def load_index(self, directory_path: str) -> None:
        """Loads a FAISS index and chunk metadata from a directory."""
        index_file = os.path.join(directory_path, "index.faiss")
        chunks_file = os.path.join(directory_path, "chunks.json")

        if not os.path.exists(index_file) or not os.path.exists(chunks_file):
            raise VectorIndexError(f"Missing index or chunk files in directory: {directory_path}")

        try:
            self.index = faiss.read_index(index_file)
            with open(chunks_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.chunks = [DocumentChunk(**chunk) for chunk in payload.get("chunks", [])]
            self.dimension = payload.get("dimension", self.index.d)
            self.model = payload.get("model", self.model)
            logger.info(f"Loaded index with {self.index.ntotal} vectors from {directory_path}")
        except Exception as e:
            raise VectorIndexError(f"Failed to load index from {directory_path}: {e}") from e
