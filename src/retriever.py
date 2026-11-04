import json
import logging
import os
from typing import List, Optional
import faiss
import numpy as np
from openai import OpenAI
from .schemas import DocumentChunk

logger = logging.getLogger(__name__)

class Retriever:
    """
    Handles FAISS vector indexing, persistence, and semantic similarity retrieval
    using OpenAI embeddings.
    """
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
        self.client = OpenAI(api_key=self.api_key)
        self.batch_size = batch_size
        self.index: Optional[faiss.Index] = None
        self.chunks: List[DocumentChunk] = []
        self.dimension: int = 1536

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts using OpenAI in batches.
        Prevents batch payload errors when handling large documents.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def build_index(self, chunks: List[DocumentChunk]):
        """Builds a FAISS index from the provided document chunks."""
        if not chunks:
            logger.warning("No chunks provided to build_index. Index not created.")
            return

        self.chunks = list(chunks)
        texts = [chunk.content for chunk in self.chunks]
        embeddings = self._get_embeddings(texts)
        
        embeddings_np = np.array(embeddings).astype("float32")
        self.dimension = embeddings_np.shape[1]
        
        # Use IndexFlatL2 for Euclidean distance
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_np)
        logger.info("Built FAISS index with %d vectors (dimension: %d)", self.index.ntotal, self.dimension)

    def retrieve(self, query: str, k: int = 5) -> List[DocumentChunk]:
        """Retrieves top-k most relevant chunks for a given query."""
        if self.index is None:
            raise ValueError("Index has not been built. Call build_index first.")
        if not query.strip():
            return []
        
        query_embedding = self._get_embeddings([query])[0]
        query_embedding_np = np.array([query_embedding]).astype("float32")
        
        top_k = min(k, len(self.chunks))
        distances, indices = self.index.search(query_embedding_np, top_k)
        
        retrieved_chunks: List[DocumentChunk] = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])
        
        return retrieved_chunks

    def save_index(self, directory: str):
        """
        Persists the FAISS index and associated chunks metadata to the given directory.
        Creates the directory if it does not exist.
        """
        if self.index is None:
            raise ValueError("Cannot save an empty or uninitialized index.")

        os.makedirs(directory, exist_ok=True)
        index_file = os.path.join(directory, "index.faiss")
        meta_file = os.path.join(directory, "chunks.json")

        faiss.write_index(self.index, index_file)

        chunks_data = [chunk.model_dump() for chunk in self.chunks]
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": self.model,
                    "dimension": self.dimension,
                    "chunks": chunks_data,
                },
                f,
                indent=2
            )
        logger.info("Saved FAISS index and %d chunks to %s", len(self.chunks), directory)

    def load_index(self, directory: str):
        """
        Loads the FAISS index and associated chunks from the specified directory.
        """
        index_file = os.path.join(directory, "index.faiss")
        meta_file = os.path.join(directory, "chunks.json")

        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            raise FileNotFoundError(f"Index files not found in directory: {directory}")

        self.index = faiss.read_index(index_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.model = data.get("model", self.model)
        self.dimension = data.get("dimension", self.index.d)
        self.chunks = [DocumentChunk(**item) for item in data.get("chunks", [])]
        logger.info("Loaded FAISS index with %d items from %s", self.index.ntotal, directory)
