import faiss
import numpy as np
import pickle
import os
from typing import List, Optional
from openai import OpenAI
from .schemas import DocumentChunk

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 100

class Retriever:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        timeout: Optional[float] = None
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Retriever")
        
        self.model = model
        self.timeout = timeout
        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[DocumentChunk] = []
        self.dimension = 1536  # Default for text-embedding-3-small

    def _get_embeddings(self, texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[List[float]]:
        """Generates embeddings in batches for a list of texts using OpenAI."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            # Ensure embeddings match batch order
            batch_embeddings = [data.embedding for data in sorted(response.data, key=lambda d: d.index)]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def build_index(self, chunks: List[DocumentChunk]):
        """Builds a FAISS index from the provided document chunks."""
        if not chunks:
            self.chunks = []
            self.index = None
            return

        self.chunks = chunks
        texts = [chunk.content for chunk in chunks]
        embeddings = self._get_embeddings(texts)
        
        embeddings_np = np.array(embeddings).astype('float32')
        self.dimension = embeddings_np.shape[1]
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_np)

    def retrieve(self, query: str, k: int = 5) -> List[DocumentChunk]:
        """Retrieves top-k most relevant chunks for a given query."""
        if not query or not query.strip():
            return []
        if self.index is None or not self.chunks:
            return []
        
        # Clamp k to available chunks
        k = max(1, min(k, len(self.chunks)))
        query_embeddings = self._get_embeddings([query])
        if not query_embeddings:
            return []

        query_embedding_np = np.array(query_embeddings).astype('float32')
        distances, indices = self.index.search(query_embedding_np, k)
        
        retrieved_chunks = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])
        
        return retrieved_chunks

    def save_index(self, directory: str):
        """Persists the FAISS index and chunk metadata to a local directory."""
        if self.index is None:
            raise ValueError("No index to save. Build an index first.")
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "index.faiss")
        chunks_path = os.path.join(directory, "chunks.pkl")
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load_index(self, directory: str):
        """Loads a FAISS index and chunk metadata from a local directory."""
        index_path = os.path.join(directory, "index.faiss")
        chunks_path = os.path.join(directory, "chunks.pkl")
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Index or chunks file not found in {directory}")
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        self.dimension = self.index.d
