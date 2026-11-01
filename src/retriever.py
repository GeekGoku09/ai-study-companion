import json
import os
from typing import List, Optional
import faiss
import numpy as np
from openai import OpenAI
from .schemas import DocumentChunk

class Retriever:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        dimension: int = 1536
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Retriever")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.batch_size = max(1, batch_size)
        self.index: Optional[faiss.Index] = None
        self.chunks: List[DocumentChunk] = []
        self.dimension = dimension

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings in batches for a list of texts using OpenAI."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            all_embeddings.extend([data.embedding for data in response.data])

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

        actual_k = min(max(1, k), len(self.chunks))
        query_embedding = self._get_embeddings([query.strip()])[0]
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.index.search(query_embedding_np, actual_k)
        
        retrieved_chunks = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])
        
        return retrieved_chunks

    def save_index(self, directory_path: str):
        """Persists the FAISS vector index and chunk metadata to disk."""
        if self.index is None:
            raise ValueError("Cannot save index: index is not built.")

        os.makedirs(directory_path, exist_ok=True)
        index_file = os.path.join(directory_path, "index.faiss")
        metadata_file = os.path.join(directory_path, "metadata.json")

        faiss.write_index(self.index, index_file)

        chunks_data = [chunk.model_dump() for chunk in self.chunks]
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump({
                "model": self.model,
                "dimension": self.dimension,
                "chunks": chunks_data
            }, f, indent=2)

    def load_index(self, directory_path: str):
        """Loads a persisted FAISS vector index and chunk metadata from disk."""
        index_file = os.path.join(directory_path, "index.faiss")
        metadata_file = os.path.join(directory_path, "metadata.json")

        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            raise FileNotFoundError(f"FAISS index or metadata not found in {directory_path}")

        self.index = faiss.read_index(index_file)
        self.dimension = self.index.d

        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.model = data.get("model", self.model)
            self.chunks = [DocumentChunk(**chunk) for chunk in data.get("chunks", [])]
