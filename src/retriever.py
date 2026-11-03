import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Union
from openai import OpenAI
from .schemas import DocumentChunk
import os

class Retriever:
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small", batch_size: int = 100):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Retriever")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.batch_size = batch_size
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.dimension = 1536 # Default for text-embedding-3-small

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings in batches using OpenAI to prevent token/payload limits."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
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
        if self.index is None:
            raise ValueError("Index has not been built or loaded. Call build_index or load_index first.")
        if not query or not query.strip():
            return []
        if not self.chunks:
            return []

        query_embedding = self._get_embeddings([query])[0]
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        effective_k = min(k, len(self.chunks))
        distances, indices = self.index.search(query_embedding_np, effective_k)
        
        retrieved_chunks = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])
        
        return retrieved_chunks

    def save_index(self, directory: Union[str, Path]) -> None:
        """Persists the FAISS index and associated chunks/metadata to disk."""
        if self.index is None or not self.chunks:
            raise ValueError("Cannot save an empty or unbuilt index.")

        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / "index.faiss"
        faiss.write_index(self.index, str(index_file))

        metadata_file = target_dir / "chunks.json"
        chunks_data = [chunk.model_dump() for chunk in self.chunks]
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump({
                "model": self.model,
                "dimension": self.dimension,
                "chunks": chunks_data
            }, f, indent=2)

    def load_index(self, directory: Union[str, Path]) -> None:
        """Loads a persisted FAISS index and chunk metadata from disk."""
        target_dir = Path(directory)
        index_file = target_dir / "index.faiss"
        metadata_file = target_dir / "chunks.json"

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Chunks metadata file not found: {metadata_file}")

        self.index = faiss.read_index(str(index_file))
        self.dimension = self.index.d

        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.model = data.get("model", self.model)
            self.chunks = [DocumentChunk(**item) for item in data.get("chunks", [])]
