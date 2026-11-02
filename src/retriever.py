import faiss
import json
import logging
import numpy as np
import os
from typing import List, Optional
from openai import OpenAI
from .schemas import DocumentChunk

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small", batch_size: int = 128):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Retriever")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.batch_size = batch_size
        self.index: Optional[faiss.Index] = None
        self.chunks: List[DocumentChunk] = []
        self.dimension = 1536 # Default for text-embedding-3-small

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts using OpenAI with batching."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            # OpenAI returns embeddings in response.data, ordered by index
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([data.embedding for data in sorted_data])

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
        logger.info(f"Built FAISS index with {len(chunks)} chunks, dimension {self.dimension}.")

    def retrieve(self, query: str, k: int = 5) -> List[DocumentChunk]:
        """Retrieves top-k most relevant chunks for a given query."""
        if self.index is None or not self.chunks:
            raise ValueError("Index has not been built or is empty. Call build_index or load_index first.")
        
        query_embedding = self._get_embeddings([query])[0]
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        k = min(k, len(self.chunks))
        distances, indices = self.index.search(query_embedding_np, k)
        
        retrieved_chunks = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])
        
        return retrieved_chunks

    def save_index(self, dir_path: str):
        """Saves the FAISS index and chunk metadata to a directory."""
        if self.index is None or not self.chunks:
            raise ValueError("No index or chunks to save.")

        os.makedirs(dir_path, exist_ok=True)
        index_path = os.path.join(dir_path, "index.faiss")
        chunks_path = os.path.join(dir_path, "chunks.json")

        faiss.write_index(self.index, index_path)
        chunks_data = [chunk.model_dump() for chunk in self.chunks]
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved FAISS index and {len(self.chunks)} chunks to {dir_path}")

    def load_index(self, dir_path: str):
        """Loads the FAISS index and chunk metadata from a directory."""
        index_path = os.path.join(dir_path, "index.faiss")
        chunks_path = os.path.join(dir_path, "chunks.json")

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Index or chunks file not found in {dir_path}")

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        self.chunks = [DocumentChunk(**c) for c in chunks_data]
        self.dimension = self.index.d
        logger.info(f"Loaded FAISS index with {len(self.chunks)} chunks from {dir_path}")
