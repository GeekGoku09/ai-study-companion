import faiss
import numpy as np
from typing import List, Tuple
from openai import OpenAI
from .schemas import DocumentChunk
import os
import logging

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, api_key: str = None, model: str = None, batch_size: int = 128):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Retriever")
        
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.batch_size = max(1, batch_size)
        self.client = OpenAI(api_key=self.api_key)
        self.index = None
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
            all_embeddings.extend([data.embedding for data in response.data])

        return all_embeddings

    def build_index(self, chunks: List[DocumentChunk]):
        """Builds a FAISS index from the provided document chunks."""
        if not chunks:
            self.chunks = []
            self.index = None
            return

        self.chunks = list(chunks)
        texts = [chunk.content for chunk in self.chunks]
        embeddings = self._get_embeddings(texts)
        
        embeddings_np = np.array(embeddings).astype('float32')
        self.dimension = embeddings_np.shape[1]
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_np)
        logger.info(f"Built FAISS IndexFlatL2 with {len(self.chunks)} items, dim {self.dimension}")

    def retrieve(self, query: str, k: int = 5) -> List[DocumentChunk]:
        """Retrieves top-k most relevant chunks for a given query."""
        if self.index is None or not self.chunks:
            return []

        cleaned_query = query.strip()
        if not cleaned_query:
            return self.chunks[:k]

        k = max(1, min(k, len(self.chunks)))
        
        query_embedding = self._get_embeddings([cleaned_query])[0]
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.index.search(query_embedding_np, k)
        
        retrieved_chunks = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])
        
        return retrieved_chunks
