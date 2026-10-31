import faiss
import numpy as np
from typing import List, Optional
from openai import OpenAI
from .schemas import DocumentChunk
import os
import json

class Retriever:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        embedding_batch_size: int = 64,
        similarity_metric: str = "l2"
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Retriever")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.embedding_batch_size = embedding_batch_size
        self.similarity_metric = similarity_metric.lower()
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.dimension = 1536 # Default for text-embedding-3-small

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts using OpenAI with batching."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i:i + self.embedding_batch_size]
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

        self.chunks = list(chunks)
        texts = [chunk.content for chunk in chunks]
        embeddings = self._get_embeddings(texts)
        
        embeddings_np = np.array(embeddings).astype('float32')
        self.dimension = embeddings_np.shape[1]
        
        if self.similarity_metric in ("cosine", "inner_product", "ip"):
            faiss.normalize_L2(embeddings_np)
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(embeddings_np)

        for chunk, emb in zip(self.chunks, embeddings):
            chunk.embedding = emb

    def retrieve(self, query: str, k: int = 5) -> List[DocumentChunk]:
        """Retrieves top-k most relevant chunks for a given query."""
        if self.index is None:
            raise ValueError("Index has not been built. Call build_index or load_index first.")
        
        if not self.chunks:
            return []

        query_embedding = self._get_embeddings([query])[0]
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        if self.similarity_metric in ("cosine", "inner_product", "ip"):
            faiss.normalize_L2(query_embedding_np)

        effective_k = min(k, len(self.chunks))
        distances, indices = self.index.search(query_embedding_np, effective_k)
        
        retrieved_chunks = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])
        
        return retrieved_chunks

    def save_index(self, directory_or_prefix: str) -> None:
        """
        Saves the FAISS index and associated chunks/metadata to disk.
        directory_or_prefix can be a directory path or base file prefix.
        """
        if self.index is None:
            raise ValueError("Cannot save index: index has not been built.")

        if os.path.isdir(directory_or_prefix) or directory_or_prefix.endswith(("/", "\\")):
            os.makedirs(directory_or_prefix, exist_ok=True)
            index_path = os.path.join(directory_or_prefix, "index.faiss")
            chunks_path = os.path.join(directory_or_prefix, "chunks.json")
        else:
            base_dir = os.path.dirname(directory_or_prefix)
            if base_dir:
                os.makedirs(base_dir, exist_ok=True)
            index_path = f"{directory_or_prefix}.faiss"
            chunks_path = f"{directory_or_prefix}_chunks.json"

        faiss.write_index(self.index, index_path)
        chunks_data = [chunk.model_dump() for chunk in self.chunks]
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)

    def load_index(self, directory_or_prefix: str) -> None:
        """Loads the FAISS index and associated chunks from disk."""
        if os.path.isdir(directory_or_prefix) or directory_or_prefix.endswith(("/", "\\")):
            index_path = os.path.join(directory_or_prefix, "index.faiss")
            chunks_path = os.path.join(directory_or_prefix, "chunks.json")
        else:
            index_path = f"{directory_or_prefix}.faiss"
            chunks_path = f"{directory_or_prefix}_chunks.json"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found at {index_path}")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks metadata file not found at {chunks_path}")

        self.index = faiss.read_index(index_path)
        self.dimension = self.index.d

        with open(chunks_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
            self.chunks = [DocumentChunk(**item) for item in raw_chunks]
