import os
import re
from typing import List
from pypdf import PdfReader
from .schemas import DocumentChunk
import tiktoken

def extract_text_from_pdf(pdf_path: str) -> List[DocumentChunk]:
    """Extracts text from a PDF and returns a list of DocumentChunks."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    reader = PdfReader(pdf_path)
    chunks = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        
        # Simple semantic chunking by splitting into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < 50:  # Skip very short paragraphs that might be noise
                continue
            
            chunks.append(DocumentChunk(
                content=para,
                metadata={
                    "source": pdf_path,
                    "page": page_num + 1,
                    "chunk_id": f"p{page_num+1}_c{i}"
                }
            ))
    
    return chunks

def semantic_chunking(chunks: List[DocumentChunk], max_tokens: int = 500) -> List[DocumentChunk]:
    """
    Refines chunks to ensure they are semantically coherent and within token limits.
    Groups small consecutive paragraphs from the same page if they fit in max_tokens.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    refined_chunks = []
    
    if not chunks:
        return []

    current_chunk_content = ""
    current_metadata = chunks[0].metadata
    
    for chunk in chunks:
        # If adding the next paragraph exceeds max_tokens or we are on a new page, 
        # save the current chunk and start a new one
        tokens_current = len(encoding.encode(current_chunk_content))
        tokens_new = len(encoding.encode(chunk.content))
        
        if (tokens_current + tokens_new > max_tokens) or (chunk.metadata["page"] != current_metadata["page"]):
            if current_chunk_content:
                refined_chunks.append(DocumentChunk(
                    content=current_chunk_content.strip(),
                    metadata=current_metadata
                ))
            current_chunk_content = chunk.content
            current_metadata = chunk.metadata
        else:
            if current_chunk_content:
                current_chunk_content += "\n\n" + chunk.content
            else:
                current_chunk_content = chunk.content
                current_metadata = chunk.metadata
                
    # Add the last chunk
    if current_chunk_content:
        refined_chunks.append(DocumentChunk(
            content=current_chunk_content.strip(),
            metadata=current_metadata
        ))
        
    return refined_chunks

def ingest_pdf(pdf_path: str) -> List[DocumentChunk]:
    """High-level function to ingest a PDF and return semantically chunked content."""
    initial_chunks = extract_text_from_pdf(pdf_path)
    final_chunks = semantic_chunking(initial_chunks)
    return final_chunks
