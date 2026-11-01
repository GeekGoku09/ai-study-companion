import os
import re
from typing import List
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from .schemas import DocumentChunk
from .utils import count_tokens, split_text_by_tokens

def extract_text_from_pdf(pdf_path: str) -> List[DocumentChunk]:
    """
    Extracts text from a PDF and returns a list of DocumentChunks.
    Handles encrypted, corrupt, and empty PDFs with descriptive exceptions or warnings.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to read PDF file '{pdf_path}': {e}") from e

    if reader.is_encrypted:
        try:
            # Attempt decryption with empty password for PDFs with empty/default permissions
            decrypted = reader.decrypt("")
            if decrypted == 0:
                raise ValueError(f"PDF file '{pdf_path}' is encrypted and password-protected.")
        except Exception as e:
            raise ValueError(f"Cannot process encrypted PDF '{pdf_path}': {e}") from e

    chunks = []
    
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            # Skip unreadable pages or corrupted page streams
            continue

        if not text:
            continue
        
        # Simple semantic chunking by splitting into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < 50:  # Skip very short paragraphs that might be noise/page numbers
                continue
            
            chunks.append(DocumentChunk(
                content=para,
                metadata={
                    "source": os.path.abspath(pdf_path),
                    "page": page_num + 1,
                    "chunk_id": f"p{page_num+1}_c{i}"
                }
            ))
    
    return chunks

def semantic_chunking(chunks: List[DocumentChunk], max_tokens: int = 500) -> List[DocumentChunk]:
    """
    Refines chunks to ensure they are semantically coherent and within token limits.
    Groups small consecutive paragraphs from the same page if they fit in max_tokens.
    Also splits any single paragraph that exceeds max_tokens to prevent oversized chunks.
    """
    if not chunks:
        return []

    # First pass: split any individual chunks that already exceed max_tokens
    normalized_chunks: List[DocumentChunk] = []
    for chunk in chunks:
        tokens = count_tokens(chunk.content)
        if tokens > max_tokens:
            sub_texts = split_text_by_tokens(chunk.content, max_tokens=max_tokens)
            for idx, sub_text in enumerate(sub_texts):
                sub_meta = dict(chunk.metadata)
                sub_meta["chunk_id"] = f"{chunk.metadata.get('chunk_id', 'chunk')}_sub{idx}"
                normalized_chunks.append(DocumentChunk(content=sub_text, metadata=sub_meta))
        else:
            normalized_chunks.append(chunk)

    refined_chunks: List[DocumentChunk] = []
    current_chunk_content = ""
    current_metadata = normalized_chunks[0].metadata
    
    for chunk in normalized_chunks:
        tokens_current = count_tokens(current_chunk_content)
        tokens_new = count_tokens(chunk.content)
        
        if (tokens_current + tokens_new > max_tokens) or (chunk.metadata.get("page") != current_metadata.get("page")):
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

def ingest_pdf(pdf_path: str, max_tokens: int = 500) -> List[DocumentChunk]:
    """High-level function to ingest a PDF and return semantically chunked content."""
    initial_chunks = extract_text_from_pdf(pdf_path)
    if not initial_chunks:
        return []
    final_chunks = semantic_chunking(initial_chunks, max_tokens=max_tokens)
    return final_chunks
