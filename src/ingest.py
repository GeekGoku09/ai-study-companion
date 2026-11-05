import os
import re
import logging
from typing import List, Optional
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from .schemas import DocumentChunk
import tiktoken

logger = logging.getLogger(__name__)

class IngestionError(Exception):
    """Raised when PDF extraction or chunking encounters an unrecoverable error."""
    pass

class EmptyDocumentError(IngestionError):
    """Raised when the document contains no extractable text."""
    pass

def _subchunk_oversized_text(text: str, max_tokens: int, encoding: tiktoken.Encoding) -> List[str]:
    """
    Sub-chunks an oversized text block that exceeds max_tokens by splitting at
    sentence boundaries or whitespace.
    """
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    # Split by sentences (period followed by whitespace or newline)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # If regex did not split (e.g. no punctuation), try paragraphs or spaces
    if len(sentences) <= 1:
        sentences = text.splitlines(keepends=True) or [text]

    subchunks: List[str] = []
    current_sentences: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = len(encoding.encode(sentence))
        if s_tokens > max_tokens:
            # Sentence itself is bigger than max_tokens; fall back to token slicing
            if current_sentences:
                subchunks.append(" ".join(current_sentences).strip())
                current_sentences = []
                current_tokens = 0

            raw_tokens = encoding.encode(sentence)
            for i in range(0, len(raw_tokens), max_tokens):
                slice_tokens = raw_tokens[i:i + max_tokens]
                subchunks.append(encoding.decode(slice_tokens).strip())
            continue

        if current_tokens + s_tokens > max_tokens:
            if current_sentences:
                subchunks.append(" ".join(current_sentences).strip())
            current_sentences = [sentence]
            current_tokens = s_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += s_tokens

    if current_sentences:
        subchunks.append(" ".join(current_sentences).strip())

    return [s for s in subchunks if s.strip()]

def extract_text_from_pdf(pdf_path: str, min_char_len: int = 20) -> List[DocumentChunk]:
    """
    Extracts text from a PDF and returns a list of initial DocumentChunks.
    Raises IngestionError or EmptyDocumentError on issues.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
    except PdfReadError as e:
        raise IngestionError(f"Failed to read PDF (corrupted or unreadable format): {e}") from e
    except Exception as e:
        raise IngestionError(f"Unexpected error opening PDF: {e}") from e

    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
            if decrypted == 0:
                raise IngestionError("PDF is password protected and cannot be decrypted.")
        except Exception as e:
            raise IngestionError(f"Failed to decrypt password-protected PDF: {e}") from e

    chunks: List[DocumentChunk] = []
    total_pages = len(reader.pages)
    
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
            continue

        if not text:
            continue
        
        # Split into paragraphs (delimited by blank lines)
        paragraphs = re.split(r'\n\s*\n', text)
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            # Retain equations, short formulas or code blocks as long as >= min_char_len
            if len(para) < min_char_len:
                continue
            
            chunks.append(DocumentChunk(
                content=para,
                metadata={
                    "source": os.path.abspath(pdf_path),
                    "page": page_num + 1,
                    "chunk_id": f"p{page_num+1}_c{i}",
                    "total_pages": total_pages
                }
            ))

    if not chunks:
        raise EmptyDocumentError(
            f"No text could be extracted from {pdf_path}. The PDF may be scanned/image-only, empty, or below threshold."
        )

    return chunks

def semantic_chunking(chunks: List[DocumentChunk], max_tokens: int = 500, split_oversized: bool = False) -> List[DocumentChunk]:
    """
    Refines chunks to ensure they are semantically coherent and within token limits.
    Groups small consecutive paragraphs from the same page if they fit in max_tokens.
    If split_oversized is True, breaks down individual chunks that exceed max_tokens.

    ⚡ Bolt optimization maintained:
    Tracks running token count to avoid O(N^2) repeated encoding.
    """
    if not chunks:
        return []

    encoding = tiktoken.get_encoding("cl100k_base")

    input_chunks = chunks
    if split_oversized:
        normalized_chunks: List[DocumentChunk] = []
        for chunk in chunks:
            tokens_len = len(encoding.encode(chunk.content))
            if tokens_len > max_tokens:
                sub_texts = _subchunk_oversized_text(chunk.content, max_tokens, encoding)
                for idx, sub_text in enumerate(sub_texts):
                    sub_meta = dict(chunk.metadata)
                    sub_meta["sub_chunk"] = idx
                    normalized_chunks.append(DocumentChunk(content=sub_text, metadata=sub_meta))
            else:
                normalized_chunks.append(chunk)
        input_chunks = normalized_chunks

    refined_chunks: List[DocumentChunk] = []
    current_parts: List[str] = []
    current_tokens: int = 0
    current_metadata = input_chunks[0].metadata
    
    for chunk in input_chunks:
        tokens_new = len(encoding.encode(chunk.content))
        
        # If adding paragraph exceeds max_tokens or we transitioned to a new page, flush
        if (current_tokens + tokens_new > max_tokens) or (chunk.metadata.get("page") != current_metadata.get("page")):
            if current_parts:
                refined_chunks.append(DocumentChunk(
                    content="\n\n".join(current_parts).strip(),
                    metadata=current_metadata
                ))
            current_parts = [chunk.content]
            current_tokens = tokens_new
            current_metadata = chunk.metadata
        else:
            current_parts.append(chunk.content)
            current_tokens += tokens_new
                
    # Add trailing chunk
    if current_parts:
        refined_chunks.append(DocumentChunk(
            content="\n\n".join(current_parts).strip(),
            metadata=current_metadata
        ))
        
    return refined_chunks

def ingest_pdf(pdf_path: str, max_tokens: int = 500, min_char_len: int = 20, split_oversized: bool = True) -> List[DocumentChunk]:
    """High-level function to ingest a PDF and return semantically chunked content."""
    initial_chunks = extract_text_from_pdf(pdf_path, min_char_len=min_char_len)
    final_chunks = semantic_chunking(initial_chunks, max_tokens=max_tokens, split_oversized=split_oversized)
    return final_chunks
