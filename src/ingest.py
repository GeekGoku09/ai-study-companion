import os
import re
from typing import List, Optional
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .schemas import DocumentChunk
from .utils import get_tokenizer, sanitize_text, setup_logger
from .exceptions import DocumentProcessingError

logger = setup_logger(__name__)

def split_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int = 50, encoding_name: str = "cl100k_base") -> List[str]:
    """Splits a large piece of text into token-bounded chunks with optional token overlap."""
    encoding = get_tokenizer(encoding_name)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    # Guard overlap against equaling or exceeding max_tokens
    actual_overlap = min(overlap_tokens, max(0, max_tokens // 4))
    stride = max(1, max_tokens - actual_overlap)

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_token_slice = tokens[start:end]
        chunk_text = encoding.decode(chunk_token_slice).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(tokens):
            break
        start += stride

    return chunks

def extract_text_from_pdf(pdf_path: str, min_paragraph_length: int = 50) -> List[DocumentChunk]:
    """
    Extracts text from a PDF file, sanitizes it, and returns initial DocumentChunks.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        DocumentProcessingError: If the PDF is unreadable, encrypted without password, or empty.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    if os.path.exists(pdf_path) and os.path.isdir(pdf_path):
        raise DocumentProcessingError(f"Specified path is a directory, not a file: {pdf_path}")

    if os.path.isfile(pdf_path) and os.path.getsize(pdf_path) == 0:
        raise DocumentProcessingError(f"PDF file is empty (0 bytes): {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
    except PdfReadError as e:
        raise DocumentProcessingError(f"Failed to parse PDF file '{pdf_path}': {e}") from e
    except Exception as e:
        raise DocumentProcessingError(f"Unexpected error opening PDF '{pdf_path}': {e}") from e

    if reader.is_encrypted:
        try:
            # Try decrypting with empty password if possible
            decrypted = reader.decrypt("")
            if decrypted == 0:
                raise DocumentProcessingError(f"PDF file '{pdf_path}' is encrypted and requires a password.")
        except Exception as e:
            raise DocumentProcessingError(f"PDF file '{pdf_path}' is encrypted and could not be decrypted: {e}") from e

    chunks: List[DocumentChunk] = []
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise DocumentProcessingError(f"PDF file '{pdf_path}' contains no pages.")

    logger.info(f"Extracting text from {total_pages} page(s) in {pdf_path}")

    for page_num, page in enumerate(reader.pages):
        try:
            raw_text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Error extracting text from page {page_num + 1} of {pdf_path}: {e}")
            continue

        text = sanitize_text(raw_text)
        if not text:
            continue

        # Split into paragraphs by double newlines or multiple whitespace-separated newlines
        paragraphs = re.split(r'\n\s*\n', text)

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < min_paragraph_length:
                continue

            chunks.append(DocumentChunk(
                content=para,
                metadata={
                    "source": os.path.abspath(pdf_path),
                    "filename": os.path.basename(pdf_path),
                    "page": page_num + 1,
                    "chunk_id": f"p{page_num+1}_c{i}"
                }
            ))

    if not chunks:
        logger.warning(f"No extractable text chunks found in {pdf_path}. The PDF may be scanned/image-only.")

    return chunks

def semantic_chunking(
    chunks: List[DocumentChunk],
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    encoding_name: str = "cl100k_base"
) -> List[DocumentChunk]:
    """
    Refines chunks to ensure they are semantically coherent and within token limits.
    Groups small consecutive paragraphs from the same page if they fit in max_tokens.
    If an individual paragraph exceeds max_tokens, it is sub-chunked with token overlap.
    """
    if not chunks:
        return []

    encoding = get_tokenizer(encoding_name)
    refined_chunks: List[DocumentChunk] = []

    current_chunk_content = ""
    current_metadata = dict(chunks[0].metadata)

    for chunk in chunks:
        tokens_current = len(encoding.encode(current_chunk_content)) if current_chunk_content else 0
        tokens_new = len(encoding.encode(chunk.content))
        same_page = chunk.metadata.get("page") == current_metadata.get("page")

        # If adding the next paragraph exceeds max_tokens or we are on a new page,
        # save the current chunk and start a new one
        if (tokens_current + tokens_new > max_tokens) or (not same_page):
            if current_chunk_content:
                # If current_chunk_content itself exceeds max_tokens, sub-chunk it
                if len(encoding.encode(current_chunk_content)) > max_tokens:
                    sub_chunks = split_text_by_tokens(
                        current_chunk_content,
                        max_tokens=max_tokens,
                        overlap_tokens=overlap_tokens,
                        encoding_name=encoding_name
                    )
                    for s_idx, s_text in enumerate(sub_chunks):
                        sub_meta = dict(current_metadata)
                        sub_meta["sub_chunk"] = s_idx
                        refined_chunks.append(DocumentChunk(content=s_text.strip(), metadata=sub_meta))
                else:
                    refined_chunks.append(DocumentChunk(
                        content=current_chunk_content.strip(),
                        metadata=dict(current_metadata)
                    ))
            current_chunk_content = chunk.content
            current_metadata = dict(chunk.metadata)
        else:
            if current_chunk_content:
                current_chunk_content += "\n\n" + chunk.content
            else:
                current_chunk_content = chunk.content
                current_metadata = dict(chunk.metadata)

    # Add the last chunk
    if current_chunk_content:
        if len(encoding.encode(current_chunk_content)) > max_tokens:
            sub_chunks = split_text_by_tokens(
                current_chunk_content,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                encoding_name=encoding_name
            )
            for s_idx, s_text in enumerate(sub_chunks):
                sub_meta = dict(current_metadata)
                sub_meta["sub_chunk"] = s_idx
                refined_chunks.append(DocumentChunk(content=s_text.strip(), metadata=sub_meta))
        else:
            refined_chunks.append(DocumentChunk(
                content=current_chunk_content.strip(),
                metadata=dict(current_metadata)
            ))

    return refined_chunks

def ingest_pdf(pdf_path: str, max_tokens: int = 500) -> List[DocumentChunk]:
    """High-level function to ingest a PDF and return semantically chunked content."""
    initial_chunks = extract_text_from_pdf(pdf_path)
    final_chunks = semantic_chunking(initial_chunks, max_tokens=max_tokens)
    return final_chunks
