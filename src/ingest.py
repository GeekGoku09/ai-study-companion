import logging
import os
import re
from typing import List
from pypdf import PdfReader
from pypdf.errors import PdfReadError
import tiktoken
from .schemas import DocumentChunk

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> List[DocumentChunk]:
    """
    Extracts text from a PDF and returns a list of DocumentChunks.
    Handles missing files, encrypted files, and pages without extractable text.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        raise PdfReadError(f"Failed to read PDF file at {pdf_path}: {e}") from e

    if reader.is_encrypted:
        try:
            # Attempt decrypt with empty password for default-encrypted PDFs
            decrypted = reader.decrypt("")
            if decrypted == 0:
                raise PdfReadError(f"PDF at {pdf_path} is encrypted with a password and cannot be read.")
        except Exception as e:
            raise PdfReadError(f"PDF at {pdf_path} is encrypted: {e}") from e

    chunks: List[DocumentChunk] = []

    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            logger.warning("Failed to extract text from page %d of %s: %s", page_num + 1, pdf_path, e)
            continue

        if not text:
            continue

        # Split into paragraphs by blank lines
        paragraphs = re.split(r'\n\s*\n', text)

        for i, para in enumerate(paragraphs):
            para = para.strip()
            # Skip very short paragraphs that might be headers, footers, or noise
            if len(para) < 50:
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
    """
    if not chunks:
        return []

    encoding = tiktoken.get_encoding("cl100k_base")
    refined_chunks: List[DocumentChunk] = []

    # Optimization (⚡ Bolt):
    # Maintain running token count and buffer paragraph contents in a list.
    # Tracking current_tokens incrementally and joining parts on flush reduces tokenization from O(N^2)
    # to O(N), avoiding repeated string copying and re-tokenization.
    current_parts: List[str] = []
    current_tokens: int = 0
    current_metadata = chunks[0].metadata

    for chunk in chunks:
        tokens_new = len(encoding.encode(chunk.content))

        # If adding the next paragraph exceeds max_tokens or we are on a new page,
        # save the current chunk and start a new one
        chunk_page = chunk.metadata.get("page")
        curr_page = current_metadata.get("page")
        if (current_tokens + tokens_new > max_tokens) or (chunk_page != curr_page):
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

    # Add the last chunk
    if current_parts:
        refined_chunks.append(DocumentChunk(
            content="\n\n".join(current_parts).strip(),
            metadata=current_metadata
        ))

    return refined_chunks


def ingest_pdf(pdf_path: str, max_tokens: int = 500) -> List[DocumentChunk]:
    """
    High-level function to ingest a PDF and return semantically chunked content.
    Returns an empty list if no text could be extracted.
    """
    initial_chunks = extract_text_from_pdf(pdf_path)
    if not initial_chunks:
        logger.warning("No readable text chunks found in PDF: %s", pdf_path)
        return []
    final_chunks = semantic_chunking(initial_chunks, max_tokens=max_tokens)
    return final_chunks
