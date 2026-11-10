import logging
import os
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .schemas import DocumentChunk
from .utils import count_tokens

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str, min_para_length: int = 50) -> list[DocumentChunk]:
    """Extracts text from a PDF and returns a list of DocumentChunks."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
    except PdfReadError as e:
        raise ValueError(f"Failed to parse PDF {pdf_path}: {e}") from e

    if reader.is_encrypted:
        try:
            # Try decrypting with empty password if encrypted
            decrypt_result = reader.decrypt("")
            if decrypt_result == 0:
                raise ValueError(f"PDF {pdf_path} is password-protected and cannot be read.")
        except Exception as e:
            raise ValueError(f"PDF {pdf_path} is password-protected: {e}") from e

    chunks: list[DocumentChunk] = []

    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except (PdfReadError, TypeError, ValueError) as e:
            logger.warning("Error extracting text from page %d in %s: %s", page_num + 1, pdf_path, e)
            continue

        if not text:
            continue

        # Semantic chunking by splitting into paragraphs
        paragraphs = re.split(r"\n\s*\n", text)

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < min_para_length:  # Skip very short paragraphs that might be noise
                continue

            chunks.append(
                DocumentChunk(
                    content=para,
                    metadata={
                        "source": os.path.abspath(pdf_path),
                        "page": page_num + 1,
                        "chunk_id": f"p{page_num + 1}_c{i}",
                    },
                )
            )

    return chunks


def semantic_chunking(chunks: list[DocumentChunk], max_tokens: int = 500) -> list[DocumentChunk]:
    """Refines chunks to ensure they are semantically coherent and within token limits.

    Groups small consecutive paragraphs from the same page if they fit in max_tokens.
    """
    if not chunks:
        return []

    refined_chunks: list[DocumentChunk] = []

    # Optimization (⚡ Bolt):
    # Maintain running token count and buffer paragraph contents in a list.
    # Linear O(N) tokenization and single join per chunk.
    current_parts: list[str] = []
    current_tokens: int = 0
    current_metadata = chunks[0].metadata

    for chunk in chunks:
        tokens_new = count_tokens(chunk.content)

        # If adding the next paragraph exceeds max_tokens or we are on a new page,
        # save the current chunk and start a new one
        if (current_tokens + tokens_new > max_tokens) or (
            chunk.metadata.get("page") != current_metadata.get("page")
        ):
            if current_parts:
                refined_chunks.append(
                    DocumentChunk(
                        content="\n\n".join(current_parts).strip(),
                        metadata=current_metadata,
                    )
                )
            current_parts = [chunk.content]
            current_tokens = tokens_new
            current_metadata = chunk.metadata
        else:
            current_parts.append(chunk.content)
            current_tokens += tokens_new

    # Add the last chunk
    if current_parts:
        refined_chunks.append(
            DocumentChunk(
                content="\n\n".join(current_parts).strip(),
                metadata=current_metadata,
            )
        )

    return refined_chunks


def ingest_pdf(pdf_path: str, max_tokens: int = 500) -> list[DocumentChunk]:
    """High-level function to ingest a PDF and return semantically chunked content."""
    initial_chunks = extract_text_from_pdf(pdf_path)
    if not initial_chunks:
        raise ValueError(
            f"No extractable text found in '{pdf_path}'. Please ensure the PDF is not scanned or empty."
        )
    return semantic_chunking(initial_chunks, max_tokens=max_tokens)
