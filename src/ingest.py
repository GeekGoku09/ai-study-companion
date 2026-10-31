import os
import re
from typing import List
from pypdf import PdfReader
from .schemas import DocumentChunk
from .utils import get_tokenizer, count_tokens

def _split_oversized_text(text: str, max_tokens: int, encoding) -> List[str]:
    """Splits an oversized paragraph into pieces within max_tokens at sentence/sub-boundaries."""
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    pieces = []
    sentences = re.split(r'(?<=[.?!])\s+', text)
    current_sentence_group = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = len(encoding.encode(sentence))
        if s_tokens > max_tokens:
            if current_sentence_group:
                pieces.append(" ".join(current_sentence_group))
                current_sentence_group = []
                current_tokens = 0
            raw_tokens = encoding.encode(sentence)
            for i in range(0, len(raw_tokens), max_tokens):
                chunk_tokens = raw_tokens[i:i + max_tokens]
                pieces.append(encoding.decode(chunk_tokens))
        elif current_tokens + s_tokens > max_tokens:
            pieces.append(" ".join(current_sentence_group))
            current_sentence_group = [sentence]
            current_tokens = s_tokens
        else:
            current_sentence_group.append(sentence)
            current_tokens += s_tokens

    if current_sentence_group:
        pieces.append(" ".join(current_sentence_group))

    return pieces

def extract_text_from_pdf(pdf_path: str) -> List[DocumentChunk]:
    """
    Extracts text from a PDF and returns a list of DocumentChunks.
    Handles missing files, scanned/empty PDFs, and parses pages safely.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    if os.path.isdir(pdf_path):
        raise ValueError(f"Path '{pdf_path}' is a directory, not a file.")

    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF at {pdf_path}: {e}") from e

    if len(reader.pages) == 0:
        return []

    chunks = []
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        if not text.strip():
            continue

        paragraphs = re.split(r'\n\s*\n', text)
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < 50:
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

    return chunks

def semantic_chunking(
    chunks: List[DocumentChunk],
    max_tokens: int = 500,
    split_oversized: bool = False
) -> List[DocumentChunk]:
    """
    Refines chunks to ensure they are semantically coherent and within token limits.
    Groups small consecutive paragraphs from the same page if they fit in max_tokens.
    If split_oversized is True, breaks down individual chunks that exceed max_tokens.
    """
    if not chunks:
        return []

    encoding = get_tokenizer("cl100k_base")

    if split_oversized:
        normalized_chunks: List[DocumentChunk] = []
        for chunk in chunks:
            tokens = len(encoding.encode(chunk.content))
            if tokens > max_tokens:
                sub_texts = _split_oversized_text(chunk.content, max_tokens, encoding)
                for idx, sub_text in enumerate(sub_texts):
                    sub_meta = dict(chunk.metadata)
                    sub_meta["sub_index"] = idx
                    normalized_chunks.append(DocumentChunk(
                        content=sub_text.strip(),
                        metadata=sub_meta
                    ))
            else:
                normalized_chunks.append(chunk)
    else:
        normalized_chunks = chunks

    refined_chunks: List[DocumentChunk] = []
    current_chunk_content = ""
    current_metadata = dict(normalized_chunks[0].metadata)

    for chunk in normalized_chunks:
        tokens_current = len(encoding.encode(current_chunk_content)) if current_chunk_content else 0
        tokens_new = len(encoding.encode(chunk.content))

        same_page = chunk.metadata.get("page") == current_metadata.get("page")
        if (tokens_current + tokens_new > max_tokens) or not same_page:
            if current_chunk_content:
                refined_chunks.append(DocumentChunk(
                    content=current_chunk_content.strip(),
                    metadata=current_metadata
                ))
            current_chunk_content = chunk.content
            current_metadata = dict(chunk.metadata)
        else:
            if current_chunk_content:
                current_chunk_content += "\n\n" + chunk.content
            else:
                current_chunk_content = chunk.content
                current_metadata = dict(chunk.metadata)

    if current_chunk_content:
        refined_chunks.append(DocumentChunk(
            content=current_chunk_content.strip(),
            metadata=current_metadata
        ))

    return refined_chunks

def ingest_pdf(pdf_path: str, max_tokens: int = 500, split_oversized: bool = False) -> List[DocumentChunk]:
    """High-level function to ingest a PDF and return semantically chunked content."""
    initial_chunks = extract_text_from_pdf(pdf_path)
    if not initial_chunks:
        return []
    final_chunks = semantic_chunking(initial_chunks, max_tokens=max_tokens, split_oversized=split_oversized)
    return final_chunks
