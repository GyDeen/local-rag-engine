from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from rag_engine.settings import DEFAULT_DOCS_ROOT

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from .base import VectorDatabaseAdapter
from .database_metadata_model import Chunk


def read_chunk_text(
    docs_root: Path,
    source_path: list[str],
    start_offset: int,
    end_offset: int,
) -> str:
    """Read a validated source span for a chunk without stored text."""
    path = (docs_root / "/".join(source_path)).resolve()
    docs_root = docs_root.resolve()

    if not path.is_relative_to(docs_root):
        raise ValueError("Source path is outside the documentation directory")

    text = path.read_bytes().decode("utf-8")
    if not 0 <= start_offset <= end_offset <= len(text):
        raise ValueError(f"Invalid source offsets for {source_path}")
    return text[start_offset:end_offset]


def embed_and_store_chunks(
    chunks: list[Chunk],
    embedder: "SentenceTransformer",
    database: VectorDatabaseAdapter,
    docs_root: str | Path = DEFAULT_DOCS_ROOT,
    batch_size: int = 128,
) -> None:
    """Embed and store source chunks in bounded batches during database creation."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    for start in range(0, len(chunks), batch_size):
        _embed_and_store_batch(
            chunks[start : start + batch_size], embedder, database, Path(docs_root)
        )


def _embed_and_store_batch(
    chunks: list[Chunk],
    embedder: "SentenceTransformer",
    database: VectorDatabaseAdapter,
    docs_root: Path,
) -> None:
    texts = [
        (
            chunk.text
            if chunk.text is not None
            else read_chunk_text(
                docs_root=docs_root,
                source_path=chunk.source_path,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
            )
        )
        for chunk in chunks
    ]

    embeddings = embedder.encode(texts, normalize_embeddings=True)

    indexed_chunks = [
        replace(chunk, text=text, embedding=[float(value) for value in embedding])
        for chunk, text, embedding in zip(chunks, texts, embeddings, strict=True)
    ]

    database.upsert_chunks(indexed_chunks)
