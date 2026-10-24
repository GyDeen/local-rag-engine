from hashlib import sha256
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
    expected_hash: str,
) -> str:
    """Read the chunk section text from source file and verify its hash to ensure it hasn't changed"""
    path = (docs_root / "/".join(source_path)).resolve()
    docs_root = docs_root.resolve()

    if not path.is_relative_to(docs_root):
        raise ValueError("Source path is outside the documentation directory")

    file_bytes = path.read_bytes()

    if sha256(file_bytes).hexdigest() != expected_hash:
        raise ValueError(f"Source file has changed: {source_path}")

    text = file_bytes.decode("utf-8")
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
        read_chunk_text(
            docs_root=docs_root,
            source_path=chunk.source_path,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            expected_hash=chunk.source_hash,
        )
        for chunk in chunks
    ]

    embeddings = embedder.encode(texts, normalize_embeddings=True)

    indexed_chunks = [
        Chunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            # Chroma rejects numpy scalars, so normalize to Python floats
            # (the same conversion chunk_to_output and _to_chunk use).
            embedding=[float(value) for value in embedding],
            source_path=chunk.source_path,
            section=chunk.section,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_hash=chunk.source_hash,
            metadata=chunk.metadata,
        )
        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        )
    ]

    database.upsert_chunks(indexed_chunks)
