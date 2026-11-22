from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_engine.logging_config import log_operation
from rag_engine.settings import DEFAULT_DATABASE_BACKEND, DEFAULT_DOCS_ROOT

from .backends import get_database_backend
from .data_chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MINIMUM_CHUNK_SIZE,
    chunk_documents,
)
from .database_metadata_model import MetadataValue
from .process_chunks import embed_and_store_chunks

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@log_operation("Index construction")
def create_database(
    embedder: "SentenceTransformer",
    docs_root: str | Path = DEFAULT_DOCS_ROOT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    metadata_by_document: Mapping[str, Mapping[str, MetadataValue]] | None = None,
    backend: str = DEFAULT_DATABASE_BACKEND,
    minimum_chunk_size: int = DEFAULT_MINIMUM_CHUNK_SIZE,
    manifest_path: str | Path | None = None,
    **settings: Any,
) -> int:
    """Create a fresh index, chunk and embed documents, and return the chunk count.

    Existing collections are rejected. Use a new collection for each experiment.
    """

    docs_path = Path(docs_root).resolve()
    if not docs_path.is_dir():
        raise FileNotFoundError(f"Documentation directory does not exist: {docs_path}")
    chunks = chunk_documents(
        docs_root=docs_path,
        chunk_size=chunk_size,
        overlap=overlap,
        metadata_by_document=metadata_by_document,
        minimum_chunk_size=minimum_chunk_size,
    )
    if not chunks:
        raise ValueError(f"No documentation chunks found in {docs_path}")

    manifest_locations = (
        docs_path.parent / "debug" / "manifest.json",
        docs_path.parent / "manifest.json",
    )
    provenance = (
        Path(manifest_path)
        if manifest_path is not None
        else next((path for path in manifest_locations if path.is_file()), None)
    )
    if provenance is not None:
        from rag_engine.ingestion.convert import attach_provenance

        attach_provenance(chunks, provenance)

    database = get_database_backend(backend).create(**settings)
    embed_and_store_chunks(chunks, embedder, database, docs_root=docs_path)
    return len(chunks)
