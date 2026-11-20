import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.errors import ChromaError, NotFoundError

from rag_engine.exceptions import DatabaseError, SourceDocumentError
from rag_engine.logging_config import log_operation
from rag_engine.settings import DEFAULT_COLLECTION_NAME, DEFAULT_DATABASE_PATH

from .base import VectorDatabaseAdapter
from .database_metadata_model import Chunk, SearchFilters

DEFAULT_CHROMA_DATABASE_PATH = DEFAULT_DATABASE_PATH
DEFAULT_CHROMA_COLLECTION_NAME = DEFAULT_COLLECTION_NAME


class ChromaVectorDatabase(VectorDatabaseAdapter):
    """Translate the project's vector database interface into Chroma calls."""

    def __init__(
        self,
        client: ClientAPI,
        collection_name: str = DEFAULT_CHROMA_COLLECTION_NAME,
    ) -> None:

        self._collection = client.get_collection(
            name=collection_name,
            embedding_function=None,
        )

    @log_operation(
        "Vector database upsert",
        error_type=DatabaseError,
        handled_errors=(ChromaError, OSError, RuntimeError),
    )
    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Insert new chunks or replace existing chunks with the same IDs."""

        if not chunks:
            return

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            documents=[self._serialize_chunk_info(chunk) for chunk in chunks],
            metadatas=[chunk.metadata or None for chunk in chunks],
        )

    @log_operation(
        "Vector database delete",
        error_type=DatabaseError,
        handled_errors=(ChromaError, OSError, RuntimeError),
    )
    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        """Delete all chunks whose IDs are supplied."""

        if chunk_ids:
            self._collection.delete(ids=list(chunk_ids))

    @log_operation(
        "Vector database lookup",
        error_type=DatabaseError,
        handled_errors=(ChromaError, OSError, RuntimeError),
    )
    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Return one chunk by its exact ID, or None when it does not exist."""

        result = self._collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas", "embeddings"],
        )

        if result["embeddings"] is None:
            raise DatabaseError("Chroma did not return the requested embedding")

        if not result["ids"]:
            return None

        metadatas = result["metadatas"] or []
        metadata = metadatas[0] if metadatas else {}

        return self._to_chunk(
            chunk_id=result["ids"][0],
            embedding=result["embeddings"][0],
            metadata=metadata or {},
            document=(result["documents"] or [None])[0],
        )

    @log_operation(
        "Vector database search",
        error_type=DatabaseError,
        handled_errors=(ChromaError, OSError, RuntimeError),
    )
    def similarity_search(
        self,
        query_embedding: Sequence[float],
        filters: SearchFilters,
        limit: int = 5,
    ) -> list[Chunk]:
        """Return the nearest chunks after applying optional metadata filters."""

        arguments: dict[str, Any] = {
            "query_embeddings": [list(query_embedding)],
            "n_results": limit,
            "include": ["documents", "metadatas", "embeddings"],
        }

        where = self._extract_filters(filters)
        if where is not None:
            arguments["where"] = where

        result = self._collection.query(**arguments)
        ids = result["ids"][0]
        documents = (result["documents"] or [[None] * len(ids)])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        if result["embeddings"] is None:
            raise DatabaseError("Chroma did not return the requested embedding")
        embeddings = result["embeddings"][0]

        return [
            self._to_chunk(
                chunk_id=chunk_id,
                metadata=metadata or {},
                embedding=embedding,
                document=document,
            )
            for chunk_id, document, metadata, embedding in zip(
                ids,
                documents,
                metadatas,
                embeddings,
                strict=True,
            )
        ]

    @staticmethod
    def _serialize_chunk_info(chunk: Chunk) -> str:
        """Persist chunk properties separately from filterable metadata."""

        return json.dumps(
            {
                "document_id": chunk.document_id,
                "source_path": chunk.source_path,
                "section": chunk.section,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                "text": chunk.text,
            }
        )

    @staticmethod
    def _extract_filters(filters: SearchFilters) -> dict[str, Any] | None:
        """Translate project search filters into a Chroma where expression."""

        clauses: list[dict[str, Any]] = []

        for field_name, value in filters.active_fields():
            values = (value,) if isinstance(value, str) else value
            for item in values:
                clauses.append(
                    {
                        "$or": [
                            {field_name: {"$eq": item}},
                            {field_name: {"$contains": item}},
                        ]
                    }
                )

        if not clauses:
            return None

        if len(clauses) == 1:
            return clauses[0]

        return {"$and": clauses}

    @staticmethod
    def _to_chunk(
        chunk_id: str,
        embedding: Sequence[float],
        metadata: Mapping[str, Any],
        document: str | None = None,
    ) -> Chunk:
        """Reconstruct a complete chunk from its separately stored fields."""

        if document is None:
            raise SourceDocumentError(
                "Stored chunk properties are missing. Rebuild the index "
                "into a fresh collection."
            )
        chunk_info = json.loads(document)
        if "text" not in chunk_info:
            raise SourceDocumentError(
                "Rebuild the legacy index into a fresh collection."
            )
        return Chunk(
            chunk_id=chunk_id,
            embedding=[float(value) for value in embedding],
            document_id=chunk_info["document_id"],
            source_path=chunk_info["source_path"],
            section=chunk_info["section"],
            start_offset=chunk_info["start_offset"],
            end_offset=chunk_info["end_offset"],
            text=chunk_info["text"],
            metadata=dict(metadata),
        )


def create_chroma_database(
    database_path: str | Path = DEFAULT_CHROMA_DATABASE_PATH,
    collection_name: str = DEFAULT_CHROMA_COLLECTION_NAME,
) -> VectorDatabaseAdapter:
    """Create a new collection; reject existing collections without changing them."""

    client = chromadb.PersistentClient(path=str(Path(database_path)))
    client.create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
    return ChromaVectorDatabase(
        client=client,
        collection_name=collection_name,
    )


def open_chroma_database(
    database_path: str | Path = DEFAULT_CHROMA_DATABASE_PATH,
    collection_name: str = DEFAULT_CHROMA_COLLECTION_NAME,
) -> VectorDatabaseAdapter:
    """Open an indexed collection without creating a database or collection."""

    path = Path(database_path)
    if not (path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            f"No Chroma database at {path}. Run the database creation command first."
        )
    client = chromadb.PersistentClient(path=str(path))
    try:
        database = ChromaVectorDatabase(client, collection_name)
    except NotFoundError as error:
        raise ValueError(
            f"Collection {collection_name!r} does not exist. "
            "Run the database creation command first."
        ) from error
    if database._collection.count() == 0:
        raise ValueError(f"Collection {collection_name!r} contains no indexed chunks")
    return database
