from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol

from .database_metadata_model import Chunk, SearchFilters

"""Database handler interface to be implemented for different database"""


class VectorSearch(Protocol):
    """Read-only search boundary consumed by the retrieval pipeline."""

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        filters: SearchFilters,
        limit: int = 5,
    ) -> list[Chunk]:
        """Return matching chunks without creating or updating the index."""
        ...


class VectorDatabaseAdapter(ABC):
    """Database-independent interface for storing and retrieving chunks."""

    @abstractmethod
    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
    ) -> None:
        """Insert chunks or replace existing chunks with matching IDs."""

    @abstractmethod
    def delete_chunks(
        self,
        chunk_ids: Sequence[str],
    ) -> None:
        """Delete chunks from the database based on their chunk_ids"""

    @abstractmethod
    def get_chunk_by_id(
        self,
        chunk_id: str,
    ) -> Chunk | None:
        """Get a chunk from the database based on its chunk_id"""

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: Sequence[float],
        filters: SearchFilters,
        limit: int = 5,
    ) -> list[Chunk]:
        """Perform similarity search in the database based on their embeddings.
        Return a list of chunks that are most similar to the query"""
