from collections.abc import Set
from typing import TYPE_CHECKING

from rag_engine.engine.vector_store.base import VectorSearch
from rag_engine.engine.vector_store.database_metadata_model import Chunk, SearchFilters

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def search_unseen(
    query: str,
    embedder: "SentenceTransformer",
    database: VectorSearch,
    seen_ids: Set[str],
    *,
    limit: int = 5,
    max_candidates: int = 100,
    filters: SearchFilters | None = None,
) -> tuple[list[Chunk], int]:
    """Return up to ``limit`` unseen chunks, widening the ranking cut-off.

    The embedding is computed once; only how far down the ranked list we read
    changes. That cut-off starts at an initial window and doubles until enough
    unseen chunks are found, the collection is exhausted, or ``max_candidates``
    is reached.
    """
    if not query.strip() or limit < 1 or max_candidates < limit:
        raise ValueError("Invalid search query or candidate limits")
    vector = embedder.encode(query, normalize_embeddings=True).tolist()
    ranking_cutoff = min(max_candidates, max(10, limit))
    while True:
        results = database.similarity_search(
            query_embedding=vector,
            filters=filters if filters is not None else SearchFilters(),
            limit=ranking_cutoff,
        )
        fresh: list[Chunk] = []
        excluded = set(seen_ids)
        for chunk in results:
            if chunk.chunk_id not in excluded:
                fresh.append(chunk)
                excluded.add(chunk.chunk_id)
            if len(fresh) == limit:
                break
        if (
            len(fresh) >= limit
            or len(results) < ranking_cutoff
            or ranking_cutoff == max_candidates
        ):
            return fresh, ranking_cutoff
        ranking_cutoff = min(ranking_cutoff * 2, max_candidates)
