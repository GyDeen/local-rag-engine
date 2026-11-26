from dataclasses import dataclass, field

from rag_engine.engine.vector_store.database_metadata_model import SearchFilters
from rag_engine.logging_config import log_operation


@dataclass(frozen=True, slots=True)
class PreparedQuery:
    """Query text and metadata filters ready for vector search."""

    query_text: str
    metadata_filter: SearchFilters = field(default_factory=SearchFilters)


@log_operation("Input preparation")
def prepare_retrieval(
    query_text: str,
    *,
    filters: SearchFilters | None = None,
) -> PreparedQuery:
    """Prepare question text and explicit filters for search."""
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("Question must contain nonempty text")
    return PreparedQuery(
        query_text, filters if filters is not None else SearchFilters()
    )
