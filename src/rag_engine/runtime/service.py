from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag_engine.engine.rerank.llm_reranker.pipeline import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_POOL_CHAR_BUDGET,
    DEFAULT_POOL_CHUNK_CHARS,
    DEFAULT_SUPPLEMENT_LIMIT,
)
from rag_engine.engine.retrieve.pipeline import RetrievalResult, retrieve_evidence
from rag_engine.engine.vector_store.database_metadata_model import SearchFilters
from rag_engine.settings import (
    DEFAULT_DOCS_ROOT,
    DEFAULT_EVIDENCE_CHAR_BUDGET,
    DEFAULT_EVIDENCE_MIN_CHUNK_CHARS,
    validate_evidence_rank_limit,
)

from .resources import RagRuntime


def question_result(question: str, retrieval: RetrievalResult) -> dict[str, Any]:
    """Prepare ranked passages with stable citation IDs for the current question."""
    evidence = []
    for index, candidate in enumerate(retrieval.evidence, 1):
        metadata = candidate.chunk.metadata
        evidence.append(
            {
                "id": f"E{index}",
                "source": metadata.get("source_document", candidate.chunk.document_id),
                "location": metadata.get(
                    "source_location", "/".join(candidate.chunk.source_path)
                ),
                "pages": metadata.get("source_pages", []),
                "section": candidate.chunk.section,
                "text": candidate.text,
                "chunk_ids": list(candidate.all_chunk_ids()),
            }
        )
    return {
        "question": question,
        "retrieval": retrieval.as_dict(),
        "evidence": evidence,
        "answer": None,
        "status": None,
    }


@dataclass(frozen=True, slots=True)
class RetrievalOptions:
    """Fixed retrieval behavior for requests sharing one service."""

    chunk_limit: int = 50
    llm_chunk_limit: int | None = None
    max_rounds: int = 0
    docs_root: str | Path = DEFAULT_DOCS_ROOT
    evidence_char_budget: int = DEFAULT_EVIDENCE_CHAR_BUDGET
    evidence_min_chunk_chars: int = DEFAULT_EVIDENCE_MIN_CHUNK_CHARS
    llm_pool_char_budget: int = DEFAULT_POOL_CHAR_BUDGET
    llm_pool_chunk_chars: int = DEFAULT_POOL_CHUNK_CHARS
    llm_supplement_limit: int = DEFAULT_SUPPLEMENT_LIMIT
    llm_max_candidates: int = DEFAULT_MAX_CANDIDATES
    evidence_rank_limit: int | None = None

    def __post_init__(self) -> None:
        validate_evidence_rank_limit(self.evidence_rank_limit)
        for name in (
            "evidence_char_budget",
            "llm_pool_char_budget",
            "llm_pool_chunk_chars",
            "llm_supplement_limit",
            "llm_max_candidates",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.chunk_limit < 1:
            raise ValueError("chunk_limit must be at least 1")
        if self.llm_chunk_limit is not None and self.llm_chunk_limit < 1:
            raise ValueError("llm_chunk_limit must be at least 1")
        if self.llm_max_candidates < self.llm_supplement_limit:
            raise ValueError("llm_max_candidates must be at least llm_supplement_limit")
        if self.max_rounds < 0 or self.evidence_min_chunk_chars < 0:
            raise ValueError(
                "max_rounds and evidence_min_chunk_chars cannot be negative"
            )


@dataclass(frozen=True, slots=True)
class RagService:
    """Reuse retrieval resources while keeping evidence local to each request."""

    runtime: RagRuntime
    options: RetrievalOptions = field(default_factory=RetrievalOptions)

    def __post_init__(self) -> None:
        if self.options.chunk_limit < self.runtime.file_limit:
            raise ValueError("chunk_limit must be at least the configured file_limit")
        if (
            self.options.llm_chunk_limit is not None
            and self.options.llm_chunk_limit < self.runtime.file_limit
        ):
            raise ValueError(
                "llm_chunk_limit must be at least the configured file_limit"
            )

    def retrieve_question(
        self, question: str, *, filters: SearchFilters | None = None
    ) -> RetrievalResult:
        """Search indexed documentation for the caller's question."""
        return retrieve_evidence(
            query_text=question,
            filters=filters,
            embedder=self.runtime.embedder,
            database=self.runtime.database,
            reranker=self.runtime.reranker,
            rerank_strategy=self.runtime.rerank_strategy,
            file_limit=self.runtime.file_limit,
            **asdict(self.options),
        )

    def ask(
        self,
        question: str,
        *,
        filters: SearchFilters | None = None,
    ) -> dict:
        """The document retrieval path shared by HTTP requests and evaluation."""
        return question_result(
            question, self.retrieve_question(question, filters=filters)
        )
