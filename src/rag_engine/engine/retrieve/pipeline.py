from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from rag_engine.engine.rerank.llm_reranker.strategy import EvidenceAssessment

from rag_engine.engine.parse_input.input_processing import (
    PreparedQuery,
    prepare_retrieval,
)
from rag_engine.engine.rerank.base import RerankStrategy
from rag_engine.engine.rerank.candidate import RerankCandidate, build_rerank_candidates
from rag_engine.engine.rerank.factory import create_rerank_strategy
from rag_engine.engine.rerank.llm_reranker.pipeline import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_POOL_CHAR_BUDGET,
    DEFAULT_POOL_CHUNK_CHARS,
    DEFAULT_SUPPLEMENT_LIMIT,
)
from rag_engine.engine.rerank.selection import stitch_evidence
from rag_engine.engine.vector_store.base import VectorSearch
from rag_engine.engine.vector_store.database_metadata_model import Chunk, SearchFilters
from rag_engine.exceptions import RetrievalError
from rag_engine.logging_config import log_operation
from rag_engine.settings import (
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_DOCS_ROOT,
    DEFAULT_EVIDENCE_CHAR_BUDGET,
    DEFAULT_EVIDENCE_MIN_CHUNK_CHARS,
    DEFAULT_FILE_LIMIT,
    validate_evidence_rank_limit,
)


@dataclass(slots=True)
class RetrievalResult:
    """The evidence selected for one request, plus how it was selected.

    The result is after reranking and evidence stitching.
    """

    evidence: list[RerankCandidate]
    strategy: str
    stop_reason: str
    assessment: EvidenceAssessment | None = None
    trace: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Build cleaner retrieval result and convert into a form that can be stored as json file"""

        assessment = self.assessment
        return {
            "strategy": self.strategy,
            "stop_reason": self.stop_reason,
            "sufficient": None if assessment is None else assessment.sufficient,
            "selected_chunk_ids": [
                chunk_id
                for candidate in self.evidence
                for chunk_id in candidate.all_chunk_ids()
            ],
            "selected_files": list(
                dict.fromkeys("/".join(c.chunk.source_path) for c in self.evidence)
            ),
            "missing": [] if assessment is None else list(assessment.missing),
            "operator_questions": (
                [] if assessment is None else list(assessment.operator_questions)
            ),
            "search_queries": (
                [] if assessment is None else list(assessment.search_queries)
            ),
            "searches": self.trace.get("searches", []),
            "judgments": self.trace.get("judgments", []),
            "decision": self.trace.get("decision"),
        }


@log_operation("Vector retrieval", error_type=RetrievalError)
def retrieve_candidate_chunks(
    query_text: str | PreparedQuery,
    embedder: "SentenceTransformer",
    database: VectorSearch,
    chunk_limit: int = 50,
) -> list[Chunk]:
    """Return ranked candidates for query text or an already prepared query."""
    if chunk_limit < 1:
        raise ValueError("Chunk limit must be at least 1")
    prepared = (
        query_text
        if isinstance(query_text, PreparedQuery)
        else prepare_retrieval(query_text)
    )
    query = prepared.query_text
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Prepared query must contain non-empty query_text")
    return search_candidate_chunks(
        query, embedder, database, chunk_limit, prepared.metadata_filter
    )


def search_candidate_chunks(
    query: str,
    embedder: "SentenceTransformer",
    database: VectorSearch,
    chunk_limit: int,
    filters: SearchFilters | None = None,
) -> list[Chunk]:
    """Search ordinary document text with an explicit query and optional filters."""
    if not query.strip() or chunk_limit < 1:
        raise ValueError("A nonempty query and positive chunk limit are required")
    vector = embedder.encode(query, normalize_embeddings=True)
    return database.similarity_search(
        query_embedding=vector.tolist(),
        filters=filters or SearchFilters(),
        limit=chunk_limit,
    )


@log_operation("Retrieval", error_type=RetrievalError)
def retrieve_evidence(
    *,
    query_text: str,
    embedder: "SentenceTransformer",
    database: VectorSearch,
    reranker: str = "cross_encoder",
    chunk_limit: int = 50,
    llm_chunk_limit: int | None = None,
    file_limit: int = DEFAULT_FILE_LIMIT,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
    rerank_strategy: RerankStrategy | None = None,
    llm_caller=None,
    max_rounds: int = 0,
    input_context: Mapping[str, Any] | None = None,
    docs_root: str | Path = DEFAULT_DOCS_ROOT,
    evidence_char_budget: int = DEFAULT_EVIDENCE_CHAR_BUDGET,
    evidence_min_chunk_chars: int = DEFAULT_EVIDENCE_MIN_CHUNK_CHARS,
    evidence_rank_limit: int | None = None,
    llm_pool_char_budget: int = DEFAULT_POOL_CHAR_BUDGET,
    llm_pool_chunk_chars: int = DEFAULT_POOL_CHUNK_CHARS,
    llm_supplement_limit: int = DEFAULT_SUPPLEMENT_LIMIT,
    llm_max_candidates: int = DEFAULT_MAX_CANDIDATES,
    filters: SearchFilters | None = None,
) -> RetrievalResult:
    """Retrieve and select evidence for a free-form question.

    ``reranker`` selects how the retrieved candidates are ordered:

    ``"cross_encoder"``
        Rerank with a local cross-encoder. This is the default. Supply an existing
        ``rerank_strategy`` to reuse its model; otherwise construct one for this call.
    ``"llm"``
        Let the LLM rank the candidates and supplement them when it reports
        missing evidence. Assessment and exploration stay in the LLM implementation.
    ``"none"``
        Skip reranking and keep vector order. No reranker is imported or
        constructed, so nothing is downloaded or loaded into memory.

    Whichever strategy orders the candidates, the same rule turns that order
    into evidence: the chosen sections are completed from the candidate pool and
    stitched into contiguous text, up to ``evidence_char_budget`` characters over
    at most ``file_limit`` files. When ``evidence_rank_limit`` is set, only the
    first N ordered chunks may contribute, including during section completion.
    With no reranker this cutoff uses vector order.

    The four ``llm_*`` budgets bound what the LLM judge is shown: the character
    budget for its whole candidate pool, the per-chunk ceiling inside that pool,
    how many chunks a supplementary query returns, and how far those searches may
    widen. They apply only to that strategy.

    ``llm_chunk_limit`` optionally gives the LLM reranker its own initial
    candidate count. The two strategies want different values: the cross-encoder
    scores a longer shortlist for almost nothing and benefits from the room,
    while the LLM judge must read every candidate it is asked to rank, and its
    pool is capped by characters before the count matters. When it is None the
    LLM reranker uses ``chunk_limit`` like every other strategy.

    Supplied strategies are trusted here; ``RagRuntime`` validates their shared
    interface at startup. Standalone calls without a strategy use the shared factory.
    """
    validate_evidence_rank_limit(evidence_rank_limit)
    if file_limit < 1:
        raise ValueError("File limit must be at least 1")
    if chunk_limit < file_limit:
        raise ValueError("Chunk limit must be at least the file limit")
    if llm_chunk_limit is not None and llm_chunk_limit < file_limit:
        raise ValueError("LLM chunk limit must be at least the file limit")

    prepared = prepare_retrieval(query_text, filters=filters)
    if rerank_strategy is None and reranker != "none":
        rerank_strategy = create_rerank_strategy(
            reranker,
            cross_encoder_model=cross_encoder_model,
            llm_caller=llm_caller,
            file_limit=file_limit,
        )

    candidate_limit = (
        llm_chunk_limit
        if reranker == "llm" and llm_chunk_limit is not None
        else chunk_limit
    )
    chunks = retrieve_candidate_chunks(
        query_text=prepared,
        embedder=embedder,
        database=database,
        chunk_limit=candidate_limit,
    )
    pool = build_rerank_candidates(chunks, docs_root=docs_root)

    def evidence_from(
        ranked: list[RerankCandidate],
        evidence_pool: list[RerankCandidate] | None = None,
    ) -> list[RerankCandidate]:
        return stitch_evidence(
            ranked,
            pool if evidence_pool is None else evidence_pool,
            file_limit=file_limit,
            char_budget=evidence_char_budget,
            min_chunk_chars=evidence_min_chunk_chars,
            evidence_rank_limit=evidence_rank_limit,
        )

    if reranker == "none":
        # Vector order is the ranking here, so no model is loaded at all.
        return RetrievalResult(
            evidence=evidence_from(pool),
            strategy="none",
            stop_reason="strategy_selected",
        )

    if reranker == "llm":
        from rag_engine.engine.rerank.llm_reranker.pipeline import explore_evidence

        exploration = explore_evidence(
            query=prepared.query_text,
            initial_chunks=chunks,
            embedder=embedder,
            database=database,
            rerank_strategy=rerank_strategy,
            file_limit=file_limit,
            max_rounds=max_rounds,
            supplement_limit=llm_supplement_limit,
            max_candidates=llm_max_candidates,
            pool_char_budget=llm_pool_char_budget,
            pool_chunk_chars=llm_pool_chunk_chars,
            docs_root=docs_root,
            input_context=dict(input_context) if input_context else None,
            filters=prepared.metadata_filter,
        )
        # The model selects chunks its own searches discovered, which the initial
        # pool never held. Stitching from what the exploration gathered keeps
        # those sections completable, and full text keeps their offsets honest.
        gathered = build_rerank_candidates(
            list(chunks) + list(exploration["discovered"]), docs_root=docs_root
        )
        return RetrievalResult(
            evidence=evidence_from(
                build_rerank_candidates(exploration["chunks"], docs_root=docs_root),
                gathered,
            ),
            strategy="llm",
            stop_reason=exploration["stop_reason"],
            assessment=exploration["assessment"],
            trace={
                "decision": exploration["decision"],
                "searches": exploration["searches"],
                "judgments": exploration["judgments"],
            },
        )

    ranked = rerank_strategy.rerank(prepared.query_text, pool)
    return RetrievalResult(
        evidence=evidence_from(ranked),
        strategy=reranker,
        stop_reason="strategy_selected",
    )
