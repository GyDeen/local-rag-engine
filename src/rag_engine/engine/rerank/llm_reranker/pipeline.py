from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_engine.engine.rerank.base import RerankStrategy
from rag_engine.engine.rerank.candidate import (
    RerankCandidate,
    build_rerank_candidates,
    to_stored,
)
from rag_engine.engine.rerank.llm_reranker.search import search_unseen
from rag_engine.engine.rerank.llm_reranker.strategy import LLMCaller, LLMReranker
from rag_engine.engine.vector_store.base import VectorSearch
from rag_engine.engine.vector_store.database_metadata_model import Chunk, SearchFilters
from rag_engine.settings import DEFAULT_DOCS_ROOT, DEFAULT_FILE_LIMIT

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MAX_SEARCH_ROUNDS = 2

# Total passage text the judge may see in one call. The judge can only rank what
# it is shown, so this has to cover the candidate pool, but it also sets the
# prompt size and the generator's context is finite (16,384 tokens by default).
# 24,000 characters is roughly 6,500 tokens of evidence, leaving room for the
# instructions, the alarm and the JSON reply.
DEFAULT_POOL_CHAR_BUDGET = 24000

# Ceiling on one chunk's contribution. The corpus holds chunks up to 7,400
# characters, so without a per-chunk cap three long passages would consume the
# whole budget above and starve every other candidate. Truncated chunks are
# flagged so the model treats their text as incomplete.
DEFAULT_POOL_CHUNK_CHARS = 4000

# Chunks fetched per supplementary query, and how far `search_unseen` may widen
# the ranked list while looking for passages the model has not seen yet.
DEFAULT_SUPPLEMENT_LIMIT = 5
DEFAULT_MAX_CANDIDATES = 100

#: Anything the exploration loop can turn into rerank candidates.
ChunkSource = Chunk | Mapping[str, Any] | RerankCandidate


def _normalise_query(query: str) -> str:
    """Fold a search query so near-identical repeats compare equal."""

    return " ".join(query.casefold().split())


def _reformulation_instruction(attempted: Sequence[str]) -> str:
    """Tell the model exactly which queries it may not propose again."""

    already = "; ".join(attempted) if attempted else "none"
    return (
        "Every search query you proposed has already been searched and returned "
        f"nothing new: {already}. Propose different queries that approach the gap "
        "from another angle, naming other components or commands. If the selected "
        "passages already explain and resolve the alarm, set sufficient true. If "
        "only runtime facts are missing, leave search_queries empty and list the "
        "operator questions instead. Never repeat a query listed above."
    )


def _load_bounded(
    chunks: Iterable[ChunkSource],
    docs_root: str | Path,
    max_chars: int,
    chunk_chars: int,
) -> list[RerankCandidate]:
    """Materialise candidates and bound how much passage text reaches the LLM."""

    loaded: list[RerankCandidate] = []
    used = 0
    ids: set[str] = set()

    for candidate in build_rerank_candidates(chunks, docs_root=docs_root):
        chunk_id = candidate.chunk.chunk_id
        if chunk_id in ids:
            continue
        if used >= max_chars:
            break
        available = min(chunk_chars, max_chars - used)
        text = candidate.text[:available]
        loaded.append(
            RerankCandidate(
                chunk=candidate.chunk,
                text=text,
                score=candidate.score,
                truncated=candidate.truncated or len(candidate.text) > available,
                chunk_ids=candidate.chunk_ids,
            )
        )
        used += len(text)
        ids.add(chunk_id)

    return loaded


def explore_evidence(
    *,
    alarm_code: str,
    alarm_text: str,
    initial_chunks: Iterable[ChunkSource],
    embedder: "SentenceTransformer",
    database: VectorSearch,
    llm_caller: LLMCaller | None = None,
    rerank_strategy: RerankStrategy | None = None,
    file_limit: int = DEFAULT_FILE_LIMIT,
    max_rounds: int = MAX_SEARCH_ROUNDS,
    supplement_limit: int = DEFAULT_SUPPLEMENT_LIMIT,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    pool_char_budget: int = DEFAULT_POOL_CHAR_BUDGET,
    pool_chunk_chars: int = DEFAULT_POOL_CHUNK_CHARS,
    docs_root: str | Path = DEFAULT_DOCS_ROOT,
    user_question: str | None = None,
    input_context: dict[str, Any] | None = None,
    filters: SearchFilters | None = None,
) -> dict[str, Any]:
    """Return selected stored chunks plus an auditable exploration trace.

    ``max_rounds`` counts supplementary search rounds after the initial
    judgement, so by default there is one initial judgement and up to two rounds
    of query-then-search. A round ends early only when the model offers no
    usable query even after being asked to reformulate, or when it reports the
    evidence sufficient.

    What the judge can see is bounded on two axes, and the character budget binds
    first: ``pool_char_budget`` caps the whole pool and ``pool_chunk_chars`` caps
    one chunk, so widening ``initial_chunks`` past the character budget adds
    retrieval work without adding anything the model reads. File count is capped
    separately by ``file_limit``.
    """
    if (
        file_limit < 1
        or max_rounds < 0
        or supplement_limit < 1
        or max_candidates < supplement_limit
        or pool_char_budget < 1
        or pool_chunk_chars < 1
    ):
        raise ValueError("Invalid exploration limits")

    if rerank_strategy is None:
        if llm_caller is None:
            raise ValueError("The 'llm' reranker requires an llm_caller or strategy")
        rerank_strategy = LLMReranker(llm_caller, file_limit=file_limit)
    elif not isinstance(rerank_strategy, LLMReranker):
        raise TypeError("LLM exploration requires an LLMReranker")
    elif rerank_strategy.file_limit != file_limit:
        raise ValueError("file_limit must match the configured LLM reranker")
    pool = _load_bounded(initial_chunks, docs_root, pool_char_budget, pool_chunk_chars)
    seen = {candidate.chunk.chunk_id for candidate in pool}
    attempted: set[str] = set()
    history: list[dict[str, Any]] = []
    judgments: list[dict[str, Any]] = []
    selected: list[RerankCandidate] = []
    # Chunks the supplementary searches surfaced. They entered the pool shown to
    # the model, so callers can stitch their sections from full text.
    discovered: list[Chunk] = []
    decision: dict[str, Any] = {
        "sufficient": False,
        "missing": ["No candidate evidence"],
        "operator_questions": [],
        "coverage": {},
        "ranked_chunk_ids": [],
    }
    assessment = None
    stop_reason = "no_candidates"

    if not pool:
        return {
            "chunks": [],
            "discovered": [],
            "sufficient": False,
            "stop_reason": stop_reason,
            "decision": decision,
            "assessment": None,
            "searches": [],
            "judgments": [],
        }

    for round_index in range(max_rounds + 1):
        outcome = rerank_strategy.judge(
            alarm_text,
            pool,
            alarm_code=alarm_code,
            alarm_text=alarm_text,
            user_question=user_question,
            input_context=input_context,
            remaining_search_rounds=max_rounds - round_index,
            previous_decision=judgments[-1]["decision"] if judgments else None,
            search_history=history,
        )
        decision, selected = outcome.raw, outcome.ranked
        assessment = outcome.assessment
        judgments.append(
            {
                "round": round_index,
                "visible_chunk_ids": [c.chunk.chunk_id for c in pool],
                "decision": decision,
                "validation_errors": outcome.validation_errors,
            }
        )
        if decision["sufficient"]:
            stop_reason = "sufficient"
            break
        if round_index == max_rounds:
            stop_reason = "round_budget"
            break

        # Only search with queries that have not been tried before.
        queries = [
            query
            for query in assessment.search_queries
            if _normalise_query(query) not in attempted
        ]
        if not queries:
            # Every proposed query was already searched, so ask once for a
            # reformulation rather than ending the exploration on a repeat.
            outcome = rerank_strategy.judge(
                alarm_text,
                pool,
                alarm_code=alarm_code,
                alarm_text=alarm_text,
                user_question=user_question,
                input_context=input_context,
                remaining_search_rounds=max_rounds - round_index,
                previous_decision=decision,
                search_history=history,
                instruction=_reformulation_instruction(sorted(attempted)),
            )
            decision, selected = outcome.raw, outcome.ranked
            assessment = outcome.assessment
            judgments.append(
                {
                    "round": round_index,
                    "reformulation": True,
                    "visible_chunk_ids": [c.chunk.chunk_id for c in pool],
                    "decision": decision,
                    "validation_errors": outcome.validation_errors,
                }
            )
            if decision["sufficient"]:
                stop_reason = "sufficient"
                break
            queries = [
                query
                for query in assessment.search_queries
                if _normalise_query(query) not in attempted
            ]
        if not queries:
            stop_reason = (
                "operator_input" if assessment.operator_questions else "no_new_query"
            )
            break

        fresh: list[Chunk] = []
        for query in queries:
            query_key = _normalise_query(query)
            if query_key in attempted:
                continue
            attempted.add(query_key)
            chunks, ranking_cutoff = search_unseen(
                query,
                embedder,
                database,
                seen | {chunk.chunk_id for chunk in fresh},
                limit=supplement_limit,
                max_candidates=max_candidates,
                filters=filters,
            )
            fresh.extend(chunks)
            history.append(
                {
                    "query": query,
                    "searched_depth": ranking_cutoff,
                    "returned_chunk_ids": [chunk.chunk_id for chunk in chunks],
                }
            )
        # Reserve half the passage budget for discoveries; retain selected
        # evidence ahead of unselected old candidates in the remaining half.
        if fresh:
            new_pool = _load_bounded(
                fresh, docs_root, pool_char_budget // 2, pool_chunk_chars
            )
            discovered.extend(candidate.chunk for candidate in new_pool)
            remaining = pool_char_budget - sum(len(c.text) for c in new_pool)
            retained = _load_bounded(
                selected + pool, docs_root, remaining, pool_chunk_chars
            )
            pool = retained + new_pool
            seen.update(candidate.chunk.chunk_id for candidate in pool)
        # Even an empty search is shown once so the LLM can reformulate its query.

    return {
        "chunks": [to_stored(candidate) for candidate in selected],
        "discovered": discovered,
        "sufficient": decision["sufficient"],
        "stop_reason": stop_reason,
        "decision": decision,
        "assessment": assessment,
        "searches": history,
        "judgments": judgments,
    }
