"""Map engine results to the public HTTP response contract."""

from collections.abc import Mapping
from typing import Any

from rag_engine.engine.llm.evidence import normalise_chunks
from rag_engine.engine.rerank.candidate import to_stored
from rag_engine.engine.retrieve.pipeline import RetrievalResult

from .models import ResolveResponse, RetrievedEvidence, RetrieveResponse, Tier


def format_retrieval(
    code: str, retrieval: RetrievalResult, alarm: Mapping[str, Any]
) -> RetrieveResponse:
    """Expose exact supplied passages and provenance without generating text."""
    evidence = normalise_chunks([to_stored(c) for c in retrieval.evidence])
    return RetrieveResponse(
        code=code,
        alarm_text=alarm["alarm_text"],
        evidence=[
            RetrievedEvidence(
                id=chunk["llm_chunk_id"],
                **{
                    key: chunk[key]
                    for key in ("doc", "path", "section", "text", "chunk_ids")
                },
            )
            for chunk in evidence
        ],
        strategy=retrieval.strategy,
        stop_reason=retrieval.stop_reason,
    )


def format_resolution(
    result: Mapping[str, Any], alarm: Mapping[str, Any], tier: Tier
) -> ResolveResponse:
    """Map an engine result and catalogue metadata, preserving step order."""
    answer = result["answer"]
    return ResolveResponse(
        code=result["code"],
        summary=answer["summary"],
        steps=answer["resolution_steps"],
        citations=answer["doc_refs"],
        tier=tier,
        title=alarm["title"],
        severity=alarm["severity"],
        severity_category=alarm["severity_category"],
        domain=alarm["domain"],
        likely_causes=(answer.get("likely_causes", []) if tier != "operator" else None),
    )
