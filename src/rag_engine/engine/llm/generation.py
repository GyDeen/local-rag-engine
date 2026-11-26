"""Grounded answer generation from already-retrieved documentation evidence."""

from collections.abc import Callable
from typing import Any

from rag_engine.engine.llm.answer_validation import (
    extract_json_object,
    validate_llm_answer,
)
from rag_engine.engine.llm.evidence import UNKNOWN, build_doc_refs, normalise_chunks
from rag_engine.engine.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from rag_engine.engine.rerank.candidate import to_stored
from rag_engine.engine.rerank.llm_reranker.strategy import LLMCaller
from rag_engine.engine.retrieve.pipeline import RetrievalResult
from rag_engine.exceptions import GenerationError
from rag_engine.logging_config import log_operation


def build_unknown_answer(reason: str = "") -> dict[str, Any]:
    """Return a refusal answer in the reference-answer structure."""

    notes = reason.strip() or "Supplied chunks were insufficient; refusing to guess."
    return {
        "summary": UNKNOWN,
        "likely_causes": [UNKNOWN],
        "resolution_steps": [UNKNOWN],
        "doc_refs": [],
        "doc_coverage": "partial",
        "notes": notes,
    }


def build_ollama_final_answer(
    ollama_caller: LLMCaller,
) -> Callable[..., dict[str, Any]]:
    """Return a ``final_answer_fn`` that generates the answer with Ollama."""

    def final_answer_fn(
        question: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        evidence = normalise_chunks(chunks)
        if not evidence:
            return build_unknown_answer("No document chunks were provided.")

        raw_response = ollama_caller(
            SYSTEM_PROMPT,
            build_user_prompt(question, evidence),
        )
        return validate_llm_answer(extract_json_object(raw_response))

    return final_answer_fn


@log_operation("Answer generation", error_type=GenerationError)
def generate_answer(
    *,
    question: str,
    retrieval: RetrievalResult,
    final_answer_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Write one grounded answer from an already-selected evidence set.

    Generation neither ranks nor retrieves. It writes only the answer text;
    source references, coverage, and notes come from selected evidence and the
    retrieval verdict.
    """

    selected = retrieval.evidence
    chunks = [to_stored(candidate) for candidate in selected]

    if chunks:
        generated = final_answer_fn(question, chunks)
        answer = {
            "summary": generated.get("summary", UNKNOWN),
            "likely_causes": generated.get("likely_causes", [UNKNOWN]),
            "resolution_steps": generated.get("resolution_steps", [UNKNOWN]),
        }
        evidence = normalise_chunks(chunks)
        doc_refs = build_doc_refs(
            [chunk["llm_chunk_id"] for chunk in evidence], evidence
        )
    else:
        answer = {
            "summary": UNKNOWN,
            "likely_causes": [UNKNOWN],
            "resolution_steps": [UNKNOWN],
        }
        doc_refs = []

    assessment = retrieval.assessment
    sufficient = assessment is not None and assessment.sufficient
    has_unknown = (
        answer["summary"] == UNKNOWN
        or UNKNOWN in answer["likely_causes"]
        or UNKNOWN in answer["resolution_steps"]
    )
    answer["doc_refs"] = doc_refs
    answer["doc_coverage"] = (
        "full" if doc_refs and sufficient and not has_unknown else "partial"
    )
    answer["notes"] = (
        "Generated only from the provided chunks."
        if chunks
        else "No supported evidence selected."
    )
    if assessment is not None and not assessment.sufficient:
        gaps = assessment.missing + assessment.operator_questions
        answer["notes"] = (
            answer["notes"]
            + " Unresolved: "
            + "; ".join(gaps or [retrieval.stop_reason])
        ).strip()
    return {
        "answer": answer,
        "retrieval": retrieval.as_dict(),
    }
