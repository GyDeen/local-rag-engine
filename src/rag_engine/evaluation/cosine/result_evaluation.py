"""Secondary cosine evaluation, preserving the existing scores and schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from rag_engine.evaluation.cosine.metrics import (
    citation_key,
    citation_match_details,
    exact_match_score,
    match_items,
    ordered_item_matches,
    ranking_scores,
    semantic_similarity,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def evaluate_answer(
    reference: dict,
    generated: dict,
    embedder: SentenceTransformer,
    k: int,
) -> dict:
    """Evaluate one answer with at-k metrics and retain match diagnostics."""

    reference_citations = reference["doc_refs"]
    generated_citations = generated.get("doc_refs", generated.get("citations"))
    if generated_citations is None:
        raise KeyError("Generated answer requires 'doc_refs' or 'citations'")
    citation_match_result = citation_match_details(
        reference_citations,
        generated_citations,
    )

    summary_reference = reference["summary"]
    summary_generated = generated["summary"]
    likely_causes_reference = reference["likely_causes"]
    likely_causes_generated = generated["likely_causes"]
    resolution_steps_reference = reference["resolution_steps"]
    resolution_steps_generated = generated.get(
        "resolution_steps", generated.get("steps")
    )
    if resolution_steps_generated is None:
        raise KeyError("Generated answer requires 'resolution_steps' or 'steps'")

    likely_causes_result = match_items(
        likely_causes_reference,
        likely_causes_generated,
        embedder,
    )
    resolution_steps_result = ordered_item_matches(
        resolution_steps_reference,
        resolution_steps_generated,
        embedder,
    )

    reference_citation_keys = list(
        dict.fromkeys(citation_key(citation) for citation in reference_citations)
    )
    generated_citation_keys = list(
        dict.fromkeys(citation_key(citation) for citation in generated_citations)
    )

    scores = {
        "summary": ranking_scores(
            [summary_reference],
            [summary_generated],
            1,
            embedder=embedder,
        ),
        "likely_causes": ranking_scores(
            likely_causes_reference,
            likely_causes_generated,
            k,
            embedder=embedder,
        ),
        "resolution_steps": ranking_scores(
            resolution_steps_reference,
            resolution_steps_generated,
            k,
            embedder=embedder,
            preserve_order=True,
        ),
        "citations": ranking_scores(
            reference_citation_keys,
            generated_citation_keys,
            k,
            similarity=exact_match_score,
        ),
    }
    return {
        "cutoffs": {
            "summary": 1,
            "likely_causes": k,
            "resolution_steps": k,
            "citations": k,
        },
        "scores": scores,
        "match_details": {
            "summary": {
                "reference": summary_reference,
                "generated": summary_generated,
                "score": semantic_similarity(
                    summary_reference,
                    summary_generated,
                    embedder,
                ),
            },
            "likely_causes": likely_causes_result,
            "resolution_steps": resolution_steps_result,
            "citations": citation_match_result,
        },
    }


def evaluate_output_files(
    reference_path: Path,
    generated_path: Path,
    embedder: SentenceTransformer,
    k: int,
) -> dict:
    """Evaluate generated output JSON against golden-standard JSON."""

    with reference_path.open(encoding="utf-8") as reference_file:
        reference_document = json.load(reference_file)

    with generated_path.open(encoding="utf-8") as generated_file:
        generated_document = json.load(generated_file)

    return evaluate_result(
        reference_document,
        generated_document,
        embedder,
        k,
    )


def evaluate_result(
    reference_document: dict,
    generated_document: dict,
    embedder: SentenceTransformer,
    k: int,
) -> dict:
    """Evaluate one generated result against its golden answer."""

    code = generated_document["code"]
    reference_answer = reference_document["answers"][code]
    generated_answer = generated_document.get("answer", generated_document)

    result = {
        "code": code,
        "k": k,
        "evaluation": evaluate_answer(
            reference_answer,
            generated_answer,
            embedder,
            k,
        ),
    }
    if "configuration" in generated_document:
        result["configuration"] = generated_document["configuration"]
    return result
