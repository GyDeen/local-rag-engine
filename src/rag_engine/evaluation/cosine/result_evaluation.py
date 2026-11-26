"""Evaluate references and answers for document questions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from rag_engine.evaluation.cases import reference_answers, result_id
from rag_engine.evaluation.cosine.metrics import (
    citation_key,
    citation_match_details,
    exact_match_score,
    match_items,
    ordered_item_matches,
    ranking_scores,
    recall_score,
    semantic_similarity,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

EVIDENCE_TAG = re.compile(r"\[(E\d+)\]")
ANSWERED_BY_STATUS = {"answered": True, "insufficient_evidence": False}


def _section_name(heading: str) -> str:
    heading = re.sub(r"^\s*(?:\d+\.)+\s*", "", heading)
    return " ".join(re.findall(r"[^\W_]+", heading.casefold()))


def reference_coverage(case: dict, evidence: list[dict]) -> dict[str, float | None]:
    """Measure source, page, and section recall at the labelled granularity."""
    references = case.get("references", [])
    if not references:
        return {
            "gold_source_recall": None,
            "gold_page_recall": None,
            "gold_section_recall": None,
        }

    sources = {reference["source"] for reference in references}
    pages = {
        (reference["source"], reference["page"])
        for reference in references
        if "page" in reference
    }
    sections = {
        (reference["source"], _section_name(reference["section"]))
        for reference in references
        if "section" in reference
    }

    passages = [
        (passage.get("source", passage.get("doc")), passage) for passage in evidence
    ]
    retrieved = {source for source, _ in passages}
    retrieved_pages = {
        (source, page)
        for source, passage in passages
        for page in passage.get("pages", [])
    }
    retrieved_sections = {
        (source, _section_name(section.rsplit(" > ", 1)[-1]))
        for source, passage in passages
        for section in passage.get("section", "").split(" | ")
    }
    return {
        "gold_source_recall": recall_score(sources, retrieved) if sources else None,
        "gold_page_recall": recall_score(pages, retrieved_pages) if pages else None,
        "gold_section_recall": (
            recall_score(sections, retrieved_sections) if sections else None
        ),
    }


def evaluate_question_answer(
    reference: dict,
    response: dict,
    embedder: SentenceTransformer,
) -> dict:
    """Score the shared question response against a labelled benchmark case."""
    answered = ANSWERED_BY_STATUS[response["status"]]
    answerable = reference.get("answerable")
    answerability_correct = (
        answered == answerable if isinstance(answerable, bool) else None
    )
    result = {
        "answer_cosine_similarity": None,
        "answerability_correct": answerability_correct,
        "citations": None,
    }
    if not answered or answerability_correct is False:
        return result

    answer = response["answer"]
    cited_ids = list(dict.fromkeys(EVIDENCE_TAG.findall(answer)))
    evidence = {passage["id"]: passage for passage in response.get("evidence", [])}
    cited_passages = [evidence[citation] for citation in cited_ids if citation in evidence]
    result["citations"] = {
        "ids": cited_ids,
        "unresolved_ids": [citation for citation in cited_ids if citation not in evidence],
        **reference_coverage(reference, cited_passages),
    }
    expected = reference.get("answer")
    if isinstance(expected, str) and expected.strip():
        result["answer_cosine_similarity"] = semantic_similarity(
            expected.strip(), EVIDENCE_TAG.sub("", answer).strip(), embedder
        )
    return result


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

    identifier = result_id(generated_document)
    reference_answer = reference_answers(reference_document)[identifier]
    if "status" in generated_document:
        evaluation = evaluate_question_answer(
            reference_answer, generated_document, embedder
        )
    else:
        generated_answer = generated_document.get("answer", generated_document)
        evaluation = evaluate_answer(reference_answer, generated_answer, embedder, k)

    result = {
        "id" if "id" in generated_document else "code": identifier,
        "k": k,
        "evaluation": evaluation,
    }
    if "configuration" in generated_document:
        result["configuration"] = generated_document["configuration"]
    return result
