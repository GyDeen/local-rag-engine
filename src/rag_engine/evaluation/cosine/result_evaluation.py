"""Evaluate references and answers for document questions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rag_engine.evaluation import DEFAULT_EVALUATION_K
from rag_engine.evaluation.cases import reference_answers
from rag_engine.evaluation.cosine.metrics import recall_score, semantic_similarity

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

EVIDENCE_TAG = re.compile(r"\[(E\d+)\]")
ANSWERED_BY_STATUS = {"answered": True, "insufficient_evidence": False}
ResultFormat = Literal["benchmark_row", "question_response"]


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
    *,
    k: int = DEFAULT_EVALUATION_K,
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
    cited_passages = [
        evidence[citation] for citation in cited_ids[:k] if citation in evidence
    ]
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


def evaluate_output_files(
    reference_path: Path,
    generated_path: Path,
    embedder: SentenceTransformer | None,
    k: int = DEFAULT_EVALUATION_K,
    *,
    result_format: ResultFormat = "question_response",
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
        k=k,
        result_format=result_format,
    )


def evaluate_result(
    reference_document: dict,
    generated_document: dict,
    embedder: SentenceTransformer | None,
    k: int = DEFAULT_EVALUATION_K,
    *,
    result_format: ResultFormat,
) -> dict:
    """Score a benchmark row or a saved answer.

    Benchmark rows contain an application response and use one reference case.
    Saved answers look up their reference by case ID.
    """
    if result_format == "benchmark_row":
        response = generated_document["application"]
        scores: dict = {
            "k": k,
            **reference_coverage(reference_document, response["evidence"][:k]),
        }
        if embedder is not None:
            scores["evaluation"] = {
                "cosine": (
                    evaluate_question_answer(reference_document, response, embedder, k=k)
                    if generated_document["generation_error"] is None
                    else None
                ),
            }
        return scores

    if embedder is None:
        raise TypeError("Evaluating a saved answer requires an embedder")
    identifier = generated_document["id"]
    reference_answer = reference_answers(reference_document)[identifier]

    result = {
        "id": identifier,
        "k": k,
        "evaluation": evaluate_question_answer(
            reference_answer, generated_document, embedder, k=k
        ),
    }
    if "configuration" in generated_document:
        result["configuration"] = generated_document["configuration"]
    return result
