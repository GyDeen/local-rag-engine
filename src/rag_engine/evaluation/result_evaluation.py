"""Score and summarize document-question results."""

import statistics

from rag_engine.evaluation.cosine.result_evaluation import (
    evaluate_output_files,
    evaluate_result,
    reference_coverage,
)


def summarize_question_results(results: list[dict]) -> dict:
    """Aggregate populated scores and count cases by answerability."""
    summary = {"cases": len(results)}
    for metric in ("gold_source_recall", "gold_page_recall", "gold_section_recall"):
        values = [row[metric] for row in results if row[metric] is not None]
        summary[f"mean_{metric}"] = statistics.mean(values) if values else None
        summary[f"{metric}_cases"] = len(values)
    answerable = [row for row in results if row["expected_answerable"] is True]
    other = [row for row in results if row["expected_answerable"] is False]
    summary.update(
        answerable_cases=len(answerable),
        unanswerable_cases=len(other),
        unlabelled_cases=len(results) - len(answerable) - len(other),
        mean_evidence_characters=statistics.mean(
            sum(len(item["text"]) for item in row["application"]["evidence"])
            for row in results
        ),
        generation_errors=sum(row["generation_error"] is not None for row in results),
    )
    generated = any(
        row["application"].get("status") or row["generation_error"] for row in results
    )
    for name, rows, status in (
        ("answered_on_answerable", answerable, "answered"),
        ("abstained_on_unanswerable", other, "insufficient_evidence"),
    ):
        summary[name] = (
            sum(row["application"].get("status") == status for row in rows)
            if generated
            else None
        )
    if any("evaluation" in row for row in results):
        similarities = [
            evaluation["answer_cosine_similarity"]
            for row in results
            if (evaluation := row.get("evaluation", {}).get("cosine")) is not None
            and evaluation["answer_cosine_similarity"] is not None
        ]
        summary["mean_answer_cosine_similarity"] = (
            statistics.mean(similarities) if similarities else None
        )
        summary["answer_cosine_similarity_cases"] = len(similarities)
    return summary


__all__ = [
    "evaluate_output_files",
    "evaluate_result",
    "reference_coverage",
    "summarize_question_results",
]
