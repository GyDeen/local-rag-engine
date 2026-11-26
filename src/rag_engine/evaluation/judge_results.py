from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_engine.evaluation.cases import reference_answers, result_id
from rag_engine.evaluation.config import EvaluationSettings, JudgeSettings
from rag_engine.evaluation.cosine import evaluate_result
from rag_engine.evaluation.llm_judge import evaluate_answer as judge_answer
from rag_engine.evaluation.llm_judge.callers import JudgeAPIError, make_judge_caller
from rag_engine.logging_config import log_operation

SOURCE_TAG = re.compile(r"\s*\[C[1-9]\d*\]")


def build_judge_caller(settings: JudgeSettings) -> Any:
    """Build the API caller for the LLM judge from its own settings."""
    return make_judge_caller(
        provider=settings.provider,
        model=settings.model,
        base_url=settings.base_url,
        api_key_env=settings.api_key_env,
        api=settings.api,
        reasoning_effort=settings.reasoning_effort,
        max_output_tokens=settings.max_output_tokens,
        timeout=settings.timeout,
    )


def strip_source_tags(answer: dict[str, Any]) -> dict[str, Any]:
    """Copy an answer with "[C#]" source tags removed from the text fields.

    Generation appends these tags to resolution_steps for traceability;
    scoring must see the same text an untagged answer would produce.
    """
    stripped = dict(answer)
    summary = stripped.get("summary")
    if isinstance(summary, str):
        stripped["summary"] = SOURCE_TAG.sub("", summary)
    for key in ("likely_causes", "resolution_steps"):
        items = stripped.get(key)
        if isinstance(items, list):
            stripped[key] = [
                SOURCE_TAG.sub("", item) if isinstance(item, str) else item
                for item in items
            ]
    return stripped


@log_operation("Answer evaluation")
def judge(
    reference_document: dict[str, Any],
    result: dict[str, Any],
    evidence: list[dict[str, Any]],
    settings: EvaluationSettings,
    evaluation_embedder: Any,
    judge_caller: Any,
) -> dict[str, Any]:
    """Attach independent LLM and cosine results to one generated answer."""
    evaluations: dict[str, Any] = {}
    answer = strip_source_tags(result["answer"])
    if "llm" in settings.judges:
        try:
            evaluations["llm"] = {
                "status": "completed",
                **judge_answer(
                    reference_answers(reference_document)[result_id(result)],
                    answer,
                    evidence,
                    judge_caller,
                ),
            }
        except (ValueError, OSError, KeyError, JudgeAPIError) as error:
            logging.getLogger(__name__).exception("LLM evaluation failed")
            evaluations["llm"] = {"status": "error", "error": str(error)}
    if "cosine" in settings.judges:
        evaluations["cosine"] = evaluate_result(
            reference_document,
            {**result, "answer": answer},
            evaluation_embedder,
            settings.k,
        )["evaluation"]
    return evaluations


def load_evidence(result: dict[str, Any], judges: list[str]) -> list[dict[str, Any]]:
    """Return the generation evidence saved next to an answer."""
    evidence = result.get("evidence")
    if evidence is not None:
        return evidence
    # Support artifacts from the earlier dual-judge implementation, which nested
    # the same chunks under the evaluation block instead of the top level.
    saved = (
        result.get("evaluation", {})
        .get("llm", {})
        .get("inputs", {})
        .get("retrieved_documentation")
    )
    if saved is not None:
        return [{**chunk, "llm_chunk_id": chunk["chunk_id"]} for chunk in saved]
    if "llm" in judges:
        identifier = result.get("id", result.get("code", "?"))
        raise KeyError(
            f"Result {identifier} has no 'evidence', so the document "
            "judge has nothing to check the answer against. Generate it with this "
            "command, which saves the evidence it passes to the model."
        )
    return []


def judge_saved_results(
    input_results: list[Path],
    output_dir: Path,
    settings: EvaluationSettings,
    reference_document: dict[str, Any],
    judge_caller: Any,
    case_ids: list[str] | None = None,
) -> int:
    """Judge saved answers with their original generation evidence."""
    evaluation_embedder = None
    if "cosine" in settings.judges:
        from sentence_transformers import SentenceTransformer

        evaluation_embedder = SentenceTransformer(settings.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    judge_slug = settings.llm.model.replace(":", "-").replace("/", "-")
    for input_path in input_results:
        with input_path.open(encoding="utf-8") as input_file:
            result = json.load(input_file)
        identifier = result_id(result)
        if case_ids and identifier not in case_ids:
            continue
        evidence = load_evidence(result, settings.judges)
        result["evidence"] = evidence
        result["evaluation_configuration"] = asdict(settings)
        result["evaluation"] = judge(
            reference_document,
            result,
            evidence,
            settings,
            evaluation_embedder,
            judge_caller,
        )
        output_path = output_dir / f"{input_path.stem}-{judge_slug}-evaluation.json"
        output_path.write_text(
            f"{json.dumps(result, indent=2, ensure_ascii=False)}\n", encoding="utf-8"
        )
        print(f"Evaluated {identifier}: {output_path}")
    return 0
