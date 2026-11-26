from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_engine.evaluation.cases import reference_answers
from rag_engine.evaluation.config import EvaluationSettings, JudgeSettings
from rag_engine.evaluation.cosine import evaluate_question_answer
from rag_engine.evaluation.llm_judge import evaluate_answer as judge_answer
from rag_engine.evaluation.llm_judge.callers import JudgeAPIError, make_judge_caller
from rag_engine.logging_config import log_operation


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


@log_operation("Answer evaluation")
def judge(
    reference: dict[str, Any],
    result: dict[str, Any],
    settings: EvaluationSettings,
    evaluation_embedder: Any,
    judge_caller: Any,
) -> dict[str, Any]:
    """Attach independent LLM and cosine results to one generated answer."""
    evaluations: dict[str, Any] = {}
    if "llm" in settings.judges:
        try:
            evaluations["llm"] = judge_answer(reference, result, judge_caller)
        except (ValueError, OSError, KeyError, JudgeAPIError) as error:
            logging.getLogger(__name__).exception("LLM evaluation failed")
            evaluations["llm"] = {"status": "error", "error": str(error)}
    if "cosine" in settings.judges:
        evaluations["cosine"] = evaluate_question_answer(
            reference, result, evaluation_embedder, k=settings.k
        )
    return evaluations


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
    references = reference_answers(reference_document)
    for input_path in input_results:
        with input_path.open(encoding="utf-8") as input_file:
            result = json.load(input_file)
        identifier = result["id"]
        if case_ids and identifier not in case_ids:
            continue
        result["evaluation_configuration"] = asdict(settings)
        result["evaluation"] = judge(
            references[identifier],
            result,
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
