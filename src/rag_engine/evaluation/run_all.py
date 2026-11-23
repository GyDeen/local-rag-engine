import argparse
import json
import re
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

from rag_engine.cli.errors import run_cli
from rag_engine.engine.llm.evidence import normalise_chunks
from rag_engine.engine.llm.generation import (
    build_ollama_final_answer,
    generate_answer,
)
from rag_engine.engine.llm.llm_callers import call_ollama_llm
from rag_engine.engine.rerank.llm_reranker.callers import make_ollama_reranker
from rag_engine.evaluation import judge_results
from rag_engine.evaluation.cases import load_question_cases, reference_answers
from rag_engine.evaluation.config import EvaluationConfig, load_config
from rag_engine.logging_config import configure_logging
from rag_engine.runtime import RagRuntime, RagService, RetrievalOptions


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and evaluate answers for a batch of questions. Run settings "
            "come from --config; only what to run is set here."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Run settings JSON. Defaults to configs/evaluation.json, or the "
        "built-in defaults when that file is absent.",
    )
    parser.add_argument(
        "--input-results",
        type=Path,
        nargs="+",
        help="Judge saved results instead of generating. Each file needs an "
        "'answer' and the 'evidence' that generated it.",
    )
    parser.add_argument(
        "--cases", type=Path, help="JSONL questions. Defaults to generation.cases."
    )
    parser.add_argument(
        "--case-ids", nargs="+", help="Optional subset by evaluation case ID."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Where results are written (default: output).",
    )
    return parser.parse_args(argv)


def _generate_and_judge(
    arguments: argparse.Namespace,
    config: EvaluationConfig,
    reference_document: dict[str, Any],
    judge_caller: Any,
) -> int:
    """Retrieve and generate for each question, judging those with references."""
    generation = config.generation
    cases = load_question_cases(arguments.cases or Path(generation.cases))
    if not cases:
        raise ValueError("No evaluation questions found")
    if arguments.case_ids:
        wanted = set(arguments.case_ids)
        unknown = wanted - {case["id"] for case in cases}
        if unknown:
            raise ValueError(
                f"Unknown evaluation case IDs: {', '.join(sorted(unknown))}"
            )
        cases = [case for case in cases if case["id"] in wanted]
    references = reference_answers(reference_document)

    llm_caller = None
    if generation.reranker == "llm":
        llm_caller = make_ollama_reranker(
            model=generation.model,
            host=generation.host,
            context_length=generation.context_length,
        )

    runtime = RagRuntime.load(
        embedding_model=generation.embedding_model,
        reranker=generation.reranker,
        cross_encoder_model=generation.cross_encoder_model,
        llm_caller=llm_caller,
        file_limit=generation.file_limit,
        database_backend=generation.database_backend,
        database_path=Path(generation.database_path),
        collection=generation.collection,
    )

    evaluation_embedder = runtime.embedder
    if (
        "cosine" in config.evaluation.judges
        and config.evaluation.model != generation.embedding_model
        and any(case["id"] in references for case in cases)
    ):
        from sentence_transformers import SentenceTransformer

        evaluation_embedder = SentenceTransformer(config.evaluation.model)

    ollama_caller = partial(
        call_ollama_llm,
        model=generation.model,
        host=generation.host,
        context_length=generation.context_length,
    )
    final_answer_fn = build_ollama_final_answer(ollama_caller)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = generation.model.replace(":", "-").replace("/", "-")

    service = RagService(
        runtime,
        options=RetrievalOptions(
            chunk_limit=generation.chunk_limit,
            llm_chunk_limit=generation.llm_chunk_limit,
            max_rounds=generation.max_rounds,
            evidence_char_budget=generation.evidence_char_budget,
            evidence_min_chunk_chars=generation.evidence_min_chunk_chars,
            evidence_rank_limit=generation.evidence_rank_limit,
            llm_pool_char_budget=generation.llm_pool_char_budget,
            llm_pool_chunk_chars=generation.llm_pool_chunk_chars,
            llm_supplement_limit=generation.llm_supplement_limit,
            llm_max_candidates=generation.llm_max_candidates,
        ),
    )

    for position, case in enumerate(cases, start=1):
        identifier, question = case["id"], case["question"]
        # Capture the exact final-generator input, including any supplementary
        # hits, so the judge sees what the answer was actually based on.
        evidence: list[dict[str, Any]] = []

        def generate_and_capture(
            question: str,
            chunks: list[dict[str, Any]],
        ) -> dict[str, Any]:
            evidence[:] = normalise_chunks(chunks)
            return final_answer_fn(question, chunks)

        retrieval = service.retrieve_question(question)
        generated = generate_answer(
            question=question,
            retrieval=retrieval,
            final_answer_fn=generate_and_capture,
        )
        result = {
            "id": identifier,
            "question": question,
            "configuration": asdict(generation),
            "evidence": evidence,
            **generated,
        }
        if identifier in references:
            result["evaluation_configuration"] = asdict(config.evaluation)
            result["evaluation"] = judge_results.judge(
                reference_document,
                result,
                evidence,
                config.evaluation,
                evaluation_embedder,
                judge_caller,
            )

        case_slug = re.sub(r"[^a-zA-Z0-9_.-]", "-", identifier)
        output_path = arguments.output_dir / (
            f"{position:04d}-{case_slug}-{model_slug}-{generation.reranker}-all.json"
        )
        output_path.write_text(
            f"{json.dumps(result, indent=2, ensure_ascii=False)}\n",
            encoding="utf-8",
        )
        print(f"Wrote result for {identifier}: {output_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    arguments = _parse_arguments(argv)
    config = load_config(arguments.config)

    judge_caller = None
    if "llm" in config.evaluation.judges:
        judge_caller = judge_results.build_judge_caller(config.evaluation.llm)

    with Path(config.evaluation.reference_answers).open(encoding="utf-8") as handle:
        reference_document = json.load(handle)
    if arguments.input_results:
        return judge_results.judge_saved_results(
            input_results=arguments.input_results,
            output_dir=arguments.output_dir,
            settings=config.evaluation,
            reference_document=reference_document,
            judge_caller=judge_caller,
            case_ids=arguments.case_ids,
        )
    return _generate_and_judge(arguments, config, reference_document, judge_caller)


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
