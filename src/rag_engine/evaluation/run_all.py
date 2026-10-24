import argparse
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

from rag_engine.engine.llm.evidence import normalise_chunks
from rag_engine.engine.llm.generation import (
    build_ollama_final_answer,
    generate_answer,
)
from rag_engine.engine.llm.llm_callers import call_ollama_llm
from rag_engine.engine.parse_input.input_processing import load_merged_catalog
from rag_engine.engine.rerank.llm_reranker.callers import make_ollama_reranker
from rag_engine.evaluation import judge_results
from rag_engine.evaluation.config import EvaluationConfig, load_config
from rag_engine.runtime import RagRuntime, RagService, RetrievalOptions


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and evaluate answers for an alarm catalogue. Run settings "
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
        "--codes",
        nargs="*",
        default=None,
        help="Alarm-code subset. Defaults to every catalogue code, or filters "
        "the saved results when --input-results is used.",
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
    """Retrieve, generate and judge one, several, or every alarm code."""
    generation = config.generation
    catalog = load_merged_catalog([Path(path) for path in generation.catalog])
    codes = arguments.codes or list(catalog)
    unknown_codes = sorted(set(codes) - set(catalog))
    if unknown_codes:
        raise KeyError(f"Unknown alarm codes: {', '.join(unknown_codes)}")

    llm_caller = None
    if generation.reranker == "llm":
        llm_caller = make_ollama_reranker(
            model=generation.model,
            host=generation.host,
            context_length=generation.context_length,
        )

    runtime = RagRuntime.load(
        catalog=catalog,
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
    if config.evaluation.model != generation.embedding_model:
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
            llm_pool_char_budget=generation.llm_pool_char_budget,
            llm_pool_chunk_chars=generation.llm_pool_chunk_chars,
            llm_supplement_limit=generation.llm_supplement_limit,
            llm_max_candidates=generation.llm_max_candidates,
        ),
    )

    for code in codes:
        # Capture the exact final-generator input, including any supplementary
        # hits, so the judge sees what the answer was actually based on.
        evidence: list[dict[str, Any]] = []

        def generate_and_capture(
            alarm_code: str,
            alarm_text: str,
            chunks: list[dict[str, Any]],
            user_question: str | None = None,
        ) -> dict[str, Any]:
            evidence[:] = normalise_chunks(chunks)
            return final_answer_fn(
                alarm_code, alarm_text, chunks, user_question=user_question
            )

        retrieval = service.retrieve({"code": code})
        generated = generate_answer(
            alarm_code=code,
            alarm_text=catalog[code]["alarm_text"],
            retrieval=retrieval,
            final_answer_fn=generate_and_capture,
            alarm_data=catalog[code],
        )
        result = {
            "code": code,
            "configuration": asdict(generation),
            "evidence": evidence,
            **generated,
        }
        result["evaluation_configuration"] = asdict(config.evaluation)
        result["evaluation"] = judge_results.judge(
            reference_document,
            result,
            evidence,
            config.evaluation,
            evaluation_embedder,
            judge_caller,
        )

        output_path = arguments.output_dir / (
            f"{code}-{model_slug}-{generation.reranker}-all.json"
        )
        output_path.write_text(
            f"{json.dumps(result, indent=2, ensure_ascii=False)}\n",
            encoding="utf-8",
        )
        print(f"Generated and evaluated {code}: {output_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
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
            codes=arguments.codes,
        )
    return _generate_and_judge(arguments, config, reference_document, judge_caller)


if __name__ == "__main__":
    raise SystemExit(main())
