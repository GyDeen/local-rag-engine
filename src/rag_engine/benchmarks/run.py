"""Evaluate the ordinary document-Q&A application using held-out labels."""

import argparse
import json
import random
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from rag_engine.cli.errors import run_cli
from rag_engine.engine.llm.llm_callers import OLLAMA_HOST
from rag_engine.evaluation import DEFAULT_EVALUATION_K
from rag_engine.evaluation.cases import load_question_cases
from rag_engine.evaluation.result_evaluation import (
    evaluate_result,
    summarize_question_results,
)
from rag_engine.exceptions import GenerationError
from rag_engine.logging_config import configure_logging
from rag_engine.runtime import RagRuntime, RagService, RetrievalOptions
from rag_engine.serving.answers import answer_question
from rag_engine.settings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EVIDENCE_CHAR_BUDGET,
    DEFAULT_EVIDENCE_MIN_CHUNK_CHARS,
    DEFAULT_FILE_LIMIT,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def _cases(path: Path, limit: int | None, seed: int) -> list[dict]:
    cases = load_question_cases(path)
    if limit is None or limit >= len(cases):
        return cases
    rng = random.Random(seed)
    if any(case.get("answerable") is None for case in cases):
        return rng.sample(cases, limit)
    answerable = [case for case in cases if case["answerable"]]
    other = [case for case in cases if not case["answerable"]]
    if not answerable or not other or limit == 1:
        return rng.sample(cases, limit)
    count = round(limit * len(answerable) / len(cases))
    count = min(len(answerable), limit - 1, max(1, limit - len(other), count))
    sample = rng.sample(answerable, count) + rng.sample(other, limit - count)
    return sorted(sample, key=lambda case: case["id"])


def _run_case(
    case: dict,
    retrieve: Callable[[str], dict],
    generate: Callable[[str, list[dict]], dict] | None = None,
    evaluation_embedder: "SentenceTransformer | None" = None,
    *,
    k: int = DEFAULT_EVALUATION_K,
) -> dict:
    response = retrieve(case["question"])
    error = None
    if generate is not None:
        try:
            response.update(generate(case["question"], response["evidence"]))
        except (ValueError, GenerationError) as failure:
            error = str(failure)
    result = {
        "id": case["id"],
        "expected_answerable": case.get("answerable"),
        "reference_answer": case.get("answer"),
        "references": case.get("references", []),
        "required_steps": case.get("required_steps", []),
        "category": case.get("category"),
        "application": response,
        "generation_error": error,
    }
    result.update(
        evaluate_result(
            case, result, evaluation_embedder, k=k, result_format="benchmark_row"
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Processed benchmark corpus")
    parser.add_argument(
        "--cases", type=Path, help="Benchmark questions and labels in JSONL"
    )
    parser.add_argument("--docs-root", type=Path, help="Normalized application corpus")
    parser.add_argument("--database-path", type=Path)
    parser.add_argument("--collection", default="benchmark")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker", choices=("none", "cross_encoder"), default="none")
    parser.add_argument("--chunk-limit", type=int, default=50)
    parser.add_argument("--file-limit", type=int, default=DEFAULT_FILE_LIMIT)
    parser.add_argument("--evidence-rank-limit", type=int)
    parser.add_argument(
        "--evidence-min-chunk-chars",
        type=int,
        default=DEFAULT_EVIDENCE_MIN_CHUNK_CHARS,
        help="Minimum evidence length in characters. Use 0 to score short passages.",
    )
    parser.add_argument("--char-budget", type=int, default=DEFAULT_EVIDENCE_CHAR_BUDGET)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_EVALUATION_K,
        help="Scoring cutoff for ranked evidence and answer citations (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", help="Omit to measure retrieval only")
    parser.add_argument(
        "--cosine",
        action="store_true",
        help="Score generated answers against reference text, citations, and answerability",
    )
    parser.add_argument("--host", default=OLLAMA_HOST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configure_logging()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.k < 1:
        parser.error("--k must be positive")
    if args.cosine and not args.model:
        parser.error("--cosine requires --model")
    if args.output and args.output.exists():
        parser.error("Output exists; choose a new filename")
    root = args.dataset_root.resolve()
    docs_root = args.docs_root or next(
        (
            path
            for path in (root / "ingested" / "docs", root / "documents", root / "docs")
            if path.is_dir()
        ),
        root / "documents",
    )
    cases = _cases(args.cases or root / "cases.jsonl", args.limit, args.seed)
    if not cases:
        parser.error("No evaluation cases found")
    service = RagService(
        RagRuntime.load(
            database_path=args.database_path or root / "index",
            collection=args.collection,
            embedding_model=args.embedding_model,
            reranker=args.reranker,
            file_limit=args.file_limit,
        ),
        RetrievalOptions(
            docs_root=docs_root,
            chunk_limit=args.chunk_limit,
            evidence_char_budget=args.char_budget,
            evidence_min_chunk_chars=args.evidence_min_chunk_chars,
            evidence_rank_limit=args.evidence_rank_limit,
        ),
    )
    generate = (
        partial(answer_question, model=args.model, host=args.host)
        if args.model
        else None
    )
    evaluation_embedder = service.runtime.embedder if args.cosine else None
    results = [
        _run_case(case, service.ask, generate, evaluation_embedder, k=args.k)
        for case in cases
    ]
    report = {
        "dataset": root.name,
        "application_path": "RagService.ask",
        "summary": summarize_question_results(results),
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, ensure_ascii=False)
            output.write("\n")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
