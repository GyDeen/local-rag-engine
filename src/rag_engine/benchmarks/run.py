"""Evaluate the ordinary document-Q&A application using held-out labels."""

import argparse
import json
import random
import re
import statistics
from collections.abc import Callable
from pathlib import Path

from rag_engine.cli.errors import run_cli
from rag_engine.evaluation.cases import load_question_cases
from rag_engine.logging_config import configure_logging
from rag_engine.runtime import RagRuntime, RagService, RetrievalOptions
from rag_engine.settings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EVIDENCE_CHAR_BUDGET,
    DEFAULT_FILE_LIMIT,
)


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


def _section_name(heading: str) -> str:
    heading = re.sub(r"^\s*(?:\d+\.)+\s*", "", heading)
    return " ".join(re.findall(r"[^\W_]+", heading.casefold()))


def _coverage(case: dict, evidence: list[dict]) -> dict:
    references = case.get("references", [])
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
    retrieved = {passage["source"] for passage in evidence}
    retrieved_pages = {
        (passage["source"], page) for passage in evidence for page in passage["pages"]
    }
    retrieved_sections = {
        (passage["source"], _section_name(section.rsplit(" > ", 1)[-1]))
        for passage in evidence
        for section in passage.get("section", "").split(" | ")
    }
    return {
        "gold_source_recall": (
            len(sources & retrieved) / len(sources) if sources else None
        ),
        "gold_page_recall": (
            len(pages & retrieved_pages) / len(pages) if pages else None
        ),
        "gold_section_recall": (
            len(sections & retrieved_sections) / len(sections) if sections else None
        ),
    }


def _run_case(case: dict, retrieve: Callable[[str], dict]) -> dict:
    application = retrieve(case["question"])
    return {
        "id": case["id"],
        "expected_answerable": case.get("answerable"),
        "reference_answer": case.get("answer"),
        "references": case.get("references", []),
        "required_steps": case.get("required_steps", []),
        "category": case.get("category"),
        "application": application,
        **_coverage(case, application["evidence"]),
    }


def _summary(results: list[dict]) -> dict:
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
    )
    return summary


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
    parser.add_argument("--char-budget", type=int, default=DEFAULT_EVIDENCE_CHAR_BUDGET)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configure_logging()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if (
        args.file_limit < 1
        or args.chunk_limit < args.file_limit
        or args.char_budget < 1
    ):
        parser.error(
            "Positive budgets are required and --chunk-limit must be at least --file-limit"
        )
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
            evidence_rank_limit=args.evidence_rank_limit,
        ),
    )
    results = [_run_case(case, service.ask) for case in cases]
    report = {
        "dataset": root.name,
        "application_path": "RagService.ask",
        "summary": _summary(results),
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
