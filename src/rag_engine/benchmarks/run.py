"""Measure retrieval over normalized documents using held-out labels."""

import argparse
import json
import random
import re
import statistics
from collections.abc import Callable
from functools import partial
from pathlib import Path

from rag_engine.cli.errors import run_cli
from rag_engine.engine.rerank.candidate import build_rerank_candidates
from rag_engine.engine.rerank.selection import stitch_evidence
from rag_engine.engine.retrieve.pipeline import RetrievalResult
from rag_engine.engine.vector_store.database_metadata_model import SearchFilters
from rag_engine.logging_config import configure_logging
from rag_engine.runtime import RagRuntime
from rag_engine.settings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EVIDENCE_CHAR_BUDGET,
    DEFAULT_EVIDENCE_MIN_CHUNK_CHARS,
    DEFAULT_FILE_LIMIT,
)


def _cases(path: Path, limit: int | None, seed: int) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        cases = [json.loads(line) for line in source]
    if limit is None or limit >= len(cases):
        return cases
    rng = random.Random(seed)
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
    sources = {reference["source"] for reference in case["references"]}
    pages = {
        (reference["source"], reference["page"])
        for reference in case["references"]
        if "page" in reference
    }
    sections = {
        (reference["source"], _section_name(reference["section"]))
        for reference in case["references"]
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


def _retrieve(
    question: str,
    runtime: RagRuntime,
    docs_root: Path,
    args: argparse.Namespace,
) -> dict:
    vector = runtime.embedder.encode(question, normalize_embeddings=True)
    chunks = runtime.database.similarity_search(
        query_embedding=vector.tolist(), filters=SearchFilters(), limit=args.chunk_limit
    )
    pool = build_rerank_candidates(chunks, docs_root)
    strategy = runtime.rerank_strategy
    ranked = pool if strategy is None else strategy.rerank(question, pool)
    selected = stitch_evidence(
        ranked,
        pool,
        runtime.file_limit,
        args.char_budget,
        min_chunk_chars=DEFAULT_EVIDENCE_MIN_CHUNK_CHARS,
        evidence_rank_limit=args.evidence_rank_limit,
    )
    evidence = []
    for item in selected:
        chunk = item.chunk
        evidence.append(
            {
                "source": chunk.metadata.get("source_document", chunk.document_id),
                "location": chunk.metadata.get(
                    "source_location", "/".join(chunk.source_path)
                ),
                "pages": chunk.metadata.get("source_pages", []),
                "section": chunk.section,
                "text": item.text,
                "chunk_ids": list(item.all_chunk_ids()),
            }
        )
    return {
        "question": question,
        "retrieval": RetrievalResult(
            selected, args.reranker, "strategy_selected"
        ).as_dict(),
        "evidence": evidence,
    }


def _run_case(case: dict, retrieve: Callable[[str], dict]) -> dict:
    application = retrieve(case["question"])
    return {
        "id": case["id"],
        "expected_answerable": case["answerable"],
        "reference_answer": case["answer"],
        "references": case["references"],
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
    answerable = [row for row in results if row["expected_answerable"]]
    other = [row for row in results if not row["expected_answerable"]]
    summary.update(
        answerable_cases=len(answerable),
        unanswerable_cases=len(other),
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
    runtime = RagRuntime.load(
        catalog={},
        database_path=args.database_path or root / "index",
        collection=args.collection,
        embedding_model=args.embedding_model,
        reranker=args.reranker,
        file_limit=args.file_limit,
    )
    retrieve = partial(_retrieve, runtime=runtime, docs_root=docs_root, args=args)
    results = [_run_case(case, retrieve) for case in cases]
    report = {
        "dataset": root.name,
        "application_path": "vector retrieval and evidence stitching",
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
