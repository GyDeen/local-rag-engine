"""Run structured answer generation for one development question.

Builds the index once with ``python -m rag_engine.cli.index``, then ask:

    PYTHONPATH=src python -m rag_engine.cli.answer "How do I restart the worker?"

Requires a local Ollama server with the default model pulled:

    ollama pull qwen3:4b
"""

import argparse
import hashlib
import json
import logging
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

from rag_engine.cli.errors import run_cli
from rag_engine.engine.llm.generation import build_ollama_final_answer, generate_answer
from rag_engine.engine.llm.llm_callers import OLLAMA_HOST, OLLAMA_MODEL, call_ollama_llm
from rag_engine.engine.rerank.llm_reranker.callers import make_ollama_reranker
from rag_engine.engine.rerank.llm_reranker.pipeline import MAX_SEARCH_ROUNDS
from rag_engine.engine.retrieve.pipeline import retrieve_evidence
from rag_engine.engine.vector_store.backends import open_database
from rag_engine.logging_config import configure_logging
from rag_engine.settings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_DATABASE_BACKEND,
    DEFAULT_DATABASE_PATH,
    DEFAULT_DOCS_ROOT,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FILE_LIMIT,
    PROJECT_ROOT,
    RERANKERS,
)

DEFAULT_ANSWER_CACHE_DIR = PROJECT_ROOT / "data/answer_cache"
CACHE_FORMAT_VERSION = 2


def _file_versions(root: Path, pattern: str) -> list[tuple[str, int, int]]:
    """Collect file paths, modification times and sizes for cache invalidation."""

    if not root.is_dir():
        raise FileNotFoundError(f"Cache version source does not exist: {root}")
    versions = []
    for path in sorted(root.rglob(pattern)):
        if path.is_file():
            stat = path.stat()
            versions.append(
                (path.relative_to(root).as_posix(), stat.st_mtime_ns, stat.st_size)
            )
    return versions


def _answer_cache_key(arguments: argparse.Namespace) -> str:
    """Include every CLI input and the corpus/code versions that can change an answer."""

    payload = {
        "format": CACHE_FORMAT_VERSION,
        "question": arguments.question,
        "docs": _file_versions(DEFAULT_DOCS_ROOT, "*.md"),
        "code": _file_versions(Path(__file__).resolve().parents[1], "*.py"),
        "settings": {
            "database_backend": arguments.database_backend,
            "database_path": str(arguments.database_path.resolve()),
            "collection": arguments.collection,
            "embedding_model": arguments.embedding_model,
            "reranker": arguments.reranker,
            "cross_encoder_model": arguments.cross_encoder_model,
            "model": arguments.model,
            "host": arguments.host,
            "context_length": arguments.context_length,
            "file_limit": arguments.limit,
            "chunk_limit": arguments.chunk_limit,
            "evidence_rank_limit": arguments.evidence_rank_limit,
            "max_rounds": arguments.max_rounds,
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cached_answer(key: str) -> dict[str, Any] | None:
    try:
        record = json.loads(
            (DEFAULT_ANSWER_CACHE_DIR / f"{key}.json").read_text(encoding="utf-8")
        )
    except OSError, ValueError:
        return None
    if not isinstance(record, dict) or record.get("key") != key:
        return None
    result = record.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("answer"), dict):
        return None
    answer = result["answer"]
    summary = answer.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return None
    for field in ("likely_causes", "resolution_steps"):
        items = answer.get(field)
        if not isinstance(items, list) or not items:
            return None
        if any(not isinstance(item, str) or not item.strip() for item in items):
            return None
    if not isinstance(result.get("retrieval"), dict):
        return None
    return result


def _save_cached_answer(key: str, result: dict[str, Any]) -> None:
    """Publish a completed answer atomically for future CLI processes."""

    DEFAULT_ANSWER_CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=DEFAULT_ANSWER_CACHE_DIR,
            prefix=f".{key}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump({"key": key, "result": result}, temporary, ensure_ascii=False)
        temporary_path.replace(DEFAULT_ANSWER_CACHE_DIR / f"{key}.json")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a structured answer for one question.",
    )
    parser.add_argument("question", help="Question to submit to document retrieval.")
    parser.add_argument(
        "--database-backend",
        default=DEFAULT_DATABASE_BACKEND,
        help="Registered vector database backend.",
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="Path used by the selected vector database backend.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help="Vector collection or index name.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence Transformers model used when the collection was indexed.",
    )
    parser.add_argument(
        "--reranker",
        choices=RERANKERS,
        default="cross_encoder",
        help=(
            "Evidence ranking strategy (default: cross_encoder, a local "
            "single-pass ranker). 'llm' selects the slower sufficiency-gated "
            "exploration loop; 'none' disables reranking and keeps vector order."
        ),
    )
    parser.add_argument(
        "--cross-encoder-model",
        default=DEFAULT_CROSS_ENCODER_MODEL,
        help="Cross-encoder used by --reranker cross_encoder.",
    )
    parser.add_argument(
        "--model",
        default=OLLAMA_MODEL,
        help="Ollama model used for answering and optional LLM reranking.",
    )
    parser.add_argument("--host", default=OLLAMA_HOST, help="Ollama server address.")
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument(
        "--file-limit",
        "--limit",
        dest="limit",
        type=int,
        default=DEFAULT_FILE_LIMIT,
        help="Maximum selected source files; multiple chunks per file are allowed.",
    )
    parser.add_argument("--chunk-limit", type=int, default=50)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=MAX_SEARCH_ROUNDS,
        help="Maximum supplementary retrieval rounds; only used by --reranker llm.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Skip the stored answer and regenerate. The fresh answer "
            "overwrites the same cache entry so later repeats reuse it."
        ),
    )
    parser.add_argument(
        "--evidence-rank-limit",
        type=int,
        default=None,
        help="Only stitch the first N ranked chunks (default: no rank cutoff).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Retrieve evidence for one question and print the structured answer."""

    configure_logging()
    arguments = _parse_arguments(argv)
    cache_key = _answer_cache_key(arguments)
    if arguments.fresh:
        print(
            "Answer cache refresh requested; regenerating answer.",
            file=sys.stderr,
        )
    else:
        cached_answer = _load_cached_answer(cache_key)
        if cached_answer is not None:
            print(
                "Answer cache hit; reusing stored answer.",
                file=sys.stderr,
            )
            print(json.dumps(cached_answer, indent=2, ensure_ascii=False))
            return 0
        print("Answer cache miss; generating answer.", file=sys.stderr)

    database = open_database(
        backend=arguments.database_backend,
        database_path=arguments.database_path,
        collection_name=arguments.collection,
    )

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(arguments.embedding_model)

    ollama_caller = partial(
        call_ollama_llm,
        model=arguments.model,
        host=arguments.host,
        context_length=arguments.context_length,
    )
    final_answer_fn = build_ollama_final_answer(ollama_caller)

    llm_caller = None
    if arguments.reranker == "llm":
        llm_caller = make_ollama_reranker(
            model=arguments.model,
            host=arguments.host,
            context_length=arguments.context_length,
        )

    retrieval = retrieve_evidence(
        query_text=arguments.question,
        embedder=embedder,
        database=database,
        reranker=arguments.reranker,
        chunk_limit=arguments.chunk_limit,
        evidence_rank_limit=arguments.evidence_rank_limit,
        file_limit=arguments.limit,
        cross_encoder_model=arguments.cross_encoder_model,
        llm_caller=llm_caller,
        max_rounds=arguments.max_rounds,
    )
    result = generate_answer(
        question=arguments.question,
        retrieval=retrieval,
        final_answer_fn=final_answer_fn,
    )

    response = {"question": arguments.question, **result}
    try:
        _save_cached_answer(cache_key, response)
    except OSError as error:
        logging.getLogger("rag_engine.cli.answer").warning(
            "Could not write answer cache: %s", error, exc_info=True
        )
    print(json.dumps(response, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
