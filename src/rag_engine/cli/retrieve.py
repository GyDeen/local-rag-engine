import argparse
import json
from pathlib import Path

from rag_engine.cli.errors import run_cli
from rag_engine.engine.llm.llm_callers import OLLAMA_HOST, OLLAMA_MODEL
from rag_engine.engine.rerank.llm_reranker.callers import make_ollama_reranker
from rag_engine.engine.rerank.llm_reranker.pipeline import MAX_SEARCH_ROUNDS
from rag_engine.engine.retrieve.pipeline import retrieve_evidence
from rag_engine.engine.vector_store.backends import open_database
from rag_engine.engine.vector_store.database_metadata_model import Chunk
from rag_engine.logging_config import configure_logging
from rag_engine.settings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_DATABASE_BACKEND,
    DEFAULT_DATABASE_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FILE_LIMIT,
    RERANKERS,
)


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve and rerank documentation chunks for a question.",
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
            "Reranking is part of retrieval: it selects the returned evidence "
            "(default: cross_encoder, a local single-pass ranker). 'none' "
            "disables reranking and keeps vector order."
        ),
    )
    parser.add_argument(
        "--cross-encoder-model",
        default=DEFAULT_CROSS_ENCODER_MODEL,
        help="Cross-encoder used by --reranker cross_encoder.",
    )
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--host", default=OLLAMA_HOST)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=MAX_SEARCH_ROUNDS,
        help="Maximum supplementary retrieval rounds; only used by --reranker llm.",
    )
    parser.add_argument(
        "--file-limit",
        "--section-limit",
        "--limit",
        dest="file_limit",
        type=int,
        default=DEFAULT_FILE_LIMIT,
        help=(
            "Maximum number of unique source files to return "
            f"(default: {DEFAULT_FILE_LIMIT})."
        ),
    )
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=50,
        help="Number of highest-matching chunks to consider (default: 50).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/retrieval_results.json"),
        help="JSON output path used for retrieval evaluation.",
    )
    parser.add_argument(
        "--evidence-rank-limit",
        type=int,
        default=None,
        help="Only stitch the first N ranked chunks (default: no rank cutoff).",
    )
    return parser.parse_args(argv)


def _chunk_to_evaluation_result(chunk: Chunk) -> dict[str, object]:
    """Serialize one retrieved chunk without its vector or indexed text."""

    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_path": chunk.source_path,
        "section": chunk.section,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "metadata": chunk.metadata,
    }


def main(argv: list[str] | None = None) -> int:
    """Assemble dependencies and run one retrieval request."""

    configure_logging()
    arguments = _parse_arguments(argv)
    database = open_database(
        backend=arguments.database_backend,
        database_path=arguments.database_path,
        collection_name=arguments.collection,
    )
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(arguments.embedding_model)

    llm_caller = None
    if arguments.reranker == "llm":
        llm_caller = make_ollama_reranker(model=arguments.model, host=arguments.host)

    retrieval = retrieve_evidence(
        query_text=arguments.question,
        embedder=embedder,
        database=database,
        reranker=arguments.reranker,
        chunk_limit=arguments.chunk_limit,
        evidence_rank_limit=arguments.evidence_rank_limit,
        file_limit=arguments.file_limit,
        cross_encoder_model=arguments.cross_encoder_model,
        llm_caller=llm_caller,
        max_rounds=arguments.max_rounds,
    )
    output = {
        "question": arguments.question,
        "results": [
            _chunk_to_evaluation_result(candidate.chunk)
            for candidate in retrieval.evidence
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        f"{json.dumps(output, indent=2)}\n",
        encoding="utf-8",
    )
    print(f"Retrieval results written to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
