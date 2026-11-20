"""Command-line entry point for one-time documentation indexing."""

import argparse
import json
from pathlib import Path

from rag_engine.cli.errors import run_cli
from rag_engine.engine.vector_store.create_database import create_database
from rag_engine.engine.vector_store.data_chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MINIMUM_CHUNK_SIZE,
)
from rag_engine.logging_config import configure_logging
from rag_engine.settings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATABASE_BACKEND,
    DEFAULT_DATABASE_PATH,
    DEFAULT_DOCS_ROOT,
    DEFAULT_EMBEDDING_MODEL,
)


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a new documentation index once. Existing collections are rejected.",
    )
    parser.add_argument("--docs-root", type=Path, default=DEFAULT_DOCS_ROOT)
    parser.add_argument("--database-backend", default=DEFAULT_DATABASE_BACKEND)
    parser.add_argument("--database-path", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--minimum-chunk-size",
        type=int,
        default=DEFAULT_MINIMUM_CHUNK_SIZE,
        help=(
            "Pieces smaller than this are merged into a neighbouring chunk; "
            "0 disables the merge."
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="JSON mapping of document paths relative to docs-root to filter metadata.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load indexing dependencies and build one new collection."""

    configure_logging()
    arguments = _parse_arguments(argv)
    metadata = None
    if arguments.metadata is not None:
        metadata = json.loads(arguments.metadata.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or any(
            not isinstance(value, dict) for value in metadata.values()
        ):
            raise ValueError(
                "Document metadata must map document paths to JSON objects"
            )

    from sentence_transformers import SentenceTransformer

    count = create_database(
        embedder=SentenceTransformer(arguments.embedding_model),
        docs_root=arguments.docs_root,
        chunk_size=arguments.chunk_size,
        overlap=arguments.overlap,
        minimum_chunk_size=arguments.minimum_chunk_size,
        metadata_by_document=metadata,
        backend=arguments.database_backend,
        database_path=arguments.database_path,
        collection_name=arguments.collection,
    )
    print(json.dumps({"collection": arguments.collection, "indexed_chunks": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
