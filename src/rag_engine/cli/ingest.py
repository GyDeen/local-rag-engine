"""Convert PDF, Word, HTML, text, and Markdown using the shared ingestion path."""

import argparse
import json
from pathlib import Path

from rag_engine.cli.errors import run_cli
from rag_engine.ingestion import ingest_documents
from rag_engine.logging_config import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Corpus root for documents/ and debug/",
    )
    parser.add_argument(
        "--sources", type=Path, help="Optional original source URL mapping"
    )
    args = parser.parse_args(argv)
    configure_logging()
    sources = json.loads(args.sources.read_text()) if args.sources else None
    manifest = ingest_documents(args.source, args.output, sources=sources)
    print(
        json.dumps(
            {"documents": len(manifest["documents"]), "output": str(args.output)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
