"""Run retrieval with python -m rag_engine.cli."""

from .errors import run_cli
from .retrieve import main

raise SystemExit(run_cli(main))
