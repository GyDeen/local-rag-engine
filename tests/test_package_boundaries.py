"""Protect package direction and executable entry points after reorganization."""

import ast
import os
import subprocess
import sys
from collections.abc import Iterator
from importlib.util import resolve_name
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"


def _imported_modules(path: Path) -> Iterator[str]:
    package = ".".join(path.relative_to(SOURCE_ROOT).parts[:-1])
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (name.name for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = resolve_name("." * node.level + module, package)
            yield module
            yield from (f"{module}.{name.name}" for name in node.names)


@pytest.mark.parametrize("layer", ["engine", "runtime"])
def test_engine_and_runtime_do_not_depend_on_application_adapters(layer):
    forbidden = (
        "rag_engine.serving",
        "rag_engine.cli",
        "rag_engine.evaluation",
        "fastapi",
        "starlette",
        "uvicorn",
    )
    if layer == "engine":
        forbidden += ("rag_engine.runtime",)
    for path in (SOURCE_ROOT / "rag_engine" / layer).rglob("*.py"):
        for module in _imported_modules(path):
            assert not any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in forbidden
            ), f"{path.relative_to(SOURCE_ROOT)} imports forbidden dependency {module}"


@pytest.mark.parametrize(
    "module",
    [
        "rag_engine.cli",
        "rag_engine.cli.index",
        "rag_engine.cli.retrieve",
        "rag_engine.cli.answer",
        "rag_engine.evaluation.run_all",
    ],
)
def test_cli_entry_points_provide_help_without_http_or_model_startup(module):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import runpy
import sys

sys.argv = [sys.argv[1], '--help']
try:
    runpy.run_module(sys.argv[0], run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert 'fastapi' not in sys.modules
assert 'sentence_transformers' not in sys.modules
assert 'chromadb' not in sys.modules
""",
            module,
        ],
        env={**os.environ, "PYTHONPATH": str(SOURCE_ROOT)},
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
