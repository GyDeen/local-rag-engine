"""Failure isolation, diagnostic logging, and public error boundaries."""

import asyncio
import logging
import subprocess
import sys
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest
from chromadb.errors import InternalError
from fastapi.testclient import TestClient

from rag_engine.cli.errors import run_cli
from rag_engine.engine.llm import llm_callers
from rag_engine.engine.llm.answer_validation import extract_json_object
from rag_engine.engine.retrieve.pipeline import retrieve_candidate_chunks
from rag_engine.engine.vector_store import backends
from rag_engine.engine.vector_store.Chroma_API import ChromaVectorDatabase
from rag_engine.engine.vector_store.create_database import create_database
from rag_engine.exceptions import (
    DatabaseError,
    GenerationError,
    IndexingError,
    ModelResponseError,
    RerankingError,
    RerankResponseError,
    RetrievalError,
    ServiceBusyError,
    SourceDocumentError,
)
from rag_engine.logging_config import configure_logging, log_operation, request_id
from rag_engine.serving.app import create_app
from rag_engine.serving.backend import BackendHealth


@pytest.fixture
def recorded_logs(monkeypatch, caplog):
    logger = logging.getLogger("rag_engine")
    monkeypatch.setattr(logger, "handlers", [caplog.handler])
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "level", logging.DEBUG)
    return caplog


@pytest.mark.parametrize(
    "error,status",
    [
        (RetrievalError("private database path"), 503),
        (RerankingError("private model configuration"), 503),
        (GenerationError("private server URL"), 503),
        (SourceDocumentError("private document path"), 503),
        (DatabaseError("private index path"), 503),
        (ModelResponseError("private model output"), 502),
        (RerankResponseError("private model output"), 502),
        (KeyError("private programming error"), 500),
    ],
)
@pytest.mark.parametrize("endpoint", ["retrieve", "resolve"])
def test_failed_request_is_safe_logged_and_next_request_succeeds(
    error, status, endpoint, recorded_logs
):
    backend = SimpleNamespace(
        error=error, health=lambda: BackendHealth(True, False, True)
    )

    async def fail_once(*args):
        if backend.error is not None:
            failure, backend.error = backend.error, None
            raise failure
        # A working health route checks that the server still handles requests;
        # a separate subsequent retrieval exercises the same operation too.
        return {
            "code": "a",
            "alarm_text": "Alarm",
            "evidence": [],
            "strategy": "none",
            "stop_reason": "empty",
        }

    backend.retrieve = fail_once
    backend.resolve = fail_once

    @asynccontextmanager
    async def factory():
        yield backend

    body = {"code": "a", "env": {"versions": {}, "locale": "en"}}
    with TestClient(create_app(factory), raise_server_exceptions=False) as client:
        response = client.post(f"/api/v1/{endpoint}", json=body)
        assert response.status_code == status
        identifier = response.headers["x-request-id"]
        assert identifier
        assert "private" not in response.text
        assert client.get("/health/live").status_code == 200
        assert client.post("/api/v1/retrieve", json=body).status_code == 200
    failures = [record for record in recorded_logs.records if record.exc_info]
    assert len(failures) == 1
    assert failures[0].request_id == identifier
    assert failures[0].exc_info[1] is error


def test_retrieval_timeout_has_a_typed_cause_and_does_not_search():
    cause = TimeoutError("embedding dependency offline")
    embedder = Mock()
    embedder.encode.side_effect = cause
    database = Mock()
    with pytest.raises(RetrievalError) as caught:
        retrieve_candidate_chunks(
            {"code": "a"}, {"a": {"title": "Alarm"}}, embedder, database
        )
    assert caught.value.__cause__ is cause
    database.similarity_search.assert_not_called()


def test_database_driver_failure_has_a_typed_cause():
    collection = Mock()
    cause = InternalError("driver unavailable")
    collection.query.side_effect = cause
    client = Mock()
    client.get_collection.return_value = collection
    from rag_engine.engine.vector_store.database_metadata_model import SearchFilters

    with pytest.raises(DatabaseError) as caught:
        ChromaVectorDatabase(client).similarity_search([1.0], SearchFilters())
    assert caught.value.__cause__ is cause


def test_indexing_embedding_failure_stops_before_database_writes(tmp_path, monkeypatch):
    (tmp_path / "manual.md").write_text("# Recovery\nCheck the drive wiring.\n")
    database = Mock()
    monkeypatch.setitem(
        backends._BACKENDS,
        "failure-test",
        backends.DatabaseBackend(lambda: database, Mock()),
    )
    embedder = Mock()
    cause = RuntimeError("embedding model unavailable")
    embedder.encode.side_effect = cause
    with pytest.raises(IndexingError) as caught:
        create_database(embedder, tmp_path, backend="failure-test")
    assert caught.value.__cause__ is cause
    database.upsert_chunks.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b"{}",
        b'{"message":{"content":null}}',
        b'{"message":{"content":""}}',
    ],
)
def test_invalid_ollama_envelope_is_typed(payload, monkeypatch):
    monkeypatch.setattr(
        llm_callers.urllib.request, "urlopen", lambda *args, **kwargs: BytesIO(payload)
    )
    with pytest.raises(ModelResponseError):
        llm_callers.call_ollama_llm("system", "user")


def test_invalid_model_output_does_not_enter_error_message():
    with pytest.raises(ModelResponseError) as caught:
        extract_json_object("private model text")
    assert "private" not in str(caught.value)


def test_dependency_busy_remains_busy_and_programming_errors_propagate():
    @log_operation("test generation", error_type=GenerationError)
    def fail(error):
        raise error

    with pytest.raises(ServiceBusyError):
        fail(HTTPError("http://model", 503, "busy", {}, None))
    bug = KeyError("bug")
    with pytest.raises(KeyError) as caught:
        fail(bug)
    assert caught.value is bug
    with pytest.raises(asyncio.CancelledError):
        fail(asyncio.CancelledError())


def test_configuration_is_idempotent_and_respects_level(monkeypatch, capsys):
    logger = logging.getLogger("rag_engine")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", True)
    monkeypatch.setattr(logger, "level", logging.NOTSET)
    monkeypatch.setenv("RAG_LOG_LEVEL", "WARNING")
    configure_logging()
    configure_logging()
    assert len(logger.handlers) == 1
    child = logging.getLogger("rag_engine.test")
    child.info("hidden")
    token = request_id.set("test-request")
    try:
        child.warning("visible")
    finally:
        request_id.reset(token)
    output = capsys.readouterr()
    assert not output.out
    assert output.err.count("visible") == 1
    assert "hidden" not in output.err
    assert "request_id=test-request" in output.err


def test_cli_boundary_logs_failure_and_preserves_parser_exit(recorded_logs):
    def fail():
        raise IndexingError("index incomplete")

    assert run_cli(fail) == 1
    assert any(record.exc_info for record in recorded_logs.records)

    def help_exit():
        raise SystemExit(0)

    with pytest.raises(SystemExit) as caught:
        run_cli(help_exit)
    assert caught.value.code == 0


def test_cli_failure_exits_nonzero_without_success_output(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_engine.cli.retrieve",
            str(tmp_path / "missing.json"),
        ],
        env={"PATH": str(Path(sys.executable).parent), "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert not result.stdout
    assert "Command failed" in result.stderr
    assert "FileNotFoundError" in result.stderr
