"""Verify the HTTP integration boundary without models or a real scheduler."""

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_engine.serving.app import create_app
from rag_engine.serving.backend import (
    BackendHealth,
    ServiceBusyError,
    ServiceUnavailableError,
    unconfigured_backend,
)
from rag_engine.serving.models import (
    QuestionResponse,
    ResolveRequest,
)

QUESTION = "How should I respond to the alarm?"
REPORT = {
    "question": QUESTION,
    "code": "demo.test.0001",
    "env": {"versions": {"core": "1.11.0"}, "locale": "en-US"},
    "context": {"program": "example.task", "line": 2},
    "text": "Service fault",
}
RESULT = {
    "question": QUESTION,
    "answer": "Consult the documented procedure. [E1]",
    "status": "answered",
    "evidence": [
        {
            "id": "E1",
            "doc": "Manual",
            "path": "manual.md",
            "pages": [],
            "section": "Recovery",
            "text": "Consult the documented procedure.",
            "chunk_ids": ["manual.md#1"],
        }
    ],
}


class StubBackend:
    def __init__(self):
        self.calls = []
        self.error = None
        self.snapshot = BackendHealth(True, True, True)

    def health(self):
        return self.snapshot

    async def ask(self, question, *, generate, filters=None, context=None):
        self.calls.append((question, generate, filters, context))
        if self.error:
            raise self.error
        answer = RESULT["answer"] if generate else None
        status = RESULT["status"] if generate else None
        return QuestionResponse(
            question=question, answer=answer, status=status, evidence=RESULT["evidence"]
        )


@pytest.fixture
def configured_app():
    backend = StubBackend()
    events = []

    @asynccontextmanager
    async def factory():
        events.append("load")
        try:
            yield backend
        finally:
            events.append("drain_and_release")

    return create_app(factory), backend, events


def test_explicit_unconfigured_backend_is_live_but_unready():
    with TestClient(create_app(unconfigured_backend)) as client:
        assert client.get("/health/live").status_code == 200
        health = client.get("/health/ready")
        assert health.status_code == 503
        assert health.json()["cache_ready"] is False
        assert health.json()["inference_ready"] is False
        response = client.post("/api/v1/resolve", json=REPORT)
        assert response.status_code == 503
        assert "not connected" in response.json()["detail"]
        assert (
            client.post("/api/v1/retrieve", json={"question": QUESTION}).status_code
            == 503
        )
        assert client.get("/").status_code == 200
        assert client.get("/api/v1/alarms").status_code == 404


def test_one_backend_is_retained_until_shutdown(configured_app):
    app, backend, events = configured_app
    with TestClient(app) as client:
        for code in ("demo.test.0001", "demo.test.0002"):
            assert (
                client.post(
                    "/api/v1/resolve", json={**REPORT, "code": code}
                ).status_code
                == 200
            )
        assert events == ["load"]
        assert app.state.backend is backend
        assert client.get("/health/ready").status_code == 200
    assert events == ["load", "drain_and_release"]
    assert app.state.backend is None
    assert app.state.phase == "stopped"


def test_public_mapping_and_request_context(configured_app):
    app, backend, _ = configured_app
    original = deepcopy(RESULT)
    with TestClient(app) as client:
        response = client.post("/api/v1/resolve", json=REPORT)
    assert response.status_code == 200
    assert response.json() == RESULT
    details = {key: value for key, value in REPORT.items() if key != "question"}
    assert backend.calls == [(QUESTION, True, None, details)]
    assert RESULT == original


def test_question_routes_share_schemas():
    schema = create_app().openapi()
    response = schema["components"]["schemas"]["QuestionResponse"]
    retrieve_response = schema["components"]["schemas"]["RetrieveResponse"]
    assert set(response["properties"]) == {
        "question",
        "evidence",
        "answer",
        "status",
    }
    assert set(response["required"]) == {"question", "evidence", "answer", "status"}
    assert set(retrieve_response["properties"]) == {"question", "evidence"}
    for model in (response, retrieve_response):
        assert model["properties"]["evidence"]["items"] == {
            "$ref": "#/components/schemas/RetrievedEvidence"
        }
    for path in ("/api/v1/ask", "/api/v1/resolve"):
        assert schema["paths"][path]["post"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"] == {"$ref": "#/components/schemas/QuestionResponse"}
    for path in ("/api/v1/ask", "/api/v1/retrieve"):
        assert schema["paths"][path]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"] == {"$ref": "#/components/schemas/QuestionRequest"}


def test_mockapi_environment_and_nested_context_reach_backend(configured_app):
    app, backend, _ = configured_app
    report = {
        **REPORT,
        "env": {
            **REPORT["env"],
            "os": "Windows",
            "machine_variant": "FX5",
            "serial": "machine-123",
            "customer_context": {"site": "north"},
        },
        "context": {
            "program": "example.task",
            "line": 2,
            "pp_snippet": ['N1 electronic_cam_enable("X","A")'],
            "device": {"name": "Worker_3", "error_code": "DEMO_TIMEOUT"},
        },
    }
    request = ResolveRequest.model_validate(report)
    assert request.model_dump(exclude_none=True, exclude_unset=True) == report

    with TestClient(app) as client:
        assert client.post("/api/v1/resolve", json=report).status_code == 200
    details = {key: value for key, value in report.items() if key != "question"}
    assert backend.calls == [(QUESTION, True, None, details)]


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"code": REPORT["code"]},
        {"code": REPORT["code"], "env": None},
        {"code": REPORT["code"], "env": {}},
        {"code": REPORT["code"], "env": {"firmware": 3}},
        {"code": REPORT["code"], "env": REPORT["env"]},
        {"code": REPORT["code"], "env": REPORT["env"], "context": {}, "text": ""},
        {
            "code": REPORT["code"],
            "env": REPORT["env"],
            "context": None,
            "text": None,
        },
    ],
)
def test_mockapi_optional_fields_can_be_absent_empty_or_null(configured_app, report):
    app, backend, _ = configured_app
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/resolve", json={**report, "question": QUESTION}
            ).status_code
            == 200
        )
    details = {name: value for name, value in report.items() if value is not None}
    assert backend.calls == [(QUESTION, True, None, details or None)]


def test_http_schema_matches_mockapi_request_requirements():
    schema = create_app().openapi()
    request = schema["components"]["schemas"]["ResolveRequest"]
    assert set(request["properties"]) == {
        "question",
        "filters",
        "generate",
        "code",
        "env",
        "context",
        "text",
    }
    assert request["required"] == ["question"]
    assert request["properties"]["env"]["anyOf"][0]["type"] == "object"
    assert "parameters" not in schema["paths"]["/api/v1/resolve"]["post"]
    assert "SearchFilters" not in schema["components"]["schemas"]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"code": REPORT["code"]},
        {**REPORT, "question": " \t"},
        {**REPORT, "env": []},
        {**REPORT, "code": ""},
        {**REPORT, "context": "program example.task"},
        {**REPORT, "context": []},
    ],
)
def test_invalid_requests_do_not_reach_backend(configured_app, body):
    app, backend, _ = configured_app
    with TestClient(app) as client:
        assert client.post("/api/v1/resolve", json=body).status_code == 422
    assert backend.calls == []


@pytest.mark.parametrize(
    "error,status",
    [
        (ServiceBusyError("Admission limit reached"), 503),
        (ServiceUnavailableError("Dependency unavailable"), 503),
    ],
)
def test_backend_errors_have_explicit_http_responses(configured_app, error, status):
    app, backend, _ = configured_app
    backend.error = error
    with TestClient(app) as client:
        response = client.post("/api/v1/resolve", json=REPORT)
    assert response.status_code == status
    assert response.json()["detail"] == str(error)


def test_inference_unready_does_not_prevent_backend_serving_cache(configured_app):
    app, backend, _ = configured_app
    backend.snapshot = BackendHealth(False, True, False, "Inference unavailable")
    with TestClient(app) as client:
        health = client.get("/health/ready")
        assert health.status_code == 503
        assert health.json()["cache_ready"] is True
        assert client.post("/api/v1/resolve", json=REPORT).status_code == 200


def test_health_remains_responsive_while_resolve_waits():
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        class WaitingBackend(StubBackend):
            async def ask(self, question, **kwargs):
                started.set()
                await release.wait()
                return await super().ask(question, **kwargs)

        @asynccontextmanager
        async def factory():
            yield WaitingBackend()

        app = create_app(factory)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                request = asyncio.create_task(
                    client.post("/api/v1/resolve", json=REPORT)
                )
                try:
                    await asyncio.wait_for(started.wait(), timeout=2)
                    health = await asyncio.wait_for(
                        client.get("/health/ready"), timeout=2
                    )
                    assert health.status_code == 200
                    assert not request.done()
                finally:
                    release.set()
                    response = await asyncio.wait_for(request, timeout=2)
                assert response.status_code == 200

    asyncio.run(scenario())


def test_failed_startup_releases_partial_backend_resources():
    events = []

    @asynccontextmanager
    async def factory():
        try:
            events.append("partial_load")
            raise RuntimeError("Startup failed")
            yield  # The context never exposes a partially initialized backend.
        finally:
            events.append("release")

    app = create_app(factory)
    with pytest.raises(RuntimeError, match="Startup failed"):
        with TestClient(app):
            pass
    assert events == ["partial_load", "release"]
    assert app.state.backend is None


def test_template_import_does_not_load_model_libraries():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from rag_engine.serving.app import create_app; "
            "create_app(); assert 'sentence_transformers' not in sys.modules; "
            "assert 'chromadb' not in sys.modules",
        ],
        env={**os.environ, "PYTHONPATH": "src"},
        check=True,
    )


def test_retrieval_uses_supplied_question_and_document_filters(configured_app):
    app, backend, _ = configured_app
    request = {"question": QUESTION, "filters": {"product": "North"}, "generate": True}
    with TestClient(app) as client:
        response = client.post("/api/v1/retrieve", json=request)
    assert response.status_code == 200
    assert response.json() == {"question": QUESTION, "evidence": RESULT["evidence"]}
    question, generate, filters, context = backend.calls[0]
    assert question == QUESTION
    assert generate is False
    assert filters.criteria == request["filters"]
    assert context is None


@pytest.mark.parametrize(
    "request_body", [{}, {"code": REPORT["code"]}, {"question": " "}]
)
def test_invalid_retrieval_input_is_rejected(configured_app, request_body):
    app, backend, _ = configured_app
    with TestClient(app) as client:
        assert client.post("/api/v1/retrieve", json=request_body).status_code == 422
    assert backend.calls == []


@pytest.mark.parametrize(
    "error,status",
    [
        (ServiceUnavailableError("Offline"), 503),
    ],
)
def test_retrieval_errors_are_mapped_to_http(configured_app, error, status):
    app, backend, _ = configured_app
    backend.error = error
    with TestClient(app) as client:
        response = client.post("/api/v1/retrieve", json={"question": QUESTION})
    assert response.status_code == status
    assert response.json()["detail"] == str(error)
