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

from rag_engine.engine.parse_input.input_processing import PreparedRetrieval
from rag_engine.engine.vector_store.database_metadata_model import SearchFilters
from rag_engine.serving.app import create_app
from rag_engine.serving.backend import (
    BackendHealth,
    ServiceBusyError,
    ServiceUnavailableError,
    UnknownAlarmError,
    unconfigured_backend,
)
from rag_engine.serving.models import AlarmSummary, ResolveRequest, RetrieveResponse
from rag_engine.serving.response import format_resolution

REPORT = {
    "code": "demo.test.0001",
    "env": {"versions": {"core": "1.11.0"}, "locale": "en-US"},
    "context": {"program": "example.task", "line": 2},
    "text": "Service fault",
}
RESULT = {
    "code": "demo.test.0001",
    "answer": {
        "summary": "A service reported a fault.",
        "resolution_steps": ["Record the code.", "Consult the documented procedure."],
        "likely_causes": ["Service fault"],
        "doc_refs": [{"doc": "Manual", "path": "manual.md", "section": "Recovery"}],
    },
    "retrieval": {"strategy": "cross_encoder"},
}
ALARM = {
    "title": "Service fault",
    "severity": 833,
    "severity_category": "Error",
    "domain": "Service",
}


class StubBackend:
    def __init__(self):
        self.calls = []
        self.retrieval_calls = []
        self.error = None
        self.snapshot = BackendHealth(True, True, True)

    def health(self):
        return self.snapshot

    def list_alarms(self):
        if self.error:
            raise self.error
        return [AlarmSummary(code=REPORT["code"], alarm_text="Service fault", **ALARM)]

    async def resolve(self, input_report, tier):
        self.calls.append((dict(input_report), tier))
        if self.error:
            raise self.error
        return format_resolution({**RESULT, "code": input_report["code"]}, ALARM, tier)

    async def retrieve(self, input_report):
        self.retrieval_calls.append(dict(input_report))
        if self.error:
            raise self.error
        return RetrieveResponse(
            code=input_report["code"],
            alarm_text="Service fault",
            evidence=[],
            strategy="cross_encoder",
            stop_reason="strategy_selected",
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
            client.post("/api/v1/retrieve", json={"code": REPORT["code"]}).status_code
            == 503
        )
        for path in ("/", "/api/v1/alarms"):
            assert client.get(path).status_code == 503


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


@pytest.mark.parametrize("tier", ["operator", "technician", "partner"])
def test_public_mapping_and_request_context(configured_app, tier):
    app, backend, _ = configured_app
    original = deepcopy(RESULT)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/resolve", json=REPORT, headers={"X-RAG-Tier": tier}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["steps"] == RESULT["answer"]["resolution_steps"]
    assert data["citations"] == RESULT["answer"]["doc_refs"]
    assert data["severity"] == 833
    assert data["title"] == ALARM["title"]
    assert data["severity_category"] == ALARM["severity_category"]
    assert data["domain"] == ALARM["domain"]
    assert data["tier"] == tier
    assert ("likely_causes" in data) == (tier != "operator")
    expected_fields = {
        "code",
        "title",
        "severity",
        "severity_category",
        "domain",
        "summary",
        "steps",
        "citations",
        "tier",
        "ai_chat_available",
    }
    if tier != "operator":
        expected_fields.add("likely_causes")
    assert set(data) == expected_fields
    assert "confidence" not in data
    assert data["ai_chat_available"] is False
    assert "answer" not in data and "retrieval" not in data
    assert backend.calls == [(REPORT, tier)]
    assert RESULT == original


def test_http_response_schema_declares_interface_fields_without_confidence():
    schema = create_app().openapi()
    response = schema["components"]["schemas"]["ResolveResponse"]
    assert set(response["properties"]) == {
        "code",
        "title",
        "severity",
        "severity_category",
        "domain",
        "summary",
        "steps",
        "likely_causes",
        "citations",
        "tier",
        "ai_chat_available",
    }
    assert set(response["required"]) == {
        "code",
        "title",
        "severity",
        "severity_category",
        "domain",
        "summary",
        "steps",
        "citations",
        "tier",
    }
    assert response["properties"]["ai_chat_available"]["const"] is False
    assert schema["paths"]["/api/v1/resolve"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/ResolveResponse"}


def test_mockapi_environment_and_nested_context_reach_backend(configured_app):
    app, backend, _ = configured_app
    report = {
        **REPORT,
        "env": {
            **REPORT["env"],
            "os": "Windows",
            "machine_variant": "FX5",
            "serial": "machine-123",
        },
        "context": {
            "program": "example.task",
            "line": 2,
            "pp_snippet": ['N1 electronic_cam_enable("X","A")'],
            "device": {"name": "Worker_3", "error_code": "DEMO_TIMEOUT"},
        },
    }
    request = ResolveRequest.model_validate(report)
    assert request.model_dump(exclude_none=True) == report
    prepared = PreparedRetrieval.model_validate(request.model_dump(exclude_none=True))
    assert isinstance(prepared.env, SearchFilters)
    assert list(prepared.env.active_fields()) == []

    with TestClient(app) as client:
        assert client.post("/api/v1/resolve", json=report).status_code == 200
    assert backend.calls == [(report, "operator")]


@pytest.mark.parametrize(
    "report",
    [
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
        assert client.post("/api/v1/resolve", json=report).status_code == 200
    assert backend.calls == [
        (
            {name: value for name, value in report.items() if value is not None},
            "operator",
        )
    ]


def test_http_schema_matches_mockapi_request_requirements():
    schema = create_app().openapi()
    request = schema["components"]["schemas"]["ResolveRequest"]
    assert set(request["properties"]) == {"code", "env", "context", "text"}
    assert request["required"] == ["code", "env"]
    env = schema["components"]["schemas"]["Env"]
    assert env["required"] == ["versions", "locale"]
    assert set(env["properties"]) == {
        "versions",
        "locale",
        "os",
        "machine_variant",
        "serial",
    }
    assert "SearchFilters" not in schema["components"]["schemas"]


@pytest.mark.parametrize(
    "body,headers",
    [
        ({}, {}),
        ({"code": REPORT["code"]}, {}),
        ({**REPORT, "env": None}, {}),
        ({**REPORT, "env": {}}, {}),
        ({**REPORT, "env": {"locale": "en-US"}}, {}),
        ({**REPORT, "env": {"versions": {"core": "1.11.0"}}}, {}),
        ({**REPORT, "code": ""}, {}),
        ({**REPORT, "env": {**REPORT["env"], "versions": []}}, {}),
        ({**REPORT, "env": {**REPORT["env"], "versions": {"core": 123}}}, {}),
        ({**REPORT, "env": {**REPORT["env"], "locale": 123}}, {}),
        ({**REPORT, "context": "program example.task"}, {}),
        ({**REPORT, "context": []}, {}),
        (REPORT, {"X-RAG-Tier": "invalid"}),
    ],
)
def test_invalid_requests_do_not_reach_backend(configured_app, body, headers):
    app, backend, _ = configured_app
    with TestClient(app) as client:
        assert (
            client.post("/api/v1/resolve", json=body, headers=headers).status_code
            == 422
        )
    assert backend.calls == []


@pytest.mark.parametrize(
    "error,status",
    [
        (UnknownAlarmError("Unknown alarm code"), 404),
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
            async def resolve(self, request, tier):
                started.set()
                await release.wait()
                return await super().resolve(request, tier)

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


@pytest.mark.parametrize(
    "report", [{"code": REPORT["code"]}, {**REPORT, "question": "How to recover?"}]
)
def test_retrieval_accepts_code_only_and_existing_alarm_context(configured_app, report):
    app, backend, _ = configured_app
    with TestClient(app) as client:
        response = client.post("/api/v1/retrieve", json=report)
    assert response.status_code == 200
    assert response.json()["evidence"] == []
    assert "answer" not in response.json() and "summary" not in response.json()
    assert backend.retrieval_calls == [report]
    assert backend.calls == []


@pytest.mark.parametrize(
    "report", [{}, {"code": ""}, {"code": REPORT["code"], "context": []}]
)
def test_invalid_retrieval_input_is_rejected(configured_app, report):
    app, backend, _ = configured_app
    with TestClient(app) as client:
        assert client.post("/api/v1/retrieve", json=report).status_code == 422
    assert backend.retrieval_calls == []


@pytest.mark.parametrize(
    "error,status",
    [
        (UnknownAlarmError("Unknown alarm"), 404),
        (ServiceUnavailableError("Offline"), 503),
    ],
)
def test_retrieval_errors_are_mapped_to_http(configured_app, error, status):
    app, backend, _ = configured_app
    backend.error = error
    with TestClient(app) as client:
        response = client.post("/api/v1/retrieve", json={"code": REPORT["code"]})
    assert response.status_code == status
    assert response.json()["detail"] == str(error)
