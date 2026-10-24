"""Verify actual overlapping generation, admission, cancellation, and HTTP reuse."""

import asyncio
import hashlib
import json
import threading
import urllib.error
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_engine.engine.llm import llm_callers
from rag_engine.engine.rerank.candidate import RerankCandidate
from rag_engine.engine.retrieve.pipeline import RetrievalResult
from rag_engine.engine.vector_store.database_metadata_model import Chunk
from rag_engine.runtime import RagRuntime, RagService, RetrievalOptions
from rag_engine.serving.app import create_app
from rag_engine.serving.backend import ServiceUnavailableError, UnknownAlarmError
from rag_engine.serving.concurrency import (
    ConcurrencyGenerationConfig,
    GenerationBusyError,
    GenerationClosedError,
    GenerationJob,
    GenerationPool,
    create_backend,
)
from rag_engine.serving.concurrency import backend as backend_module
from rag_engine.serving.response import format_retrieval

HTTP_ENV = {"versions": {"core": "1.11.0"}, "locale": "en-US"}


def test_retrieval_endpoint_reuses_runtime_and_never_generates(runtime_service):
    runtime, options = runtime_service
    generator = Mock(side_effect=AssertionError("Retrieval must never generate"))
    app = create_app(
        lambda: create_backend(service=RagService(runtime, options, generator))
    )
    with TestClient(app) as client:
        backend = app.state.backend
        for code in "ab":
            response = client.post("/api/v1/retrieve", json={"code": code})
            assert response.status_code == 200
            result = response.json()
            assert result["code"] == code
            assert result["alarm_text"] == f"Alarm {code}"
            assert (
                result["evidence"][0]["text"]
                == f"Documented procedure for {code}. " * 12
            )
            assert result["evidence"][0]["chunk_ids"] == [f"{code}-1"]
            assert result["evidence"][0]["id"] == "C1"
            assert result["evidence"][0]["section"] == "Recovery"
            assert app.state.backend is backend
        assert (
            client.post("/api/v1/retrieve", json={"code": "unknown"}).status_code == 404
        )
        runtime.database.similarity_search.return_value = []
        runtime.database.similarity_search.side_effect = None
        assert (
            client.post("/api/v1/retrieve", json={"code": "a"}).json()["evidence"] == []
        )
    generator.assert_not_called()


def test_retrieval_response_keeps_stitched_text_and_every_source_id(runtime_service):
    runtime, _ = runtime_service
    chunk = runtime.database.similarity_search(query_embedding=[0])[0]
    candidate = RerankCandidate(
        chunk, "First passage. Second passage.", chunk_ids=("first", "second")
    )
    result = format_retrieval(
        "a", RetrievalResult([candidate], "none", "test"), runtime.catalog["a"]
    )
    assert result.evidence[0].text == candidate.text
    assert result.evidence[0].chunk_ids == ["first", "second"]
    assert result.evidence[0].path == "a.md"


def test_retrieval_only_startup_skips_ollama_and_rejects_generation(
    runtime_service, monkeypatch
):
    runtime, options = runtime_service
    loader = Mock(return_value=runtime)
    preload = Mock(side_effect=AssertionError("Ollama must not be loaded"))
    monkeypatch.setattr(backend_module.RagRuntime, "load", loader)
    monkeypatch.setattr(backend_module, "preload_ollama_model", preload)
    monkeypatch.setenv("RAG_RETRIEVAL_ONLY", "1")
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").status_code == 200
        assert client.get("/health/ready").json()["inference_ready"] is False
        app_backend = client.app.state.backend
        assert app_backend.service.final_answer_fn is None
        response = client.post("/api/v1/resolve", json={"code": "a", "env": HTTP_ENV})
        assert response.status_code == 503
        assert "disabled" in response.json()["detail"]
    loader.assert_called_once_with()
    preload.assert_not_called()
    # Also exercise retrieval through a configured service without any generator.
    with TestClient(
        create_app(
            lambda: create_backend(
                service=RagService(runtime, options), retrieval_only=True
            )
        )
    ) as client:
        assert client.post("/api/v1/retrieve", json={"code": "a"}).status_code == 200


def test_retrieval_forwards_question_and_context_without_generation(runtime_service):
    async def scenario():
        runtime, _ = runtime_service
        retrieve = Mock(return_value=RetrievalResult([], "none", "empty"))
        service = SimpleNamespace(
            runtime=runtime, retrieve=retrieve, final_answer_fn=None
        )
        report = {
            "code": "a",
            "text": "Actual fault",
            "context": {"queue": "X"},
            "question": "Which mapping?",
        }
        async with create_backend(service=service, retrieval_only=True) as backend:
            response = await backend.retrieve(report)
        assert response.evidence == []
        retrieve.assert_called_once_with(
            report, user_question=report["question"], input_context=report
        )

    asyncio.run(scenario())


async def wait(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), timeout=5)


def blocking_call(loop, started, release, value):
    loop.call_soon_threadsafe(started.set)
    if not release.wait(5):
        raise TimeoutError("Test worker was not released")
    return value


def generation_job(code: str) -> GenerationJob:
    return GenerationJob({"code": code}, RetrievalResult([], "none", "test_evidence"))


@pytest.mark.parametrize("options", [{"max_concurrent": 0}, {"max_queue": -1}])
def test_invalid_generation_capacity_is_rejected(options):
    with pytest.raises(ValueError):
        ConcurrencyGenerationConfig(**options)


def test_generation_capacity_is_read_once_from_environment(monkeypatch):
    monkeypatch.setenv("RAG_GENERATION_CONCURRENCY", "2")
    monkeypatch.setenv("RAG_GENERATION_QUEUE_SIZE", "7")
    assert ConcurrencyGenerationConfig.from_env() == ConcurrencyGenerationConfig(2, 7)


def test_pool_runs_two_calls_and_bounds_the_waiting_queue():
    async def scenario():
        loop = asyncio.get_running_loop()
        started = [asyncio.Event() for _ in range(3)]
        release = [threading.Event() for _ in range(3)]

        def generate(report, retrieval):
            index = int(report["code"])
            return blocking_call(
                loop, started[index], release[index], {"code": report["code"]}
            )

        pool = GenerationPool(generate, ConcurrencyGenerationConfig(2, 1))
        jobs = [asyncio.create_task(pool.run(generation_job(str(i)))) for i in range(3)]
        try:
            await wait(started[0])
            await wait(started[1])
            assert not started[2].is_set()
            with pytest.raises(GenerationBusyError):
                await pool.run(generation_job("overloaded"))
            release[0].set()
            await wait(started[2])
            assert not jobs[1].done()
        finally:
            for event in release:
                event.set()
            results = await asyncio.gather(*jobs)
            await pool.close()
        assert results == [{"code": str(i)} for i in range(3)]

    asyncio.run(scenario())


def test_cancelled_running_generation_retains_capacity_until_it_finishes():
    async def scenario():
        started, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        def generate(report, retrieval):
            return blocking_call(loop, started, release, {"code": report["code"]})

        pool = GenerationPool(generate, ConcurrencyGenerationConfig(1, 0))
        request = asyncio.create_task(pool.run(generation_job("active")))
        try:
            await wait(started)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            with pytest.raises(GenerationBusyError):
                await pool.run(generation_job("must not overlap"))
            shutdown = asyncio.create_task(pool.close())
            await asyncio.sleep(0)
            assert not shutdown.done()
            with pytest.raises(GenerationClosedError):
                await pool.run(generation_job("closed"))
        finally:
            release.set()
            await pool.close()
        await shutdown

    asyncio.run(scenario())


def test_cancelled_queued_generation_never_calls_the_model():
    async def scenario():
        started, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        called = []

        def generate(report, retrieval):
            called.append(report["code"])
            return blocking_call(loop, started, release, {"code": report["code"]})

        pool = GenerationPool(generate, ConcurrencyGenerationConfig(1, 1))
        active = asyncio.create_task(pool.run(generation_job("first")))
        try:
            await wait(started)
            queued = asyncio.create_task(pool.run(generation_job("skipped")))
            await asyncio.sleep(0)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            replacement = asyncio.create_task(pool.run(generation_job("replacement")))
            release.set()
            assert await active == {"code": "first"}
            assert await replacement == {"code": "replacement"}
            assert called == ["first", "replacement"]
        finally:
            release.set()
            await pool.close()

    asyncio.run(scenario())


def test_generator_failure_does_not_poison_the_pool():
    async def scenario():
        generate = Mock(side_effect=[ValueError("bad reply"), {"code": "next answer"}])
        pool = GenerationPool(generate, ConcurrencyGenerationConfig(1, 0))
        try:
            with pytest.raises(ValueError, match="bad reply"):
                await pool.run(generation_job("bad reply"))
            assert await pool.run(generation_job("next answer")) == {
                "code": "next answer"
            }
        finally:
            await pool.close()

    asyncio.run(scenario())


def test_shutdown_drains_both_active_and_queued_generation():
    async def scenario():
        loop = asyncio.get_running_loop()
        started = [asyncio.Event(), asyncio.Event()]
        release = [threading.Event(), threading.Event()]

        def generate(report, retrieval):
            index = int(report["code"])
            return blocking_call(
                loop, started[index], release[index], {"code": report["code"]}
            )

        pool = GenerationPool(generate, ConcurrencyGenerationConfig(1, 1))
        jobs = [asyncio.create_task(pool.run(generation_job(str(i)))) for i in range(2)]
        try:
            await wait(started[0])
            shutdown = asyncio.create_task(pool.close())
            await asyncio.sleep(0)
            release[0].set()
            await wait(started[1])
            assert not shutdown.done()
        finally:
            for event in release:
                event.set()
            assert await asyncio.gather(*jobs) == [{"code": "0"}, {"code": "1"}]
            await pool.close()
        await shutdown

    asyncio.run(scenario())


@pytest.fixture
def runtime_service(tmp_path):
    catalog, chunks = {}, {}
    for index, code in enumerate("abcd"):
        text = f"Documented procedure for {code}. " * 12
        (tmp_path / f"{code}.md").write_text(text)
        catalog[code] = {
            "alarm_text": f"Alarm {code}",
            "title": f"Alarm {code}",
            "severity": 833,
            "severity_category": "Error",
            "domain": "Service",
        }
        chunks[index] = Chunk(
            chunk_id=f"{code}-1",
            document_id=f"{code}.md",
            source_path=[f"{code}.md"],
            section="Recovery",
            embedding=[],
            start_offset=0,
            end_offset=len(text),
            source_hash=hashlib.sha256(text.encode()).hexdigest(),
            metadata={},
        )
    embedder = Mock()
    embedder.encode.side_effect = lambda query, **kwargs: SimpleNamespace(
        tolist=lambda: ["abcd".index(query[-1])]
    )
    database = Mock()
    database.similarity_search.side_effect = lambda **kwargs: [
        chunks[kwargs["query_embedding"][0]]
    ]
    strategy = SimpleNamespace(rerank=Mock(side_effect=lambda query, pool: pool))
    runtime = RagRuntime(catalog, embedder, database, rerank_strategy=strategy)
    return runtime, RetrievalOptions(docs_root=tmp_path)


def test_http_reuses_runtime_and_retrieves_while_generations_are_busy(
    runtime_service, monkeypatch
):
    async def scenario():
        runtime, retrieval_options = runtime_service
        loop = asyncio.get_running_loop()
        started = {code: asyncio.Event() for code in "abc"}
        release = threading.Event()
        supplied = {}
        third_queued = asyncio.Event()
        original_run = GenerationPool.run

        async def observed_run(pool, job):
            if job.input_report["code"] == "c":
                third_queued.set()
            return await original_run(pool, job)

        monkeypatch.setattr(GenerationPool, "run", observed_run)

        def generator(code, alarm_text, chunks, **kwargs):
            supplied[code] = chunks
            loop.call_soon_threadsafe(started[code].set)
            if not release.wait(5):
                raise TimeoutError("Generation was not released")
            return {
                "summary": f"Guidance for {code}",
                "likely_causes": [f"Cause {code}"],
                "resolution_steps": [f"Reset {code}"],
            }

        service = RagService(runtime, retrieval_options, generator)
        app = create_app(
            lambda: create_backend(
                service=service, options=ConcurrencyGenerationConfig(2, 1)
            )
        )
        async with app.router.lifespan_context(app):
            retained = app.state.backend
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                requests = [
                    asyncio.create_task(
                        client.post(
                            "/api/v1/resolve",
                            json={"code": code, "env": HTTP_ENV},
                        )
                    )
                    for code in "ab"
                ]
                try:
                    await wait(started["a"])
                    await wait(started["b"])
                    ready = await asyncio.wait_for(client.get("/health/ready"), 2)
                    assert ready.status_code == 200
                    assert ready.json()["cache_ready"] is False
                    third = asyncio.create_task(
                        client.post(
                            "/api/v1/resolve",
                            json={"code": "c", "env": HTTP_ENV},
                        )
                    )
                    requests.append(third)
                    await wait(third_queued)
                    evidence_only = await asyncio.wait_for(
                        client.post("/api/v1/retrieve", json={"code": "d"}), 2
                    )
                    assert evidence_only.status_code == 200
                    assert evidence_only.json()["evidence"][0]["path"] == "d.md"
                    # A fourth request completes retrieval too, then hits the full
                    # generation queue while a/b remain blocked in their model calls.
                    fourth = await asyncio.wait_for(
                        client.post(
                            "/api/v1/resolve",
                            json={"code": "d", "env": HTTP_ENV},
                        ),
                        2,
                    )
                    assert fourth.status_code == 503
                    assert "queue is full" in fourth.json()["detail"]
                    assert runtime.database.similarity_search.call_count == 5
                    assert not started["c"].is_set()
                    assert app.state.backend is retained
                finally:
                    release.set()
                    responses = await asyncio.gather(*requests)
                for code, response in zip("abc", responses, strict=True):
                    assert response.status_code == 200
                    answer = response.json()
                    assert answer["summary"] == f"Guidance for {code}"
                    assert answer["citations"][0]["path"] == f"{code}.md"
                    assert supplied[code][0]["chunk_id"] == f"{code}-1"
        assert app.state.backend is None
        assert retained.service.runtime is runtime
        assert not retained.health().ready

    asyncio.run(scenario())


def test_mockapi_runtime_details_reach_actual_retrieval(runtime_service):
    runtime, options = runtime_service
    runtime.embedder.encode.side_effect = lambda *args, **kwargs: SimpleNamespace(
        tolist=lambda: [0]
    )
    generator = Mock(
        return_value={
            "summary": "Worker guidance",
            "likely_causes": ["Service fault"],
            "resolution_steps": ["Consult the documented recovery procedure."],
        }
    )
    report = {
        "code": "a",
        "env": {
            "versions": {"core": "1.11.0", "oem": "3.2.1"},
            "locale": "en-US",
            "os": "Example OS 1.0",
            "machine_variant": "Demo",
            "serial": "DEMO-0001",
        },
        "text": "Emergency on Worker_3",
        "context": {"device_name": "Worker_3", "error_code": "DEMO_TIMEOUT"},
    }
    service = RagService(runtime, options, generator)
    app = create_app(lambda: create_backend(service=service))
    with TestClient(app) as client:
        response = client.post("/api/v1/resolve", json=report)
    assert response.status_code == 200
    assert response.json()["citations"][0]["path"] == "a.md"
    assert response.json()["ai_chat_available"] is False
    assert "confidence" not in response.json()
    query = runtime.embedder.encode.call_args.args[0]
    assert runtime.catalog["a"]["alarm_text"] in query
    assert report["text"] in query
    assert report["context"]["error_code"] in query
    filters = runtime.database.similarity_search.call_args.kwargs["filters"]
    assert list(filters.active_fields()) == []
    generator.assert_called_once()


def test_backend_rejects_unknown_alarms_before_retrieval(runtime_service):
    async def scenario():
        runtime, options = runtime_service
        generator = Mock()
        async with create_backend(
            service=RagService(runtime, options, generator)
        ) as backend:
            with pytest.raises(UnknownAlarmError):
                await backend.resolve({"code": "unknown"}, "operator")
        runtime.database.similarity_search.assert_not_called()
        generator.assert_not_called()

    asyncio.run(scenario())


def test_backend_reports_transport_failure_and_recovers(runtime_service):
    async def scenario():
        runtime, options = runtime_service
        generator = Mock(
            side_effect=[
                urllib.error.URLError("offline"),
                {
                    "summary": "Recovered",
                    "likely_causes": ["Cause"],
                    "resolution_steps": ["Reset"],
                },
            ]
        )
        async with create_backend(
            service=RagService(runtime, options, generator)
        ) as backend:
            with pytest.raises(ServiceUnavailableError):
                await backend.resolve({"code": "a"}, "operator")
            assert not backend.health().ready
            search = runtime.database.similarity_search.side_effect
            runtime.database.similarity_search.side_effect = None
            runtime.database.similarity_search.return_value = []
            assert (await backend.resolve({"code": "b"}, "operator")).citations == []
            assert not backend.health().ready  # A refusal did not contact the model.
            runtime.database.similarity_search.side_effect = search
            assert (
                await backend.resolve({"code": "a"}, "operator")
            ).summary == "Recovered"
            assert backend.health().ready

    asyncio.run(scenario())


def test_no_evidence_returns_without_waiting_for_busy_generation(runtime_service):
    async def scenario():
        runtime, options = runtime_service
        started, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        generator = Mock(
            side_effect=lambda *args, **kwargs: blocking_call(
                loop,
                started,
                release,
                {
                    "summary": "Answer",
                    "likely_causes": ["Cause"],
                    "resolution_steps": ["Reset"],
                },
            )
        )
        async with create_backend(
            service=RagService(runtime, options, generator),
            options=ConcurrencyGenerationConfig(1, 0),
        ) as backend:
            active = asyncio.create_task(backend.resolve({"code": "a"}, "operator"))
            try:
                await wait(started)
                runtime.database.similarity_search.side_effect = None
                runtime.database.similarity_search.return_value = []
                refusal = await asyncio.wait_for(
                    backend.resolve({"code": "b"}, "operator"), 2
                )
                assert refusal.citations == []
                generator.assert_called_once()
                assert not active.done()
            finally:
                release.set()
                await active

    asyncio.run(scenario())


def test_backend_shutdown_waits_for_cancelled_inflight_generation(runtime_service):
    async def scenario():
        runtime, options = runtime_service
        started, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        def generator(*args, **kwargs):
            return blocking_call(loop, started, release, {})

        async with create_backend(
            service=RagService(runtime, options, generator)
        ) as backend:
            request = asyncio.create_task(backend.resolve({"code": "a"}, "operator"))
            try:
                await wait(started)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                closing = asyncio.create_task(backend.close())
                await asyncio.sleep(0)
                assert not closing.done()
                assert not backend.health().ready
                with pytest.raises(ServiceUnavailableError, match="shutting down"):
                    await backend.resolve({"code": "b"}, "operator")
            finally:
                release.set()
                await backend.close()
            await closing

    asyncio.run(scenario())


def test_default_app_loads_one_runtime_and_preloads_generation(
    runtime_service, monkeypatch
):
    runtime, _options = runtime_service
    loader = Mock(return_value=runtime)
    preload = Mock()
    monkeypatch.setattr(backend_module.RagRuntime, "load", loader)
    monkeypatch.setattr(backend_module, "preload_ollama_model", preload)
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    monkeypatch.setenv("OLLAMA_HOST", "http://model-server:11434")
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").status_code == 200
        loader.assert_called_once_with()
        preload.assert_called_once_with(
            model="test-model", host="http://model-server:11434"
        )


def test_preload_failure_prevents_startup(runtime_service, monkeypatch):
    runtime, _options = runtime_service
    monkeypatch.setattr(backend_module.RagRuntime, "load", Mock(return_value=runtime))
    monkeypatch.setattr(
        backend_module,
        "preload_ollama_model",
        Mock(side_effect=urllib.error.URLError("offline")),
    )
    app = create_app()
    with pytest.raises(urllib.error.URLError):
        with TestClient(app):
            pass
    assert app.state.backend is None
    assert app.state.phase == "stopped"


def test_ollama_preload_and_queries_keep_the_model_but_not_chat_history(monkeypatch):
    payloads = []

    def urlopen(request, timeout):
        payloads.append((request.full_url, json.loads(request.data)))
        body = (
            {"done": True}
            if request.full_url.endswith("/generate")
            else {"message": {"content": "answer"}}
        )
        return BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(llm_callers.urllib.request, "urlopen", urlopen)
    llm_callers.preload_ollama_model(model="local", host="http://model")
    for query in ("first query", "second query"):
        assert (
            llm_callers.call_ollama_llm(
                "instructions", query, model="local", host="http://model", keep_alive=-1
            )
            == "answer"
        )
    assert payloads[0][1] == {"model": "local", "keep_alive": -1, "stream": False}
    assert all(payload[1]["keep_alive"] == -1 for payload in payloads)
    assert [
        message["content"]
        for message in payloads[2][1]["messages"]
        if message["role"] == "user"
    ] == ["second query"]
