"""Verify overlapping retrieval and generation, admission, cancellation, and HTTP reuse."""

import asyncio
import json
import threading
import urllib.error
from functools import partial
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
from rag_engine.runtime.service import question_result
from rag_engine.serving import answers as question_answers
from rag_engine.serving.app import create_app
from rag_engine.serving.backend import ServiceUnavailableError
from rag_engine.serving.concurrency import (
    ConcurrencyGenerationConfig,
    GenerationBusyError,
    GenerationClosedError,
    GenerationPool,
    create_backend,
)
from rag_engine.serving.concurrency import backend as backend_module
from rag_engine.serving.models import RetrieveResponse

HTTP_ENV = {"versions": {"core": "1.11.0"}, "locale": "en-US"}


def cited_answer(text):
    return json.dumps(
        {
            "status": "answered",
            "answer": f"{text} [E1]",
        }
    )


def test_retrieval_endpoint_reuses_runtime_and_never_generates(
    runtime_service, monkeypatch
):
    runtime, options = runtime_service
    generator = Mock(side_effect=AssertionError("Retrieval must never generate"))
    monkeypatch.setattr(question_answers, "call_ollama_llm", generator)
    app = create_app(lambda: create_backend(service=RagService(runtime, options)))
    with TestClient(app) as client:
        backend = app.state.backend
        for code in "ab":
            response = client.post(
                "/api/v1/retrieve",
                json={"question": f"Which procedure documents {code}"},
            )
            assert response.status_code == 200
            result = response.json()
            assert set(result) == {"question", "evidence"}
            assert result["question"] == f"Which procedure documents {code}"
            assert (
                result["evidence"][0]["text"]
                == f"Documented procedure for {code}. " * 12
            )
            assert result["evidence"][0]["chunk_ids"] == [f"{code}-1"]
            assert result["evidence"][0]["id"] == "E1"
            assert result["evidence"][0]["section"] == "Recovery"
            assert app.state.backend is backend
        assert (
            client.post("/api/v1/retrieve", json={"code": "unknown"}).status_code == 422
        )
        runtime.database.similarity_search.return_value = []
        runtime.database.similarity_search.side_effect = None
        assert (
            client.post(
                "/api/v1/retrieve", json={"question": "Which procedure documents a"}
            ).json()["evidence"]
            == []
        )
    generator.assert_not_called()


def test_http_retrieval_runs_four_workers_and_rejects_overflow(
    runtime_service, monkeypatch
):
    monkeypatch.setenv("RAG_RETRIEVAL_CONCURRENCY", "4")
    monkeypatch.setenv("RAG_RETRIEVAL_QUEUE_SIZE", "0")

    async def scenario():
        runtime, options = runtime_service
        service = RagService(runtime, options)
        loop = asyncio.get_running_loop()
        entered = {code: asyncio.Event() for code in "abcd"}
        release = threading.Event()

        def retrieve(question, *, filters=None):
            code = question.split()[-1]
            loop.call_soon_threadsafe(entered[code].set)
            if not release.wait(5):
                raise TimeoutError("Retrieval was not released")
            return service.ask(question, filters=filters)

        app = create_app(
            lambda: create_backend(
                service=SimpleNamespace(ask=retrieve), retrieval_only=True
            )
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                requests = [
                    asyncio.create_task(
                        client.post(
                            "/api/v1/retrieve",
                            json={"question": f"Which procedure documents {code}"},
                        )
                    )
                    for code in "abcd"
                ]
                try:
                    await asyncio.gather(*(wait(event) for event in entered.values()))
                    rejected = await client.post(
                        "/api/v1/retrieve", json={"question": "Which procedure documents a"}
                    )
                    assert rejected.status_code == 503
                    assert rejected.json()["detail"] == "Retrieval queue is full"
                finally:
                    release.set()
                    responses = await asyncio.gather(*requests)
                for code, response in zip("abcd", responses, strict=True):
                    assert response.status_code == 200
                    assert response.json()["evidence"][0]["path"] == f"{code}.md"

    asyncio.run(scenario())


def test_retrieval_response_keeps_stitched_text_and_every_source_id(runtime_service):
    runtime, _ = runtime_service
    chunk = runtime.database.similarity_search(query_embedding=[0])[0]
    chunk.metadata["source_pages"] = [3, 4]
    candidate = RerankCandidate(
        chunk, "First passage. Second passage.", chunk_ids=("first", "second")
    )
    result = RetrieveResponse.model_validate(
        question_result(
            "Which procedure documents a", RetrievalResult([candidate], "none", "test")
        )
    )
    assert result.evidence[0].text == candidate.text
    assert result.evidence[0].chunk_ids == ["first", "second"]
    assert result.evidence[0].path == "a.md"
    assert result.evidence[0].pages == [3, 4]


def test_retrieval_only_startup_skips_ollama_and_returns_evidence(
    runtime_service, monkeypatch
):
    runtime, options = runtime_service
    loader = Mock(return_value=runtime)
    preload = Mock(side_effect=AssertionError("Ollama must not be loaded"))
    monkeypatch.setattr(backend_module.RagRuntime, "load", loader)
    monkeypatch.setattr(backend_module, "preload_ollama_model", preload)
    monkeypatch.setenv("RAG_RETRIEVAL_ONLY", "1")
    monkeypatch.setenv("RAG_DOCS_ROOT", str(options.docs_root))
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").status_code == 200
        assert client.get("/health/ready").json()["inference_ready"] is False
        app_backend = client.app.state.backend
        assert app_backend._generation is None
        response = client.post(
            "/api/v1/resolve",
            json={
                "question": "Which procedure documents a",
                "code": "a",
                "env": HTTP_ENV,
            },
        )
        assert response.status_code == 200
        assert response.json()["answer"] is None
        assert response.json()["evidence"]
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
        assert (
            client.post(
                "/api/v1/retrieve", json={"question": "Which procedure documents a"}
            ).status_code
            == 200
        )


def test_ask_keeps_model_settings_on_server_and_validates_answers(
    runtime_service, monkeypatch
):
    runtime, options = runtime_service
    answer = {
        "status": "answered",
        "answer": "Follow the documented procedure. [E1]",
    }
    caller = Mock(
        side_effect=[
            json.dumps({**answer, "configuration": {"model": "server-model"}}),
            "not JSON",
        ]
    )
    monkeypatch.setattr(question_answers, "call_ollama_llm", caller)
    monkeypatch.setenv("OLLAMA_MODEL", "server-model")
    monkeypatch.setenv("OLLAMA_HOST", "http://server-ollama")
    app = create_app(lambda: create_backend(service=RagService(runtime, options)))
    question = "Which procedure documents a"
    with TestClient(app) as client:
        result = client.post(
            "/api/v1/ask", json={"question": question, "generate": False}
        )
        assert result.status_code == 200
        assert result.json()["answer"] is None
        assert result.json()["status"] is None
        caller.assert_not_called()

        result = client.post("/api/v1/ask", json={"question": question})
        assert result.status_code == 200
        payload = result.json()
        assert set(payload) == {"question", "status", "answer", "evidence"}
        assert payload["answer"] == answer["answer"]
        assert payload["status"] == answer["status"]
        assert payload["evidence"][0]["id"] == "E1"
        assert caller.call_args.kwargs["model"] == "server-model"
        assert caller.call_args.kwargs["host"] == "http://server-ollama"

        failure = client.post("/api/v1/ask", json={"question": question})
        assert failure.status_code == 502
        assert client.get("/health/ready").status_code == 200


async def wait(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), timeout=5)


def blocking_call(loop, started, release, value):
    loop.call_soon_threadsafe(started.set)
    if not release.wait(5):
        raise TimeoutError("Test worker was not released")
    return value


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

        def generate(index):
            return blocking_call(
                loop, started[index], release[index], {"code": str(index)}
            )

        pool = GenerationPool(ConcurrencyGenerationConfig(2, 1))
        jobs = [
            asyncio.create_task(pool.run_call(partial(generate, i))) for i in range(3)
        ]
        try:
            await wait(started[0])
            await wait(started[1])
            assert not started[2].is_set()
            with pytest.raises(GenerationBusyError):
                await pool.run_call(partial(generate, 0))
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

        def generate():
            return blocking_call(loop, started, release, {"code": "active"})

        pool = GenerationPool(ConcurrencyGenerationConfig(1, 0))
        request = asyncio.create_task(pool.run_call(generate))
        try:
            await wait(started)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            with pytest.raises(GenerationBusyError):
                await pool.run_call(generate)
            shutdown = asyncio.create_task(pool.close())
            await asyncio.sleep(0)
            assert not shutdown.done()
            with pytest.raises(GenerationClosedError):
                await pool.run_call(generate)
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

        def generate(code):
            called.append(code)
            return blocking_call(loop, started, release, {"code": code})

        pool = GenerationPool(ConcurrencyGenerationConfig(1, 1))
        active = asyncio.create_task(pool.run_call(partial(generate, "first")))
        try:
            await wait(started)
            queued = asyncio.create_task(pool.run_call(partial(generate, "skipped")))
            await asyncio.sleep(0)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            replacement = asyncio.create_task(
                pool.run_call(partial(generate, "replacement"))
            )
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
        pool = GenerationPool(ConcurrencyGenerationConfig(1, 0))
        try:
            with pytest.raises(ValueError, match="bad reply"):
                await pool.run_call(generate)
            assert await pool.run_call(generate) == {"code": "next answer"}
        finally:
            await pool.close()

    asyncio.run(scenario())


def test_shutdown_drains_both_active_and_queued_generation():
    async def scenario():
        loop = asyncio.get_running_loop()
        started = [asyncio.Event(), asyncio.Event()]
        release = [threading.Event(), threading.Event()]

        def generate(index):
            return blocking_call(
                loop, started[index], release[index], {"code": str(index)}
            )

        pool = GenerationPool(ConcurrencyGenerationConfig(1, 1))
        jobs = [
            asyncio.create_task(pool.run_call(partial(generate, i))) for i in range(2)
        ]
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
    chunks = {}
    for index, code in enumerate("abcd"):
        text = f"Documented procedure for {code}. " * 12
        (tmp_path / f"{code}.md").write_text(text)
        chunks[index] = Chunk(
            chunk_id=f"{code}-1",
            document_id=f"{code}.md",
            source_path=[f"{code}.md"],
            section="Recovery",
            embedding=[],
            start_offset=0,
            end_offset=len(text),
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
    runtime = RagRuntime(embedder, database, rerank_strategy=strategy)
    return runtime, RetrievalOptions(docs_root=tmp_path)


def test_http_repeated_resolve_generates_each_answer(runtime_service, monkeypatch):
    runtime, options = runtime_service
    generator = Mock(
        side_effect=[cited_answer(text) for text in ("First answer", "Fresh answer")]
    )
    monkeypatch.setattr(question_answers, "call_ollama_llm", generator)
    app = create_app(lambda: create_backend(service=RagService(runtime, options)))
    with TestClient(app) as client:
        answers = [
            client.post(
                "/api/v1/resolve",
                json={
                    "question": "Which procedure documents a",
                    "code": "a",
                    "env": HTTP_ENV,
                },
            )
            for _ in range(2)
        ]
        assert all(answer.status_code == 200 for answer in answers)
        assert [answer.json()["answer"] for answer in answers] == [
            "First answer [E1]",
            "Fresh answer [E1]",
        ]
        assert client.get("/health/ready").json()["inference_ready"] is True
    assert generator.call_count == 2
    assert runtime.database.similarity_search.call_count == 2


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
        original_run = GenerationPool.run_call

        async def observed_run(pool, call):
            context = call.keywords["context"]
            if context and context["code"] == "c":
                third_queued.set()
            return await original_run(pool, call)

        monkeypatch.setattr(GenerationPool, "run_call", observed_run)

        def generator(_system, payload, **kwargs):
            request = json.loads(payload)
            code = request["context"]["code"]
            supplied[code] = request["evidence"]
            loop.call_soon_threadsafe(started[code].set)
            if not release.wait(5):
                raise TimeoutError("Generation was not released")
            return cited_answer(f"Guidance for {code}")

        monkeypatch.setattr(question_answers, "call_ollama_llm", generator)
        service = RagService(runtime, retrieval_options)
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
                            json={
                                "question": f"Which procedure documents {code}",
                                "code": code,
                                "env": HTTP_ENV,
                            },
                        )
                    )
                    for code in "ab"
                ]
                try:
                    await wait(started["a"])
                    await wait(started["b"])
                    ready = await asyncio.wait_for(client.get("/health/ready"), 2)
                    assert ready.status_code == 200
                    assert ready.json()["inference_ready"] is True
                    third = asyncio.create_task(
                        client.post(
                            "/api/v1/resolve",
                            json={
                                "question": "Which procedure documents c",
                                "code": "c",
                                "env": HTTP_ENV,
                            },
                        )
                    )
                    requests.append(third)
                    await wait(third_queued)
                    evidence_only = await asyncio.wait_for(
                        client.post(
                            "/api/v1/retrieve",
                            json={"question": "Which procedure documents d"},
                        ),
                        2,
                    )
                    assert evidence_only.status_code == 200
                    assert evidence_only.json()["evidence"][0]["path"] == "d.md"
                    # Document questions share the generation queue with alarms.
                    fourth = await asyncio.wait_for(
                        client.post(
                            "/api/v1/ask",
                            json={"question": "Which procedure documents d"},
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
                    assert answer["answer"] == f"Guidance for {code} [E1]"
                    assert answer["evidence"][0]["path"] == f"{code}.md"
                    assert supplied[code][0]["chunk_ids"] == [f"{code}-1"]
        assert app.state.backend is None
        assert retained.service.runtime is runtime
        assert not retained.health().ready

    asyncio.run(scenario())


def test_resolve_uses_only_question_for_retrieval_and_passes_report_to_generation(
    runtime_service, monkeypatch
):
    runtime, options = runtime_service
    runtime.embedder.encode.side_effect = lambda *args, **kwargs: SimpleNamespace(
        tolist=lambda: [0]
    )
    generator = Mock(return_value=cited_answer("Worker guidance"))
    monkeypatch.setattr(question_answers, "call_ollama_llm", generator)
    report = {
        "question": "What should I check after a worker timeout?",
        "code": "uncatalogued-code",
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
    service = RagService(runtime, options)
    app = create_app(lambda: create_backend(service=service))
    with TestClient(app) as client:
        response = client.post("/api/v1/resolve", json=report)
    assert response.status_code == 200
    assert response.json()["evidence"][0]["path"] == "a.md"
    assert response.json()["answer"] == "Worker guidance [E1]"
    payload = json.loads(generator.call_args.args[1])
    assert payload["question"] == report["question"]
    assert payload["context"] == {
        key: value for key, value in report.items() if key != "question"
    }
    runtime.embedder.encode.assert_called_once_with(
        report["question"], normalize_embeddings=True
    )
    runtime.rerank_strategy.rerank.assert_called_once()
    assert runtime.rerank_strategy.rerank.call_args.args[0] == report["question"]
    filters = runtime.database.similarity_search.call_args.kwargs["filters"]
    assert list(filters.active_fields()) == []
    generator.assert_called_once()


@pytest.mark.parametrize("invalid_response", ["not JSON", '{"summary":'])
def test_invalid_model_response_keeps_readiness_open_for_retry(
    runtime_service, invalid_response, monkeypatch
):
    runtime, options = runtime_service
    caller = Mock(
        side_effect=[
            invalid_response,
            invalid_response,
            cited_answer("Recovered"),
        ]
    )
    monkeypatch.setattr(question_answers, "call_ollama_llm", caller)
    service = RagService(runtime, options)
    app = create_app(lambda: create_backend(service=service))
    with TestClient(app) as client:
        # Match clients that check readiness before every generation attempt.
        for expected_status in (502, 502, 200):
            health = client.get("/health/ready")
            assert health.status_code == 200
            assert health.json()["ready"] is True
            assert health.json()["inference_ready"] is True
            assert health.json()["detail"] is None
            response = client.post(
                "/api/v1/resolve",
                json={
                    "question": "Which procedure documents a",
                    "code": "a",
                    "env": HTTP_ENV,
                },
            )
            assert response.status_code == expected_status
        assert response.json()["answer"] == "Recovered [E1]"
        assert client.get("/health/ready").status_code == 200
    assert caller.call_count == 3


def test_backend_reports_transport_failure_and_recovers(runtime_service, monkeypatch):
    async def scenario():
        runtime, options = runtime_service
        generator = Mock(
            side_effect=[
                urllib.error.URLError("offline"),
                cited_answer("Recovered"),
            ]
        )
        monkeypatch.setattr(question_answers, "call_ollama_llm", generator)
        async with create_backend(service=RagService(runtime, options)) as backend:
            with pytest.raises(ServiceUnavailableError):
                await backend.ask("Which procedure documents a", generate=True)
            assert not backend.health().ready
            search = runtime.database.similarity_search.side_effect
            runtime.database.similarity_search.side_effect = None
            runtime.database.similarity_search.return_value = []
            assert (
                await backend.ask("Which procedure documents b", generate=True)
            ).status == "insufficient_evidence"
            assert not backend.health().ready  # A refusal did not contact the model.
            runtime.database.similarity_search.side_effect = search
            assert (
                await backend.ask("Which procedure documents a", generate=True)
            ).answer == "Recovered [E1]"
            assert backend.health().ready

    asyncio.run(scenario())


def test_no_evidence_returns_without_waiting_for_busy_generation(
    runtime_service, monkeypatch
):
    async def scenario():
        runtime, options = runtime_service
        started, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        generator = Mock(
            side_effect=lambda *args, **kwargs: blocking_call(
                loop,
                started,
                release,
                cited_answer("Answer"),
            )
        )
        monkeypatch.setattr(question_answers, "call_ollama_llm", generator)
        async with create_backend(
            service=RagService(runtime, options),
            options=ConcurrencyGenerationConfig(1, 0),
        ) as backend:
            active = asyncio.create_task(
                backend.ask("Which procedure documents a", generate=True)
            )
            try:
                await wait(started)
                runtime.database.similarity_search.side_effect = None
                runtime.database.similarity_search.return_value = []
                refusal = await asyncio.wait_for(
                    backend.ask("Which procedure documents b", generate=True), 2
                )
                assert refusal.status == "insufficient_evidence"
                assert refusal.answer == "No evidence was available for this question."
                generator.assert_called_once()
                assert not active.done()
            finally:
                release.set()
                await active

    asyncio.run(scenario())


def test_backend_shutdown_waits_for_cancelled_inflight_generation(
    runtime_service, monkeypatch
):
    async def scenario():
        runtime, options = runtime_service
        started, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        def generator(*args, **kwargs):
            return blocking_call(loop, started, release, cited_answer("Answer"))

        monkeypatch.setattr(question_answers, "call_ollama_llm", generator)
        async with create_backend(service=RagService(runtime, options)) as backend:
            request = asyncio.create_task(
                backend.ask("Which procedure documents a", generate=True)
            )
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
                    await backend.ask("Which procedure documents b", generate=True)
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


def test_request_id_reaches_retrieval_and_generation_workers(
    runtime_service, monkeypatch
):
    from rag_engine.logging_config import request_id

    runtime, options = runtime_service
    observed = []
    original_search = runtime.database.similarity_search.side_effect

    def search(**kwargs):
        observed.append(("retrieval", request_id.get()))
        return original_search(**kwargs)

    runtime.database.similarity_search.side_effect = search

    def generate(*args, **kwargs):
        observed.append(("generation", request_id.get()))
        return cited_answer("Check wiring")

    monkeypatch.setattr(question_answers, "call_ollama_llm", generate)
    app = create_app(lambda: create_backend(service=RagService(runtime, options)))
    with TestClient(app) as client:
        for code in ("a", "b"):
            response = client.post(
                "/api/v1/resolve",
                json={"question": f"Which procedure documents {code}"},
            )
            assert response.status_code == 200
            assert observed[-2:] == [
                ("retrieval", response.headers["x-request-id"]),
                ("generation", response.headers["x-request-id"]),
            ]
    assert observed[0][1] != observed[2][1]
    assert request_id.get() == "-"


def test_abandoned_generation_failure_is_logged():
    import logging

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("rag_engine.serving.concurrency.generation")
    handler = Capture()
    logger.addHandler(handler)

    async def scenario():
        loop = asyncio.get_running_loop()
        started, release = asyncio.Event(), threading.Event()

        def generate():
            loop.call_soon_threadsafe(started.set)
            release.wait(5)
            raise RuntimeError("generation failed after disconnect")

        pool = GenerationPool()
        caller = asyncio.create_task(pool.run_call(generate))
        try:
            await wait(started)
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
        finally:
            release.set()
            await pool.close()

    try:
        asyncio.run(scenario())
    finally:
        logger.removeHandler(handler)
    assert any(
        record.exc_info and "caller disconnected" in record.getMessage()
        for record in records
    )
