import asyncio
import os
import urllib.error
from collections.abc import AsyncIterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from copy import deepcopy
from functools import partial
from typing import Any

from rag_engine.engine.llm.generation import build_ollama_final_answer
from rag_engine.engine.llm.llm_callers import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    call_ollama_llm,
    preload_ollama_model,
)
from rag_engine.runtime import RagRuntime, RagService
from rag_engine.serving.backend import (
    BackendHealth,
    ServiceBusyError,
    ServiceUnavailableError,
    UnknownAlarmError,
)
from rag_engine.serving.models import (
    AlarmSummary,
    ResolveResponse,
    RetrieveResponse,
    Tier,
)
from rag_engine.serving.response import format_resolution, format_retrieval

from .generation import (
    ConcurrencyGenerationConfig,
    GenerationBusyError,
    GenerationClosedError,
    GenerationJob,
    GenerationPool,
)


class GenerationBackend:

    def __init__(
        self,
        service: RagService,
        options: ConcurrencyGenerationConfig = ConcurrencyGenerationConfig(),
        *,
        retrieval_only: bool = False,
    ) -> None:
        if not retrieval_only and service.final_answer_fn is None:
            raise ValueError("GenerationBackend requires a final_answer_fn")
        self.service = service
        self._generation = (
            None if retrieval_only else GenerationPool(service.generate, options)
        )
        self._retrieval = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rag-retrieval"
        )
        self._requests: set[asyncio.Task[Any]] = set()
        self._closed = False
        self._failure: str | None = None
        self._close_task: asyncio.Task[None] | None = None

    def health(self) -> BackendHealth:
        """Report availability without waiting for either inference stage."""
        ready = not self._closed and self._failure is None
        return BackendHealth(
            ready, False, ready and self._generation is not None, self._failure
        )

    def list_alarms(self) -> list[AlarmSummary]:
        """Read alarm metadata from the same catalogue used by resolution."""
        if self._closed:
            raise ServiceUnavailableError("Backend is shutting down")
        return [
            AlarmSummary(
                code=code,
                **{
                    field: alarm[field]
                    for field in (
                        "title",
                        "domain",
                        "severity",
                        "severity_category",
                        "alarm_text",
                    )
                },
            )
            for code, alarm in self.service.runtime.catalog.items()
        ]

    def _prepare_report(self, input_report: Mapping[str, Any]) -> dict[str, Any]:
        """Copy and validate input before admitting a request."""
        if self._closed:
            raise ServiceUnavailableError("Backend is shutting down")
        report = deepcopy(dict(input_report))
        code = report["code"]
        if code not in self.service.runtime.catalog:
            raise UnknownAlarmError(f"Unknown alarm code: {code}")
        return report

    async def retrieve(self, input_report: Mapping[str, Any]) -> RetrieveResponse:
        """Reuse the retrieval worker while bypassing answer generation."""
        report = self._prepare_report(input_report)
        task = asyncio.current_task()
        assert task is not None
        self._requests.add(task)
        try:
            retrieval = await asyncio.get_running_loop().run_in_executor(
                self._retrieval,
                partial(
                    self.service.retrieve,
                    report,
                    user_question=report.get("question"),
                    input_context=report,
                ),
            )
            return format_retrieval(
                report["code"], retrieval, self.service.runtime.catalog[report["code"]]
            )
        except (urllib.error.URLError, TimeoutError) as error:
            raise ServiceUnavailableError("Retrieval server is unavailable") from error
        finally:
            self._requests.discard(task)

    async def resolve(
        self, input_report: Mapping[str, Any], tier: Tier
    ) -> ResolveResponse:
        """Keep request state local and queue only the final generation stage."""
        report = self._prepare_report(input_report)
        code = report["code"]
        if self._generation is None:
            raise ServiceUnavailableError(
                "Answer generation is disabled in retrieval-only mode"
            )

        # Register the incoming task and run the retrieval
        task = asyncio.current_task()
        assert task is not None
        self._requests.add(task)
        try:
            retrieval = await asyncio.get_running_loop().run_in_executor(
                self._retrieval,
                partial(self.service.retrieve, report, input_context=report),
            )
            if retrieval.evidence:
                result = await self._generation.run(GenerationJob(report, retrieval))
                self._failure = None
            else:
                # No evidence produces a refusal without calling the LLM.
                result = self.service.generate(report, retrieval)
            return format_resolution(result, self.service.runtime.catalog[code], tier)
        except GenerationBusyError as error:
            raise ServiceBusyError(str(error)) from error
        except GenerationClosedError as error:
            raise ServiceUnavailableError(str(error)) from error
        except urllib.error.HTTPError as error:
            if error.code == 503:
                raise ServiceBusyError("Generation server is busy") from error
            self._failure = "Generation server rejected the request"
            raise ServiceUnavailableError(self._failure) from error
        except (urllib.error.URLError, TimeoutError) as error:
            self._failure = "Generation server is unavailable"
            raise ServiceUnavailableError(self._failure) from error
        finally:
            self._requests.discard(task)

    async def close(self) -> None:
        """Drain accepted requests and orphaned calls before releasing resources"""
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._drain())
        await asyncio.shield(self._close_task)

    async def _drain(self) -> None:
        await asyncio.gather(*tuple(self._requests), return_exceptions=True)
        if self._generation is not None:
            await self._generation.close()
        await asyncio.to_thread(self._retrieval.shutdown, wait=True)


async def _load_service(*, retrieval_only: bool = False) -> RagService:
    """Load the shared runtime and preload the server-owned generation model."""
    runtime = await asyncio.to_thread(RagRuntime.load)
    if retrieval_only:
        return RagService(runtime)
    model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    host = os.getenv("OLLAMA_HOST", OLLAMA_HOST)
    await asyncio.to_thread(preload_ollama_model, model=model, host=host)
    caller = partial(call_ollama_llm, model=model, host=host, keep_alive=-1)
    return RagService(runtime, final_answer_fn=build_ollama_final_answer(caller))


@asynccontextmanager
async def create_backend(
    *,
    service: RagService | None = None,
    options: ConcurrencyGenerationConfig | None = None,
    retrieval_only: bool | None = None,
) -> AsyncIterator[GenerationBackend]:
    """Create the manager for the shared runtime and generation pool"""
    configured_options = (
        options if options is not None else ConcurrencyGenerationConfig.from_env()
    )
    only_retrieve = (
        retrieval_only
        if retrieval_only is not None
        else os.getenv("RAG_RETRIEVAL_ONLY", "0").lower() in {"1", "true", "yes"}
    )
    backend = GenerationBackend(
        (
            service
            if service is not None
            else await _load_service(retrieval_only=only_retrieve)
        ),
        configured_options,
        retrieval_only=only_retrieve,
    )
    try:
        yield backend
    finally:
        await backend.close()
