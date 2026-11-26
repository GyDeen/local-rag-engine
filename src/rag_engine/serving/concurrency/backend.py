import asyncio
import logging
import os
import urllib.error
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from functools import partial
from typing import Any

from rag_engine.engine.llm.llm_callers import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    preload_ollama_model,
)
from rag_engine.engine.vector_store.database_metadata_model import SearchFilters
from rag_engine.exceptions import GenerationError, ModelResponseError
from rag_engine.runtime import RagRuntime, RagService, RetrievalOptions
from rag_engine.serving.answers import answer_question
from rag_engine.serving.backend import (
    BackendHealth,
    ServiceBusyError,
    ServiceUnavailableError,
)
from rag_engine.serving.models import QuestionResponse
from rag_engine.settings import DEFAULT_DOCS_ROOT

from .generation import (
    ConcurrencyGenerationConfig,
    GenerationBusyError,
    GenerationClosedError,
    GenerationPool,
)
from .retrieval import ConcurrencyRetrievalConfig, RetrievalPool
from .workers import WorkerBusyError, WorkerClosedError

logger = logging.getLogger(__name__)


class GenerationBackend:

    def __init__(
        self,
        service: RagService,
        options: ConcurrencyGenerationConfig = ConcurrencyGenerationConfig(),
        *,
        retrieval_options: ConcurrencyRetrievalConfig = ConcurrencyRetrievalConfig(),
        retrieval_only: bool = False,
        model: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
    ) -> None:
        self.service = service
        self._model = model
        self._host = host
        self._generation = None if retrieval_only else GenerationPool(options=options)
        self._retrieval = RetrievalPool(retrieval_options)
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

    @contextmanager
    def _track_request(self) -> Iterator[None]:
        if self._closed:
            raise ServiceUnavailableError("Backend is shutting down")
        task = asyncio.current_task()
        assert task is not None
        self._requests.add(task)
        try:
            yield
        finally:
            self._requests.discard(task)

    async def ask(
        self,
        question: str,
        *,
        generate: bool,
        filters: SearchFilters | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> QuestionResponse:
        """Retrieve from the supplied question and optionally generate an answer."""
        with self._track_request():
            try:
                result = await self._retrieval.run_call(
                    partial(self.service.ask, question, filters=filters)
                )
            except WorkerBusyError as error:
                raise ServiceBusyError(str(error)) from error
            except WorkerClosedError as error:
                raise ServiceUnavailableError(str(error)) from error
            except (urllib.error.URLError, TimeoutError) as error:
                raise ServiceUnavailableError(
                    "Retrieval server is unavailable"
                ) from error

            if generate and self._generation is not None:
                result.update(await self._generate_answer(result, context=context))
            return QuestionResponse.model_validate(result)

    async def _generate_answer(
        self, result: dict[str, Any], *, context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        assert self._generation is not None
        call = partial(
            answer_question,
            result["question"],
            result["evidence"],
            model=self._model,
            host=self._host,
            context=context,
        )
        with self._generation_errors():
            if not result["evidence"]:
                return call()
            answer = await self._generation.run_call(call)
            self._failure = None
            return answer

    @contextmanager
    def _generation_errors(self) -> Iterator[None]:
        try:
            yield
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
        except ModelResponseError:
            self._failure = None
            raise
        except GenerationError as error:
            self._failure = error.public_message
            raise

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
        await self._retrieval.close()


async def _load_service(*, retrieval_only: bool, model: str, host: str) -> RagService:
    """Load the shared runtime and preload the server-owned generation model."""
    rank_limit = os.getenv("RAG_EVIDENCE_RANK_LIMIT")
    options = RetrievalOptions(
        docs_root=os.getenv("RAG_DOCS_ROOT") or DEFAULT_DOCS_ROOT,
        evidence_rank_limit=int(rank_limit) if rank_limit else None,
    )
    settings = {
        name: value
        for name in ("database_path", "collection", "embedding_model", "reranker")
        if (value := os.getenv(f"RAG_{name.upper()}"))
    }
    runtime = await asyncio.to_thread(RagRuntime.load, **settings)
    if not retrieval_only:
        await asyncio.to_thread(preload_ollama_model, model=model, host=host)
    return RagService(runtime, options=options)


@asynccontextmanager
async def create_backend(
    *,
    service: RagService | None = None,
    options: ConcurrencyGenerationConfig | None = None,
    retrieval_options: ConcurrencyRetrievalConfig | None = None,
    retrieval_only: bool | None = None,
) -> AsyncIterator[GenerationBackend]:
    """Create the backend for the shared runtime and separate worker pools."""
    configured_options = (
        options if options is not None else ConcurrencyGenerationConfig.from_env()
    )
    retrieval_config = retrieval_options or ConcurrencyRetrievalConfig.from_env()
    only_retrieve = (
        retrieval_only
        if retrieval_only is not None
        else os.getenv("RAG_RETRIEVAL_ONLY", "0").lower() in {"1", "true", "yes"}
    )
    model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    host = os.getenv("OLLAMA_HOST", OLLAMA_HOST)
    backend = GenerationBackend(
        (
            service
            if service is not None
            else await _load_service(
                retrieval_only=only_retrieve, model=model, host=host
            )
        ),
        configured_options,
        retrieval_options=retrieval_config,
        retrieval_only=only_retrieve,
        model=model,
        host=host,
    )
    try:
        yield backend
    finally:
        await backend.close()
