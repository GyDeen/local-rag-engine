import asyncio
import logging
import os
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any

from rag_engine.engine.retrieve.pipeline import RetrievalResult
from rag_engine.settings import (
    DEFAULT_GENERATION_CONCURRENCY,
    DEFAULT_GENERATION_QUEUE_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GenerationJob:
    """One request's input report and already-retrieved evidence"""

    input_report: Mapping[str, Any]
    retrieval: RetrievalResult


@dataclass(frozen=True, slots=True)
class ConcurrencyGenerationConfig:
    """Startup limits for active generation workers and waiting jobs."""

    # The maximum concurrent generation workers
    max_concurrent: int = DEFAULT_GENERATION_CONCURRENCY
    max_queue: int = DEFAULT_GENERATION_QUEUE_SIZE

    def __post_init__(self) -> None:
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if self.max_queue < 0:
            raise ValueError("max_queue cannot be negative")

    @classmethod
    def from_env(cls) -> ConcurrencyGenerationConfig:
        """Read generation capacity once, before loading runtime resources"""
        defaults = cls()
        return cls(
            max_concurrent=int(
                os.getenv("RAG_GENERATION_CONCURRENCY", str(defaults.max_concurrent))
            ),
            max_queue=int(
                os.getenv("RAG_GENERATION_QUEUE_SIZE", str(defaults.max_queue))
            ),
        )


class GenerationBusyError(Exception):
    """Generation capacity and its waiting queue are full."""


class GenerationClosedError(Exception):
    """The generation pool is shutting down and cannot accept new work."""


class GenerationPool:
    """Run generation jobs using one configured handler and bounded capacity.

    A cancelled waiter can cancel a queued job, but cannot stop a running HTTP
    call. That call retains its worker and is included in shutdown draining.
    """

    def __init__(
        self,
        generate: Callable[[Mapping[str, Any], RetrievalResult], dict[str, Any]],
        options: ConcurrencyGenerationConfig = ConcurrencyGenerationConfig(),
    ) -> None:
        self.options = options
        self._generate = generate
        self._executor = ThreadPoolExecutor(
            max_workers=options.max_concurrent, thread_name_prefix="rag-generation"
        )
        self._pending: set[asyncio.Future[dict[str, Any]]] = set()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def closed(self) -> bool:
        """Whether new generation work is no longer accepted"""
        return self._closed

    async def run(self, job: GenerationJob) -> dict[str, Any]:
        """Submit one generation or reject it immediately when capacity is full"""
        if self._closed:
            raise GenerationClosedError("Generation pool is closed")
        if len(self._pending) >= self.options.max_concurrent + self.options.max_queue:
            raise GenerationBusyError("Generation queue is full")
        submitted = self._executor.submit(
            copy_context().run, self._generate, job.input_report, job.retrieval
        )
        future = asyncio.wrap_future(submitted)
        self._pending.add(future)
        future.add_done_callback(self._completed)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if submitted.cancel():  # Running jobs retain their worker and capacity.
                future.cancel()
            else:
                future.add_done_callback(self._log_abandoned_failure)
            raise
        finally:
            # shield() can return immediately for an already-completed future,
            # before its scheduled callback has released the admission slot.
            if future.done():
                self._completed(future)

    def _completed(self, future: asyncio.Future[dict[str, Any]]) -> None:
        self._pending.discard(future)
        if not future.cancelled():
            future.exception()  # Observe failures even after the caller disconnected.

    def _log_abandoned_failure(self, future: asyncio.Future[dict[str, Any]]) -> None:
        if not future.cancelled() and (error := future.exception()) is not None:
            logger.error(
                "Generation failed after caller disconnected",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def close(self) -> None:
        """Stop admission and finish accepted work before releasing the workers."""
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._drain())
        await asyncio.shield(self._close_task)

    async def _drain(self) -> None:
        await asyncio.gather(*tuple(self._pending), return_exceptions=True)
        await asyncio.to_thread(self._executor.shutdown, wait=True)
