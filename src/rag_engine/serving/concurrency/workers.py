"""Run blocking calls with bounded admission and graceful shutdown."""

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context


class WorkerBusyError(RuntimeError):
    """All worker and queue slots are occupied."""


class WorkerClosedError(RuntimeError):
    """The pool has stopped accepting work."""


class WorkerPool[T]:
    busy_error: type[WorkerBusyError] = WorkerBusyError
    closed_error: type[WorkerClosedError] = WorkerClosedError

    def __init__(self, stage: str, max_concurrent: int, max_queue: int) -> None:
        self._stage = stage
        self._capacity = max_concurrent + max_queue
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix=f"rag-{stage.lower()}"
        )
        self._pending: set[asyncio.Future[T]] = set()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    async def run_call(self, call: Callable[[], T]) -> T:
        if self._closed:
            raise self.closed_error(f"{self._stage} pool is closed")
        if len(self._pending) >= self._capacity:
            raise self.busy_error(f"{self._stage} queue is full")
        submitted = self._executor.submit(copy_context().run, call)
        future = asyncio.wrap_future(submitted)
        self._pending.add(future)
        future.add_done_callback(self._completed)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if submitted.cancel():
                future.cancel()
            else:
                # Running jobs retain capacity until the actual call finishes.
                future.add_done_callback(self._log_abandoned_failure)
            raise
        finally:
            if future.done():
                self._completed(future)

    def _completed(self, future: asyncio.Future[T]) -> None:
        self._pending.discard(future)
        if not future.cancelled():
            future.exception()

    def _log_abandoned_failure(self, future: asyncio.Future[T]) -> None:
        if not future.cancelled() and (error := future.exception()) is not None:
            logging.getLogger(type(self).__module__).error(
                "%s failed after caller disconnected", self._stage, exc_info=error
            )

    async def close(self) -> None:
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._drain())
        await asyncio.shield(self._close_task)

    async def _drain(self) -> None:
        await asyncio.gather(*tuple(self._pending), return_exceptions=True)
        await asyncio.to_thread(self._executor.shutdown, wait=True)
