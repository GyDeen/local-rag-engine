import os
from dataclasses import dataclass
from typing import Any

from rag_engine.settings import (
    DEFAULT_GENERATION_CONCURRENCY,
    DEFAULT_GENERATION_QUEUE_SIZE,
)

from .workers import WorkerBusyError, WorkerClosedError, WorkerPool


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


class GenerationBusyError(WorkerBusyError):
    """Generation capacity and its waiting queue are full."""


class GenerationClosedError(WorkerClosedError):
    """The generation pool is shutting down and cannot accept new work."""


class GenerationPool(WorkerPool[dict[str, Any]]):
    """Run generation calls with bounded capacity.

    A cancelled waiter can cancel a queued job, but cannot stop a running HTTP
    call. That call retains its worker and is included in shutdown draining.
    """

    busy_error = GenerationBusyError
    closed_error = GenerationClosedError

    def __init__(
        self,
        options: ConcurrencyGenerationConfig = ConcurrencyGenerationConfig(),
    ) -> None:
        self.options = options
        super().__init__("Generation", options.max_concurrent, options.max_queue)
