"""Admission limits for blocking retrieval calls."""

import os
from dataclasses import dataclass
from typing import Any

from rag_engine.settings import (
    DEFAULT_RETRIEVAL_CONCURRENCY,
    DEFAULT_RETRIEVAL_QUEUE_SIZE,
)

from .workers import WorkerPool


@dataclass(frozen=True, slots=True)
class ConcurrencyRetrievalConfig:
    max_concurrent: int = DEFAULT_RETRIEVAL_CONCURRENCY
    max_queue: int = DEFAULT_RETRIEVAL_QUEUE_SIZE

    def __post_init__(self) -> None:
        if self.max_concurrent < 1:
            raise ValueError("Retrieval concurrency must be at least 1")
        if self.max_queue < 0:
            raise ValueError("Retrieval queue size cannot be negative")

    @classmethod
    def from_env(cls) -> ConcurrencyRetrievalConfig:
        return cls(
            max_concurrent=int(
                os.getenv("RAG_RETRIEVAL_CONCURRENCY", str(DEFAULT_RETRIEVAL_CONCURRENCY))
            ),
            max_queue=int(
                os.getenv("RAG_RETRIEVAL_QUEUE_SIZE", str(DEFAULT_RETRIEVAL_QUEUE_SIZE))
            ),
        )


class RetrievalPool(WorkerPool[dict[str, Any]]):
    def __init__(self, options: ConcurrencyRetrievalConfig) -> None:
        super().__init__("Retrieval", options.max_concurrent, options.max_queue)
