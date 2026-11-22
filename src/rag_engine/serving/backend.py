from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from rag_engine.engine.vector_store.database_metadata_model import SearchFilters
from rag_engine.exceptions import (
    ServiceBusyError as ServiceBusyError,
)
from rag_engine.exceptions import (
    ServiceUnavailableError as ServiceUnavailableError,
)
from rag_engine.serving.models import QuestionResponse


@dataclass(frozen=True, slots=True)
class BackendHealth:

    ready: bool
    cache_ready: bool
    inference_ready: bool
    detail: str | None = None


class ServiceBackend(Protocol):

    def health(self) -> BackendHealth:
        """Return service health status"""
        ...

    async def ask(
        self,
        question: str,
        *,
        generate: bool,
        filters: SearchFilters | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> QuestionResponse:
        """Search the configured documents and optionally generate a cited answer."""
        ...


BackendFactory = Callable[[], AbstractAsyncContextManager[ServiceBackend]]


class UnconfiguredBackend:
    """Explicit placeholder: starts the HTTP shell without loading RAG models."""

    def health(self) -> BackendHealth:
        """Keep readiness false until a real backend is connected."""
        return BackendHealth(False, False, False, "Concurrency backend not connected")

    async def ask(
        self,
        question: str,
        *,
        generate: bool,
        filters: SearchFilters | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> QuestionResponse:
        raise ServiceUnavailableError("Concurrency backend not connected")


@asynccontextmanager
async def unconfigured_backend() -> AsyncIterator[ServiceBackend]:
    """Provide an explicitly selected placeholder without loading models."""
    yield UnconfiguredBackend()
