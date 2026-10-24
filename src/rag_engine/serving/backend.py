from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from rag_engine.serving.models import (
    AlarmSummary,
    ResolveResponse,
    RetrieveResponse,
    Tier,
)


@dataclass(frozen=True, slots=True)
class BackendHealth:

    ready: bool
    cache_ready: bool
    inference_ready: bool
    detail: str | None = None


class UnknownAlarmError(Exception):
    """The requested alarm is absent from the loaded catalogue."""


class ServiceBusyError(Exception):
    """The backend cannot admit another request within its configured limits."""


class ServiceUnavailableError(Exception):
    """The backend cannot serve this request in its current dependency state."""


class ServiceBackend(Protocol):

    def list_alarms(self) -> list[AlarmSummary]:
        """Return the loaded alarm catalogue."""
        ...

    def health(self) -> BackendHealth:
        """Return service health status"""
        ...

    async def retrieve(self, input_report: Mapping[str, Any]) -> RetrieveResponse:
        """Return selected documentation without generating an answer."""
        ...

    async def resolve(
        self, input_report: Mapping[str, Any], tier: Tier
    ) -> ResolveResponse:
        """Await inference and return the shared resolution response."""
        ...


BackendFactory = Callable[[], AbstractAsyncContextManager[ServiceBackend]]


class UnconfiguredBackend:
    """Explicit placeholder: starts the HTTP shell without loading RAG models."""

    def health(self) -> BackendHealth:
        """Keep readiness false until a real backend is connected."""
        return BackendHealth(False, False, False, "Concurrency backend not connected")

    def list_alarms(self) -> list[AlarmSummary]:
        raise ServiceUnavailableError("Concurrency backend not connected")

    async def retrieve(self, input_report: Mapping[str, Any]) -> RetrieveResponse:
        """Reject retrieval when no runtime is connected."""
        raise ServiceUnavailableError("Concurrency backend not connected")

    async def resolve(
        self, input_report: Mapping[str, Any], tier: Tier
    ) -> ResolveResponse:
        """Reject inference when the placeholder is explicitly selected."""
        raise ServiceUnavailableError("Concurrency backend not connected")


@asynccontextmanager
async def unconfigured_backend() -> AsyncIterator[ServiceBackend]:
    """Provide an explicitly selected placeholder without loading models."""
    yield UnconfiguredBackend()
