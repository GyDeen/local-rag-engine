from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, cast

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from .backend import (
    BackendFactory,
    BackendHealth,
    ServiceBackend,
    ServiceBusyError,
    ServiceUnavailableError,
    UnknownAlarmError,
)
from .models import (
    AlarmSummary,
    ResolveRequest,
    ResolveResponse,
    RetrieveRequest,
    RetrieveResponse,
    Tier,
)


def create_app(backend_factory: BackendFactory | None = None) -> FastAPI:
    """Create one application whose lifespan owns one backend context"""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        factory = backend_factory
        if factory is None:
            from rag_engine.serving.concurrency import create_backend

            factory = create_backend
        application.state.phase = "starting"
        try:
            async with factory() as backend:
                application.state.backend = backend
                application.state.phase = "running"
                try:
                    yield
                finally:
                    application.state.phase = "stopping"
                    application.state.backend = None
        finally:
            application.state.backend = None
            application.state.phase = "stopped"

    application = FastAPI(
        title="Local RAG Engine RAG service",
        version="0.1.0",
        description="Shared runtime resources with bounded generation concurrency.",
        lifespan=lifespan,
    )
    application.state.backend = None
    application.state.phase = "stopped"

    def running_backend() -> ServiceBackend:
        backend = cast(ServiceBackend | None, application.state.backend)
        if backend is None:
            raise HTTPException(503, "Backend not running")
        return backend

    @application.get("/")
    async def index() -> dict[str, str | int]:
        try:
            alarms = running_backend().list_alarms()
        except ServiceUnavailableError as error:
            raise HTTPException(503, str(error)) from error
        return {
            "service": "Local RAG Engine RAG service",
            "docs": "/docs",
            "alarms": len(alarms),
        }

    @application.get("/api/v1/alarms", response_model=list[AlarmSummary])
    async def list_alarms() -> list[AlarmSummary]:
        """Supply catalogue entries for the example alarm selector."""
        try:
            return running_backend().list_alarms()
        except ServiceUnavailableError as error:
            raise HTTPException(503, str(error)) from error

    @application.get("/health/live")
    async def liveness() -> dict[str, str]:
        """Report HTTP process liveness without waiting for inference."""
        return {"status": "alive", "phase": application.state.phase}

    @application.get("/health/ready")
    async def readiness() -> JSONResponse:
        """Expose separate cache and inference availability from a cheap snapshot."""
        backend = application.state.backend
        health = (
            backend.health()
            if backend is not None
            else BackendHealth(False, False, False, "Backend not running")
        )
        return JSONResponse(
            {"phase": application.state.phase, **asdict(health)},
            status_code=200 if health.ready else 503,
        )

    @application.post(
        "/api/v1/retrieve",
        response_model=RetrieveResponse,
        responses={
            404: {"description": "Unknown alarm code"},
            503: {"description": "Retrieval backend unavailable"},
        },
    )
    async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
        """Return stitched documentation for the caller's answer generator."""
        try:
            return await running_backend().retrieve(
                request.model_dump(exclude_none=True)
            )
        except UnknownAlarmError as error:
            raise HTTPException(404, str(error)) from error
        except (ServiceBusyError, ServiceUnavailableError) as error:
            raise HTTPException(503, str(error)) from error

    @application.post(
        "/api/v1/resolve",
        response_model=ResolveResponse,
        response_model_exclude_none=True,
        responses={
            404: {"description": "Unknown alarm code"},
            503: {"description": "Backend unavailable or admission capacity reached"},
        },
    )
    async def resolve(
        request: ResolveRequest,
        tier: Annotated[Tier, Header(alias="X-RAG-Tier")] = "operator",
    ) -> ResolveResponse:
        """Validate the public request and delegate generation to the backend."""
        backend = running_backend()
        try:
            return await backend.resolve(request.model_dump(exclude_none=True), tier)
        except UnknownAlarmError as error:
            raise HTTPException(404, str(error)) from error
        except (ServiceBusyError, ServiceUnavailableError) as error:
            raise HTTPException(503, str(error)) from error

    return application
