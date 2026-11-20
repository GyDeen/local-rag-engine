import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, cast

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from rag_engine.exceptions import ModelResponseError, RerankResponseError
from rag_engine.logging_config import configure_logging

from .backend import (
    BackendFactory,
    BackendHealth,
    ServiceBackend,
    ServiceBusyError,
    ServiceUnavailableError,
    UnknownAlarmError,
)
from .logging import RequestLoggingMiddleware
from .models import (
    AlarmSummary,
    ResolveRequest,
    ResolveResponse,
    RetrieveRequest,
    RetrieveResponse,
    Tier,
)

logger = logging.getLogger(__name__)


def create_app(backend_factory: BackendFactory | None = None) -> FastAPI:
    """Create one application whose lifespan owns one backend context"""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        logger.info("Service starting")

        application.state.phase = "starting"
        try:
            factory = backend_factory
            if factory is None:
                from rag_engine.serving.concurrency import create_backend

                factory = create_backend
            async with factory() as backend:
                application.state.backend = backend
                application.state.phase = "running"
                logger.info("Service ready")
                try:
                    yield
                finally:
                    application.state.phase = "stopping"
                    application.state.backend = None
        except Exception:
            logger.exception("Service lifespan failed")
            raise
        finally:
            application.state.backend = None
            application.state.phase = "stopped"
            logger.info("Service stopped")

    application = FastAPI(
        title="Local RAG Engine RAG service",
        version="0.1.0",
        description="Shared runtime resources with bounded generation concurrency.",
        lifespan=lifespan,
    )
    application.add_middleware(RequestLoggingMiddleware)
    application.state.backend = None
    application.state.phase = "stopped"

    @application.exception_handler(UnknownAlarmError)
    async def unknown_alarm(request: Request, error: UnknownAlarmError) -> JSONResponse:
        logger.info("Unknown alarm requested")
        return JSONResponse({"detail": str(error)}, status_code=404)

    @application.exception_handler(ServiceBusyError)
    @application.exception_handler(ServiceUnavailableError)
    async def unavailable(request: Request, error: Exception) -> JSONResponse:
        logger.warning("Request failed: %s", type(error).__name__, exc_info=True)
        status = (
            502 if isinstance(error, (ModelResponseError, RerankResponseError)) else 503
        )
        return JSONResponse(
            {"detail": getattr(error, "public_message", str(error))},
            status_code=status,
        )

    @application.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception) -> JSONResponse:
        # Request middleware already recorded the traceback with its request ID.
        identifier = getattr(request.state, "request_id", "-")
        return JSONResponse(
            {"detail": "Internal server error", "request_id": identifier},
            status_code=500,
            headers={"X-Request-ID": identifier},
        )

    def running_backend() -> ServiceBackend:
        backend = cast(ServiceBackend | None, application.state.backend)
        if backend is None:
            raise HTTPException(503, "Backend not running")
        return backend

    @application.get("/")
    async def index() -> dict[str, str | int]:
        alarms = running_backend().list_alarms()
        return {
            "service": "Local RAG Engine RAG service",
            "docs": "/docs",
            "alarms": len(alarms),
        }

    @application.get("/api/v1/alarms", response_model=list[AlarmSummary])
    async def list_alarms() -> list[AlarmSummary]:
        """Supply catalogue entries for the example alarm selector."""
        return running_backend().list_alarms()

    @application.get("/health/live")
    async def liveness() -> dict[str, str]:
        """Report HTTP process liveness without waiting for inference."""
        return {"status": "alive", "phase": application.state.phase}

    @application.get("/health/ready")
    async def readiness() -> JSONResponse:
        """Expose backend and inference availability from a cheap snapshot."""
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
            502: {"description": "Invalid ranking model response"},
            500: {"description": "Unexpected request failure"},
        },
    )
    async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
        """Return stitched documentation for the caller's answer generator."""
        return await running_backend().retrieve(request.model_dump(exclude_none=True))

    @application.post(
        "/api/v1/resolve",
        response_model=ResolveResponse,
        response_model_exclude_none=True,
        responses={
            404: {"description": "Unknown alarm code"},
            503: {"description": "Backend unavailable or admission capacity reached"},
            502: {"description": "Invalid model response"},
            500: {"description": "Unexpected request failure"},
        },
    )
    async def resolve(
        request: ResolveRequest,
        tier: Annotated[Tier, Header(alias="X-RAG-Tier")] = "operator",
    ) -> ResolveResponse:
        """Validate the public request and delegate generation to the backend."""
        backend = running_backend()
        return await backend.resolve(request.model_dump(exclude_none=True), tier)

    return application
