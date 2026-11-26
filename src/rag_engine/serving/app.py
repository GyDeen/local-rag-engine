import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from rag_engine.engine.vector_store.database_metadata_model import SearchFilters
from rag_engine.exceptions import ModelResponseError, RerankResponseError
from rag_engine.logging_config import configure_logging

from .backend import (
    BackendFactory,
    BackendHealth,
    ServiceBackend,
    ServiceBusyError,
    ServiceUnavailableError,
)
from .logging import RequestLoggingMiddleware
from .models import (
    QuestionRequest,
    QuestionResponse,
    ResolveRequest,
    RetrieveResponse,
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
        description="Shared runtime resources with bounded retrieval and generation concurrency.",
        lifespan=lifespan,
    )
    application.add_middleware(RequestLoggingMiddleware)
    application.state.backend = None
    application.state.phase = "stopped"

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
    async def index() -> dict[str, str]:
        return {
            "service": "Local RAG Engine RAG service",
            "docs": "/docs",
        }

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
            503: {"description": "Retrieval backend unavailable"},
            502: {"description": "Invalid ranking model response"},
            500: {"description": "Unexpected request failure"},
        },
    )
    async def retrieve(request: QuestionRequest) -> RetrieveResponse:
        """Return stitched documentation for the caller's answer generator."""
        result = await running_backend().ask(
            request.question,
            generate=False,
            filters=SearchFilters(request.filters) if request.filters else None,
        )
        return RetrieveResponse(question=result.question, evidence=result.evidence)

    @application.post(
        "/api/v1/ask",
        response_model=QuestionResponse,
        response_model_exclude_unset=True,
        responses={
            503: {"description": "Backend unavailable or admission capacity reached"},
            502: {"description": "Invalid model response"},
            500: {"description": "Unexpected request failure"},
        },
    )
    async def ask(request: QuestionRequest) -> QuestionResponse:
        """Ask a free-form question about the indexed documents."""
        return await running_backend().ask(
            request.question,
            generate=request.generate,
            filters=SearchFilters(request.filters) if request.filters else None,
            context=request.context,
        )

    @application.post(
        "/api/v1/resolve",
        response_model=QuestionResponse,
        response_model_exclude_unset=True,
        responses={
            503: {"description": "Backend unavailable or admission capacity reached"},
            502: {"description": "Invalid model response"},
            500: {"description": "Unexpected request failure"},
        },
    )
    async def resolve(request: ResolveRequest) -> QuestionResponse:
        """Answer a question with optional alarm report details."""
        context = request.model_dump(
            include={"code", "env", "context", "text"}, exclude_none=True
        )
        return await running_backend().ask(
            request.question,
            generate=request.generate,
            filters=SearchFilters(request.filters) if request.filters else None,
            context=context or None,
        )

    return application
