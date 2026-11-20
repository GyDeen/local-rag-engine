import logging
from time import perf_counter
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_engine.logging_config import request_id

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware:
    """Assign a server-generated ID and log status, duration, and uncaught errors."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        identifier = uuid4().hex
        token = request_id.set(identifier)
        scope.setdefault("state", {})["request_id"] = identifier
        started = perf_counter()
        status = 500

        async def send_response(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append(
                    (b"x-request-id", identifier.encode("ascii"))
                )
            await send(message)

        try:
            await self.app(scope, receive, send_response)
        except Exception:
            logger.exception("Unhandled request failure")
            raise
        finally:
            logger.info(
                "HTTP request completed method=%s path=%s status=%s duration_ms=%.1f",
                scope["method"],
                scope["path"],
                status,
                (perf_counter() - started) * 1000,
            )
            request_id.reset(token)
