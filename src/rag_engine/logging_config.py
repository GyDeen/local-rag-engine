import logging
import os
import sys
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from time import perf_counter
from typing import ParamSpec, TypeVar
from urllib.error import HTTPError

from rag_engine.exceptions import RagError, ServiceBusyError

request_id: ContextVar[str] = ContextVar("request_id", default="-")
P = ParamSpec("P")
R = TypeVar("R")


class RequestContextFilter(logging.Filter):
    """Attach the current request ID, including in copied worker contexts."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True


class ConsoleHandler(logging.StreamHandler):
    """Follow stderr when a host redirects or replaces its output stream."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


def configure_logging() -> None:
    """Configure RAG stderr logging once, leaving host loggers intact."""
    level = os.getenv("RAG_LOG_LEVEL", "INFO").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(
            "RAG_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL"
        )
    logger = logging.getLogger("rag_engine")
    logger.setLevel(level)
    if any(getattr(handler, "_rag_handler", False) for handler in logger.handlers):
        return
    handler = ConsoleHandler()
    handler._rag_handler = True
    handler.addFilter(RequestContextFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s: %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.propagate = False


def log_operation(
    name: str,
    *,
    error_type: type[RagError] | None = None,
    level: int = logging.INFO,
    handled_errors: tuple[type[Exception], ...] = (OSError, RuntimeError),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Log stage timing without arguments or model content.

    Optionally translate OS/network and runtime dependency failures. Existing
    application errors retain their type. Programming/validation failures
    propagate unchanged. Tracebacks belong at the API or command boundary.
    """

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        module_name = function.__module__
        if module_name == "__main__":
            spec = getattr(sys.modules.get(module_name), "__spec__", None)
            if spec is not None:
                module_name = spec.name
        logger = logging.getLogger(module_name)

        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            started = perf_counter()
            logger.log(level, "%s started", name)
            try:
                result = function(*args, **kwargs)
            except Exception as error:
                logger.warning(
                    "%s failed error_type=%s duration_ms=%.1f",
                    name,
                    type(error).__name__,
                    (perf_counter() - started) * 1000,
                )
                if (
                    error_type is not None
                    and not isinstance(error, RagError)
                    and isinstance(error, handled_errors)
                ):
                    if isinstance(error, HTTPError) and error.code == 503:
                        raise ServiceBusyError("Dependency service is busy") from error
                    raise error_type(f"{name} failed") from error
                raise
            logger.log(
                level,
                "%s completed duration_ms=%.1f",
                name,
                (perf_counter() - started) * 1000,
            )
            return result

        return wrapped

    return decorate
