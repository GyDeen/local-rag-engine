from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rag_engine.settings import DEFAULT_DATABASE_BACKEND

from .base import VectorDatabaseAdapter, VectorSearch


@dataclass(frozen=True)
class DatabaseBackend:
    """Separate factories for creating a new index and opening an existing one."""

    create: Callable[..., VectorDatabaseAdapter]
    open: Callable[..., VectorSearch]


def _create_chroma(**settings: Any) -> VectorDatabaseAdapter:
    from .Chroma_API import create_chroma_database

    return create_chroma_database(**settings)


def _open_chroma(**settings: Any) -> VectorSearch:
    from .Chroma_API import open_chroma_database

    return open_chroma_database(**settings)


_BACKENDS = {"chroma": DatabaseBackend(_create_chroma, _open_chroma)}


def _normalize_backend_name(backend: str) -> str:
    normalized = backend.strip().casefold()
    if not normalized:
        raise ValueError("Database backend name must not be empty")
    return normalized


def register_database_backend(
    backend: str,
    builder: Callable[..., VectorDatabaseAdapter],
    opener: Callable[..., VectorSearch],
    *,
    replace: bool = False,
) -> None:
    """Register independent create and open factories for a backend."""

    normalized = _normalize_backend_name(backend)
    if normalized in _BACKENDS and not replace:
        raise ValueError(f"Database backend is already registered: {normalized}")
    _BACKENDS[normalized] = DatabaseBackend(builder, opener)


def available_database_backends() -> tuple[str, ...]:
    """Return the registered backend names."""

    return tuple(sorted(_BACKENDS))


def get_database_backend(backend: str) -> DatabaseBackend:
    """Resolve a backend without connecting to it."""

    try:
        return _BACKENDS[_normalize_backend_name(backend)]
    except KeyError as error:
        available = ", ".join(available_database_backends())
        raise ValueError(
            f"Unknown database backend {backend!r}. Available backends: {available}"
        ) from error


def open_database(
    backend: str = DEFAULT_DATABASE_BACKEND,
    **settings: Any,
) -> VectorSearch:
    """Open an existing index; backend openers must never create or populate it."""

    return get_database_backend(backend).open(**settings)
