from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Annotated, TypeAlias

from pydantic import Field, FiniteFloat, StrictBool, StrictStr, StringConstraints

"""Allowed metadata value types for database metadata."""
MetadataValue: TypeAlias = (
    str | int | float | bool | list[str] | list[int] | list[float] | list[bool]
)

MetadataKey = Annotated[str, StringConstraints(pattern=r"^[^$\s](?:[^\r\n]*\S)?$")]
FilterValue: TypeAlias = (
    StrictStr
    | StrictBool
    | Annotated[int, Field(strict=True, ge=-(2**63), lt=2**63)]
    | Annotated[FiniteFloat, Field(strict=True)]
)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    embedding: list[float]

    document_id: str
    source_path: list[str]
    section: str  # Potential multiple level section hierarchy
    start_offset: int
    end_offset: int
    metadata: dict[str, MetadataValue]
    text: str | None = None


@dataclass(slots=True, frozen=True)
class SearchFilters:
    """Explicit metadata constraints for a document search."""

    criteria: Mapping[MetadataKey, FilterValue] = field(default_factory=dict)

    def active_fields(self) -> Iterator[tuple[str, FilterValue]]:
        return iter(self.criteria.items())
