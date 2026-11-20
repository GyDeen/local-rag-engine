from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Self, TypeAlias, get_args, get_origin, get_type_hints

"""Allowed metadata value types for database metadata."""
MetadataValue: TypeAlias = (
    str | int | float | bool | list[str] | list[int] | list[float] | list[bool]
)

FilterValue: TypeAlias = str | tuple[str, ...]


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
    """Shared machine environment and optional search metadata.

    Environment values can be carried by an alarm report even when the source
    documents do not contain corresponding metadata. Disabled fields are ignored
    during both filter preparation and query creation.
    """

    machine_type: str | tuple[str, ...] | None = field(
        default=None, metadata={"enabled": False}
    )
    domain: str | None = field(default=None, metadata={"enabled": False})
    component: str | None = field(default=None, metadata={"enabled": False})
    see_also: str | tuple[str, ...] | None = field(
        default=None, metadata={"enabled": False}
    )
    versions: dict[str, str] | None = field(default=None, metadata={"enabled": False})
    locale: str | None = field(default=None, metadata={"enabled": False})
    os: str | None = field(default=None, metadata={"enabled": False})
    machine_variant: str | None = field(default=None, metadata={"enabled": False})
    serial: str | None = field(default=None, metadata={"enabled": False})

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Self:
        """Select enabled filter fields and normalize their sequence values."""
        annotations = get_type_hints(cls)
        selected = {}
        for filter_field in fields(cls):
            if not filter_field.metadata.get("enabled", True):
                continue
            name = filter_field.name
            value = values.get(name)
            if not value:
                continue
            if any(
                get_origin(option) is tuple for option in get_args(annotations[name])
            ):
                value = (value,) if isinstance(value, str) else tuple(value)
            selected[name] = value
        return cls(**selected)

    def active_fields(self) -> Iterator[tuple[str, FilterValue]]:
        """Yield only fields with usable values; absent values do not filter."""
        for filter_field in fields(self):
            if not filter_field.metadata.get("enabled", True):
                continue
            value = getattr(self, filter_field.name)
            if isinstance(value, tuple):
                value = tuple(item for item in value if item and item.strip())
            if not value or (isinstance(value, str) and not value.strip()):
                continue
            yield filter_field.name, value
