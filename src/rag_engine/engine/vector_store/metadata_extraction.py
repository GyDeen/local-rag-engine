"""Extract configured document metadata from labelled Markdown lines."""

import logging
import re
from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from .database_metadata_model import FilterValue

logger = logging.getLogger(__name__)
_LABELLED_VALUE = re.compile(
    r"^\s*(?:[-*+]\s+)?(?:\*\*(?P<bold>[^:]+?)(?::\*\*|\*\*:)|"
    r"(?P<plain>[^:]+):)\s*(?P<value>\S.*)$"
)


class MetadataFieldRule(BaseModel):
    labels: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    ] = Field(min_length=1)
    type: Literal["string", "integer", "number", "boolean"] = "string"


def _parse_value(value: str, kind: str) -> FilterValue:
    if kind == "integer":
        number = int(value)
        if not -(2**63) <= number < 2**63:
            raise ValueError("Integer outside the supported metadata range")
        return number
    if kind == "number":
        number = float(value)
        if not isfinite(number):
            raise ValueError("Numeric metadata must be finite")
        return number
    if kind == "boolean":
        flag = value.casefold()
        if flag not in {"true", "false"}:
            raise ValueError("Boolean metadata must be true or false")
        return flag == "true"
    return value


def extract_document_metadata(
    lines: Iterable[str], rules: Mapping[str, MetadataFieldRule], document_id: str
) -> dict[str, FilterValue]:
    """Use a field only when its labelled values agree across the document."""
    required = {label.casefold() for rule in rules.values() for label in rule.labels}
    labelled: dict[str, set[str]] = {}
    # Searching supplied document whether it has desired pattern
    for line in lines:
        match = _LABELLED_VALUE.match(line)
        if match is None:
            continue
        label = (match["bold"] or match["plain"]).strip().casefold()
        if label in required:
            labelled.setdefault(label, set()).add(match["value"].strip())

    metadata: dict[str, FilterValue] = {}
    # Parse the extracted metadata into desire type
    for name, rule in rules.items():
        candidates = set().union(
            *(labelled.get(label.casefold(), ()) for label in rule.labels)
        )
        if not candidates:
            continue
        try:
            values = {_parse_value(value, rule.type) for value in candidates}
            if len(values) == 1:
                metadata[name] = values.pop()
            else:
                logger.warning("Conflicting metadata field %s in %s", name, document_id)
        except ValueError:
            logger.warning(
                "Invalid %s metadata field %s in %s", rule.type, name, document_id
            )
    return metadata
