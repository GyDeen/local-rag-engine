import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from rag_engine.engine.vector_store.database_metadata_model import SearchFilters


class PreparedRetrieval(BaseModel):
    code: str = Field(min_length=1)
    env: SearchFilters | None = None
    context: dict[str, Any] | str | None = None
    text: str | None = None

    _query_text: str = PrivateAttr(default="")
    _metadata_filter: SearchFilters = PrivateAttr(default_factory=SearchFilters)

    @property
    def query_text(self) -> str:
        """Return the query produced by ``prepare_retrieval``"""
        return self._query_text

    @property
    def metadata_filter(self) -> SearchFilters:
        """Return enabled search criteria produced by ``prepare_retrieval``"""
        return self._metadata_filter


def load_input_report(path: str | Path) -> dict[str, Any]:
    """Load one alarm report from a JSON file."""

    with Path(path).open(encoding="utf-8") as input_file:
        document = json.load(input_file)

    if not isinstance(document, dict):
        raise TypeError("Alarm report must be a JSON object")

    return document


def load_alarm_catalog(path: str | Path) -> dict[str, dict]:
    """Load alarm definitions indexed by alarm code"""

    with Path(path).open(encoding="utf-8") as input_file:
        document = json.load(input_file)

    return {alarm["code"]: alarm for alarm in document["alarms"]}


def get_alarm_definition(
    catalog: dict[str, dict],
    code: str,
) -> dict:
    """Return the definition associated with an alarm code.
    The definition is a dictionary with keys: code, title, alarm_text, domain."""

    if code not in catalog:
        raise KeyError(f"Unknown alarm code: {code}")

    return catalog[code]


def prepare_retrieval(
    input_report: Mapping[str, Any],
    catalog: dict[str, dict],
) -> PreparedRetrieval:
    """Prepare query text and metadata filters from an alarm report."""
    prepared = PreparedRetrieval.model_validate(input_report)
    code = prepared.code

    error_code_def = get_alarm_definition(catalog, code)
    if not error_code_def:
        raise ValueError(f"Unknown alarm code: {code}")

    # Runtime error reported by the QR code
    runtime_text = prepared.text
    if runtime_text == error_code_def.get("alarm_text"):
        runtime_text = None

    # Optional context about user behavior or environment
    context = prepared.context

    query = [
        error_code_def.get("title", ""),
        error_code_def.get("alarm_text", ""),
        runtime_text,
        context,
    ]

    environment = asdict(prepared.env) if prepared.env is not None else {}
    filter_values = {
        name: value
        for source in (error_code_def, environment, input_report)
        for name, value in source.items()
        if value
    }
    prepared._metadata_filter = SearchFilters.from_mapping(filter_values)
    prepared._query_text = "\n".join(str(value) for value in query if value)
    return prepared


def load_merged_catalog(paths: Iterable[Path]) -> dict[str, dict]:
    """Merge several alarm catalogue files into one code -> alarm mapping."""

    catalog: dict[str, dict] = {}
    for path in paths:
        for code, alarm in load_alarm_catalog(path).items():
            if code in catalog:
                raise ValueError(
                    f"Alarm code appears in more than one catalogue: {code}"
                )
            catalog[code] = alarm
    return catalog
