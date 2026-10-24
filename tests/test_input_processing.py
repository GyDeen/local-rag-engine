"""Verify the typed filter contract between input parsing and retrieval."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag_engine.engine.parse_input.input_processing import (
    PreparedRetrieval,
    prepare_retrieval,
)
from rag_engine.engine.retrieve import pipeline
from rag_engine.engine.vector_store.database_metadata_model import SearchFilters


@pytest.mark.parametrize(
    "environment",
    [
        {"machine_type": "MX7", "machine_variant": "FX5"},
        {"machine_type": ["MX7", "FX5"], "see_also": "G01"},
        {"machine_variant": "FX5"},
        {
            "versions": {"core": "1.11.0"},
            "locale": "en-US",
            "os": "Windows",
            "machine_variant": "FX5",
            "serial": "machine-123",
        },
        {},
    ],
)
def test_disabled_metadata_fields_do_not_create_filters(environment):
    prepared = prepare_retrieval(
        {
            "code": "demo.svc.0001",
            "env": {**environment, "component": "runtime-component"},
            "text": "Service fault",
            "context": "During startup",
        },
        {
            "demo.svc.0001": {
                "title": "Worker alarm",
                "alarm_text": "Service fault",
                "domain": "service",
                "component": "worker",
            }
        },
    )

    assert prepared.metadata_filter == SearchFilters()
    assert prepared.query_text == "Worker alarm\nService fault\nDuring startup"
    assert list(prepared.metadata_filter.active_fields()) == []


def test_disabled_component_does_not_create_filter():
    prepared = prepare_retrieval(
        {"code": "alarm", "env": {"component": "worker"}},
        {"alarm": {"title": "Fault"}},
    )
    assert prepared.metadata_filter == SearchFilters()


def test_absent_metadata_produces_no_active_filters():
    prepared = prepare_retrieval({"code": "alarm"}, {"alarm": {"title": "Fault"}})
    assert list(prepared.metadata_filter.active_fields()) == []


def test_preparation_creates_shared_input_without_mutating_the_report():
    report = {
        "code": "alarm",
        "text": "Service fault",
        "env": {"versions": {"core": "1.11.0"}, "locale": "en-US"},
        "context": {"operation": "startup"},
    }
    catalog = {"alarm": {"title": "Worker alarm", "alarm_text": "Service fault"}}

    first = prepare_retrieval(report, catalog)
    second = prepare_retrieval(report, {"alarm": {"title": "Queue alarm"}})

    assert isinstance(first, PreparedRetrieval)
    assert first.query_text == "Worker alarm\nService fault\n{'operation': 'startup'}"
    assert second.query_text == "Queue alarm\nService fault\n{'operation': 'startup'}"
    assert "query_text" not in report
    assert first.env.versions == report["env"]["versions"]
    assert first.context is not report["context"]
    assert first.metadata_filter == SearchFilters()
    assert first.model_dump(exclude_none=True) == report


def test_prepared_values_are_not_accepted_from_the_report():
    report = {
        "code": "alarm",
        "query_text": "Use a different query",
        "metadata_filter": {"domain": "service"},
    }
    prepared = prepare_retrieval(report, {"alarm": {"title": "Service fault"}})
    assert prepared.query_text == "Service fault"
    assert prepared.metadata_filter == SearchFilters()
    assert prepared.model_dump(exclude_none=True) == {"code": "alarm"}


def test_supplied_environment_does_not_add_database_filters():
    from rag_engine.engine.vector_store.Chroma_API import ChromaVectorDatabase

    filters = SearchFilters(
        versions={"core": "1.11.0"},
        locale="en-US",
        os="Windows",
        machine_variant="FX5",
        serial="machine-123",
    )
    assert ChromaVectorDatabase._extract_filters(filters) is None


def test_retrieval_passes_prepared_filters_directly_to_database(monkeypatch):
    filters = SearchFilters(machine_type=("MX7",), see_also=("G01",))
    monkeypatch.setattr(
        pipeline,
        "prepare_retrieval",
        lambda report, catalog: SimpleNamespace(
            code="demo.svc.0001", query_text="Worker fault", metadata_filter=filters
        ),
    )
    embedder = Mock()
    embedder.encode.return_value.tolist.return_value = [1.0, 0.0]
    database = Mock(spec=["similarity_search"])
    database.similarity_search.return_value = []

    assert pipeline.retrieve_relevant_chunks({}, {}, embedder, database) == []
    assert database.similarity_search.call_args.kwargs["filters"] is filters
