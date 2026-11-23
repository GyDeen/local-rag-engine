"""Tests for selecting unique files from a larger chunk candidate set."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag_engine.cli import retrieve as cli
from rag_engine.engine.retrieve.pipeline import retrieve_relevant_chunks
from rag_engine.engine.vector_store.database_metadata_model import Chunk


def _chunk(chunk_id: str, source_path: str, section: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        embedding=[],
        document_id=source_path,
        source_path=[source_path],
        section=section,
        start_offset=0,
        end_offset=1,
        metadata={},
    )


def _embedder() -> Mock:
    embedder = Mock()
    embedder.encode.return_value.tolist.return_value = [1.0, 0.0]
    return embedder


def test_returns_highest_ranked_chunk_from_each_unique_file(monkeypatch):
    monkeypatch.setattr(
        "rag_engine.engine.retrieve.pipeline.prepare_retrieval",
        lambda query: SimpleNamespace(query_text="fault", metadata_filter=Mock()),
    )
    candidates = [
        _chunk("manual-a#0", "manual-a.md", "Reset worker"),
        _chunk("manual-a#1", "manual-a.md", "Reset worker"),
        _chunk("manual-b#0", "manual-b.md", "Reset worker"),
        _chunk("manual-a#2", "manual-a.md", "Check wiring"),
    ]
    database = Mock(spec=["similarity_search"])
    database.similarity_search.return_value = candidates

    results = retrieve_relevant_chunks(
        "Worker fault",
        _embedder(),
        database,
        chunk_limit=4,
        file_limit=3,
    )

    assert [chunk.chunk_id for chunk in results] == [
        "manual-a#0",
        "manual-b#0",
    ]
    assert database.similarity_search.call_args.kwargs["limit"] == 4


@pytest.mark.parametrize(
    "chunk_limit,file_limit,error",
    [
        (0, 1, "Chunk limit must be at least 1"),
        (5, 0, "File limit must be at least 1"),
        (2, 3, "Chunk limit must be at least the file limit"),
    ],
)
def test_rejects_invalid_retrieval_limits(chunk_limit, file_limit, error):
    with pytest.raises(ValueError, match=error):
        retrieve_relevant_chunks(
            "Worker fault",
            _embedder(),
            Mock(),
            chunk_limit=chunk_limit,
            file_limit=file_limit,
        )


def test_cli_configures_chunk_and_file_limits():
    default_arguments = cli._parse_arguments(["How do I reset the worker?"])
    assert default_arguments.chunk_limit == 50
    assert default_arguments.file_limit == 3

    arguments = cli._parse_arguments(
        ["How do I reset the worker?", "--chunk-limit", "8", "--file-limit", "2"]
    )

    assert arguments.chunk_limit == 8
    assert arguments.file_limit == 2


def test_section_limit_remains_a_compatibility_alias():
    arguments = cli._parse_arguments(
        ["How do I reset the worker?", "--section-limit", "2"]
    )

    assert arguments.file_limit == 2
