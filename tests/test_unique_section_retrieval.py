"""Verify limits for retrieving and selecting stitched evidence."""

from unittest.mock import Mock

import pytest

from rag_engine.cli import retrieve as cli
from rag_engine.engine.retrieve import retrieve_evidence
from rag_engine.engine.vector_store.database_metadata_model import Chunk


def _chunk(chunk_id: str, source_path: str, section: str, start: int = 0) -> Chunk:
    text = f"Evidence in {chunk_id}."
    return Chunk(
        chunk_id=chunk_id,
        embedding=[],
        document_id=source_path,
        source_path=[source_path],
        section=section,
        start_offset=start,
        end_offset=start + len(text),
        metadata={},
        text=text,
    )


def _embedder() -> Mock:
    embedder = Mock()
    embedder.encode.return_value.tolist.return_value = [1.0, 0.0]
    return embedder


def test_file_limit_keeps_multiple_passages_from_the_selected_file():
    candidates = [
        _chunk("manual-a#0", "manual-a.md", "Reset worker"),
        _chunk("manual-a#1", "manual-a.md", "Reset worker", start=25),
        _chunk("manual-b#0", "manual-b.md", "Reset worker"),
        _chunk("manual-a#2", "manual-a.md", "Check wiring", start=50),
    ]
    database = Mock(spec=["similarity_search"])
    database.similarity_search.return_value = candidates

    result = retrieve_evidence(
        query_text="Worker fault",
        embedder=_embedder(),
        database=database,
        reranker="none",
        chunk_limit=4,
        file_limit=1,
        evidence_min_chunk_chars=0,
    )

    assert [candidate.all_chunk_ids() for candidate in result.evidence] == [
        ("manual-a#0", "manual-a#1"),
        ("manual-a#2",),
    ]
    assert database.similarity_search.call_args.kwargs["limit"] == 4


@pytest.mark.parametrize(
    "chunk_limit,file_limit,error",
    [
        (0, 1, "Chunk limit must be at least the file limit"),
        (5, 0, "File limit must be at least 1"),
        (2, 3, "Chunk limit must be at least the file limit"),
    ],
)
def test_rejects_invalid_retrieval_limits(chunk_limit, file_limit, error):
    with pytest.raises(ValueError, match=error):
        retrieve_evidence(
            query_text="Worker fault",
            embedder=_embedder(),
            database=Mock(),
            reranker="none",
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
