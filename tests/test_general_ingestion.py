"""Verify shared ingestion, indexing provenance and retrieval evaluation."""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from rag_engine.benchmarks.run import _coverage, _run_case
from rag_engine.engine.rerank.selection import stitch_evidence
from rag_engine.engine.vector_store.backends import open_database
from rag_engine.engine.vector_store.create_database import create_database
from rag_engine.engine.vector_store.database_metadata_model import SearchFilters
from rag_engine.ingestion import ingest_documents
from rag_engine.ingestion.convert import attach_provenance


class Embedder:
    def __init__(self):
        self.queries = []

    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            self.queries.append(texts)
            return np.array([1.0, 0.0])
        return np.array([[1.0, 0.0] for _ in texts])


def test_ingest_and_index_preserve_source_names(tmp_path):
    source = tmp_path / "incoming"
    source.mkdir()
    (source / "guide.txt").write_text(
        "Turn off the power before replacing the motor.\n"
    )
    (source / "guide.md").write_text("# Diagnostics\n\nCheck the motor wiring first.\n")
    # An existing Markdown filename must not collide with converted text.
    (source / "guide.txt.md").write_text(
        "# Separate document\n\nKeep this content too.\n"
    )
    output = tmp_path / "normalized"
    manifest = ingest_documents(source, output)
    assert len(manifest["documents"]) == 3
    assert json.loads((output / "debug/manifest.json").read_text()) == manifest
    assert (output / "documents/guide.md.md").read_text() == (
        source / "guide.md"
    ).read_text()
    embedder = Embedder()
    database_path = tmp_path / "index"
    create_database(
        embedder,
        output / "documents",
        database_path=database_path,
        collection_name="general",
        minimum_chunk_size=1,
    )
    database = open_database(database_path=database_path, collection_name="general")
    chunks = database.similarity_search([1.0, 0.0], SearchFilters(), limit=3)
    assert {chunk.metadata["source_document"] for chunk in chunks} == {
        "guide.txt",
        "guide.md",
        "guide.txt.md",
    }
    assert all(chunk.text for chunk in chunks)


def test_failed_ingestion_does_not_publish_partial_corpus(tmp_path):
    source = tmp_path / "incoming"
    source.mkdir()
    (source / "first.md").write_text("# Valid\nContent")
    (source / "second.txt").write_bytes(b"\xff\xfeinvalid utf8")
    destination = tmp_path / "normalized"
    with pytest.raises(UnicodeError):
        ingest_documents(source, destination)
    assert not destination.exists()


def test_ingest_can_publish_into_a_prepared_corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    guide = raw / "guide.txt"
    guide.write_text("Turn off the power before replacing the motor.\n")
    labels = tmp_path / "cases.jsonl"
    labels.write_text("{}\n")

    ingest_documents(raw, tmp_path)

    assert (tmp_path / "documents/guide.txt.md").read_text() == guide.read_text()
    assert (tmp_path / "debug/manifest.json").is_file()
    assert labels.read_text() == "{}\n"
    with pytest.raises(FileExistsError):
        ingest_documents(raw, tmp_path)


def test_provenance_survives_cross_page_section_stitching(tmp_path):
    from rag_engine.engine.rerank.candidate import build_rerank_candidates
    from rag_engine.engine.vector_store.data_chunking import chunk_documents

    docs = tmp_path / "docs"
    docs.mkdir()
    text = (
        "# Maintenance\n\n"
        + ("Disconnect power before work. " * 15)
        + ("Reconnect after inspection. " * 15)
    )
    (docs / "manual.pdf.md").write_text(text)
    midpoint = len(text) // 2
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": {
                    "manual.pdf.md": {
                        "source": "manual.pdf",
                        "location": "manual.pdf",
                        "page_spans": [
                            {"start": 0, "end": midpoint, "page": 3},
                            {"start": midpoint, "end": len(text), "page": 4},
                        ],
                    }
                }
            }
        )
    )
    chunks = chunk_documents(docs, chunk_size=200, overlap=20, minimum_chunk_size=1)
    attach_provenance(chunks, manifest)
    pool = build_rerank_candidates(chunks, docs)
    stitched = stitch_evidence(pool, pool, file_limit=1, char_budget=2000)
    assert len(stitched) == 1
    assert stitched[0].chunk.metadata["source_pages"] == [3, 4]


def test_benchmark_passes_only_the_question_to_retrieval():
    retrieve = Mock(return_value={"question": "What next?", "evidence": []})
    case = {
        "id": "case",
        "question": "What next?",
        "answerable": True,
        "answer": "Expected answer",
        "references": [{"source": "guide.pdf", "page": 9}],
    }
    result = _run_case(case, retrieve)
    retrieve.assert_called_once_with("What next?")
    assert result["gold_source_recall"] == 0
    assert result["gold_page_recall"] == 0


def test_correct_manual_without_correct_page_does_not_pass_page_coverage():
    coverage = _coverage(
        {"references": [{"source": "manual.pdf", "page": 9}]},
        [{"source": "manual.pdf", "pages": [1, 2]}],
    )
    assert coverage == {
        "gold_source_recall": 1.0,
        "gold_page_recall": 0.0,
        "gold_section_recall": None,
    }


def test_section_coverage_matches_only_labelled_sections_from_the_correct_file():
    coverage = _coverage(
        {"references": [{"source": "guide.html", "section": "speed_change"}]},
        [{"source": "other.html", "pages": [], "section": "2. Speed Change"}],
    )
    assert coverage["gold_section_recall"] == 0

    coverage = _coverage(
        {"references": [{"source": "guide.html", "section": "speed_change"}]},
        [
            {
                "source": "guide.html",
                "pages": [],
                "section": "Guide > 1. Starting | Guide > 2.1. Speed Change",
            }
        ],
    )
    assert coverage["gold_section_recall"] == 1
