"""Regression tests for the indexing/retrieval lifecycle boundary."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import chromadb
import numpy as np
import pytest
from chromadb.errors import InternalError

from rag_engine.cli import index as index_cli
from rag_engine.cli import retrieve as cli
from rag_engine.engine.retrieve import retrieve_evidence
from rag_engine.engine.vector_store import backends
from rag_engine.engine.vector_store.backends import (
    open_database,
    register_database_backend,
)
from rag_engine.engine.vector_store.Chroma_API import create_chroma_database
from rag_engine.engine.vector_store.create_database import create_database
from rag_engine.engine.vector_store.database_metadata_model import SearchFilters
from rag_engine.settings import DEFAULT_COLLECTION_NAME


class FakeEmbedder:
    """Deterministic embeddings without model downloads."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings=True):
        self.calls.append(texts)
        if isinstance(texts, str):
            return np.array([1.0, 0.0, 0.0])
        return np.array([[1.0, 0.0, 0.0] for _ in texts])


@pytest.fixture
def source_docs(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "manual.md").write_text(
        "# Worker\nCheck the worker settings and reset the worker.\n", encoding="utf-8"
    )
    return docs


def test_build_once_then_retrieve_without_sources_or_document_embedding(
    tmp_path, source_docs
):
    path = tmp_path / "index"
    embedder = FakeEmbedder()
    count = create_database(
        embedder,
        source_docs,
        database_path=path,
        metadata_by_document={"manual.md": {"domain": "service"}},
    )
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_collection(DEFAULT_COLLECTION_NAME, embedding_function=None)
    before = collection.get(include=["metadatas", "embeddings"])
    assert count == collection.count() == 1
    assert embedder.calls == [
        ["# Worker\nCheck the worker settings and reset the worker."]
    ]

    # Retrieval must not read or chunk the source corpus again.
    (source_docs / "manual.md").unlink()
    for _ in range(2):
        database = open_database(database_path=path)
        result = retrieve_evidence(
            query_text="Worker fault",
            embedder=embedder,
            database=database,
            reranker="none",
            file_limit=1,
            docs_root=source_docs,
            evidence_min_chunk_chars=0,
        )
        (evidence,) = result.evidence
        assert evidence.all_chunk_ids() == ("manual.md#0",)
        assert evidence.chunk.source_path == ["manual.md"]
        assert evidence.text == embedder.calls[0][0]
    assert embedder.calls[1:] == ["Worker fault", "Worker fault"]
    after = collection.get(include=["metadatas", "embeddings"])
    assert before["ids"] == after["ids"]
    assert before["metadatas"] == after["metadatas"]
    np.testing.assert_array_equal(before["embeddings"], after["embeddings"])


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="creation command"):
        open_database(database_path=path)
    assert not path.exists()


def test_missing_or_empty_collection_is_not_populated(tmp_path):
    path = tmp_path / "index"
    create_chroma_database(path, "empty_collection")
    client = chromadb.PersistentClient(path=str(path))
    with pytest.raises(ValueError, match="does not exist"):
        open_database(database_path=path, collection_name="missing_collection")
    with pytest.raises(ValueError, match="no indexed chunks"):
        open_database(database_path=path, collection_name="empty_collection")
    assert [item.name for item in client.list_collections()] == ["empty_collection"]


def test_build_rejects_existing_collection_and_allows_new_configuration(
    tmp_path, source_docs
):
    (source_docs / "manual.md").write_text(
        "# Worker\nCheck service settings. Reset the worker process. "
        "Inspect the power supply. Confirm encoder connection.\n",
        encoding="utf-8",
    )
    path = tmp_path / "index"
    embedder = FakeEmbedder()
    original_count = create_database(embedder, source_docs, database_path=path)
    with pytest.raises(InternalError, match="already exists"):
        create_database(
            embedder,
            source_docs,
            database_path=path,
            chunk_size=30,
            overlap=5,
            minimum_chunk_size=1,
        )
    assert len(embedder.calls) == 1
    new_count = create_database(
        embedder,
        source_docs,
        database_path=path,
        collection_name="experiment_c30_o5",
        chunk_size=30,
        overlap=5,
        minimum_chunk_size=1,
    )
    client = chromadb.PersistentClient(path=str(path))
    assert client.get_collection(DEFAULT_COLLECTION_NAME).count() == original_count
    assert client.get_collection("experiment_c30_o5").count() == new_count
    assert new_count > original_count


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (100, -1), (10, 10)])
def test_invalid_configuration_does_not_create_database(
    tmp_path, source_docs, chunk_size, overlap
):
    path = tmp_path / "index"
    with pytest.raises(ValueError):
        create_database(
            FakeEmbedder(), source_docs, chunk_size, overlap, database_path=path
        )
    assert not path.exists()


def test_empty_source_does_not_create_database(tmp_path):
    with pytest.raises(ValueError, match="No documentation chunks"):
        create_database(FakeEmbedder(), tmp_path, database_path=tmp_path / "index")
    assert not (tmp_path / "index").exists()


def test_retrieval_uses_search_only_backend(monkeypatch):
    monkeypatch.setattr(backends, "_BACKENDS", {})
    search = Mock(spec=["similarity_search"])
    search.similarity_search.return_value = []
    builder = Mock(side_effect=AssertionError("Retrieval attempted database creation"))
    opener = Mock(return_value=search)
    register_database_backend("other", builder, opener)
    database = open_database(" OTHER ", connection="example")
    result = retrieve_evidence(
        query_text="Worker fault",
        embedder=FakeEmbedder(),
        database=database,
        reranker="none",
    )
    assert result.evidence == []
    opener.assert_called_once_with(connection="example")
    builder.assert_not_called()
    search.similarity_search.assert_called_once_with(
        query_embedding=[1.0, 0.0, 0.0], filters=SearchFilters(), limit=50
    )


def test_retrieval_cli_only_opens_database(tmp_path, monkeypatch, capsys):
    question = "How do I restart the worker?"
    output_path = tmp_path / "retrieval.json"
    database = Mock(spec=["similarity_search"])
    database.similarity_search.return_value = []
    opener = Mock(return_value=database)
    monkeypatch.setattr(cli, "open_database", opener)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        Mock(SentenceTransformer=FakeEmbedderFactory),
    )
    assert (
        cli.main(
            [
                question,
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    opener.assert_called_once()
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "question": question,
        "results": [],
    }
    assert capsys.readouterr().out == f"Retrieval results written to {output_path}\n"


def FakeEmbedderFactory(model):
    """Accept the CLI model argument without loading a model."""
    return FakeEmbedder()


def test_index_cli_passes_build_configuration(tmp_path, monkeypatch, capsys):
    builder = Mock(return_value=12)
    monkeypatch.setattr(index_cli, "create_database", builder)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        Mock(SentenceTransformer=FakeEmbedderFactory),
    )
    assert (
        index_cli.main(
            [
                "--docs-root",
                str(tmp_path),
                "--collection",
                "experiment",
                "--chunk-size",
                "500",
                "--overlap",
                "50",
            ]
        )
        == 0
    )
    assert builder.call_args.kwargs["docs_root"] == tmp_path
    assert builder.call_args.kwargs["chunk_size"] == 500
    assert builder.call_args.kwargs["overlap"] == 50
    assert json.loads(capsys.readouterr().out)["indexed_chunks"] == 12


def test_retrieval_import_does_not_load_creation_or_chroma():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'src'); "
            "import rag_engine.cli.retrieve; "
            "assert 'chromadb' not in sys.modules; "
            "assert 'rag_engine.engine.vector_store.create_database' not in sys.modules; "
            "assert 'rag_engine.engine.vector_store.data_chunking' not in sys.modules",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
