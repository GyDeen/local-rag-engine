"""Check questions over ingested and indexed documents."""

import json
from functools import partial
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag_engine.benchmarks.run import _run_case
from rag_engine.engine.vector_store.backends import open_database
from rag_engine.engine.vector_store.create_database import create_database
from rag_engine.engine.vector_store.metadata_extraction import MetadataFieldRule
from rag_engine.ingestion import ingest_documents
from rag_engine.runtime import RagRuntime, RagService, RetrievalOptions
from rag_engine.serving import answers
from rag_engine.serving.app import create_app
from rag_engine.serving.concurrency import create_backend


class Embedder:
    def __init__(self):
        self.queries = []

    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            self.queries.append(texts)
            return np.array([1.0, 0.0])
        return np.array([[1.0, 0.0] for _ in texts])


def test_ingest_index_and_question_without_labels_or_catalogue(tmp_path):
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
    service = RagService(
        RagRuntime(
            embedder=embedder,
            database=open_database(
                database_path=database_path, collection_name="general"
            ),
            reranker="none",
            file_limit=3,
        ),
        RetrievalOptions(docs_root=output / "documents", evidence_min_chunk_chars=0),
    )
    # Retrieval uses stored text and provenance, not the original documents.
    for path in source.iterdir():
        path.unlink()
    (output / "debug/manifest.json").unlink()
    app = create_app(lambda: create_backend(service=service, retrieval_only=True))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ask", json={"question": "How do I check the motor?"}
        )
        assert response.status_code == 200
        assert client.post("/api/v1/ask", json={"question": " \t"}).status_code == 422
    result = response.json()
    assert embedder.queries == ["How do I check the motor?"]
    assert {item["doc"] for item in result["evidence"]} == {
        "guide.txt",
        "guide.md",
        "guide.txt.md",
    }
    assert result["answer"] is None


def test_document_metadata_controls_http_retrieval(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    for filename, labelled_fields in {
        "north.md": "**Model:** North\nRevision: 2\nRetired: false",
        "south.md": "Model: South\nRevision: 2\nRetired: false",
        "old.md": "Product: North\nRevision: 1\nRetired: true",
    }.items():
        (docs / filename).write_text(
            f"# Recovery\n\n{labelled_fields}\n\nCheck the motor wiring.",
            encoding="utf-8",
        )
    embedder = Embedder()
    database_path = tmp_path / "index"
    create_database(
        embedder,
        docs,
        database_path=database_path,
        collection_name="general",
        metadata_fields={
            "product": MetadataFieldRule(labels=["Product", "Model"]),
            "revision": MetadataFieldRule(labels=["Revision"], type="integer"),
            "retired": MetadataFieldRule(labels=["Retired"], type="boolean"),
        },
        metadata_by_document={
            name: {"region": ["AU", "NZ"]}
            for name in ("north.md", "south.md", "old.md")
        },
    )
    service = RagService(
        RagRuntime(
            embedder=embedder,
            database=open_database(
                database_path=database_path, collection_name="general"
            ),
            reranker="none",
        ),
        RetrievalOptions(docs_root=docs, evidence_min_chunk_chars=0),
    )
    question = "What should I check?"
    with TestClient(
        create_app(lambda: create_backend(service=service, retrieval_only=True))
    ) as client:
        response = client.post(
            "/api/v1/ask",
            json={
                "question": question,
                "filters": {
                    "product": "North",
                    "revision": 2,
                    "retired": False,
                    "region": "AU",
                },
            },
        )
        assert response.status_code == 200, response.text
        assert [item["doc"] for item in response.json()["evidence"]] == ["north.md"]
        assert (
            client.post(
                "/api/v1/ask", json={"question": question, "filters": {"$or": "North"}}
            ).status_code
            == 422
        )
    assert embedder.queries == [question]


def test_benchmark_cannot_supply_gold_information_to_application():
    retrieve = Mock(
        return_value={"question": "What next?", "evidence": [], "answer": None}
    )
    generate = Mock(
        return_value={
            "status": "insufficient_evidence",
            "answer": "Need evidence",
        }
    )
    case = {
        "id": "case",
        "question": "What next?",
        "answerable": True,
        "document": "secret-manual.pdf",
        "answer": "secret answer",
        "required_steps": ["secret step"],
        "references": [{"source": "secret-manual.pdf", "page": 9}],
    }
    result = _run_case(
        case,
        retrieve,
        partial(generate, model="local", host="http://local"),
    )
    retrieve.assert_called_once_with("What next?")
    generate.assert_called_once_with(
        "What next?",
        [],
        model="local",
        host="http://local",
    )
    assert result["gold_source_recall"] == 0
    assert result["gold_page_recall"] == 0


@pytest.mark.parametrize(
    "response",
    [
        {"status": "answered", "answer": "Do this. [E99]"},
        {"status": "answered", "answer": "Do this without evidence."},
        {"status": "insufficient_evidence", "answer": ""},
    ],
)
def test_question_generation_rejects_invalid_answers(monkeypatch, response):
    monkeypatch.setattr(
        answers, "call_ollama_llm", Mock(return_value=json.dumps(response))
    )
    with pytest.raises(ValueError):
        answers.answer_question(
            "What next?", [{"id": "E1", "text": "Check the wiring."}], model="local"
        )


def test_empty_evidence_abstains_without_calling_the_model(monkeypatch):
    caller = Mock(side_effect=AssertionError("No evidence must not call a model"))
    monkeypatch.setattr(answers, "call_ollama_llm", caller)
    assert (
        answers.answer_question("What next?", [], model="local")["status"]
        == "insufficient_evidence"
    )
