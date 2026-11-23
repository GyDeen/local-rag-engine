"""Check questions over ingested and indexed documents."""

import json

import numpy as np
from fastapi.testclient import TestClient

from rag_engine.engine.vector_store.backends import open_database
from rag_engine.engine.vector_store.create_database import create_database
from rag_engine.ingestion import ingest_documents
from rag_engine.runtime import RagRuntime, RagService, RetrievalOptions
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
