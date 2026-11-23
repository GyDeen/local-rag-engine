"""Verify the typed filter contract between input parsing and retrieval."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag_engine.engine.parse_input.input_processing import (
    PreparedQuery,
    prepare_retrieval,
)
from rag_engine.engine.retrieve import pipeline
from rag_engine.engine.vector_store.database_metadata_model import SearchFilters


def test_free_form_preparation_needs_only_question_text():
    question = "How do I restart the worker?\n"
    prepared = prepare_retrieval(question)
    assert prepared == PreparedQuery(question)

    filters = SearchFilters({"product": "North"})
    assert prepare_retrieval(question, filters=filters).metadata_filter is filters

    with pytest.raises(ValueError, match="Question must contain nonempty text"):
        prepare_retrieval(" \n")


def test_preparation_requires_question_text():
    with pytest.raises(ValueError, match="Question must contain nonempty text"):
        prepare_retrieval({"code": "alarm"})


def test_custom_values_create_explicit_database_filters():
    from rag_engine.engine.vector_store.Chroma_API import ChromaVectorDatabase

    filters = SearchFilters({"product": "kettle", "retired": False})
    assert ChromaVectorDatabase._extract_filters(filters) == {
        "$and": [
            {
                "$or": [
                    {"product": {"$eq": "kettle"}},
                    {"product": {"$contains": "kettle"}},
                ]
            },
            {"$or": [{"retired": {"$eq": False}}, {"retired": {"$contains": False}}]},
        ]
    }


def test_retrieval_passes_prepared_filters_directly_to_database(monkeypatch):
    filters = SearchFilters({"machine_type": "MX7", "see_also": "G01"})
    monkeypatch.setattr(
        pipeline,
        "prepare_retrieval",
        lambda query: SimpleNamespace(query_text=query, metadata_filter=filters),
    )
    embedder = Mock()
    embedder.encode.return_value.tolist.return_value = [1.0, 0.0]
    database = Mock(spec=["similarity_search"])
    database.similarity_search.return_value = []

    assert pipeline.retrieve_relevant_chunks("Worker fault", embedder, database) == []
    assert database.similarity_search.call_args.kwargs["filters"] is filters
