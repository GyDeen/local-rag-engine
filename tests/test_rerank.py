"""Behavioral checks for evidence exploration; no model or network required."""

import hashlib
import json
from unittest.mock import Mock

import pytest

from rag_engine.engine.rerank.base import RerankStrategy
from rag_engine.engine.rerank.candidate import build_rerank_candidates, to_stored
from rag_engine.engine.rerank.llm_reranker.pipeline import explore_evidence
from rag_engine.engine.rerank.llm_reranker.search import search_unseen
from rag_engine.engine.rerank.llm_reranker.strategy import LLMReranker
from rag_engine.engine.rerank.selection import NOT_CONTINUOUS_MARKER, stitch_evidence
from rag_engine.engine.retrieve.pipeline import retrieve_candidate_chunks
from rag_engine.engine.vector_store.database_metadata_model import Chunk

LONG_TEXT = "Documented procedure for the queue transfer. " * 8


def chunk(key, path="manual.md", text="Documented procedure"):
    """A stored chunk record that carries its own text."""
    return {
        "chunk_id": key,
        "source_path": [path],
        "section": "Procedure",
        "text": text,
        "document_id": path,
        "embedding": [],
        "start_offset": 0,
        "end_offset": len(text),
        "source_hash": "",
        "metadata": {},
    }


def chunk_object(key, path="manual.md", text="Documented procedure"):
    """A chunk exactly as the vector database returns it, with no text attached."""
    return Chunk(
        chunk_id=key,
        embedding=[],
        document_id=path,
        source_path=[path],
        section="Procedure",
        start_offset=0,
        end_offset=len(text),
        source_hash=hashlib.sha256(text.encode()).hexdigest(),
        metadata={},
    )


DOCUMENT = "".join(f"[{index:03d}]" for index in range(200))


def slice_chunk(key, document, start, end, path="manual.md", section="Procedure"):
    """A chunk that is a genuine slice of ``document``, with matching offsets."""
    return {
        "chunk_id": key,
        "source_path": [path],
        "section": section,
        "text": document[start:end],
        "document_id": path,
        "embedding": [],
        "start_offset": start,
        "end_offset": end,
        "source_hash": hashlib.sha256(document.encode()).hexdigest(),
        "metadata": {},
    }


def slice_chunk_object(
    key, document, start, end, path="manual.md", section="Procedure"
):
    """A database chunk whose slice of ``document`` is readable from disk.

    ``source_hash`` covers the whole file: that is what read_chunk_text verifies
    before slicing, so a chunk-level hash would be rejected as a changed source.
    """
    return Chunk(
        chunk_id=key,
        embedding=[],
        document_id=path,
        source_path=[path],
        section=section,
        start_offset=start,
        end_offset=end,
        source_hash=hashlib.sha256(document.encode()).hexdigest(),
        metadata={},
    )


def embedder():
    result = Mock()
    result.encode.return_value.tolist.return_value = [1.0, 0.0]
    return result


def verdict(ids, sufficient=True, queries=None):
    return json.dumps(
        {
            "ranked_chunk_ids": ids,
            "coverage": {"explanation": ids[:1], "resolution": ids[-1:]},
            "sufficient": sufficient,
            "missing": [] if sufficient else ["Need reset instructions"],
            "search_queries": queries or [],
            "operator_questions": [],
            "reason": "Selected passages explain the condition and action",
        }
    )


def test_search_reaches_unseen_hits_below_seen_top_ten():
    candidates = [chunk_object(str(index)) for index in range(15)]
    database = Mock()
    database.similarity_search.side_effect = lambda **kw: candidates[: kw["limit"]]
    encoder = embedder()
    fresh, ranking_cutoff = search_unseen(
        "queue reset procedure", encoder, database, {str(i) for i in range(10)}, limit=5
    )
    assert [chunk.chunk_id for chunk in fresh] == [str(i) for i in range(10, 15)]
    assert ranking_cutoff == 20
    encoder.encode.assert_called_once_with(
        "queue reset procedure", normalize_embeddings=True
    )


@pytest.mark.parametrize("reuse_strategy", [False, True])
def test_targeted_query_stable_ids_and_multiple_chunks_per_file(
    tmp_path, reuse_strategy
):
    (tmp_path / "manual.md").write_text("Documented procedure")
    caller = Mock(
        side_effect=[
            verdict(["initial"], False, ["queue reset procedure"]),
            verdict(["reset", "initial"]),
        ]
    )
    database = Mock()
    database.similarity_search.return_value = [
        chunk_object("initial"),
        chunk_object("reset"),
    ]
    encoder = embedder()
    result = explore_evidence(
        alarm_code="alarm",
        alarm_text="original alarm text",
        initial_chunks=[chunk("initial"), chunk("irrelevant", "b.md")],
        embedder=encoder,
        database=database,
        llm_caller=None if reuse_strategy else caller,
        rerank_strategy=LLMReranker(caller, file_limit=1) if reuse_strategy else None,
        file_limit=1,
        docs_root=tmp_path,
    )
    assert result["sufficient"]
    assert [c["chunk_id"] for c in result["chunks"]] == ["reset", "initial"]
    assert encoder.encode.call_args.args[0] == "queue reset procedure"
    assert len(json.loads(caller.call_args_list[0].args[1])["candidates"]) == 2
    assert result["searches"][0]["returned_chunk_ids"] == ["reset"]


@pytest.mark.parametrize("reuse_strategy", [False, True])
def test_repeated_query_gets_one_reformulation_attempt_then_stops(reuse_strategy):
    """A repeat is not a dead end: ask for a reformulation once, then give up."""

    caller = Mock(return_value=verdict(["a"], False, ["reset procedure"]))
    database = Mock()
    database.similarity_search.return_value = [chunk_object("a")]
    result = explore_evidence(
        alarm_code="alarm",
        alarm_text="fault",
        initial_chunks=[chunk("a")],
        embedder=embedder(),
        database=database,
        llm_caller=None if reuse_strategy else caller,
        rerank_strategy=LLMReranker(caller) if reuse_strategy else None,
        max_rounds=4,
    )
    assert result["stop_reason"] == "no_new_query"
    # Round 0 searched, round 1 repeated, then one reformulation request.
    assert caller.call_count == 3
    assert database.similarity_search.call_count == 1
    final_payload = json.loads(caller.call_args.args[1])
    assert "reset procedure" in final_payload["instruction"]
    assert final_payload["search_history"][0]["returned_chunk_ids"] == []
    assert [entry["round"] for entry in result["judgments"]] == [0, 1, 1]
    assert result["judgments"][-1]["reformulation"] is True


def test_reformulation_spends_the_second_supplementary_round(tmp_path):
    """With max_rounds=2 a repeat must not cost the run its second search."""

    (tmp_path / "manual.md").write_text("Documented procedure")
    caller = Mock(
        side_effect=[
            verdict(["a"], False, ["reset procedure"]),
            verdict(["a"], False, ["reset procedure"]),
            verdict(["a"], False, ["clear the fault latch"]),
            verdict(["a"], True, []),
        ]
    )
    database = Mock()
    database.similarity_search.side_effect = [
        [chunk_object("b")],
        [chunk_object("c")],
    ]
    result = explore_evidence(
        alarm_code="alarm",
        alarm_text="fault",
        initial_chunks=[chunk("a")],
        embedder=embedder(),
        database=database,
        llm_caller=caller,
        max_rounds=2,
        docs_root=tmp_path,
    )
    assert database.similarity_search.call_count == 2
    assert len(result["searches"]) == 2
    assert result["stop_reason"] == "sufficient"
    assert result["sufficient"] is True


@pytest.mark.parametrize("response", [verdict(["invented"]), verdict(["a", "b"])])
def test_invalid_selection_raises_before_any_generation(response):
    with pytest.raises(ValueError):
        explore_evidence(
            alarm_code="alarm",
            alarm_text="fault",
            initial_chunks=[chunk("a"), chunk("b", "other.md")],
            embedder=embedder(),
            database=Mock(),
            llm_caller=Mock(return_value=response),
            file_limit=1,
        )


def test_generation_uses_the_retrieval_verdict_for_coverage(tmp_path):
    from rag_engine.engine.llm.generation import generate_answer
    from rag_engine.engine.rerank.llm_reranker.strategy import EvidenceAssessment
    from rag_engine.engine.retrieve.pipeline import RetrievalResult

    text = "Explanation. Reset procedure."
    (tmp_path / "manual.md").write_text(text)
    materialised = build_rerank_candidates(
        [chunk_object("a", text=text)], docs_root=tmp_path
    )
    generator = Mock(return_value={"doc_coverage": "full", "notes": "Generated"})

    result = generate_answer(
        alarm_code="alarm",
        alarm_text="fault",
        retrieval=RetrievalResult(
            evidence=materialised,
            strategy="llm",
            stop_reason="round_budget",
            assessment=EvidenceAssessment(
                sufficient=False, missing=["Need reset instructions"]
            ),
        ),
        final_answer_fn=generator,
    )

    supplied = generator.call_args.args[2]
    assert [chunk["chunk_id"] for chunk in supplied] == ["a"]
    assert supplied[0]["text"] == text
    assert result["answer"]["doc_coverage"] == "partial"
    assert "Need reset instructions" in result["answer"]["notes"]
    assert result["retrieval"]["sufficient"] is False
    assert result["retrieval"]["stop_reason"] == "round_budget"


def test_initial_candidates_keep_same_file_chunks():
    database = Mock()
    database.similarity_search.return_value = [chunk_object("a"), chunk_object("b")]
    results = retrieve_candidate_chunks(
        {"code": "alarm"}, {"alarm": {"alarm_text": "fault"}}, embedder(), database
    )
    assert len(results) == 2
    assert database.similarity_search.call_args.kwargs["limit"] == 50


def test_context_budget_and_truncation_are_visible():
    caller = Mock(return_value=verdict(["a"]))
    result = explore_evidence(
        alarm_code="alarm",
        alarm_text="fault",
        initial_chunks=[chunk("a", text="x" * 30), chunk("b")],
        embedder=embedder(),
        database=Mock(),
        llm_caller=caller,
        pool_char_budget=10,
        pool_chunk_chars=10,
    )
    candidates = json.loads(caller.call_args.args[1])["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["truncated"]
    assert result["chunks"][0]["text"] == "x" * 10


def test_empty_search_can_be_reformulated_in_next_round(tmp_path):
    (tmp_path / "manual.md").write_text("Documented procedure")
    database = Mock()
    database.similarity_search.side_effect = [
        [chunk_object("a")],
        [chunk_object("b")],
    ]
    caller = Mock(
        side_effect=[
            verdict(["a"], False, ["reset worker"]),
            verdict(["a"], False, ["clear queue fault"]),
            verdict(["a", "b"]),
        ]
    )
    result = explore_evidence(
        alarm_code="alarm",
        alarm_text="fault",
        initial_chunks=[chunk("a")],
        embedder=embedder(),
        database=database,
        llm_caller=caller,
        docs_root=tmp_path,
    )
    assert result["sufficient"]
    assert [s["query"] for s in result["searches"]] == [
        "reset worker",
        "clear queue fault",
    ]
    assert len(result["judgments"]) == 3


def test_missing_resolution_evidence_cannot_be_marked_sufficient():
    response = json.loads(verdict(["a"]))
    response["coverage"]["resolution"] = []
    result = explore_evidence(
        alarm_code="alarm",
        alarm_text="fault",
        initial_chunks=[chunk("a")],
        embedder=embedder(),
        database=Mock(),
        llm_caller=Mock(return_value=json.dumps(response)),
    )
    assert not result["sufficient"]


def test_invalid_decision_gets_one_correction_attempt():
    caller = Mock(side_effect=[verdict(["invented"]), verdict(["a"])])
    result = explore_evidence(
        alarm_code="alarm",
        alarm_text="fault",
        initial_chunks=[chunk("a")],
        embedder=embedder(),
        database=Mock(),
        llm_caller=caller,
    )
    assert result["sufficient"]
    assert result["judgments"][0]["validation_errors"]
    assert "validation_error" in json.loads(caller.call_args.args[1])


def test_ollama_reranker_constrains_ids_and_output(monkeypatch):
    from rag_engine.engine.rerank.llm_reranker import callers

    transport = Mock(return_value=verdict(["a"]))
    monkeypatch.setattr(callers, "call_ollama_llm", transport)
    caller = callers.make_ollama_reranker(model="local", host="http://localhost")
    caller("instructions", json.dumps({"candidates": [{"chunk_id": "a"}]}))
    options = transport.call_args.kwargs
    schema = options["response_schema"]
    assert schema["properties"]["ranked_chunk_ids"]["items"]["enum"] == ["a"]
    assert options["max_new_tokens"] == 1200
    assert options["context_length"] == 16384


def test_rerank_candidates_are_materialised_once(tmp_path):
    text = "Explanation. Reset procedure."
    (tmp_path / "manual.md").write_text(text)
    stored = chunk("a", text=text)
    stored.pop("text")
    stored["source_hash"] = hashlib.sha256(text.encode()).hexdigest()

    candidates = build_rerank_candidates([stored, chunk("b")], docs_root=tmp_path)

    assert [candidate.chunk.chunk_id for candidate in candidates] == ["a", "b"]
    assert candidates[0].text == text
    assert candidates[0].score is None
    assert candidates[0].truncated is False


def test_llm_reranker_satisfies_the_shared_strategy_contract():
    candidates = build_rerank_candidates([chunk("a"), chunk("b", "other.md")])
    reranker = LLMReranker(Mock(return_value=verdict(["b", "a"])), file_limit=2)

    assert isinstance(reranker, RerankStrategy)

    ranked = reranker.rerank("alarm text", candidates)

    assert [candidate.chunk.chunk_id for candidate in ranked] == ["b", "a"]
    assert all(candidate.text for candidate in ranked)


def test_llm_strategy_uses_its_configured_file_limit_for_every_judgment():
    candidates = build_rerank_candidates([chunk("a"), chunk("b", "other.md")])
    caller = Mock(return_value=verdict(["a", "b"]))
    reranker = LLMReranker(caller, file_limit=2, max_attempts=1)

    ranked = reranker.rerank("alarm text", candidates)
    assert [candidate.chunk.chunk_id for candidate in ranked] == ["a", "b"]
    assert json.loads(caller.call_args.args[1])["file_limit"] == 2

    with pytest.raises(TypeError):
        reranker.judge("alarm text", candidates, file_limit=1)
    with pytest.raises(AttributeError):
        reranker.file_limit = 1

    ranked = reranker.rerank("another alarm", candidates)
    assert [candidate.chunk.chunk_id for candidate in ranked] == ["a", "b"]
    assert json.loads(caller.call_args.args[1])["file_limit"] == 2


def test_exploration_rejects_a_different_file_limit_for_a_loaded_strategy():
    caller = Mock()
    strategy = LLMReranker(caller, file_limit=1)
    with pytest.raises(ValueError, match="configured LLM reranker"):
        explore_evidence(
            alarm_code="alarm",
            alarm_text="fault",
            initial_chunks=[chunk("a")],
            embedder=embedder(),
            database=Mock(),
            rerank_strategy=strategy,
            file_limit=2,
        )
    caller.assert_not_called()


def test_cross_encoder_reranker_scores_and_sorts(monkeypatch):
    from rag_engine.engine.rerank.cross_encoder import strategy as cross_encoder

    model = Mock()
    model.predict.return_value = [0.2, 0.9, 0.5]
    monkeypatch.setattr(cross_encoder, "CrossEncoder", Mock(return_value=model))

    candidates = build_rerank_candidates([chunk("a"), chunk("b"), chunk("c")])
    reranker = cross_encoder.CrossEncoderReranker("local-cross-encoder")

    assert isinstance(reranker, RerankStrategy)

    ranked = reranker.rerank("reset the queue", candidates)

    assert [candidate.chunk.chunk_id for candidate in ranked] == ["b", "c", "a"]
    assert [candidate.score for candidate in ranked] == [0.9, 0.5, 0.2]
    assert (
        model.predict.call_args.args[0]
        == [("reset the queue", "Documented procedure")] * 3
    )
    assert [candidate.chunk.chunk_id for candidate in candidates] == ["a", "b", "c"]


def test_cross_encoder_reranker_skips_the_model_without_candidates(monkeypatch):
    from rag_engine.engine.rerank.cross_encoder import strategy as cross_encoder

    model = Mock()
    monkeypatch.setattr(cross_encoder, "CrossEncoder", Mock(return_value=model))

    reranker = cross_encoder.CrossEncoderReranker("local-cross-encoder")

    assert reranker.rerank("reset the queue", []) == []
    model.predict.assert_not_called()


def test_selection_keeps_one_chunk_per_file_in_rank_order():
    from rag_engine.engine.rerank.selection import select_top_files

    candidates = build_rerank_candidates(
        [chunk("a"), chunk("b", "second.md"), chunk("c"), chunk("d", "third.md")]
    )
    # Ranked: "c" (manual.md), then "a" from the same file, then "d", then "b".
    ranked = [candidates[2], candidates[0], candidates[3], candidates[1]]

    selected = select_top_files(ranked, file_limit=2)

    assert [candidate.chunk.chunk_id for candidate in selected] == ["c", "d"]


def test_selection_requires_a_positive_file_limit():
    from rag_engine.engine.rerank.selection import select_top_files

    with pytest.raises(ValueError):
        select_top_files([], file_limit=0)


def test_retrieval_reranks_before_generation(tmp_path, monkeypatch):
    from rag_engine.engine.llm.generation import generate_answer
    from rag_engine.engine.rerank.cross_encoder import strategy as cross_encoder
    from rag_engine.engine.retrieve.pipeline import retrieve_evidence

    (tmp_path / "manual.md").write_text(LONG_TEXT)
    model = Mock()
    model.predict.return_value = [0.9]
    monkeypatch.setattr(cross_encoder, "CrossEncoder", Mock(return_value=model))

    database = Mock()
    database.similarity_search.return_value = [chunk_object("a", text=LONG_TEXT)]
    generator = Mock(
        return_value={
            "summary": "Grounded",
            "likely_causes": ["Cause"],
            "resolution_steps": ["Step"],
        }
    )

    retrieval = retrieve_evidence(
        input_report={"code": "alarm"},
        catalog={"alarm": {"title": "Fault", "alarm_text": "fault"}},
        embedder=embedder(),
        database=database,
        chunk_limit=10,
        file_limit=3,
        docs_root=tmp_path,
    )

    assert retrieval.strategy == "cross_encoder"
    assert retrieval.stop_reason == "strategy_selected"
    assert retrieval.assessment is None
    assert [candidate.chunk.chunk_id for candidate in retrieval.evidence] == ["a"]

    result = generate_answer(
        alarm_code="alarm",
        alarm_text="fault",
        retrieval=retrieval,
        final_answer_fn=generator,
    )

    assert result["retrieval"]["selected_chunk_ids"] == ["a"]
    assert result["retrieval"]["selected_files"] == ["manual.md"]
    assert result["retrieval"]["strategy"] == "cross_encoder"
    assert result["retrieval"]["stop_reason"] == "strategy_selected"
    assert result["retrieval"]["sufficient"] is None
    assert result["retrieval"]["searches"] == []
    assert result["retrieval"]["judgments"] == []
    assert result["answer"] == {
        "summary": "Grounded",
        "likely_causes": ["Cause"],
        "resolution_steps": ["Step"],
        "doc_refs": [{"doc": "manual.md", "path": "manual.md", "section": "Procedure"}],
        "doc_coverage": "partial",
        "notes": "Generated only from the provided chunks.",
    }

    # Generation receives the stored shape, with its text already attached.
    supplied = generator.call_args.args[2]
    assert supplied[0]["chunk_id"] == "a"
    assert supplied[0]["text"] == LONG_TEXT


def test_standalone_retrieval_constructs_the_llm_strategy(tmp_path):
    from rag_engine.engine.retrieve.pipeline import retrieve_evidence

    (tmp_path / "manual.md").write_text(LONG_TEXT)
    database = Mock()
    database.similarity_search.return_value = [chunk_object("a", text=LONG_TEXT)]
    caller = Mock(return_value=verdict(["a"]))

    result = retrieve_evidence(
        input_report={"code": "alarm"},
        catalog={"alarm": {"title": "Fault", "alarm_text": "fault"}},
        embedder=embedder(),
        database=database,
        reranker="llm",
        llm_caller=caller,
        file_limit=1,
        docs_root=tmp_path,
    )

    assert result.strategy == "llm"
    assert result.stop_reason == "sufficient"
    assert [candidate.chunk.chunk_id for candidate in result.evidence] == ["a"]
    caller.assert_called_once()
    assert json.loads(caller.call_args.args[1])["file_limit"] == 1


def test_retrieval_reports_llm_gaps_and_operator_questions():
    from rag_engine.engine.rerank.llm_reranker.strategy import EvidenceAssessment
    from rag_engine.engine.retrieve.pipeline import RetrievalResult

    payload = RetrievalResult(
        evidence=[],
        strategy="llm",
        stop_reason="operator_input",
        assessment=EvidenceAssessment(
            sufficient=False,
            missing=["Reset procedure is not documented"],
            search_queries=["queue reset procedure"],
            operator_questions=["Which service instance owns the queue?"],
        ),
    ).as_dict()

    assert payload["strategy"] == "llm"
    assert payload["sufficient"] is False
    assert payload["missing"] == ["Reset procedure is not documented"]
    assert payload["operator_questions"] == ["Which service instance owns the queue?"]
    assert payload["search_queries"] == ["queue reset procedure"]
    assert payload["selected_chunk_ids"] == []


def test_retrieval_omits_llm_extras_without_an_assessment():
    from rag_engine.engine.retrieve.pipeline import RetrievalResult

    payload = RetrievalResult(
        evidence=[],
        strategy="cross_encoder",
        stop_reason="strategy_selected",
    ).as_dict()

    assert payload["sufficient"] is None
    assert payload["missing"] == []
    assert payload["operator_questions"] == []
    assert payload["search_queries"] == []
    assert payload["decision"] is None


def test_reranker_none_never_constructs_a_model(tmp_path, monkeypatch):
    from rag_engine.engine.rerank.cross_encoder import strategy as cross_encoder
    from rag_engine.engine.retrieve.pipeline import retrieve_evidence

    def explode(*args, **kwargs):
        raise AssertionError("reranker='none' must not build a cross-encoder")

    monkeypatch.setattr(cross_encoder, "CrossEncoder", explode)

    (tmp_path / "manual.md").write_text(LONG_TEXT)
    (tmp_path / "other.md").write_text(LONG_TEXT)
    database = Mock()
    database.similarity_search.return_value = [
        chunk_object("a", text=LONG_TEXT),
        chunk_object("b", "other.md", text=LONG_TEXT),
    ]

    retrieval = retrieve_evidence(
        input_report={"code": "alarm"},
        catalog={"alarm": {"title": "Fault", "alarm_text": "fault"}},
        embedder=embedder(),
        database=database,
        reranker="none",
        docs_root=tmp_path,
    )

    assert retrieval.strategy == "none"
    assert retrieval.assessment is None
    # Vector order is preserved, one chunk per file.
    assert [candidate.chunk.chunk_id for candidate in retrieval.evidence] == ["a", "b"]


def test_reranker_none_keeps_vector_order_over_the_stitched_pool(tmp_path):
    from rag_engine.engine.retrieve.pipeline import retrieve_evidence

    (tmp_path / "manual.md").write_text(LONG_TEXT)
    (tmp_path / "other.md").write_text(LONG_TEXT)
    database = Mock()
    database.similarity_search.return_value = [
        chunk_object("a", text=LONG_TEXT),
        chunk_object("b", "other.md", text=LONG_TEXT),
    ]
    report = {"code": "alarm"}
    catalog = {"alarm": {"title": "Fault", "alarm_text": "fault"}}

    bypass = retrieve_evidence(
        input_report=report,
        catalog=catalog,
        embedder=embedder(),
        database=database,
        reranker="none",
        docs_root=tmp_path,
    )

    # No reranker runs, so evidence follows the vector order. It no longer
    # mirrors retrieve_relevant_chunks: sections are stitched from the pool.
    assert [candidate.chunk.chunk_id for candidate in bypass.evidence] == ["a", "b"]


@pytest.mark.parametrize(
    "reranker,error", [("bogus", "Unknown reranker"), ("llm", "requires an llm_caller")]
)
def test_invalid_standalone_reranker_is_rejected_before_retrieval(reranker, error):
    from rag_engine.engine.retrieve.pipeline import retrieve_evidence

    database = Mock()
    with pytest.raises(ValueError, match=error):
        retrieve_evidence(
            input_report={"code": "alarm"},
            catalog={"alarm": {"title": "Fault", "alarm_text": "fault"}},
            embedder=Mock(),
            database=database,
            reranker=reranker,
        )
    database.similarity_search.assert_not_called()


def test_a_section_is_completed_from_the_pool():
    """The ranking picks one chunk; the pool supplies the rest of its section."""

    pool = build_rerank_candidates(
        [
            slice_chunk("first", DOCUMENT, 0, 200),
            slice_chunk("second", DOCUMENT, 200, 400),
        ]
    )
    stitched = stitch_evidence(
        [pool[0]], pool, file_limit=1, char_budget=10_000, min_chunk_chars=0
    )

    assert len(stitched) == 1
    assert stitched[0].text == DOCUMENT[0:400]
    assert stitched[0].chunk.chunk_id == "first"


def test_overlapping_chunks_are_stitched_without_repeating_the_join():
    """Chunks are indexed with an overlap; stitched text must not stutter."""

    pool = build_rerank_candidates(
        [
            slice_chunk("first", DOCUMENT, 0, 200),
            slice_chunk("second", DOCUMENT, 180, 400),
        ]
    )
    stitched = stitch_evidence(
        pool, pool, file_limit=1, char_budget=10_000, min_chunk_chars=0
    )

    assert len(stitched) == 1
    assert stitched[0].text == DOCUMENT[0:400]


def test_a_chunk_contained_in_another_contributes_nothing():
    pool = build_rerank_candidates(
        [
            slice_chunk("outer", DOCUMENT, 0, 300),
            slice_chunk("inner", DOCUMENT, 50, 100),
        ]
    )
    stitched = stitch_evidence(
        pool, pool, file_limit=1, char_budget=10_000, min_chunk_chars=0
    )

    assert [candidate.chunk.chunk_id for candidate in stitched] == ["outer"]
    assert stitched[0].all_chunk_ids() == ("outer",)
    assert stitched[0].text == DOCUMENT[0:300]


def test_stitched_evidence_names_every_chunk_it_was_built_from():
    """A stitched passage must account for all the text it was handed."""

    pool = build_rerank_candidates(
        [
            slice_chunk("first", DOCUMENT, 0, 200),
            slice_chunk("second", DOCUMENT, 200, 400),
        ]
    )
    stitched = stitch_evidence(
        [pool[0]], pool, file_limit=1, char_budget=10_000, min_chunk_chars=0
    )

    assert stitched[0].all_chunk_ids() == ("first", "second")
    stored = to_stored(stitched[0])
    # The anchor keeps its own indexed identity, while chunk_ids lists where the
    # whole passage came from.
    assert stored["chunk_id"] == "first"
    assert stored["chunk_ids"] == ["first", "second"]
    assert stored["text"] == DOCUMENT[0:400]


def test_single_chunk_evidence_names_only_itself():
    pool = build_rerank_candidates([slice_chunk("only", DOCUMENT, 0, 200)])
    stitched = stitch_evidence(
        pool, pool, file_limit=1, char_budget=10_000, min_chunk_chars=0
    )

    assert stitched[0].all_chunk_ids() == ("only",)
    assert to_stored(stitched[0])["chunk_ids"] == ["only"]


def test_a_truncated_chunk_makes_the_stitched_evidence_truncated():
    parts = [
        slice_chunk("first", DOCUMENT, 0, 200),
        slice_chunk("second", DOCUMENT, 200, 400),
    ]
    parts[1]["truncated"] = True
    pool = build_rerank_candidates(parts)
    stitched = stitch_evidence(
        [pool[0]], pool, file_limit=1, char_budget=10_000, min_chunk_chars=0
    )

    assert to_stored(stitched[0])["truncated"] is True


def test_reported_chunk_ids_cover_the_whole_stitched_evidence():
    from rag_engine.engine.retrieve.pipeline import RetrievalResult

    pool = build_rerank_candidates(
        [
            slice_chunk("first", DOCUMENT, 0, 200),
            slice_chunk("second", DOCUMENT, 200, 400),
        ]
    )
    stitched = stitch_evidence(
        [pool[0]], pool, file_limit=1, char_budget=10_000, min_chunk_chars=0
    )
    result = RetrievalResult(
        evidence=stitched, strategy="none", stop_reason="strategy_selected"
    )

    assert result.as_dict()["selected_chunk_ids"] == ["first", "second"]


def test_normalised_evidence_keeps_the_stitched_provenance():
    from rag_engine.engine.llm.evidence import normalise_chunks

    pool = build_rerank_candidates(
        [
            slice_chunk("first", DOCUMENT, 0, 200),
            slice_chunk("second", DOCUMENT, 200, 400),
        ]
    )
    stitched = stitch_evidence(
        [pool[0]], pool, file_limit=1, char_budget=10_000, min_chunk_chars=0
    )

    evidence = normalise_chunks([to_stored(stitched[0])])

    assert evidence[0]["chunk_ids"] == ["first", "second"]
    assert evidence[0]["text"] == DOCUMENT[0:400]


def test_a_retrieval_gap_is_marked_in_the_stitched_text(tmp_path):
    """A passage built around chunks retrieval missed says so in its own text.

    The chunks are numbered consecutively on purpose: only the width of the seam
    between them, read from the offsets, decides the marker.
    """

    from rag_engine.engine.llm.evidence import format_chunks_for_llm, normalise_chunks

    document = DOCUMENT
    (tmp_path / "manual.md").write_text(document)
    pool = build_rerank_candidates(
        [
            slice_chunk_object("manual.md#1", document, 0, 200),
            slice_chunk_object("manual.md#2", document, 260, 400),
        ],
        docs_root=tmp_path,
    )
    stitched = stitch_evidence(
        pool,
        pool,
        file_limit=1,
        char_budget=10_000,
        min_chunk_chars=0,
    )

    assert stitched[0].text == (
        document[0:200] + NOT_CONTINUOUS_MARKER + document[260:400]
    )
    # The marker survives into what answer generation and the judge read.
    rendered = format_chunks_for_llm(normalise_chunks([to_stored(stitched[0])]))
    assert NOT_CONTINUOUS_MARKER.strip() in rendered


def test_neighbouring_chunks_are_not_marked(tmp_path):
    """A paragraph break between consecutive chunks is not a retrieval gap."""

    document = "A" * 200 + "\r\n\r\n" + "B" * 200
    (tmp_path / "manual.md").write_text(document)
    pool = build_rerank_candidates(
        [
            slice_chunk_object("manual.md#5", document, 0, 200),
            slice_chunk_object("manual.md#6", document, 204, 404),
        ],
        docs_root=tmp_path,
    )
    stitched = stitch_evidence(
        pool,
        pool,
        file_limit=1,
        char_budget=10_000,
        min_chunk_chars=0,
    )

    assert stitched[0].text == "A" * 200 + "B" * 200
    assert NOT_CONTINUOUS_MARKER not in stitched[0].text


def test_a_single_chunk_is_never_marked(tmp_path):
    (tmp_path / "manual.md").write_text(DOCUMENT)
    pool = build_rerank_candidates(
        [slice_chunk_object("manual.md#3", DOCUMENT, 0, 200)], docs_root=tmp_path
    )
    stitched = stitch_evidence(
        pool,
        pool,
        file_limit=1,
        char_budget=10_000,
        min_chunk_chars=0,
    )

    assert stitched[0].text == DOCUMENT[0:200]


@pytest.mark.parametrize("seam, marked", [(4, False), (200, True)])
def test_the_seam_width_decides_the_marker(tmp_path, seam, marked):
    """A trimmed paragraph break is not missing documentation; a chunk-sized
    seam is, because no indexed chunk is anywhere near that narrow."""

    document = "A" * 200 + "x" * seam + "B" * 200
    (tmp_path / "manual.md").write_text(document)
    pool = build_rerank_candidates(
        [
            slice_chunk_object("first", document, 0, 200),
            slice_chunk_object("second", document, 200 + seam, len(document)),
        ],
        docs_root=tmp_path,
    )
    stitched = stitch_evidence(
        pool,
        pool,
        file_limit=1,
        char_budget=10_000,
        min_chunk_chars=0,
    )

    assert (NOT_CONTINUOUS_MARKER in stitched[0].text) is marked


def test_a_retrieved_chunk_covered_by_another_earns_no_marker(tmp_path):
    """A chunk that was retrieved but adds no text is not missing content.

    ``manual.md#2`` sits inside ``manual.md#1``, so stitching drops it. Its
    content is already in the passage, and the passage is continuous: nothing was
    skipped, so nothing is announced.
    """

    document = DOCUMENT
    (tmp_path / "manual.md").write_text(document)
    pool = build_rerank_candidates(
        [
            slice_chunk_object("manual.md#1", document, 0, 300),
            slice_chunk_object("manual.md#2", document, 50, 100),
            slice_chunk_object("manual.md#3", document, 300, 400),
        ],
        docs_root=tmp_path,
    )
    stitched = stitch_evidence(
        pool,
        pool,
        file_limit=1,
        char_budget=10_000,
        min_chunk_chars=0,
    )

    assert stitched[0].text == document[0:300] + document[300:400]
    assert stitched[0].all_chunk_ids() == ("manual.md#1", "manual.md#3")
    assert NOT_CONTINUOUS_MARKER not in stitched[0].text


def test_a_chunk_missing_between_two_retrieved_ones_is_marked(tmp_path):
    """Chunks 3 and 5 retrieved, 4 not: the passage announces the hole.

    The missing chunk is never fetched. Stitching joins what it was given and
    marks the seam, so the text says it is not continuous instead of pretending
    chunk 4's documentation was never written.
    """

    document = DOCUMENT
    (tmp_path / "manual.md").write_text(document)
    missing = slice_chunk_object("manual.md#4", document, 200, 260)
    pool = build_rerank_candidates(
        [
            slice_chunk_object("manual.md#3", document, 0, 200),
            slice_chunk_object("manual.md#5", document, 260, 400),
        ],
        docs_root=tmp_path,
    )
    stitched = stitch_evidence(
        pool,
        pool,
        file_limit=1,
        char_budget=10_000,
        min_chunk_chars=0,
    )

    assert stitched[0].text == (
        document[0:200] + NOT_CONTINUOUS_MARKER + document[260:400]
    )
    assert stitched[0].text.count(NOT_CONTINUOUS_MARKER) == 1
    # Chunk 4 is neither fetched nor claimed.
    assert document[missing.start_offset : missing.end_offset] not in (stitched[0].text)
    assert missing.chunk_id not in stitched[0].all_chunk_ids()


def test_the_character_budget_skips_a_section_that_would_overflow():
    pool = build_rerank_candidates(
        [
            slice_chunk("a", DOCUMENT, 0, 300, path="a.md"),
            slice_chunk("b", DOCUMENT, 0, 300, path="b.md"),
            slice_chunk("c", DOCUMENT, 0, 300, path="c.md"),
            slice_chunk("d", DOCUMENT, 0, 50, path="d.md"),
        ]
    )
    stitched = stitch_evidence(
        pool, pool, file_limit=4, char_budget=700, min_chunk_chars=0
    )

    # a and b fit; c would overflow and is skipped, so the smaller d still fits.
    assert [candidate.chunk.chunk_id for candidate in stitched] == ["a", "b", "d"]
    assert sum(len(candidate.text) for candidate in stitched) == 650


def test_sections_shorter_than_the_floor_are_not_evidence():
    pool = build_rerank_candidates(
        [
            slice_chunk("thin", DOCUMENT, 0, 100, path="thin.md"),
            slice_chunk("thick", DOCUMENT, 0, 400, path="thick.md"),
        ]
    )
    stitched = stitch_evidence(
        pool, pool, file_limit=3, char_budget=10_000, min_chunk_chars=200
    )

    assert [candidate.chunk.chunk_id for candidate in stitched] == ["thick"]


def test_the_file_limit_still_caps_distinct_files():
    pool = build_rerank_candidates(
        [
            slice_chunk("a", DOCUMENT, 0, 400, path=f"{name}.md")
            for name in ("a", "b", "c", "d")
        ]
    )
    stitched = stitch_evidence(
        pool, pool, file_limit=2, char_budget=10_000, min_chunk_chars=0
    )

    assert len({tuple(c.chunk.source_path) for c in stitched}) == 2


def test_exploration_reports_the_chunks_its_searches_found(tmp_path):
    """Discovery is returned, not discarded, so callers can stitch it."""

    (tmp_path / "manual.md").write_text(LONG_TEXT)
    caller = Mock(
        side_effect=[
            verdict(["initial"], False, ["queue reset procedure"]),
            verdict(["reset"]),
        ]
    )
    database = Mock()
    database.similarity_search.return_value = [
        chunk_object("initial", text=LONG_TEXT),
        chunk_object("reset", text=LONG_TEXT),
    ]
    result = explore_evidence(
        alarm_code="alarm",
        alarm_text="fault",
        initial_chunks=[chunk("initial", text=LONG_TEXT)],
        embedder=embedder(),
        database=database,
        llm_caller=caller,
        docs_root=tmp_path,
    )

    assert [c.chunk_id for c in result["discovered"]] == ["reset"]


def test_the_llm_path_stitches_sections_its_searches_discovered(tmp_path, monkeypatch):
    """A section only the search found still gets completed, not passed through."""

    import rag_engine.engine.rerank.llm_reranker.pipeline as reranker_pipeline
    import rag_engine.engine.retrieve.pipeline as retrieve_pipeline

    (tmp_path / "manual.md").write_text(DOCUMENT)
    initial = slice_chunk_object("first", DOCUMENT, 0, 200)
    found = slice_chunk_object("second", DOCUMENT, 200, 400)

    monkeypatch.setattr(
        retrieve_pipeline, "retrieve_candidate_chunks", lambda **kwargs: [initial]
    )

    def fake_exploration(**kwargs):
        return {
            "chunks": [slice_chunk("second", DOCUMENT, 200, 400)],
            "discovered": [found],
            "sufficient": False,
            "stop_reason": "round_budget",
            "decision": {"sufficient": False},
            "assessment": None,
            "searches": [],
            "judgments": [],
        }

    monkeypatch.setattr(reranker_pipeline, "explore_evidence", fake_exploration)

    retrieval = retrieve_pipeline.retrieve_evidence(
        input_report={"code": "alarm"},
        catalog={"alarm": {"title": "Fault", "alarm_text": "fault"}},
        embedder=embedder(),
        database=Mock(),
        reranker="llm",
        llm_caller=object(),
        file_limit=1,
        docs_root=tmp_path,
    )

    # "second" was never in the initial pool, so without the gathered pool this
    # section could only be passed through unchanged.
    assert len(retrieval.evidence) == 1
    assert retrieval.evidence[0].chunk.chunk_id == "first"
    assert retrieval.evidence[0].text == DOCUMENT[0:400]


def test_a_user_question_reranks_but_never_recalls_different_chunks(monkeypatch):
    """The question reaches ranking, not recall: the vector query stays alarm-only."""

    import rag_engine.engine.rerank.cross_encoder as ce_package
    import rag_engine.engine.retrieve.pipeline as retrieve_pipeline

    seen = {}

    class StubReranker:
        def __init__(self, *args, **kwargs):
            pass

        def rerank(self, query, candidates):
            seen["query"] = query
            return candidates

    monkeypatch.setattr(ce_package, "CrossEncoderReranker", StubReranker)

    embedder = Mock()
    embedder.encode.return_value.tolist.return_value = [1.0]
    database = Mock()
    database.similarity_search.return_value = []

    retrieve_pipeline.retrieve_evidence(
        input_report={"code": "alarm"},
        catalog={"alarm": {"title": "Fault", "alarm_text": "fault"}},
        embedder=embedder,
        database=database,
        reranker="cross_encoder",
        user_question="how do I restart the worker safely?",
    )

    embedded = embedder.encode.call_args.args[0]
    assert "Fault" in embedded
    assert "restart the worker" not in embedded
    assert seen["query"].endswith("how do I restart the worker safely?")


def test_llm_chunk_limit_only_applies_to_the_llm_reranker(tmp_path, monkeypatch):
    """The two strategies can retrieve different numbers of candidates."""

    import rag_engine.engine.rerank.cross_encoder as ce_package
    import rag_engine.engine.rerank.llm_reranker.pipeline as reranker_pipeline
    import rag_engine.engine.retrieve.pipeline as retrieve_pipeline

    (tmp_path / "manual.md").write_text(LONG_TEXT)
    asked: list[int] = []

    def fake_candidates(**kwargs):
        asked.append(kwargs["chunk_limit"])
        return [slice_chunk_object("only", LONG_TEXT, 0, len(LONG_TEXT))]

    monkeypatch.setattr(retrieve_pipeline, "retrieve_candidate_chunks", fake_candidates)
    monkeypatch.setattr(
        reranker_pipeline,
        "explore_evidence",
        lambda **kwargs: {
            "chunks": [slice_chunk("only", LONG_TEXT, 0, len(LONG_TEXT))],
            "discovered": [],
            "sufficient": False,
            "stop_reason": "stub",
            "decision": {},
            "assessment": None,
            "searches": [],
            "judgments": [],
        },
    )

    class StubReranker:
        def __init__(self, *args, **kwargs):
            pass

        def rerank(self, query, candidates):
            return candidates

    monkeypatch.setattr(ce_package, "CrossEncoderReranker", StubReranker)

    def run(reranker):
        retrieve_pipeline.retrieve_evidence(
            input_report={"code": "alarm"},
            catalog={"alarm": {"title": "Fault", "alarm_text": "fault"}},
            embedder=Mock(),
            database=Mock(),
            reranker=reranker,
            chunk_limit=50,
            llm_chunk_limit=10,
            llm_caller=object(),
            docs_root=tmp_path,
        )

    run("cross_encoder")
    run("llm")
    assert asked == [50, 10]


def test_llm_chunk_limit_falls_back_and_is_validated(tmp_path, monkeypatch):
    import rag_engine.engine.rerank.llm_reranker.pipeline as reranker_pipeline
    import rag_engine.engine.retrieve.pipeline as retrieve_pipeline

    asked: list[int] = []

    def fake_candidates(**kwargs):
        asked.append(kwargs["chunk_limit"])
        return []

    monkeypatch.setattr(retrieve_pipeline, "retrieve_candidate_chunks", fake_candidates)
    monkeypatch.setattr(
        reranker_pipeline,
        "explore_evidence",
        lambda **kwargs: {
            "chunks": [],
            "discovered": [],
            "sufficient": False,
            "stop_reason": "stub",
            "decision": {},
            "assessment": None,
            "searches": [],
            "judgments": [],
        },
    )
    catalog = {"alarm": {"title": "Fault", "alarm_text": "fault"}}

    # Unset: the LLM reranker uses chunk_limit like every other strategy.
    retrieve_pipeline.retrieve_evidence(
        input_report={"code": "alarm"},
        catalog=catalog,
        embedder=Mock(),
        database=Mock(),
        reranker="llm",
        chunk_limit=25,
        llm_caller=object(),
    )
    assert asked == [25]

    # A limit below file_limit cannot produce even one file per file slot.
    with pytest.raises(ValueError, match="LLM chunk limit"):
        retrieve_pipeline.retrieve_evidence(
            input_report={"code": "alarm"},
            catalog=catalog,
            embedder=Mock(),
            database=Mock(),
            reranker="llm",
            chunk_limit=25,
            llm_chunk_limit=2,
            file_limit=3,
            llm_caller=object(),
        )


def test_the_llm_budgets_reach_the_exploration(tmp_path, monkeypatch):
    """Each configured budget is handed to explore_evidence unchanged."""

    import rag_engine.engine.rerank.llm_reranker.pipeline as reranker_pipeline
    import rag_engine.engine.retrieve.pipeline as retrieve_pipeline

    received = {}

    def fake_exploration(**kwargs):
        received.update(kwargs)
        return {
            "chunks": [],
            "discovered": [],
            "sufficient": False,
            "stop_reason": "stub",
            "decision": {},
            "assessment": None,
            "searches": [],
            "judgments": [],
        }

    monkeypatch.setattr(retrieve_pipeline, "retrieve_candidate_chunks", lambda **k: [])
    monkeypatch.setattr(reranker_pipeline, "explore_evidence", fake_exploration)

    retrieve_pipeline.retrieve_evidence(
        input_report={"code": "alarm"},
        catalog={"alarm": {"title": "Fault", "alarm_text": "fault"}},
        embedder=Mock(),
        database=Mock(),
        reranker="llm",
        llm_caller=object(),
        llm_pool_char_budget=12345,
        llm_pool_chunk_chars=2345,
        llm_supplement_limit=7,
        llm_max_candidates=77,
    )

    assert received["pool_char_budget"] == 12345
    assert received["pool_chunk_chars"] == 2345
    assert received["supplement_limit"] == 7
    assert received["max_candidates"] == 77


@pytest.mark.parametrize(
    "setting, value, message",
    [
        ("llm_pool_char_budget", 0, "llm_pool_char_budget"),
        ("llm_pool_chunk_chars", 0, "llm_pool_chunk_chars"),
        ("llm_supplement_limit", 0, "llm_supplement_limit"),
        ("llm_max_candidates", 0, "llm_max_candidates"),
    ],
)
def test_llm_budgets_must_be_positive(tmp_path, setting, value, message):
    from rag_engine.evaluation.config import load_config

    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps({"generation": {setting: value}}))
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_max_candidates_must_cover_the_supplement_limit(tmp_path):
    from rag_engine.evaluation.config import load_config

    path = tmp_path / "evaluation.json"
    path.write_text(
        json.dumps(
            {"generation": {"llm_supplement_limit": 10, "llm_max_candidates": 5}}
        )
    )
    with pytest.raises(ValueError, match="llm_max_candidates"):
        load_config(path)


def test_a_file_that_contributes_nothing_does_not_spend_a_file_slot():
    """file_limit must count files that contributed evidence, not files seen.

    a.md's only section is larger than the budget, so it is skipped. That must
    not block d.md, which fits.
    """

    ranked = build_rerank_candidates(
        [
            slice_chunk("A", DOCUMENT, 0, 900, path="a.md"),
            slice_chunk("B", DOCUMENT, 0, 300, path="b.md"),
            slice_chunk("C", DOCUMENT, 0, 150, path="c.md"),
            slice_chunk("D", DOCUMENT, 0, 40, path="d.md"),
        ]
    )
    selected = stitch_evidence(
        ranked, ranked, file_limit=3, char_budget=500, min_chunk_chars=0
    )

    assert [candidate.chunk.chunk_id for candidate in selected] == ["B", "C", "D"]
    assert sum(len(candidate.text) for candidate in selected) == 490


def test_a_section_below_the_floor_does_not_spend_a_file_slot():
    ranked = build_rerank_candidates(
        [
            slice_chunk("thin", DOCUMENT, 0, 100, path="thin.md"),
            slice_chunk("B", DOCUMENT, 0, 300, path="b.md"),
            slice_chunk("C", DOCUMENT, 0, 300, path="c.md"),
            slice_chunk("E", DOCUMENT, 0, 300, path="e.md"),
        ]
    )
    selected = stitch_evidence(
        ranked, ranked, file_limit=3, char_budget=10_000, min_chunk_chars=200
    )

    assert [candidate.chunk.chunk_id for candidate in selected] == ["B", "C", "E"]


def test_one_file_can_contribute_several_sections():
    """The cap is on distinct contributing files, not on sections."""

    ranked = build_rerank_candidates(
        [
            slice_chunk("a1", DOCUMENT, 0, 300, path="a.md", section="S1"),
            slice_chunk("a2", DOCUMENT, 300, 600, path="a.md", section="S2"),
            slice_chunk("b1", DOCUMENT, 0, 300, path="b.md", section="S1"),
        ]
    )
    selected = stitch_evidence(
        ranked, ranked, file_limit=1, char_budget=10_000, min_chunk_chars=0
    )

    assert [candidate.chunk.chunk_id for candidate in selected] == ["a1", "a2"]
