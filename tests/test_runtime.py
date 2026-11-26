"""Exercise resource reuse through the real pipeline without downloading models."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag_engine.engine.rerank.base import RerankStrategy
from rag_engine.engine.vector_store.database_metadata_model import Chunk
from rag_engine.runtime import RagRuntime, RagService, RetrievalOptions, resources


@pytest.fixture
def runtime_dependencies(tmp_path, monkeypatch):
    import sentence_transformers

    from rag_engine.engine.rerank.cross_encoder import strategy

    text = "Consult the documented procedure before resetting the worker. " * 6
    (tmp_path / "manual.md").write_text(text)
    questions = ("How do I reset the worker?", "How do I clear the queue?")
    database = Mock()
    database.similarity_search.return_value = [
        Chunk(
            chunk_id="manual-1",
            document_id="manual.md",
            source_path=["manual.md"],
            section="Recovery",
            embedding=[1.0, 0.0],
            start_offset=0,
            end_offset=len(text),
            metadata={},
        )
    ]
    encoder = Mock()
    encoder.encode.return_value.tolist.return_value = [1.0, 0.0]
    cross_encoder = Mock()
    cross_encoder.predict.return_value = [0.9]
    encoder_factory = Mock(return_value=encoder)
    cross_encoder_factory = Mock(return_value=cross_encoder)
    database_factory = Mock(return_value=database)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", encoder_factory)
    monkeypatch.setattr(strategy, "CrossEncoder", cross_encoder_factory)
    monkeypatch.setattr(resources, "open_database", database_factory)
    return SimpleNamespace(
        questions=questions,
        docs_root=tmp_path,
        encoder=encoder,
        cross_encoder=cross_encoder,
        database=database,
        encoder_factory=encoder_factory,
        cross_encoder_factory=cross_encoder_factory,
        database_factory=database_factory,
    )


def test_different_questions_reuse_resources_and_retrieve_fresh_evidence(
    runtime_dependencies, monkeypatch
):
    from rag_engine.engine.retrieve import pipeline

    deps = runtime_dependencies
    runtime = RagRuntime.load()
    strategy_factory = Mock(side_effect=AssertionError("Reuse the runtime strategy"))
    monkeypatch.setattr(pipeline, "create_rerank_strategy", strategy_factory)
    service = RagService(runtime, RetrievalOptions(docs_root=deps.docs_root))

    results = [service.ask(question) for question in deps.questions]

    assert isinstance(runtime.rerank_strategy, RerankStrategy)
    for factory in (
        deps.encoder_factory,
        deps.cross_encoder_factory,
        deps.database_factory,
    ):
        factory.assert_called_once()
    assert deps.encoder.encode.call_count == 2
    assert deps.cross_encoder.predict.call_count == 2
    assert deps.database.similarity_search.call_count == 2
    strategy_factory.assert_not_called()
    assert [result["question"] for result in results] == list(deps.questions)
    assert all(result["evidence"][0]["location"] == "manual.md" for result in results)
    assert all(result["retrieval"]["strategy"] == "cross_encoder" for result in results)
    assert [call.args[0] for call in deps.encoder.encode.call_args_list] == list(
        deps.questions
    )


def test_runtime_accepts_and_reuses_a_custom_ranking_strategy(runtime_dependencies):
    deps = runtime_dependencies
    strategy = SimpleNamespace(
        rerank=Mock(side_effect=lambda query, candidates: candidates)
    )
    runtime = RagRuntime(
        embedder=deps.encoder,
        database=deps.database,
        rerank_strategy=strategy,
    )
    service = RagService(runtime, RetrievalOptions(docs_root=deps.docs_root))
    for question in deps.questions:
        assert service.retrieve_question(question).evidence
    assert runtime.rerank_strategy is strategy
    assert strategy.rerank.call_count == 2
    deps.cross_encoder_factory.assert_not_called()


def test_requests_do_not_share_candidate_scores_or_trace(runtime_dependencies):
    deps = runtime_dependencies
    runtime = RagRuntime.load()
    service = RagService(runtime, RetrievalOptions(docs_root=deps.docs_root))
    deps.cross_encoder.predict.side_effect = [[0.9], [0.2]]
    first = service.retrieve_question(deps.questions[0])
    second = service.retrieve_question(deps.questions[1])
    assert first.evidence[0] is not second.evidence[0]
    assert first.evidence[0].score == 0.9
    assert second.evidence[0].score == 0.2
    first.trace["request"] = "first"
    assert "request" not in second.trace


@pytest.mark.parametrize("reranker", ["none", "llm"])
@pytest.mark.parametrize("file_limit", [1, 3])
def test_optional_strategies_never_load_cross_encoder(
    runtime_dependencies, monkeypatch, reranker, file_limit
):
    from rag_engine.engine.rerank.llm_reranker import strategy

    deps = runtime_dependencies
    llm_strategy_factory = Mock(wraps=strategy.LLMReranker)
    monkeypatch.setattr(strategy, "LLMReranker", llm_strategy_factory)
    caller = Mock(
        return_value=json.dumps(
            {
                "ranked_chunk_ids": ["manual-1"],
                "coverage": {
                    "explanation": ["manual-1"],
                    "resolution": ["manual-1"],
                },
                "sufficient": True,
                "missing": [],
                "search_queries": [],
                "operator_questions": [],
                "reason": "Evidence explains the alarm and recovery",
            }
        )
    )
    runtime = RagRuntime.load(
        reranker=reranker,
        llm_caller=caller if reranker == "llm" else None,
        file_limit=file_limit,
    )
    selected_strategy = runtime.rerank_strategy
    service = RagService(
        runtime,
        RetrievalOptions(docs_root=deps.docs_root),
    )
    for question in deps.questions:
        result = service.retrieve_question(question)
        assert result.strategy == reranker
        assert result.evidence[0].chunk.chunk_id == "manual-1"
    assert runtime.rerank_strategy is selected_strategy
    if reranker == "llm":
        assert isinstance(selected_strategy, RerankStrategy)
        llm_strategy_factory.assert_called_once_with(caller, file_limit=file_limit)
        assert selected_strategy.file_limit == file_limit
        assert all(
            json.loads(call.args[1])["file_limit"] == file_limit
            for call in caller.call_args_list
        )
    else:
        assert selected_strategy is None
        llm_strategy_factory.assert_not_called()
    deps.cross_encoder_factory.assert_not_called()
    deps.encoder_factory.assert_called_once()
    assert caller.call_count == (2 if reranker == "llm" else 0)


def test_missing_index_fails_before_loading_models(runtime_dependencies):
    deps = runtime_dependencies
    deps.database_factory.side_effect = FileNotFoundError("No Chroma database")
    with pytest.raises(FileNotFoundError, match="No Chroma database"):
        RagRuntime.load()
    deps.encoder_factory.assert_not_called()
    deps.cross_encoder_factory.assert_not_called()


def test_reranker_load_failure_is_propagated(runtime_dependencies):
    deps = runtime_dependencies
    deps.cross_encoder_factory.side_effect = RuntimeError("model unavailable")
    with pytest.raises(RuntimeError, match="model unavailable"):
        RagRuntime.load()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reranker": "invalid"},
        {"file_limit": 0},
    ],
)
def test_invalid_resource_settings_fail_before_opening_database(
    runtime_dependencies, kwargs
):
    deps = runtime_dependencies
    with pytest.raises(ValueError):
        RagRuntime.load(**kwargs)
    deps.database_factory.assert_not_called()
    deps.encoder_factory.assert_not_called()


def test_blank_question_does_not_run_inference(runtime_dependencies):
    deps = runtime_dependencies
    service = RagService(RagRuntime.load())
    with pytest.raises(ValueError, match="Question must contain nonempty text"):
        service.retrieve_question("   ")
    deps.encoder.encode.assert_not_called()
    deps.cross_encoder.predict.assert_not_called()


def test_llm_runtime_requires_caller_before_loading_models(runtime_dependencies):
    deps = runtime_dependencies
    with pytest.raises(ValueError, match="llm_caller"):
        RagRuntime.load(reranker="llm")
    deps.database_factory.assert_not_called()
    deps.encoder_factory.assert_not_called()
    deps.cross_encoder_factory.assert_not_called()


@pytest.mark.parametrize(
    "reranker,strategy_kind,error",
    [
        ("cross_encoder", "absent", ValueError),
        ("llm", "absent", ValueError),
        ("none", "ranking", ValueError),
        ("cross_encoder", "invalid", TypeError),
        ("llm", "invalid", TypeError),
    ],
)
def test_runtime_validates_selected_strategy_contract(reranker, strategy_kind, error):
    strategies = {
        "absent": None,
        "ranking": SimpleNamespace(rerank=lambda query, candidates: candidates),
        "invalid": object(),
    }
    with pytest.raises(error):
        RagRuntime(
            embedder=object(),
            database=Mock(),
            reranker=reranker,
            rerank_strategy=strategies[strategy_kind],
        )


@pytest.mark.parametrize(
    "options",
    [
        {"chunk_limit": 0},
        {"llm_chunk_limit": 0},
        {"llm_max_candidates": 1, "llm_supplement_limit": 2},
        {"max_rounds": -1},
        {"evidence_min_chunk_chars": -1},
        {"evidence_rank_limit": 0},
        {"evidence_rank_limit": -1},
        {"evidence_rank_limit": 1.5},
        {"evidence_char_budget": 0},
    ],
)
def test_invalid_retrieval_options_are_rejected(options):
    with pytest.raises(ValueError):
        RetrievalOptions(**options)


@pytest.mark.parametrize("options", [{"chunk_limit": 1}, {"llm_chunk_limit": 1}])
def test_service_validates_candidate_limits_against_fixed_file_cap(
    runtime_dependencies, options
):
    runtime = RagRuntime.load(file_limit=2)
    with pytest.raises(ValueError, match="configured file_limit"):
        RagService(runtime, RetrievalOptions(**options))


def test_runtime_file_cap_is_fixed_and_used_for_every_request(runtime_dependencies):
    deps = runtime_dependencies
    first_chunk = deps.database.similarity_search.return_value[0]
    (deps.docs_root / "second.md").write_text(
        (deps.docs_root / "manual.md").read_text()
    )
    deps.database.similarity_search.return_value.append(
        replace(
            first_chunk,
            chunk_id="second-1",
            document_id="second.md",
            source_path=["second.md"],
        )
    )
    deps.cross_encoder.predict.return_value = [0.9, 0.8]
    runtime = RagRuntime.load(file_limit=1)
    service = RagService(runtime, RetrievalOptions(docs_root=deps.docs_root))
    for question in deps.questions:
        result = service.retrieve_question(question)
        assert {tuple(item.chunk.source_path) for item in result.evidence} == {
            ("manual.md",)
        }
    with pytest.raises(FrozenInstanceError):
        runtime.file_limit = 2
    with pytest.raises(TypeError):
        service.retrieve_question(deps.questions[0], file_limit=2)


def test_package_import_does_not_import_model_or_database_libraries():
    code = (
        "import sys; import rag_engine.runtime; "
        "assert 'sentence_transformers' not in sys.modules; "
        "assert 'chromadb' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
        check=True,
        capture_output=True,
        text=True,
    )


def test_evaluation_config_preserves_question_batch_path_and_cutoff(tmp_path):
    from rag_engine.evaluation.config import load_config

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"generation": {"cases": "batch.jsonl"}, "evaluation": {"k": 2}})
    )
    config = load_config(path)
    assert config.generation.cases == "batch.jsonl"
    assert config.evaluation.k == 2


@pytest.mark.parametrize("override_cases", [False, True])
def test_batch_evaluation_accepts_questions_without_ids(
    runtime_dependencies, monkeypatch, override_cases
):
    from rag_engine.evaluation import run_all
    from rag_engine.evaluation.config import EvaluationConfig

    deps = runtime_dependencies
    cases_file = deps.docs_root / "questions.jsonl"
    cases_file.write_text(
        "\n".join(json.dumps({"question": question}) for question in deps.questions)
    )
    config = EvaluationConfig()
    config.generation.cases = str(cases_file if not override_cases else "unused.jsonl")
    monkeypatch.setattr(
        run_all,
        "RetrievalOptions",
        lambda **kwargs: RetrievalOptions(docs_root=deps.docs_root, **kwargs),
    )
    caller = Mock(
        return_value=json.dumps(
            {"status": "answered", "answer": "Grounded guidance [E1]"}
        )
    )
    judge = Mock(return_value={"checked": True})
    monkeypatch.setattr("rag_engine.serving.answers.call_ollama_llm", caller)
    monkeypatch.setattr(run_all.judge_results, "judge", judge)
    output_dir = deps.docs_root / "results"
    arguments = argparse.Namespace(
        cases=cases_file if override_cases else None,
        case_ids=None,
        output_dir=output_dir,
    )
    references = {
        "answers": {"case-1": {"answer": "Grounded guidance", "answerable": True}}
    }
    assert run_all._generate_and_judge(arguments, config, references, None) == 0
    deps.encoder_factory.assert_called_once()
    deps.cross_encoder_factory.assert_called_once()
    deps.database_factory.assert_called_once()
    assert deps.cross_encoder.predict.call_count == 2
    assert [call.args[0] for call in deps.encoder.encode.call_args_list] == list(
        deps.questions
    )
    assert [
        pair[0]
        for call in deps.cross_encoder.predict.call_args_list
        for pair in call.args[0]
    ] == list(deps.questions)
    assert caller.call_count == 2
    assert caller.call_args.kwargs["context_length"] == config.generation.context_length
    judge.assert_called_once()
    assert judge.call_args.args[3] is deps.encoder
    results = [
        json.loads(path.read_text()) for path in sorted(output_dir.glob("*.json"))
    ]
    assert [result["id"] for result in results] == ["case-1", "case-2"]
    assert [result["question"] for result in results] == list(deps.questions)
    assert all(result["evidence"][0]["text"] for result in results)
    assert all(result["status"] == "answered" for result in results)
    assert all(result["answer"] == "Grounded guidance [E1]" for result in results)
    assert results[0]["evaluation"] == {"checked": True}
    assert "evaluation" not in results[1]
