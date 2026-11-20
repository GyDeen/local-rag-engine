"""Exercise resource reuse through the real pipeline without downloading models."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
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
    catalog = {
        code: {
            "code": code,
            "title": title,
            "alarm_text": title,
            "severity": 833,
            "severity_category": "Error",
        }
        for code, title in (
            ("demo.test.0001", "Service fault"),
            ("demo.test.0002", "Queue fault"),
        )
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"alarms": list(catalog.values())}))
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
    catalog_loader = Mock(wraps=resources.load_alarm_catalog)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", encoder_factory)
    monkeypatch.setattr(strategy, "CrossEncoder", cross_encoder_factory)
    monkeypatch.setattr(resources, "open_database", database_factory)
    monkeypatch.setattr(resources, "load_alarm_catalog", catalog_loader)
    return SimpleNamespace(
        catalog=catalog,
        catalog_path=catalog_path,
        docs_root=tmp_path,
        encoder=encoder,
        cross_encoder=cross_encoder,
        database=database,
        encoder_factory=encoder_factory,
        cross_encoder_factory=cross_encoder_factory,
        database_factory=database_factory,
        catalog_loader=catalog_loader,
    )


@pytest.mark.parametrize("report_factory", [dict, MappingProxyType])
def test_different_alarms_reuse_resources_and_generate_fresh_answers(
    runtime_dependencies,
    monkeypatch,
    report_factory,
):
    from rag_engine.engine.retrieve import pipeline

    deps = runtime_dependencies
    runtime = RagRuntime.load(catalog_path=deps.catalog_path)
    strategy_factory = Mock(side_effect=AssertionError("Reuse the runtime strategy"))
    monkeypatch.setattr(pipeline, "create_rerank_strategy", strategy_factory)
    generator = Mock(
        side_effect=lambda code, *_args, **_kwargs: {
            "summary": f"Guidance for {code}",
            "likely_causes": ["Service fault"],
            "resolution_steps": ["Consult the procedure"],
        }
    )
    service = RagService(
        runtime,
        RetrievalOptions(docs_root=deps.docs_root),
        final_answer_fn=generator,
    )
    first_report = report_factory({"code": "demo.test.0001"})
    second_report = report_factory({"code": "demo.test.0002"})
    first = service.resolve(first_report, user_question="How to reset?")
    second = service.resolve(second_report)

    assert isinstance(runtime.rerank_strategy, RerankStrategy)
    for factory in (
        deps.encoder_factory,
        deps.cross_encoder_factory,
        deps.database_factory,
        deps.catalog_loader,
    ):
        factory.assert_called_once()
    assert deps.encoder.encode.call_count == 2
    assert deps.cross_encoder.predict.call_count == 2
    assert deps.database.similarity_search.call_count == 2
    assert generator.call_count == 2
    strategy_factory.assert_not_called()
    assert first["answer"]["summary"] == "Guidance for demo.test.0001"
    assert second["answer"]["summary"] == "Guidance for demo.test.0002"
    assert first["answer"]["severity"] == 833
    assert first["answer"]["doc_refs"][0]["path"] == "manual.md"
    assert first["retrieval"]["strategy"] == "cross_encoder"
    assert generator.call_args_list[0].kwargs["user_question"] == "How to reset?"
    assert generator.call_args_list[1].kwargs["user_question"] is None
    assert "How to reset?" not in deps.encoder.encode.call_args_list[0].args[0]
    assert "How to reset?" in deps.cross_encoder.predict.call_args_list[0].args[0][0][0]


def test_runtime_accepts_and_reuses_a_custom_ranking_strategy(runtime_dependencies):
    deps = runtime_dependencies
    strategy = SimpleNamespace(
        rerank=Mock(side_effect=lambda query, candidates: candidates)
    )
    runtime = RagRuntime(
        catalog=deps.catalog,
        embedder=deps.encoder,
        database=deps.database,
        rerank_strategy=strategy,
    )
    service = RagService(runtime, RetrievalOptions(docs_root=deps.docs_root))
    for code in deps.catalog:
        assert service.retrieve({"code": code}).evidence
    assert runtime.rerank_strategy is strategy
    assert strategy.rerank.call_count == 2
    deps.cross_encoder_factory.assert_not_called()


def test_requests_do_not_share_candidate_scores_or_trace(runtime_dependencies):
    deps = runtime_dependencies
    runtime = RagRuntime.load(catalog=deps.catalog)
    service = RagService(runtime, RetrievalOptions(docs_root=deps.docs_root))
    deps.cross_encoder.predict.side_effect = [[0.9], [0.2]]
    first = service.retrieve({"code": "demo.test.0001"})
    second = service.retrieve({"code": "demo.test.0002"})
    assert first.evidence[0] is not second.evidence[0]
    assert first.evidence[0].score == 0.9
    assert second.evidence[0].score == 0.2
    first.trace["request"] = "first"
    assert "request" not in second.trace
    deps.catalog["demo.test.0001"]["alarm_text"] = "changed outside the runtime"
    assert runtime.catalog["demo.test.0001"]["alarm_text"] == "Service fault"


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
        catalog=deps.catalog,
        reranker=reranker,
        llm_caller=caller if reranker == "llm" else None,
        file_limit=file_limit,
    )
    selected_strategy = runtime.rerank_strategy
    service = RagService(
        runtime,
        RetrievalOptions(docs_root=deps.docs_root),
    )
    for code in deps.catalog:
        result = service.retrieve({"code": code})
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
        RagRuntime.load(catalog=deps.catalog)
    deps.encoder_factory.assert_not_called()
    deps.cross_encoder_factory.assert_not_called()


def test_reranker_load_failure_is_propagated(runtime_dependencies):
    deps = runtime_dependencies
    deps.cross_encoder_factory.side_effect = RuntimeError("model unavailable")
    with pytest.raises(RuntimeError, match="model unavailable"):
        RagRuntime.load(catalog=deps.catalog)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reranker": "invalid"},
        {"catalog_path": "catalog.json"},
        {"file_limit": 0},
    ],
)
def test_invalid_resource_settings_fail_before_opening_database(
    runtime_dependencies, kwargs
):
    deps = runtime_dependencies
    with pytest.raises(ValueError):
        RagRuntime.load(catalog=deps.catalog, **kwargs)
    deps.database_factory.assert_not_called()
    deps.encoder_factory.assert_not_called()


def test_unknown_alarm_and_missing_generator_do_not_run_inference(runtime_dependencies):
    deps = runtime_dependencies
    service = RagService(RagRuntime.load(catalog=deps.catalog))
    with pytest.raises(KeyError, match="Unknown alarm code"):
        service.retrieve({"code": "unknown"})
    with pytest.raises(ValueError, match="final_answer_fn"):
        service.resolve({"code": "demo.test.0001"})
    deps.encoder.encode.assert_not_called()
    deps.cross_encoder.predict.assert_not_called()


def test_llm_runtime_requires_caller_before_loading_models(runtime_dependencies):
    deps = runtime_dependencies
    with pytest.raises(ValueError, match="llm_caller"):
        RagRuntime.load(catalog=deps.catalog, reranker="llm")
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
            catalog={},
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
    runtime = RagRuntime.load(catalog=runtime_dependencies.catalog, file_limit=2)
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
    runtime = RagRuntime.load(catalog=deps.catalog, file_limit=1)
    service = RagService(runtime, RetrievalOptions(docs_root=deps.docs_root))
    for code in deps.catalog:
        result = service.retrieve({"code": code})
        assert {tuple(item.chunk.source_path) for item in result.evidence} == {
            ("manual.md",)
        }
    with pytest.raises(FrozenInstanceError):
        runtime.file_limit = 2
    with pytest.raises(TypeError):
        service.retrieve({"code": "demo.test.0001"}, file_limit=2)


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


@pytest.mark.parametrize("catalog", [["catalog.json"], ["a.json", "b.json"]])
def test_evaluation_config_preserves_catalogue_paths(tmp_path, catalog):
    from rag_engine.evaluation.config import load_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"generation": {"catalog": catalog}}))
    assert load_config(path).generation.catalog == catalog


@pytest.mark.parametrize("catalogue_count", [1, 2])
def test_batch_evaluation_reuses_models_for_all_alarms(
    runtime_dependencies, monkeypatch, catalogue_count
):
    from rag_engine.evaluation import run_all
    from rag_engine.evaluation.config import EvaluationConfig

    deps = runtime_dependencies
    config = EvaluationConfig()
    config.generation.catalog = [str(deps.catalog_path)]
    if catalogue_count == 2:
        config.generation.catalog = []
        for index, alarm in enumerate(deps.catalog.values()):
            path = deps.docs_root / f"catalog-{index}.json"
            path.write_text(json.dumps({"alarms": [alarm]}))
            config.generation.catalog.append(str(path))
    monkeypatch.setattr(
        run_all,
        "RetrievalOptions",
        lambda **kwargs: RetrievalOptions(docs_root=deps.docs_root, **kwargs),
    )
    caller = Mock(
        return_value=json.dumps(
            {
                "summary": "Grounded guidance",
                "likely_causes": ["Service fault"],
                "resolution_steps": ["Consult the procedure"],
            }
        )
    )
    judge = Mock(return_value={"checked": True})
    monkeypatch.setattr(run_all, "call_ollama_llm", caller)
    monkeypatch.setattr(run_all.judge_results, "judge", judge)
    output_dir = deps.docs_root / "results"
    arguments = argparse.Namespace(codes=None, output_dir=output_dir)
    assert run_all._generate_and_judge(arguments, config, {}, None) == 0
    deps.encoder_factory.assert_called_once()
    deps.cross_encoder_factory.assert_called_once()
    deps.database_factory.assert_called_once()
    assert deps.cross_encoder.predict.call_count == 2
    assert caller.call_count == 2
    assert judge.call_count == 2
    results = [json.loads(path.read_text()) for path in output_dir.glob("*.json")]
    assert {result["code"] for result in results} == set(deps.catalog)
    assert all(result["evidence"][0]["text"] for result in results)
    assert all(call.args[4] is deps.encoder for call in judge.call_args_list)
