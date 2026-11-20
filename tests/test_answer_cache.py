"""A repeated CLI alarm uses its completed answer before heavy RAG startup."""

import json
import sys
from types import ModuleType

import pytest

from rag_engine.cli import answer as answer_alarm
from rag_engine.engine.rerank.candidate import RerankCandidate
from rag_engine.engine.retrieve.pipeline import RetrievalResult
from rag_engine.engine.vector_store.database_metadata_model import Chunk


def test_cli_reuses_answer_and_invalidates_changed_inputs(
    tmp_path, monkeypatch, capsys
):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    document = docs_root / "manual.md"
    document.write_text("Reset the alarm.")
    cache_dir = tmp_path / "answer_cache"
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"code": "demo.test.0001", "env": {"locale": "en"}}))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "alarms": [
                    {
                        "code": "demo.test.0001",
                        "alarm_text": "Reset needed",
                        "severity": 833,
                        "severity_category": "Error",
                    }
                ]
            }
        )
    )
    chunk = Chunk(
        chunk_id="stored-1",
        embedding=[],
        document_id="manual",
        source_path=["guide", "manual.md"],
        section="Reset",
        start_offset=0,
        end_offset=16,
        metadata={},
    )
    retrieval = RetrievalResult(
        evidence=[RerankCandidate(chunk=chunk, text="Reset the alarm.")],
        strategy="none",
        stop_reason="strategy_selected",
    )
    operations = {"database": 0, "retrieval": 0, "llm": 0}

    def open_database(**_kwargs):
        operations["database"] += 1
        return object()

    def retrieve_evidence(**_kwargs):
        operations["retrieval"] += 1
        return retrieval

    def call_ollama_llm(_system, _user, **_kwargs):
        operations["llm"] += 1
        return json.dumps(
            {
                "summary": "Reset is required.",
                "likely_causes": ["A reset is pending."],
                "resolution_steps": ["Reset the alarm."],
            }
        )

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = lambda _model: object()
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setattr(answer_alarm, "open_database", open_database)
    monkeypatch.setattr(answer_alarm, "retrieve_evidence", retrieve_evidence)
    monkeypatch.setattr(answer_alarm, "call_ollama_llm", call_ollama_llm)
    monkeypatch.setattr(answer_alarm, "DEFAULT_DOCS_ROOT", docs_root, raising=False)
    monkeypatch.setattr(
        answer_alarm, "DEFAULT_ANSWER_CACHE_DIR", cache_dir, raising=False
    )
    args = [
        str(report),
        "--catalog",
        str(catalog),
        "--database-path",
        str(tmp_path / "db"),
        "--reranker",
        "none",
    ]

    assert answer_alarm.main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["answer"]["summary"] == "Reset is required."
    assert first["answer"]["severity"] == 833
    assert answer_alarm.main(args) == 0
    assert json.loads(capsys.readouterr().out) == first
    assert operations == {"database": 1, "retrieval": 1, "llm": 1}

    cached_file = next(cache_dir.glob("*.json"))
    cached_file.write_text("not json")
    assert answer_alarm.main(args) == 0
    assert json.loads(capsys.readouterr().out) == first
    assert operations["llm"] == 2
    assert answer_alarm.main(args) == 0
    capsys.readouterr()
    assert operations["llm"] == 2

    report.write_text(
        json.dumps(
            {
                "code": "demo.test.0001",
                "env": {"locale": "en"},
                "context": "after startup",
            }
        )
    )
    assert answer_alarm.main(args) == 0
    capsys.readouterr()
    assert operations["llm"] == 3

    document.write_text("Updated reset procedure.")
    assert answer_alarm.main(args) == 0
    capsys.readouterr()
    assert operations["llm"] == 4

    assert answer_alarm.main([*args, "--question", "What happened?"]) == 0
    capsys.readouterr()
    assert operations["llm"] == 5

    # A model other than the default must miss the cache: the model is part of
    # the cache key.
    assert answer_alarm.main([*args, "--model", "qwen3:1.7b"]) == 0
    capsys.readouterr()
    assert operations["llm"] == 6
    assert len(list(cache_dir.glob("*.json"))) == 5


def _make_fresh_harness(tmp_path, monkeypatch):
    """Build an isolated CLI harness whose LLM answer changes per call."""

    from rag_engine.cli import answer as answer_module

    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "manual.md").write_text("Reset the alarm.")
    cache_dir = tmp_path / "answer_cache"
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"code": "demo.test.0001"}))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "alarms": [
                    {
                        "code": "demo.test.0001",
                        "alarm_text": "Reset needed",
                    }
                ]
            }
        )
    )
    chunk = Chunk(
        chunk_id="stored-1",
        embedding=[],
        document_id="manual",
        source_path=["guide", "manual.md"],
        section="Reset",
        start_offset=0,
        end_offset=16,
        metadata={},
    )
    retrieval = RetrievalResult(
        evidence=[RerankCandidate(chunk=chunk, text="Reset the alarm.")],
        strategy="none",
        stop_reason="strategy_selected",
    )
    operations = {"llm": 0}

    def open_database(**_kwargs):
        return object()

    def retrieve_evidence(**_kwargs):
        return retrieval

    def call_ollama_llm(_system, _user, **_kwargs):
        operations["llm"] += 1
        return json.dumps(
            {
                "summary": f"Reset answer {operations['llm']}.",
                "likely_causes": ["A reset is pending."],
                "resolution_steps": ["Reset the alarm."],
            }
        )

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = lambda _model: object()
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setattr(answer_module, "open_database", open_database)
    monkeypatch.setattr(answer_module, "retrieve_evidence", retrieve_evidence)
    monkeypatch.setattr(answer_module, "call_ollama_llm", call_ollama_llm)
    monkeypatch.setattr(answer_module, "DEFAULT_DOCS_ROOT", docs_root, raising=False)
    monkeypatch.setattr(
        answer_module, "DEFAULT_ANSWER_CACHE_DIR", cache_dir, raising=False
    )
    args = [
        str(report),
        "--catalog",
        str(catalog),
        "--database-path",
        str(tmp_path / "db"),
        "--reranker",
        "none",
    ]
    return answer_module, args, operations, cache_dir


def test_cli_resolves_alarm_from_second_catalogue(tmp_path, monkeypatch, capsys):
    answer_alarm, args, operations, _cache_dir = _make_fresh_harness(
        tmp_path, monkeypatch
    )
    code = "demo.test.0002"
    (tmp_path / "report.json").write_text(json.dumps({"code": code}))
    extra_catalogue = tmp_path / "extra.json"
    extra_catalogue.write_text(
        json.dumps({"alarms": [{"code": code, "alarm_text": "Another service fault"}]})
    )

    assert answer_alarm.main([*args, "--catalog", str(extra_catalogue)]) == 0
    assert json.loads(capsys.readouterr().out)["code"] == code
    assert operations["llm"] == 1


def test_cli_rejects_duplicate_codes_across_catalogues(tmp_path, monkeypatch):
    answer_alarm, args, operations, _cache_dir = _make_fresh_harness(
        tmp_path, monkeypatch
    )
    extra_catalogue = tmp_path / "extra.json"
    extra_catalogue.write_text((tmp_path / "catalog.json").read_text())

    with pytest.raises(ValueError, match="more than one catalogue"):
        answer_alarm.main([*args, "--catalog", str(extra_catalogue)])
    assert operations["llm"] == 0


def test_fresh_flag_regenerates_and_overwrites_same_key(tmp_path, monkeypatch, capsys):
    answer_alarm, args, operations, cache_dir = _make_fresh_harness(
        tmp_path, monkeypatch
    )

    assert answer_alarm.main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert operations["llm"] == 1

    assert answer_alarm.main(args) == 0
    assert json.loads(capsys.readouterr().out) == first
    assert operations["llm"] == 1

    assert answer_alarm.main([*args, "--fresh"]) == 0
    refreshed = json.loads(capsys.readouterr().out)
    assert operations["llm"] == 2
    assert refreshed["answer"]["summary"] != first["answer"]["summary"]
    assert len(list(cache_dir.glob("*.json"))) == 1

    assert answer_alarm.main(args) == 0
    assert json.loads(capsys.readouterr().out) == refreshed
    assert operations["llm"] == 2


def test_cache_hit_miss_and_refresh_log_to_stderr(tmp_path, monkeypatch, capsys):
    answer_alarm, args, operations, _cache_dir = _make_fresh_harness(
        tmp_path, monkeypatch
    )

    assert answer_alarm.main(args) == 0
    assert "miss" in capsys.readouterr().err.lower()

    assert answer_alarm.main(args) == 0
    assert "hit" in capsys.readouterr().err.lower()
    assert operations["llm"] == 1

    assert answer_alarm.main([*args, "--fresh"]) == 0
    assert "refresh" in capsys.readouterr().err.lower()
    assert operations["llm"] == 2


@pytest.mark.parametrize(
    "invalid_answer",
    [
        {},
        {"summary": "", "likely_causes": ["cause"], "resolution_steps": ["step"]},
        {"summary": "ok", "likely_causes": [], "resolution_steps": ["step"]},
        {"summary": "ok", "likely_causes": ["cause"], "resolution_steps": "step"},
        {"summary": "ok", "likely_causes": [None], "resolution_steps": ["step"]},
    ],
)
def test_incomplete_cached_answer_is_regenerated(
    tmp_path, monkeypatch, capsys, invalid_answer
):
    answer_module, args, operations, cache_dir = _make_fresh_harness(
        tmp_path, monkeypatch
    )
    assert answer_module.main(args) == 0
    capsys.readouterr()
    cache_file = next(cache_dir.glob("*.json"))
    record = json.loads(cache_file.read_text())
    record["result"]["answer"] = invalid_answer
    cache_file.write_text(json.dumps(record))

    assert answer_module.main(args) == 0
    regenerated = json.loads(capsys.readouterr().out)
    assert operations["llm"] == 2
    assert regenerated["answer"]["summary"] == "Reset answer 2."
    assert answer_module.main(args) == 0
    assert json.loads(capsys.readouterr().out) == regenerated
    assert operations["llm"] == 2
