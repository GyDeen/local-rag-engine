"""The final model writes prose; answer metadata comes from trusted inputs."""

from rag_engine.engine.llm.generation import (
    build_ollama_final_answer,
    generate_answer,
)
from rag_engine.engine.rerank.candidate import RerankCandidate
from rag_engine.engine.rerank.llm_reranker.strategy import EvidenceAssessment
from rag_engine.engine.retrieve.pipeline import RetrievalResult
from rag_engine.engine.vector_store.database_metadata_model import Chunk


def _retrieval(sufficient: bool | None) -> RetrievalResult:
    chunk = Chunk(
        chunk_id="stored-1",
        embedding=[],
        document_id="manual",
        source_path=["guide", "alarms.md"],
        section="Alarm > Reset",
        start_offset=0,
        end_offset=19,
        source_hash="unused",
        metadata={},
    )
    return RetrievalResult(
        evidence=[RerankCandidate(chunk=chunk, text="Check and reset alarm")],
        strategy="llm" if sufficient is not None else "cross_encoder",
        stop_reason="sufficient" if sufficient else "strategy_selected",
        assessment=(
            EvidenceAssessment(sufficient=sufficient)
            if sufficient is not None
            else None
        ),
    )


def test_final_model_only_writes_three_text_fields():
    prompts = []

    def fake_model(system: str, user: str) -> str:
        prompts.append(system)
        return (
            '{"summary":"Alarm needs a reset",'
            '"likely_causes":["Reset pending"],'
            '"resolution_steps":["Reset the alarm"]}'
        )

    answer = build_ollama_final_answer(fake_model)(
        "demo.net.0001",
        "Reset needed",
        [
            {
                "source_path": ["guide", "alarms.md"],
                "section": "Alarm > Reset",
                "text": "Check and reset alarm",
            }
        ],
    )

    assert answer == {
        "summary": "Alarm needs a reset",
        "likely_causes": ["Reset pending"],
        "resolution_steps": ["Reset the alarm"],
    }
    for field in ("supporting_chunk_ids", "doc_coverage", "notes", "severity"):
        assert field not in prompts[0]


def test_answer_metadata_ignores_model_supplied_values():
    def fake_generator(*args, **kwargs):
        return {
            "summary": "Alarm needs a reset",
            "likely_causes": ["Reset pending"],
            "resolution_steps": ["Reset the alarm"],
            "doc_refs": [{"doc": "Fake", "path": "fake.md", "section": "Fake"}],
            "doc_coverage": "partial",
            "severity": 1,
            "notes": "Invented",
        }

    result = generate_answer(
        alarm_code="demo.net.0001",
        alarm_text="Reset needed",
        alarm_data={"severity": 833, "severity_category": "Error"},
        retrieval=_retrieval(True),
        final_answer_fn=fake_generator,
    )["answer"]

    assert result == {
        "summary": "Alarm needs a reset",
        "likely_causes": ["Reset pending"],
        "resolution_steps": ["Reset the alarm"],
        "doc_refs": [
            {
                "doc": "guide",
                "path": "guide/alarms.md",
                "section": "Alarm > Reset",
            }
        ],
        "doc_coverage": "full",
        "severity": 833,
        "severity_category": "Error",
        "notes": "Generated only from the provided chunks.",
    }


def test_unassessed_evidence_has_partial_coverage():
    result = generate_answer(
        alarm_code="demo.net.0001",
        alarm_text="Reset needed",
        retrieval=_retrieval(None),
        final_answer_fn=lambda *args, **kwargs: {
            "summary": "Alarm needs a reset",
            "likely_causes": ["Reset pending"],
            "resolution_steps": ["Reset the alarm"],
        },
    )["answer"]

    assert result["doc_coverage"] == "partial"
    assert result["doc_refs"][0]["path"] == "guide/alarms.md"
