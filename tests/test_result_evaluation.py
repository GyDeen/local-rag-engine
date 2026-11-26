import json
from pathlib import Path

import pytest

from rag_engine.evaluation.config import EvaluationSettings
from rag_engine.evaluation.cosine.metrics import ordered_item_matches
from rag_engine.evaluation.cosine.result_evaluation import (
    evaluate_output_files,
    evaluate_question_answer,
    evaluate_result,
)
from rag_engine.evaluation.judge_results import judge, judge_saved_results


class IdenticalEmbedder:
    """Return identical unit vectors so schema integration is deterministic."""

    def encode(self, texts, normalize_embeddings=True):
        return [[1.0, 0.0] for _ in texts]


REFERENCE_DOCUMENT = {
    "answers": {
        "demo.test.0001": {
            "answer": "Inspect the fan to check for a cooling failure.",
            "answerable": True,
            "required_steps": ["Inspect the fan"],
            "references": [{"source": "guide.md", "section": "Cooling"}],
        }
    }
}

EVIDENCE = {
    "id": "E1",
    "source": "guide.md",
    "location": "guide.md",
    "section": "Cooling",
    "text": "Check the fan",
}


def test_evaluate_result_scores_question_response():
    generated = {
        "id": "demo.test.0001",
        "configuration": {"collection": "config-a"},
        "answer": "Check the fan. [E1]",
        "status": "answered",
        "evidence": [EVIDENCE],
    }

    result = evaluate_result(
        REFERENCE_DOCUMENT,
        generated,
        IdenticalEmbedder(),
        result_format="question_response",
    )

    assert result["id"] == "demo.test.0001"
    assert result["k"] == 5
    assert result["configuration"] == {"collection": "config-a"}
    assert result["evaluation"]["answer_cosine_similarity"] == 1.0
    assert result["evaluation"]["answerability_correct"] is True
    assert result["evaluation"]["citations"]["gold_section_recall"] == 1.0


def test_evaluate_output_files_accepts_configuration_artifact(tmp_path: Path):
    reference_path = tmp_path / "reference.json"
    generated_path = tmp_path / "generated.json"
    reference_path.write_text(json.dumps(REFERENCE_DOCUMENT), encoding="utf-8")
    generated_path.write_text(
        json.dumps(
            {
                "id": "demo.test.0001",
                "configuration": {"llm_model": "test-model"},
                "answer": "Check the fan [E1]",
                "status": "answered",
                "evidence": [EVIDENCE],
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_output_files(
        reference_path,
        generated_path,
        IdenticalEmbedder(),
        k=1,
    )

    assert result["id"] == "demo.test.0001"
    assert result["k"] == 1
    assert result["configuration"]["llm_model"] == "test-model"
    assert result["evaluation"]["answer_cosine_similarity"] == 1.0
    assert result["evaluation"]["citations"]["gold_section_recall"] == 1.0


class StepEmbedder:
    vectors = {
        "check": [1.0, 0.0, 0.0],
        "transfer": [0.0, 1.0, 0.0],
        "reenable": [0.0, 0.0, 1.0],
    }

    def encode(self, texts, normalize_embeddings=True):
        return [self.vectors[text] for text in texts]


def test_ordered_resolution_matches_do_not_reuse_one_generated_step():
    report = ordered_item_matches(
        ["check", "transfer", "reenable"],
        ["check"],
        StepEmbedder(),
    )

    matches = report["matches"]
    assert [match["generated_index"] for match in matches] == [0, None, None]
    generated_indices = [
        match["generated_index"]
        for match in matches
        if match["generated_index"] is not None
    ]
    assert len(generated_indices) == len(set(generated_indices))
    assert report["unmatched_generated"] == []


@pytest.mark.parametrize(("k", "page_recall"), [(2, 0.0), (3, 1.0)])
def test_judge_scores_page_recall_at_the_citation_cutoff(k, page_recall):
    reference = {
        "answer": "check",
        "answerable": True,
        "references": [{"source": "guide.pdf", "page": 4}],
    }
    response = {
        "id": "manual.question",
        "answer": "check [E1] [E1] [E9] [E2]",
        "status": "answered",
        "evidence": [
            {"id": "E1", "doc": "guide.pdf", "pages": [2]},
            {"id": "E2", "doc": "guide.pdf", "pages": [4]},
        ],
    }

    evaluated = judge(
        reference,
        response,
        EvaluationSettings(judges=["cosine"], k=k),
        StepEmbedder(),
        None,
    )["cosine"]

    assert evaluated["answer_cosine_similarity"] == 1.0
    assert evaluated["answerability_correct"] is True
    assert evaluated["citations"] == {
        "ids": ["E1", "E9", "E2"],
        "unresolved_ids": ["E9"],
        "gold_source_recall": 1.0,
        "gold_page_recall": page_recall,
        "gold_section_recall": None,
    }


@pytest.mark.parametrize(
    ("answerable", "status", "correct"),
    [
        (True, "insufficient_evidence", False),
        (False, "insufficient_evidence", True),
        (False, "answered", False),
    ],
)
def test_evaluator_returns_after_scoring_answerability(answerable, status, correct):
    evaluated = evaluate_question_answer(
        {"answer": "check", "answerable": answerable},
        {"answer": "Unsupported answer. [E1]", "status": status},
        StepEmbedder(),
    )

    assert evaluated == {
        "answer_cosine_similarity": None,
        "answerability_correct": correct,
        "citations": None,
    }


def test_judge_fails_when_answer_is_not_text():
    result = {"status": "answered", "answer": ["Check the fan"], "evidence": []}

    with pytest.raises(TypeError):
        judge(
            REFERENCE_DOCUMENT["answers"]["demo.test.0001"],
            result,
            EvaluationSettings(judges=["cosine"]),
            IdenticalEmbedder(),
            None,
        )


def test_llm_judge_reads_the_same_answer_and_evidence_as_cosine():
    captured = {}

    def fake_judge_caller(system_prompt: str, user_prompt: str) -> str:
        captured["payload"] = json.loads(user_prompt)
        return json.dumps({"criterion_coverage": [], "claims": []})

    settings = EvaluationSettings(judges=["llm"])
    result = {
        "answer": "Check the fan [E1]",
        "status": "answered",
        "evidence": [EVIDENCE],
    }

    evaluation = judge(
        REFERENCE_DOCUMENT["answers"]["demo.test.0001"],
        result,
        settings,
        None,
        fake_judge_caller,
    )

    assert evaluation["llm"]["status"] == "completed"
    assert captured["payload"]["generated_answer"] == [
        {"field_id": "answer", "text": result["answer"]},
    ]
    assert captured["payload"]["retrieved_documentation"][0]["chunk_id"] == "E1"
    assert captured["payload"]["reference_criteria"][1] == {
        "criterion_id": "resolution_1",
        "text": "Inspect the fan",
    }


def test_judge_saved_results_uses_saved_answer_and_evidence(tmp_path):
    input_path = tmp_path / "answer.json"
    saved = {
        "id": "demo.test.0001",
        "answer": "Check the fan [E1]",
        "status": "answered",
        "evidence": [EVIDENCE],
    }
    input_path.write_text(json.dumps(saved))
    output_dir = tmp_path / "judged"
    settings = EvaluationSettings(judges=[])

    assert (
        judge_saved_results(
            [input_path], output_dir, settings, REFERENCE_DOCUMENT, None
        )
        == 0
    )

    evaluated = json.loads(next(output_dir.glob("*.json")).read_text())
    assert {key: evaluated[key] for key in saved} == saved
    assert evaluated["evaluation"] == {}
    assert json.loads(input_path.read_text()) == saved
