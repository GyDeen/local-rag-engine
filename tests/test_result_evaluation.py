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
from rag_engine.evaluation.judge_results import (
    judge,
    judge_saved_results,
    strip_source_tags,
)


class IdenticalEmbedder:
    """Return identical unit vectors so schema integration is deterministic."""

    def encode(self, texts, normalize_embeddings=True):
        return [[1.0, 0.0] for _ in texts]


REFERENCE_DOCUMENT = {
    "answers": {
        "demo.test.0001": {
            "summary": "Worker overheated.",
            "likely_causes": ["Cooling failure"],
            "resolution_steps": ["Inspect the fan"],
            "doc_refs": [{"path": "guide.md", "section": "Cooling"}],
        }
    }
}

CURRENT_ANSWER = {
    "summary": "The worker is too hot.",
    "likely_causes": ["The fan failed"],
    "resolution_steps": ["Check the fan"],
    "doc_refs": [{"path": "guide.md", "section": "Cooling"}],
    "doc_coverage": "full",
    "notes": "Grounded in retrieved evidence.",
}


@pytest.mark.parametrize("identifier_field", ["id", "code"])
def test_evaluate_result_accepts_current_nested_answer_schema(identifier_field):
    generated = {
        identifier_field: "demo.test.0001",
        "configuration": {"collection": "config-a"},
        "answer": CURRENT_ANSWER,
    }

    result = evaluate_result(
        REFERENCE_DOCUMENT,
        generated,
        IdenticalEmbedder(),
        k=1,
    )

    assert result[identifier_field] == "demo.test.0001"
    assert result["configuration"] == {"collection": "config-a"}
    assert result["evaluation"]["scores"]["summary"]["f1"] == 1.0
    assert result["evaluation"]["scores"]["citations"]["f1"] == 1.0


def test_evaluate_output_files_accepts_configuration_artifact(tmp_path: Path):
    reference_path = tmp_path / "reference.json"
    generated_path = tmp_path / "generated.json"
    reference_path.write_text(json.dumps(REFERENCE_DOCUMENT), encoding="utf-8")
    generated_path.write_text(
        json.dumps(
            {
                "code": "demo.test.0001",
                "configuration": {"llm_model": "test-model"},
                "answer": CURRENT_ANSWER,
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

    assert result["code"] == "demo.test.0001"
    assert result["configuration"]["llm_model"] == "test-model"


def test_evaluator_retains_legacy_generated_field_names():
    generated = {
        "code": "demo.test.0001",
        "summary": "The worker is too hot.",
        "likely_causes": ["The fan failed"],
        "steps": ["Check the fan"],
        "citations": [{"path": "guide.md", "section": "Cooling"}],
    }

    result = evaluate_result(
        REFERENCE_DOCUMENT,
        generated,
        IdenticalEmbedder(),
        k=1,
    )

    assert result["evaluation"]["scores"]["resolution_steps"]["f1"] == 1.0


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


def test_judge_scores_text_and_cited_pages_from_the_shared_response():
    reference = {
        "answers": {
            "manual.question": {
                "answer": "check",
                "answerable": True,
                "references": [{"source": "guide.pdf", "page": 4}],
            }
        }
    }
    response = {
        "id": "manual.question",
        "answer": "check [E1] [E1] [E9]",
        "status": "answered",
    }
    evidence = [
        {"id": "E1", "doc": "guide.pdf", "pages": [2]},
        {"id": "E2", "doc": "guide.pdf", "pages": [4]},
    ]

    evaluated = judge(
        reference,
        response,
        evidence,
        EvaluationSettings(judges=["cosine"]),
        StepEmbedder(),
        None,
    )["cosine"]

    assert evaluated["answer_cosine_similarity"] == 1.0
    assert evaluated["answerability_correct"] is True
    assert evaluated["citations"] == {
        "ids": ["E1", "E9"],
        "unresolved_ids": ["E9"],
        "gold_source_recall": 1.0,
        "gold_page_recall": 0.0,
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


TAGGED_ANSWER = {
    "summary": "The worker is too hot.",
    "likely_causes": ["The fan failed"],
    "resolution_steps": ["Check the fan [C2]", "Reset the worker [C1] [C3]"],
    "doc_refs": [{"path": "guide.md", "section": "Cooling"}],
    "doc_coverage": "full",
    "notes": "Grounded in retrieved evidence.",
}

UNTAGGED_ANSWER = {
    **CURRENT_ANSWER,
    "resolution_steps": ["Check the fan", "Reset the worker"],
}


def test_strip_source_tags_removes_only_citation_tags():
    stripped = strip_source_tags(TAGGED_ANSWER)

    assert stripped["summary"] == "The worker is too hot."
    assert stripped["likely_causes"] == ["The fan failed"]
    assert stripped["resolution_steps"] == ["Check the fan", "Reset the worker"]
    # The original answer keeps its tags for traceability.
    assert TAGGED_ANSWER["resolution_steps"] == [
        "Check the fan [C2]",
        "Reset the worker [C1] [C3]",
    ]


def test_strip_source_tags_leaves_untagged_text_unchanged():
    assert strip_source_tags(CURRENT_ANSWER) == CURRENT_ANSWER


def test_judge_scores_tagged_and_untagged_answers_identically():
    settings = EvaluationSettings(judges=["cosine"])
    tagged_result = {
        "code": "demo.test.0001",
        "answer": TAGGED_ANSWER,
    }
    untagged_result = {
        "code": "demo.test.0001",
        "answer": UNTAGGED_ANSWER,
    }

    tagged = judge(
        REFERENCE_DOCUMENT, tagged_result, [], settings, IdenticalEmbedder(), None
    )
    untagged = judge(
        REFERENCE_DOCUMENT, untagged_result, [], settings, IdenticalEmbedder(), None
    )

    assert tagged["cosine"]["scores"] == untagged["cosine"]["scores"]


def test_judge_fails_when_an_alarm_answer_is_a_list_of_pairs():
    result = {"id": "demo.test.0001", "answer": list(CURRENT_ANSWER.items())}

    with pytest.raises(AttributeError, match="has no attribute 'get'"):
        judge(
            REFERENCE_DOCUMENT,
            result,
            [],
            EvaluationSettings(judges=["cosine"]),
            IdenticalEmbedder(),
            None,
        )


def test_judge_sends_stripped_text_to_the_llm_judge():
    captured = {}

    def fake_judge_caller(system_prompt: str, user_prompt: str) -> str:
        captured["payload"] = json.loads(user_prompt)
        return json.dumps({"criterion_coverage": [], "claims": []})

    settings = EvaluationSettings(judges=["llm"])
    result = {"code": "demo.test.0001", "answer": TAGGED_ANSWER}

    evaluation = judge(
        REFERENCE_DOCUMENT, result, [], settings, None, fake_judge_caller
    )

    assert evaluation["llm"]["status"] == "completed"
    judged_steps = [
        field["text"]
        for field in captured["payload"]["generated_answer"]
        if field["field_id"].startswith("step_")
    ]
    assert judged_steps == ["Check the fan", "Reset the worker"]


def test_judge_saved_results_preserves_tagged_answer(tmp_path):
    input_path = tmp_path / "answer.json"
    input_path.write_text(
        json.dumps({"code": "demo.test.0001", "answer": TAGGED_ANSWER, "evidence": []})
    )
    output_dir = tmp_path / "judged"
    settings = EvaluationSettings(judges=[])

    assert (
        judge_saved_results(
            [input_path], output_dir, settings, REFERENCE_DOCUMENT, None
        )
        == 0
    )

    saved = json.loads(next(output_dir.glob("*.json")).read_text())
    assert saved["answer"] == TAGGED_ANSWER
    assert saved["evaluation"] == {}
