"""Read question batches for evaluation."""

import json
from collections.abc import Mapping
from pathlib import Path


def load_question_cases(path: Path) -> list[dict]:
    """Read JSONL questions, assigning IDs for reports when absent."""
    cases = []
    seen_ids = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            question = case.get("question") if isinstance(case, dict) else None
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{path}:{line_number} needs a nonempty question")
            identifier = case.setdefault("id", f"case-{line_number}")
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(f"{path}:{line_number} needs a nonempty string ID")
            if identifier in seen_ids:
                raise ValueError(f"{path}:{line_number} repeats case ID {identifier!r}")
            seen_ids.add(identifier)
            cases.append(case)
    return cases


def result_id(result: dict) -> str:
    """Read a case ID from current or previously saved results."""
    return result["id"] if "id" in result else result["code"]


def reference_answers(reference_document: dict) -> Mapping[str, dict]:
    """Index reference answers from current and older reference files."""
    answers = reference_document.get("answers", {})
    if isinstance(answers, list):
        return {result_id(answer): answer for answer in answers}
    return answers
