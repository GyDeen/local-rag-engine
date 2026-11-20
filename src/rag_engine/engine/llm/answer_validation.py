"""Parse and validate grounded LLM answers."""

import json
from typing import Any, Dict, List

from rag_engine.engine.llm.evidence import UNKNOWN
from rag_engine.exceptions import ModelResponseError


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from the LLM response.

    The prompt requests JSON-only output, but this makes the integration
    more robust in case the LLM accidentally emits extra text.
    """

    decoder = json.JSONDecoder()

    start = text.find("{")

    if start == -1:
        raise ModelResponseError("LLM did not return a JSON object")

    try:
        parsed, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ModelResponseError("LLM returned invalid JSON") from exc

    if not isinstance(parsed, dict):
        raise ModelResponseError("LLM output must be a JSON object.")

    return parsed


def _ensure_string(value: Any) -> str:
    """Ensure a value is a non-empty string."""

    if isinstance(value, str) and value.strip():
        return value.strip()

    return UNKNOWN


def _ensure_string_list(value: Any) -> List[str]:
    """Ensure a value is a non-empty list of strings."""

    if not isinstance(value, list):
        return [UNKNOWN]

    result = [str(item).strip() for item in value if str(item).strip()]

    return result if result else [UNKNOWN]


def validate_llm_answer(answer: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the three text fields written by the final-answer model."""

    summary = _ensure_string(answer.get("summary"))

    likely_causes = _ensure_string_list(answer.get("likely_causes"))

    resolution_steps = _ensure_string_list(answer.get("resolution_steps"))

    return {
        "summary": summary,
        "likely_causes": likely_causes,
        "resolution_steps": resolution_steps,
    }
