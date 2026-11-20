"""Record LLM judgments; keep coverage separate from grounding."""

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rag_engine.engine.llm.answer_validation import extract_json_object
from rag_engine.evaluation.llm_judge.prompts import SYSTEM_PROMPT
from rag_engine.logging_config import log_operation

COVERAGE_STATUSES = {"covered", "partially_covered", "not_covered"}
SUPPORT_STATUSES = {
    "explicitly_supported",
    "requires_inference",
    "unsupported",
    "contradicted",
}


@log_operation("LLM evaluation")
def evaluate_answer(
    reference: Mapping[str, Any],
    generated: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    llm_caller: Callable[[str, str], str],
) -> dict[str, Any]:
    """Judge summary/steps against references and supplied evidence only.

    chunks use normalise_chunks output; no retrieval or external knowledge is
    available to the judge. Pipeline-produced inputs and the judge's response
    structure are trusted; only JSON parsing is performed, without retries.
    """
    criteria = [{"criterion_id": "summary", "text": reference["summary"]}]
    criteria.extend(
        {"criterion_id": f"resolution_{i}", "text": text}
        for i, text in enumerate(reference["resolution_steps"], 1)
    )
    fields = [{"field_id": "summary", "text": generated["summary"]}]
    fields.extend(
        {"field_id": f"step_{i}", "text": text}
        for i, text in enumerate(generated["resolution_steps"], 1)
    )
    documents = [
        {
            "chunk_id": c["llm_chunk_id"],
            "path": c["path"],
            "section": c["section"],
            "text": c["text"],
        }
        for c in chunks
    ]
    payload: dict[str, Any] = {
        "reference_criteria": criteria,
        "generated_answer": fields,
        "retrieved_documentation": documents,
    }
    raw = llm_caller(SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False))
    judgment = extract_json_object(raw)
    missing = [key for key in ("criterion_coverage", "claims") if key not in judgment]
    if missing:
        # A wrong-shaped response is recorded as an error, never repaired, but
        # the caller needs the response itself to fix the model or the prompt.
        raise ValueError(
            f"Judge response is missing {', '.join(missing)}; "
            f"raw response starts: {raw.strip()[:200]}"
        )

    coverage_counts = Counter(x["status"] for x in judgment["criterion_coverage"])
    support_counts = Counter(x["status"] for x in judgment["claims"])
    resolution = [
        x
        for x in judgment["criterion_coverage"]
        if x["criterion_id"].startswith("resolution_")
    ]
    return {
        "scope": "summary and resolution_steps; supplied documentation only",
        "inputs": {
            "reference_criteria": criteria,
            "generated_answer": fields,
            "retrieved_documentation": documents,
        },
        "judgment": judgment,
        "diagnostics": {
            "coverage_counts": {
                x: coverage_counts[x] for x in sorted(COVERAGE_STATUSES)
            },
            "support_counts": {x: support_counts[x] for x in sorted(SUPPORT_STATUSES)},
            "all_resolution_criteria_covered": all(
                x["status"] == "covered" for x in resolution
            ),
            "all_claims_explicitly_supported": (
                all(x["status"] == "explicitly_supported" for x in judgment["claims"])
                if judgment["claims"]
                else None
            ),
        },
    }
