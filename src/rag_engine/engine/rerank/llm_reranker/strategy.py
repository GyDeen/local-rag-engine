import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_engine.engine.llm.answer_validation import extract_json_object
from rag_engine.engine.rerank.candidate import RerankCandidate
from rag_engine.engine.rerank.llm_reranker.prompts import SYSTEM_PROMPT
from rag_engine.settings import DEFAULT_FILE_LIMIT

DEFAULT_MAX_ATTEMPTS = 2

# Any transport that turns a (system, user) prompt pair into raw model text.
LLMCaller = Callable[[str, str], str]


@dataclass(slots=True)
class EvidenceAssessment:
    """LLM judgement about whether the supplied evidence can answer the alarm."""

    sufficient: bool
    missing: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    operator_questions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RerankDecision:
    """One LLM judgement: the ranking, its assessment and the validated reply.

    ``raw`` is the decision exactly as the LLM produced it. It is kept because
    the next round replays it as ``previous_decision`` and the exploration trace
    records it verbatim.
    """

    ranked: list[RerankCandidate]
    assessment: EvidenceAssessment
    raw: dict[str, Any]
    validation_errors: list[str] = field(default_factory=list)


def _strings(value: object, name: str) -> list[str]:
    """Validate a list of non-empty strings and drop duplicates."""

    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"Reranker {name} must be a list of non-empty strings")
    return list(dict.fromkeys(item.strip() for item in value))


class LLMReranker:
    """Rank candidates with the LLM and report what evidence is still missing."""

    def __init__(
        self,
        llm_caller: LLMCaller,
        file_limit: int = DEFAULT_FILE_LIMIT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        if file_limit < 1:
            raise ValueError("file_limit must be at least 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._llm_caller = llm_caller
        self._file_limit = file_limit
        self._max_attempts = max_attempts

    @property
    def file_limit(self) -> int:
        """Return the file cap configured when this strategy was constructed."""
        return self._file_limit

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
    ) -> list[RerankCandidate]:
        """Return the LLM's ranking of ``candidates`` for ``query``."""

        return self.judge(query, candidates).ranked

    def judge(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        alarm_code: str | None = None,
        alarm_text: str | None = None,
        user_question: str | None = None,
        input_context: dict[str, Any] | None = None,
        remaining_search_rounds: int = 0,
        previous_decision: dict[str, Any] | None = None,
        search_history: Sequence[dict[str, Any]] = (),
        instruction: str | None = None,
    ) -> RerankDecision:
        """Run one LLM judgement and return both outputs of that single call.

        ``query`` is the retrieval query. When the caller does not supply the
        alarm wording explicitly, the query itself stands in for it.
        ``instruction`` adds a one-off directive to this request; the exploration
        loop uses it to demand a reformulation instead of accepting a reply that
        repeats work already done.
        """

        payload: dict[str, Any] = {
            "alarm_code": alarm_code if alarm_code is not None else "",
            "alarm_text": alarm_text if alarm_text is not None else query,
            "user_question": user_question,
            "input_context": input_context,
            "file_limit": self._file_limit,
            "remaining_search_rounds": remaining_search_rounds,
            "previous_decision": previous_decision,
            "search_history": list(search_history),
            "candidates": [
                {
                    "chunk_id": candidate.chunk.chunk_id,
                    "path": "/".join(candidate.chunk.source_path),
                    "section": candidate.chunk.section,
                    "text": candidate.text,
                    "truncated": candidate.truncated,
                }
                for candidate in candidates
            ],
        }
        if instruction:
            payload["instruction"] = instruction
        validation_errors: list[str] = []

        for attempt in range(self._max_attempts):
            raw_response = self._llm_caller(
                SYSTEM_PROMPT,
                json.dumps(payload, ensure_ascii=False),
            )
            try:
                return self._parse(raw_response, candidates, validation_errors)
            except ValueError as error:
                if attempt == self._max_attempts - 1:
                    raise
                validation_errors.append(str(error))
                payload["validation_error"] = str(error)
                payload["instruction"] = (
                    "Correct the invalid response and return the required JSON."
                )

        raise AssertionError("unreachable")

    def _parse(
        self,
        raw_response: str,
        candidates: list[RerankCandidate],
        validation_errors: list[str],
    ) -> RerankDecision:
        """Validate one raw LLM response or raise ``ValueError``."""

        decision = extract_json_object(raw_response)
        ranked = _strings(decision.get("ranked_chunk_ids"), "ranked_chunk_ids")
        lookup = {candidate.chunk.chunk_id: candidate for candidate in candidates}

        if any(chunk_id not in lookup for chunk_id in ranked):
            raise ValueError("Reranker referenced an unseen chunk")
        files = {tuple(lookup[chunk_id].chunk.source_path) for chunk_id in ranked}
        if len(files) > self._file_limit:
            raise ValueError("Reranker exceeded the file limit")

        coverage = decision.get("coverage")
        if not isinstance(coverage, dict):
            raise ValueError("Reranker requires evidence coverage")
        for field_name in ("explanation", "resolution"):
            coverage[field_name] = _strings(coverage.get(field_name), field_name)
            if not set(coverage[field_name]).issubset(ranked):
                raise ValueError("Coverage must reference selected chunks only")

        for field_name in ("missing", "search_queries", "operator_questions"):
            decision[field_name] = _strings(decision.get(field_name), field_name)

        if not isinstance(decision.get("sufficient"), bool):
            raise ValueError("Reranker sufficient must be a boolean")
        if not isinstance(decision.get("reason"), str):
            raise ValueError("Reranker requires a justification")

        decision["ranked_chunk_ids"] = ranked
        decision["search_queries"] = decision["search_queries"][:2]

        # Structural checks cannot prove semantic completeness, but they prevent
        # obvious contradictions such as claiming sufficiency with no resolution
        # evidence.
        decision["sufficient"] = bool(
            decision["sufficient"]
            and coverage["explanation"]
            and coverage["resolution"]
            and not decision["missing"]
            and not decision["operator_questions"]
        )

        return RerankDecision(
            ranked=[lookup[chunk_id] for chunk_id in ranked],
            assessment=EvidenceAssessment(
                sufficient=decision["sufficient"],
                missing=list(decision["missing"]),
                search_queries=list(decision["search_queries"]),
                operator_questions=list(decision["operator_questions"]),
            ),
            raw=decision,
            validation_errors=list(validation_errors),
        )
