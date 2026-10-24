from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag_engine.engine.llm.generation import generate_answer
from rag_engine.engine.rerank.llm_reranker.pipeline import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_POOL_CHAR_BUDGET,
    DEFAULT_POOL_CHUNK_CHARS,
    DEFAULT_SUPPLEMENT_LIMIT,
)
from rag_engine.engine.retrieve.pipeline import RetrievalResult, retrieve_evidence
from rag_engine.settings import (
    DEFAULT_DOCS_ROOT,
    DEFAULT_EVIDENCE_CHAR_BUDGET,
    DEFAULT_EVIDENCE_MIN_CHUNK_CHARS,
)

from .resources import RagRuntime


@dataclass(frozen=True, slots=True)
class RetrievalOptions:
    """Fixed retrieval behavior for requests sharing one service."""

    chunk_limit: int = 50
    llm_chunk_limit: int | None = None
    max_rounds: int = 0
    docs_root: str | Path = DEFAULT_DOCS_ROOT
    evidence_char_budget: int = DEFAULT_EVIDENCE_CHAR_BUDGET
    evidence_min_chunk_chars: int = DEFAULT_EVIDENCE_MIN_CHUNK_CHARS
    llm_pool_char_budget: int = DEFAULT_POOL_CHAR_BUDGET
    llm_pool_chunk_chars: int = DEFAULT_POOL_CHUNK_CHARS
    llm_supplement_limit: int = DEFAULT_SUPPLEMENT_LIMIT
    llm_max_candidates: int = DEFAULT_MAX_CANDIDATES

    def __post_init__(self) -> None:
        for name in (
            "evidence_char_budget",
            "llm_pool_char_budget",
            "llm_pool_chunk_chars",
            "llm_supplement_limit",
            "llm_max_candidates",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.chunk_limit < 1:
            raise ValueError("chunk_limit must be at least 1")
        if self.llm_chunk_limit is not None and self.llm_chunk_limit < 1:
            raise ValueError("llm_chunk_limit must be at least 1")
        if self.llm_max_candidates < self.llm_supplement_limit:
            raise ValueError("llm_max_candidates must be at least llm_supplement_limit")
        if self.max_rounds < 0 or self.evidence_min_chunk_chars < 0:
            raise ValueError(
                "max_rounds and evidence_min_chunk_chars cannot be negative"
            )


@dataclass(frozen=True, slots=True)
class RagService:
    """Reuse resources while keeping evidence and answers local to each request.

    The runtime owns the rerank strategy. The caller configures answer generation.
    """

    runtime: RagRuntime
    options: RetrievalOptions = field(default_factory=RetrievalOptions)
    final_answer_fn: Callable[..., dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.options.chunk_limit < self.runtime.file_limit:
            raise ValueError("chunk_limit must be at least the configured file_limit")
        if (
            self.options.llm_chunk_limit is not None
            and self.options.llm_chunk_limit < self.runtime.file_limit
        ):
            raise ValueError(
                "llm_chunk_limit must be at least the configured file_limit"
            )

    def retrieve(
        self,
        input_report: Mapping[str, Any],
        *,
        user_question: str | None = None,
        input_context: Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        """Retrieve fresh evidence with the existing models and database handle."""
        return retrieve_evidence(
            input_report=input_report,
            catalog=self.runtime.catalog,
            embedder=self.runtime.embedder,
            database=self.runtime.database,
            reranker=self.runtime.reranker,
            rerank_strategy=self.runtime.rerank_strategy,
            file_limit=self.runtime.file_limit,
            user_question=user_question,
            input_context=input_context,
            **asdict(self.options),
        )

    def resolve(
        self,
        input_report: Mapping[str, Any],
        *,
        user_question: str | None = None,
    ) -> dict[str, Any]:
        """Generatefinal answer"""
        if self.final_answer_fn is None:
            raise ValueError("resolve requires a final_answer_fn")
        retrieval = self.retrieve(
            input_report, user_question=user_question, input_context=input_report
        )
        return self.generate(input_report, retrieval, user_question=user_question)

    def generate(
        self,
        input_report: Mapping[str, Any],
        retrieval: RetrievalResult,
        *,
        user_question: str | None = None,
    ) -> dict[str, Any]:
        """Generate from existing evidence using the configured final_answer_fn.

        The serving backend can schedule this stage without repeating retrieval.
        Callers must configure final_answer_fn before using this method.
        """
        code = input_report["code"]
        alarm = self.runtime.catalog[code]
        return {
            "code": code,
            **generate_answer(
                alarm_code=code,
                alarm_text=alarm["alarm_text"],
                retrieval=retrieval,
                final_answer_fn=self.final_answer_fn,
                user_question=user_question,
                alarm_data=alarm,
            ),
        }
