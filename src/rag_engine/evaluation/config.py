from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from rag_engine.engine.llm.llm_callers import OLLAMA_HOST, OLLAMA_MODEL
from rag_engine.engine.rerank.llm_reranker.pipeline import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_POOL_CHAR_BUDGET,
    DEFAULT_POOL_CHUNK_CHARS,
    DEFAULT_SUPPLEMENT_LIMIT,
    MAX_SEARCH_ROUNDS,
)
from rag_engine.evaluation import DEFAULT_EVALUATION_K
from rag_engine.evaluation.llm_judge.callers import (
    DEFAULT_API_BASE_URL,
    DEFAULT_JUDGE_MODEL,
)
from rag_engine.settings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_DATABASE_BACKEND,
    DEFAULT_DATABASE_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EVIDENCE_CHAR_BUDGET,
    DEFAULT_EVIDENCE_MIN_CHUNK_CHARS,
    DEFAULT_FILE_LIMIT,
    PROJECT_ROOT,
    RERANKERS,
    validate_evidence_rank_limit,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/evaluation.json"
DEFAULT_REFERENCE_ANSWERS = PROJECT_ROOT / "examples/reference-answers.json"

JUDGES = ("llm", "cosine")
JUDGE_PROVIDERS = ("openai", "openai-compatible", "ollama")
JUDGE_APIS = ("responses", "chat_completions")

_SECTIONS = ("generation", "evaluation")


@dataclass(slots=True)
class GenerationSettings:
    """How one answer is retrieved, reranked and generated."""

    model: str = OLLAMA_MODEL
    host: str = OLLAMA_HOST
    context_length: int = 16384
    reranker: str = "cross_encoder"
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL
    chunk_limit: int = 50
    llm_chunk_limit: int | None = None
    file_limit: int = DEFAULT_FILE_LIMIT
    evidence_char_budget: int = DEFAULT_EVIDENCE_CHAR_BUDGET
    evidence_min_chunk_chars: int = DEFAULT_EVIDENCE_MIN_CHUNK_CHARS
    max_rounds: int = MAX_SEARCH_ROUNDS
    llm_pool_char_budget: int = DEFAULT_POOL_CHAR_BUDGET
    llm_pool_chunk_chars: int = DEFAULT_POOL_CHUNK_CHARS
    llm_supplement_limit: int = DEFAULT_SUPPLEMENT_LIMIT
    llm_max_candidates: int = DEFAULT_MAX_CANDIDATES
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    database_backend: str = DEFAULT_DATABASE_BACKEND
    database_path: str = str(DEFAULT_DATABASE_PATH)
    collection: str = DEFAULT_COLLECTION_NAME
    cases: str = str(PROJECT_ROOT / "examples/questions.jsonl")
    evidence_rank_limit: int | None = None


@dataclass(slots=True)
class JudgeSettings:
    """The LLM judge endpoint, selected independently of the generator."""

    model: str = DEFAULT_JUDGE_MODEL
    provider: str = "ollama"
    base_url: str = DEFAULT_API_BASE_URL
    api: str = "chat_completions"
    api_key_env: str = "OPENAI_API_KEY"
    reasoning_effort: str | None = None
    max_output_tokens: int = 8192
    timeout: float = 300


@dataclass(slots=True)
class EvaluationSettings:
    """Which judges run, and how the cosine judge compares answers."""

    judges: list[str] = field(default_factory=lambda: ["llm", "cosine"])
    model: str = DEFAULT_EMBEDDING_MODEL
    k: int = DEFAULT_EVALUATION_K
    reference_answers: str = str(DEFAULT_REFERENCE_ANSWERS)
    llm: JudgeSettings = field(default_factory=JudgeSettings)


@dataclass(slots=True)
class EvaluationConfig:
    """The complete run configuration."""

    generation: GenerationSettings = field(default_factory=GenerationSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)


def _build_section(section: type, data: Any, name: str) -> Any:
    if not isinstance(data, Mapping):
        raise ValueError(f"'{name}' must be a JSON object")
    known = {setting.name for setting in fields(section)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown setting(s) in '{name}': {', '.join(unknown)}")
    values = dict(data)
    if "llm" in values:
        values["llm"] = _build_section(JudgeSettings, values["llm"], f"{name}.llm")
    return section(**values)


def validate(config: EvaluationConfig) -> None:
    """Reject settings that would fail later, with the setting named."""
    generation = config.generation
    evaluation = config.evaluation

    if generation.reranker not in RERANKERS:
        raise ValueError(f"generation.reranker must be one of {', '.join(RERANKERS)}")
    if generation.file_limit < 1:
        raise ValueError("generation.file_limit must be at least 1")
    if generation.chunk_limit < generation.file_limit:
        raise ValueError("generation.chunk_limit must be at least file_limit")
    if (
        generation.llm_chunk_limit is not None
        and generation.llm_chunk_limit < generation.file_limit
    ):
        raise ValueError(
            "generation.llm_chunk_limit must be at least file_limit when set"
        )
    for name in (
        "llm_pool_char_budget",
        "llm_pool_chunk_chars",
        "llm_supplement_limit",
        "llm_max_candidates",
    ):
        if getattr(generation, name) < 1:
            raise ValueError(f"generation.{name} must be at least 1")
    if generation.llm_max_candidates < generation.llm_supplement_limit:
        raise ValueError(
            "generation.llm_max_candidates must be at least llm_supplement_limit"
        )
    validate_evidence_rank_limit(generation.evidence_rank_limit)
    if generation.evidence_char_budget < 1:
        raise ValueError("generation.evidence_char_budget must be at least 1")
    if generation.evidence_min_chunk_chars < 0:
        raise ValueError("generation.evidence_min_chunk_chars cannot be negative")
    if type(evaluation.k) is not int or evaluation.k < 1:
        raise ValueError("evaluation.k must be a positive integer")
    if not evaluation.judges:
        raise ValueError("evaluation.judges must name at least one judge")
    unknown_judges = sorted(set(evaluation.judges) - set(JUDGES))
    if unknown_judges:
        raise ValueError(
            f"evaluation.judges must be from {', '.join(JUDGES)}; "
            f"got {', '.join(unknown_judges)}"
        )
    if evaluation.llm.provider not in JUDGE_PROVIDERS:
        raise ValueError(
            f"evaluation.llm.provider must be one of {', '.join(JUDGE_PROVIDERS)}"
        )
    if evaluation.llm.api not in JUDGE_APIS:
        raise ValueError(f"evaluation.llm.api must be one of {', '.join(JUDGE_APIS)}")
    if evaluation.llm.max_output_tokens < 1:
        raise ValueError("evaluation.llm.max_output_tokens must be at least 1")
    if evaluation.llm.timeout <= 0:
        raise ValueError("evaluation.llm.timeout must be greater than 0")


def load_config(path: str | Path | None = None) -> EvaluationConfig:
    """Load run settings from JSON, or use the defaults when there is no file."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
        if not Path(path).exists():
            return EvaluationConfig()
    path = Path(path)
    with path.open(encoding="utf-8") as settings_file:
        raw = json.load(settings_file)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ValueError(
            f"Unknown section(s) in {path}: {', '.join(unknown)}; "
            f"expected {', '.join(_SECTIONS)}"
        )
    config = EvaluationConfig(
        generation=_build_section(
            GenerationSettings, raw.get("generation", {}), "generation"
        ),
        evaluation=_build_section(
            EvaluationSettings, raw.get("evaluation", {}), "evaluation"
        ),
    )
    validate(config)
    return config
