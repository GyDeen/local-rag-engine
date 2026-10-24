from __future__ import annotations

from typing import TYPE_CHECKING

from rag_engine.engine.rerank.base import RerankStrategy
from rag_engine.settings import (
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_FILE_LIMIT,
    RERANKERS,
)

if TYPE_CHECKING:
    from rag_engine.engine.rerank.llm_reranker.strategy import LLMCaller


def create_rerank_strategy(
    reranker: str,
    *,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
    llm_caller: LLMCaller | None = None,
    file_limit: int = DEFAULT_FILE_LIMIT,
) -> RerankStrategy | None:
    """Create a supported strategy; disabled reranking returns no strategy."""
    if reranker not in RERANKERS:
        raise ValueError(f"Unknown reranker {reranker!r}")
    if file_limit < 1:
        raise ValueError("file_limit must be at least 1")
    if reranker == "none":
        return None
    if reranker == "llm":
        from rag_engine.engine.rerank.llm_reranker.strategy import LLMReranker

        if llm_caller is None:
            raise ValueError("The 'llm' reranker requires an llm_caller")
        return LLMReranker(llm_caller, file_limit=file_limit)

    if reranker == "cross_encoder":
        from rag_engine.engine.rerank.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(cross_encoder_model)
    raise ValueError(f"No strategy factory configured for {reranker!r}")
