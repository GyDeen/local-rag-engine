from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rag_engine.engine.rerank.base import RerankStrategy
from rag_engine.engine.rerank.factory import create_rerank_strategy
from rag_engine.engine.vector_store.backends import open_database
from rag_engine.engine.vector_store.base import VectorSearch
from rag_engine.logging_config import log_operation
from rag_engine.settings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_DATABASE_BACKEND,
    DEFAULT_DATABASE_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FILE_LIMIT,
    RERANKERS,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from rag_engine.engine.rerank.llm_reranker.strategy import LLMCaller


@dataclass(frozen=True, slots=True)
class RagRuntime:
    """Resources retained while this runtime is referenced by its caller.

    Create once with ``load`` and reuse for sequential requests.
    """

    embedder: SentenceTransformer
    database: VectorSearch
    reranker: str = "cross_encoder"
    rerank_strategy: RerankStrategy | None = None
    file_limit: int = DEFAULT_FILE_LIMIT

    def __post_init__(self) -> None:
        if self.file_limit < 1:
            raise ValueError("file_limit must be at least 1")
        if self.reranker not in RERANKERS:
            raise ValueError(f"Unknown reranker {self.reranker!r}")
        if (self.rerank_strategy is not None) != (self.reranker != "none"):
            raise ValueError(
                "Enabled reranking requires a strategy; 'none' disables it"
            )
        if self.rerank_strategy is not None and not isinstance(
            self.rerank_strategy, RerankStrategy
        ):
            raise TypeError("rerank_strategy must implement RerankStrategy")

    @classmethod
    @log_operation("Runtime load")
    def load(
        cls,
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker: str = "cross_encoder",
        cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
        llm_caller: LLMCaller | None = None,
        file_limit: int = DEFAULT_FILE_LIMIT,
        database_backend: str = DEFAULT_DATABASE_BACKEND,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        collection: str = DEFAULT_COLLECTION_NAME,
    ) -> RagRuntime:
        """Load once, opening an existing index without creating or repairing it."""
        if reranker not in RERANKERS:
            raise ValueError(f"Unknown reranker {reranker!r}")
        if file_limit < 1:
            raise ValueError("file_limit must be at least 1")
        if reranker == "llm" and llm_caller is None:
            raise ValueError("The 'llm' reranker requires an llm_caller")
        database = open_database(
            backend=database_backend,
            database_path=database_path,
            collection_name=collection,
        )
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(embedding_model)
        rerank_strategy = create_rerank_strategy(
            reranker,
            cross_encoder_model=cross_encoder_model,
            llm_caller=llm_caller,
            file_limit=file_limit,
        )
        return cls(embedder, database, reranker, rerank_strategy, file_limit)
