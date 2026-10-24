from typing import Protocol, runtime_checkable

from rag_engine.engine.rerank.candidate import RerankCandidate


@runtime_checkable
class RerankStrategy(Protocol):
    """Rank retrieved candidates for one query."""

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
    ) -> list[RerankCandidate]:
        """Return ``candidates`` ordered from most to least relevant."""
        ...
