from sentence_transformers import CrossEncoder

from rag_engine.engine.rerank.candidate import RerankCandidate
from rag_engine.settings import DEFAULT_CROSS_ENCODER_MODEL


class CrossEncoderReranker:
    """Rank candidates by their cross-encoder relevance to the query."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
    ) -> list[RerankCandidate]:
        """Score every candidate against ``query`` and return the best first."""

        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores, strict=True):
            candidate.score = float(score)

        return sorted(
            candidates,
            key=lambda candidate: (
                candidate.score if candidate.score is not None else float("-inf")
            ),
            reverse=True,
        )
