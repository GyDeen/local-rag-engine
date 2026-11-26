"""Public requests and responses for the RAG service."""

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field

from rag_engine.engine.vector_store.database_metadata_model import (
    FilterValue,
    MetadataKey,
)

AnswerStatus = Literal["answered", "insufficient_evidence"]


class QuestionRequest(BaseModel):
    question: str = Field(
        pattern=r"\S", description="Question about the indexed documents."
    )
    filters: dict[MetadataKey, FilterValue] | None = Field(
        default=None,
        description="Metadata values that all retrieved chunks must match.",
    )
    generate: bool = Field(
        default=True,
        description="Request a cited answer where supported. /retrieve is evidence-only.",
    )
    context: dict[str, Any] | None = Field(
        default=None, description="Optional context for the answer."
    )


class ResolveRequest(QuestionRequest):
    """A question that can include details from an alarm report."""

    code: str | None = Field(default=None, min_length=1, description="Reported code.")
    env: dict[str, Any] | None = Field(
        default=None, description="Product-specific environment context."
    )
    text: str | None = Field(
        default=None,
        description="Extra context for the alarm",
    )


class RetrievedEvidence(BaseModel):
    """One supplied passage, including all stitched source chunk IDs."""

    id: str
    doc: str = Field(validation_alias=AliasChoices("doc", "source"))
    path: str = Field(validation_alias=AliasChoices("path", "location"))
    pages: list[int] = Field(default_factory=list)
    section: str
    text: str
    chunk_ids: list[str]


class RetrieveResponse(BaseModel):
    """Documentation evidence without a generated answer."""

    question: str
    evidence: list[RetrievedEvidence]


class QuestionResponse(RetrieveResponse):
    answer: str | None
    status: AnswerStatus | None
