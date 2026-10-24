"""Public alarm-resolution requests and responses matching the service interface contract."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Env(BaseModel):
    """Machine environment carried by the frontend's alarm instance."""

    versions: dict[str, str] = Field(
        description="Software versions, keyed by component such as core or oem."
    )
    locale: str = Field(description="Locale of the on-screen alarm text.")
    os: str | None = None
    machine_variant: str | None = None
    serial: str | None = None


class ResolveRequest(BaseModel):
    """Frontend payload for POST /api/v1/resolve; code identifies the alarm."""

    code: str = Field(min_length=1, description="Stable alarm catalogue code.")
    env: Env
    context: dict[str, Any] | None = Field(
        default=None, description="Optional alarm-specific runtime information."
    )
    text: str | None = Field(
        default=None,
        description="Extra context for the alarm",
    )


class RetrieveRequest(BaseModel):
    """Retrieve alarm evidence for an external answer generator."""

    code: str = Field(min_length=1, description="Stable alarm catalogue code.")
    env: Env | None = None
    context: dict[str, Any] | None = None
    text: str | None = None
    question: str | None = Field(
        default=None, description="Optional question used to rerank alarm evidence."
    )


Tier = Literal["operator", "technician", "partner"]
SeverityCategory = Literal["Debug", "Info", "Warning", "Error", "Fatal"]


class Citation(BaseModel):
    """Source reference derived from retrieved evidence."""

    doc: str
    path: str
    section: str


class RetrievedEvidence(Citation):
    """One supplied passage, including all stitched source chunk IDs."""

    id: str
    text: str
    chunk_ids: list[str]


class RetrieveResponse(BaseModel):
    """Documentation evidence without a generated answer."""

    code: str
    alarm_text: str
    evidence: list[RetrievedEvidence]
    strategy: str
    stop_reason: str


class ResolveResponse(BaseModel):
    """Grounded guidance with citations and catalogue metadata."""

    code: str
    summary: str
    steps: list[str]
    citations: list[Citation]
    tier: Tier
    title: str
    severity: int = Field(ge=1, le=1000)
    severity_category: SeverityCategory
    domain: str
    likely_causes: list[str] | None = None
    ai_chat_available: Literal[False] = Field(
        default=False, description="Follow-up AI chat is not currently available."
    )


class AlarmSummary(BaseModel):
    """Catalogue entry exposed for alarm selection and QR generation."""

    code: str
    title: str
    domain: str
    severity: int = Field(ge=1, le=1000)
    severity_category: SeverityCategory
    alarm_text: str
