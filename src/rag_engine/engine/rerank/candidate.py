from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from rag_engine.engine.vector_store.database_metadata_model import Chunk
from rag_engine.engine.vector_store.process_chunks import read_chunk_text
from rag_engine.settings import DEFAULT_DOCS_ROOT

_CHUNK_FIELDS = tuple(chunk_field.name for chunk_field in fields(Chunk))


@dataclass(slots=True)
class RerankCandidate:
    """A retrieved chunk whose document text is ready for reranking.

    ``score`` is deliberately strategy-defined (a cross-encoder similarity, an LLM rank position, or nothing at all), so shared code never reads it.

    ``chunk_ids`` names every indexed chunk the text was stitched from, in
    offset order, and stays empty while the candidate is exactly ``chunk``. A
    stitched candidate therefore reports all of its sources instead of claiming
    to be only the chunk it is anchored to.
    """

    chunk: Chunk
    text: str
    score: float | None = None
    truncated: bool = False
    chunk_ids: tuple[str, ...] = ()

    def all_chunk_ids(self) -> tuple[str, ...]:
        """Every indexed chunk behind this candidate's text."""

        return self.chunk_ids or (self.chunk.chunk_id,)


def to_chunk(value: Chunk | Mapping[str, Any]) -> Chunk:
    """Get the required chunk index"""

    if isinstance(value, Chunk):
        return value

    record = asdict(value) if is_dataclass(value) else dict(value)
    missing = [name for name in _CHUNK_FIELDS if name not in record]
    if missing:
        raise ValueError(
            f"Stored chunk is missing required fields: {', '.join(missing)}"
        )

    return Chunk(**{name: record[name] for name in _CHUNK_FIELDS})


def to_stored(candidate: RerankCandidate) -> dict[str, Any]:
    """Return the stored chunk record plus the text the candidate already holds.

    Answer generation consumes stored chunk records, so this is the single place
    where a reranked candidate crosses back into that shape. ``chunk_id`` and the
    offsets stay the anchor's own indexed values, so the record can still be
    checked against the index; ``chunk_ids`` carries the full provenance of the
    text, which spans several chunks once the section has been stitched.
    """

    return {
        **asdict(candidate.chunk),
        "text": candidate.text,
        "truncated": candidate.truncated,
        "chunk_ids": list(candidate.all_chunk_ids()),
    }


def build_rerank_candidates(
    chunks: Iterable[Chunk | Mapping[str, Any] | RerankCandidate],
    docs_root: str | Path = DEFAULT_DOCS_ROOT,
) -> list[RerankCandidate]:
    """Retrieve the text for offered Chunk therefore build the RerankCandidates"""

    candidates: list[RerankCandidate] = []

    for value in chunks:
        if isinstance(value, RerankCandidate):
            candidates.append(value)
            continue

        chunk = to_chunk(value)
        if isinstance(value, Mapping):
            text = value.get("text")
            truncated = bool(value.get("truncated", False))
            stored_ids = value.get("chunk_ids") or ()
        else:
            text = None
            truncated = False
            stored_ids = ()

        if not text:
            text = read_chunk_text(
                docs_root=Path(docs_root),
                source_path=[str(part) for part in chunk.source_path],
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                expected_hash=chunk.source_hash,
            )

        candidates.append(
            RerankCandidate(
                chunk=chunk,
                text=text,
                truncated=truncated,
                chunk_ids=tuple(str(chunk_id) for chunk_id in stored_ids),
            )
        )

    return candidates
