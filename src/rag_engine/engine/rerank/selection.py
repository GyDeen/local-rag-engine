from collections.abc import Sequence
from dataclasses import replace

from rag_engine.engine.rerank.candidate import RerankCandidate
from rag_engine.settings import validate_evidence_rank_limit

SectionKey = tuple[tuple[str, ...], str]

# Marks where retrieval skipped documentation inside one stitched passage, so
# whoever reads the text can see that it is not continuous.
NOT_CONTINUOUS_MARKER = "\n\n[... content not retrieved here ...]\n\n"

# The chunker trims the whitespace between pieces, so two consecutive chunks of a
# document sit only a few characters apart, while a chunk that was never
# retrieved leaves at least its own length behind: the smallest indexed chunk is
# 80 characters.
BOUNDARY_GAP_CHARS = 32


def _skipped_documentation(
    previous: RerankCandidate,
    following: RerankCandidate,
) -> bool:
    """Does documentation sit between these two chunks that the passage omits?

    The offsets answer this on their own. Chunks from the same document are
    indexed in order, so a chunk that was never retrieved leaves its own span
    uncovered between the two that were.
    """

    return following.chunk.start_offset - previous.chunk.end_offset > BOUNDARY_GAP_CHARS


def _file_key(candidate: RerankCandidate) -> tuple[str, ...]:
    return tuple(candidate.chunk.source_path) or (candidate.chunk.document_id,)


def _section_key(candidate: RerankCandidate) -> SectionKey:
    return _file_key(candidate), candidate.chunk.section


def select_top_files(
    ranked: list[RerankCandidate],
    file_limit: int,
) -> list[RerankCandidate]:
    """Keep the best-ranked chunk of each of the first ``file_limit`` files.

    At most one chunk per source file, chosen by rank, so a single long document cannot consume the
    whole evidence budget.
    """

    if file_limit < 1:
        raise ValueError("file_limit must be at least 1")

    selected: list[RerankCandidate] = []
    seen: set[tuple[str, ...]] = set()

    for candidate in ranked:
        file_key = _file_key(candidate)
        if file_key in seen:
            continue
        seen.add(file_key)
        selected.append(candidate)
        if len(selected) == file_limit:
            break

    return selected


def _stitch(
    chunks: Sequence[RerankCandidate],
) -> tuple[str, list[RerankCandidate]]:
    """Join one section's chunks in offset order, dropping overlapped tails.

    Slicing that overlap off each chunk yields contiguous prose instead of
    a passage with the same sentences twice. A chunk already covered by its
    predecessor contributes nothing and is dropped entirely.

    Where retrieval skipped documentation between two chunks, the join carries
    ``NOT_CONTINUOUS_MARKER`` so the passage says so itself instead of reading as
    one unbroken text.

    Returns the text together with the chunks that actually contributed to it,
    so callers can report every source rather than only the first one.
    """

    ordered = sorted(chunks, key=lambda candidate: candidate.chunk.start_offset)
    parts: list[str] = []
    contributors: list[RerankCandidate] = []
    covered_until: int | None = None

    for candidate in ordered:
        text = candidate.text
        start = candidate.chunk.start_offset
        if covered_until is not None and start < covered_until:
            overlap = covered_until - start
            if overlap >= len(text):
                continue
            text = text[overlap:]
        if text:
            # The marker only ever precedes text that follows it, so a gap is
            # announced once and never left dangling.
            if contributors and _skipped_documentation(contributors[-1], candidate):
                parts.append(NOT_CONTINUOUS_MARKER)
            parts.append(text)
            contributors.append(candidate)
        covered_until = max(covered_until or start, candidate.chunk.end_offset)

    return "".join(parts), contributors


def stitch_evidence(
    ranked: Sequence[RerankCandidate],
    pool: Sequence[RerankCandidate],
    file_limit: int,
    char_budget: int,
    min_chunk_chars: int = 0,
    evidence_rank_limit: int | None = None,
) -> list[RerankCandidate]:
    """Turn a ranking into per-section evidence, widened from the whole pool.

    Every section the ranking chose is completed with the other pool chunks
    that belong to it, stitched in offset order, which hands answer generation
    contiguous documentation rather than isolated fragments.

    ``file_limit`` caps the distinct source files that actually contribute
    evidence, and ``char_budget`` caps the total text. A section that would overflow is
    skipped. Sections whose stitched text is
    shorter than ``min_chunk_chars`` are dropped.

    ``evidence_rank_limit`` optionally restricts both selection and stitching to
    the first N ranked chunks. Later and unranked pool chunks are excluded, even
    from selected files or sections. None preserves whole-pool completion.

    Every chunk merged into one result shares its path and section, so the
    citations built from it stay exact. Each result also names the chunks it was
    stitched from in ``chunk_ids``, and a section whose retrieved chunks do not
    meet carries ``NOT_CONTINUOUS_MARKER`` in its text wherever documentation was
    skipped, so nothing downstream reads the passage as unbroken.
    """

    if file_limit < 1:
        raise ValueError("file_limit must be at least 1")
    if char_budget < 1:
        raise ValueError("char_budget must be at least 1")
    if min_chunk_chars < 0:
        raise ValueError("min_chunk_chars cannot be negative")

    validate_evidence_rank_limit(evidence_rank_limit)
    if evidence_rank_limit is not None:
        # Cut off individual ranks before grouping files or completing sections.
        # Use only accepted candidates so the full pool cannot restore late chunks.
        ranked = ranked[:evidence_rank_limit]
        pool = ranked

    sections: dict[SectionKey, list[RerankCandidate]] = {}
    seen_ids: set[str] = set()
    for candidate in pool:
        chunk_id = candidate.chunk.chunk_id
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        sections.setdefault(_section_key(candidate), []).append(candidate)

    selected: list[RerankCandidate] = []
    accepted_files: list[tuple[str, ...]] = []
    taken: set[SectionKey] = set()
    used = 0

    for candidate in ranked:
        key = _section_key(candidate)
        if key in taken:
            continue
        taken.add(key)

        # Only files that contribute evidence count against the limit, so a file
        # whose sections are all rejected cannot block a later useful one.
        file_key = key[0]
        known_file = file_key in accepted_files
        if not known_file and len(accepted_files) >= file_limit:
            continue

        bucket = sorted(
            sections.get(key) or _anchor_for(key, ranked),
            key=lambda candidate: candidate.chunk.start_offset,
        )
        text, contributors = _stitch(bucket)
        if not contributors:
            continue
        if len(text) < min_chunk_chars:
            continue
        if used + len(text) > char_budget:
            continue

        if not known_file:
            accepted_files.append(file_key)
        selected.append(
            RerankCandidate(
                chunk=replace(
                    bucket[0].chunk,
                    metadata={
                        **bucket[0].chunk.metadata,
                        **_page_metadata(contributors),
                    },
                ),
                text=text,
                score=bucket[0].score,
                truncated=any(candidate.truncated for candidate in contributors),
                chunk_ids=tuple(candidate.chunk.chunk_id for candidate in contributors),
            )
        )
        used += len(text)

    return selected


def _page_metadata(contributors: Sequence[RerankCandidate]) -> dict:
    pages = sorted(
        {
            page
            for candidate in contributors
            for page in candidate.chunk.metadata.get("source_pages", [])
        }
    )
    return {"source_pages": pages} if pages else {}


def _anchor_for(
    key: SectionKey,
    ranked: Sequence[RerankCandidate],
) -> list[RerankCandidate]:
    """The ranked candidate itself, for a section the pool did not carry"""

    return [candidate for candidate in ranked if _section_key(candidate) == key]
