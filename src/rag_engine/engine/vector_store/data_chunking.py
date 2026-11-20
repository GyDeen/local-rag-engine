from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from rag_engine.engine.vector_store.chunk_metadata import (
    LABEL_LINE_PATTERN,
    SEE_ALSO_LINE_PATTERN,
    WARNING_CALLOUT_LINE_PATTERN,
    extract_alarm_codes,
    extract_related_commands,
    extract_see_also,
    extract_warnings,
)
from rag_engine.engine.vector_store.database_metadata_model import Chunk, MetadataValue

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MINIMUM_CHUNK_SIZE = 200
OVERSIZED_CHUNK_THRESHOLD = 2500  # flag chunks larger than this for manual review

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
SENTENCE_END_PATTERN = re.compile(r"""[.!?。！？](?:["'”’)\]}]*)?(?=\s|$)""")
SENTENCE_END_BEFORE_BOUNDARY_PATTERN = re.compile(
    r"""[.!?。！？](?:["'”’)\]}]*)?\s*$"""
)

# CommonMark fences: up to three leading spaces, three or more backticks or tildes.
FENCE_PATTERN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
FenceMarker = tuple[str, int]
ChunkPiece = tuple[str, int, int]
DERIVED_METADATA_KEYS = frozenset(
    {"see_also", "related_commands", "alarm_codes", "warnings", "oversized"}
)

TABLE_ROW_PATTERN = re.compile(r"^\|.*\|$")
CALLOUT_START_PATTERN = re.compile(
    r"^ {0,3}>\s*\[!(?:NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*$",
    re.IGNORECASE,
)
BLOCKQUOTE_LINE_PATTERN = re.compile(r"^ {0,3}>")


def find_markdown_files(docs_root: str | Path) -> list[Path]:
    """Find all Markdown files inside the documentation folder."""

    docs_path = Path(docs_root)
    return sorted(docs_path.rglob("*.md"))


def get_heading_level(line: str) -> int | None:
    """Return the heading depth when a line is a Markdown heading."""

    match = HEADING_PATTERN.match(line.strip())
    return len(match.group(1)) if match else None


def get_heading_text(line: str) -> str | None:
    """Return the heading text when a line is a Markdown heading."""

    match = HEADING_PATTERN.match(line.strip())
    return match.group(2).strip() if match else None


def split_text_with_overlap(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    minimum_chunk_size: int = DEFAULT_MINIMUM_CHUNK_SIZE,
) -> list[str]:
    """Split text without cutting sentences and merge undersized chunks."""

    _validate_chunk_settings(chunk_size, overlap)
    _validate_minimum_chunk_size(minimum_chunk_size)

    pieces = _split_with_offsets_protected(
        text,
        0,
        chunk_size,
        overlap,
        [],
    )
    pieces = _merge_small_pieces(text, 0, pieces, minimum_chunk_size)
    return [piece_text for piece_text, _piece_start, _piece_end in pieces]


def _validate_chunk_settings(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")


def _validate_minimum_chunk_size(minimum_chunk_size: int) -> None:
    if minimum_chunk_size <= 0:
        raise ValueError("minimum_chunk_size must be greater than zero")


def _parse_fence_marker(line: str) -> FenceMarker | None:
    """Return (marker character, minimum closer length) when the line opens a fence."""

    match = FENCE_PATTERN.match(line.rstrip("\r\n"))
    if match is None:
        return None

    marker = match.group(1)
    return marker[0], len(marker)


def _closes_fence(line: str, opener: FenceMarker) -> bool:
    """Return True when the line is a valid closing fence for the opener."""

    match = re.match(r"^ {0,3}(`{3,}|~{3,})[ \t]*$", line.rstrip("\r\n"))
    if match is None:
        return False

    marker = match.group(1)
    marker_character, minimum_length = opener
    return marker[0] == marker_character and len(marker) >= minimum_length


def _is_fence_line(line: str) -> bool:
    return _parse_fence_marker(line) is not None


def _is_table_row(line: str) -> bool:
    return bool(TABLE_ROW_PATTERN.match(line.strip()))


def _is_see_also_line(line: str) -> bool:
    return bool(SEE_ALSO_LINE_PATTERN.match(line.strip()))


def _is_callout_start(line: str) -> bool:
    return bool(CALLOUT_START_PATTERN.match(line.rstrip("\r\n")))


def _is_blockquote_line(line: str) -> bool:
    return bool(BLOCKQUOTE_LINE_PATTERN.match(line.rstrip("\r\n")))


def find_callout_spans(text: str) -> list[tuple[int, int]]:
    """Return complete Markdown callout ranges, including marker and body."""

    lines = text.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if not _is_callout_start(lines[i]):
            i += 1
            continue

        start_line = i
        i += 1
        while i < len(lines) and _is_blockquote_line(lines[i]):
            i += 1
        end = line_starts[i] if i < len(lines) else len(text)
        spans.append((line_starts[start_line], end))

    return spans


def find_protected_spans(text: str) -> list[tuple[int, int]]:
    """Find code, tables, callouts and See Also lines that must stay whole."""

    lines = text.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0

    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    spans: list[tuple[int, int]] = []
    i = 0

    while i < len(lines):
        opener = _parse_fence_marker(lines[i])
        if opener is not None:
            start_line = i
            i += 1
            while i < len(lines) and not _closes_fence(lines[i], opener):
                i += 1
            if i < len(lines):
                i += 1
            spans.append(_extend_with_label(lines, line_starts, start_line, i))
            continue

        if _is_callout_start(lines[i]):
            start_line = i
            i += 1
            while i < len(lines) and _is_blockquote_line(lines[i]):
                i += 1
            end = line_starts[i] if i < len(lines) else len(text)
            spans.append((line_starts[start_line], end))
            continue

        if _is_table_row(lines[i]):
            start_line = i
            while i < len(lines) and _is_table_row(lines[i]):
                i += 1
            spans.append(_extend_with_label(lines, line_starts, start_line, i))
            continue

        if _is_see_also_line(lines[i]):
            start = line_starts[i]
            spans.append((start, start + len(lines[i])))
            i += 1
            continue

        i += 1

    return spans


def _find_warning_spans(
    text: str,
    protected_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Select WARNING callouts from spans already protected by the parser."""

    warning_spans: list[tuple[int, int]] = []
    for span_start, span_end in protected_spans:
        protected_text = text[span_start:span_end]
        first_line = protected_text.splitlines()[0] if protected_text else ""
        if WARNING_CALLOUT_LINE_PATTERN.match(first_line):
            warning_spans.append((span_start, span_end))

    return warning_spans


def _extend_with_label(
    lines: list[str],
    line_starts: list[int],
    start_line: int,
    end_line: int,
) -> tuple[int, int]:
    """Include a bold label line before a protected block when present."""

    probe = start_line - 1
    while probe >= 0 and lines[probe].strip() == "":
        probe -= 1

    if probe >= 0 and LABEL_LINE_PATTERN.match(lines[probe].strip()):
        start_line = probe

    if end_line < len(lines):
        end_offset = line_starts[end_line]
    else:
        end_offset = sum(len(line) for line in lines)

    return line_starts[start_line], end_offset


def _trim_offsets(text: str, start: int) -> tuple[str, int, int]:
    """Strip whitespace while keeping offsets aligned to the original file."""

    stripped = text.strip()
    if not stripped:
        return "", start, start

    leading_space = len(text) - len(text.lstrip())
    clean_start = start + leading_space
    clean_end = clean_start + len(stripped)
    return stripped, clean_start, clean_end


def _extend_to_sentence_end(text: str, end: int) -> int:
    """Move a proposed chunk end forward so it never cuts a sentence."""

    if end >= len(text):
        return len(text)
    if SENTENCE_END_BEFORE_BOUNDARY_PATTERN.search(text[:end]):
        return end

    match = SENTENCE_END_PATTERN.search(text, end)
    return match.end() if match is not None else len(text)


def _next_complete_sentence_start(text: str, start: int, end: int) -> int:
    """Return an overlap start that does not begin in the middle of a sentence."""

    if start >= end:
        return end

    for match in SENTENCE_END_PATTERN.finditer(text, start, end):
        candidate = match.end()
        while candidate < end and text[candidate].isspace():
            candidate += 1
        if candidate < end:
            return candidate

    return end


def _split_with_offsets_protected(
    text: str,
    base_offset: int,
    chunk_size: int,
    overlap: int,
    protected_spans: list[tuple[int, int]],
) -> list[tuple[str, int, int]]:
    """Split text while keeping protected blocks whole.

    A WARNING closes the current piece so it stays attached to the content
    immediately before it, even when the resulting piece exceeds chunk_size.
    """

    _validate_chunk_settings(chunk_size, overlap)

    if not text:
        return []

    pieces: list[tuple[str, int, int]] = []
    warning_spans = _find_warning_spans(text, protected_spans)
    position = 0

    while position < len(text):
        end = min(position + chunk_size, len(text))
        end = _extend_to_sentence_end(text, end)

        for warning_start, warning_end in warning_spans:
            if warning_start < position:
                continue
            if warning_start <= end or not text[end:warning_start].strip():
                end = warning_end
                break

        for span_start, span_end in protected_spans:
            if span_start < end < span_end:
                end = span_end
                break

        raw_piece = text[position:end]
        piece_text, piece_start, piece_end = _trim_offsets(
            raw_piece,
            base_offset + position,
        )

        if piece_text:
            pieces.append((piece_text, piece_start, piece_end))

        if end >= len(text):
            break

        next_position = _next_complete_sentence_start(
            text,
            max(end - overlap, position + 1),
            end,
        )
        for span_start, span_end in protected_spans:
            if span_start <= next_position < span_end:
                next_position = end
                break

        position = max(next_position, position + 1)

    return pieces


def _split_oversized_piece(
    piece_text: str,
    piece_start: int,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int, int]]:
    """Split a piece that a protected span forced oversize, at line boundaries.

    Used when a protected piece is larger than OVERSIZED_CHUNK_THRESHOLD. It
    is divided into line-quantized pieces
    of at most chunk_size characters (a single longer line stays whole) so the
    downstream embedding stage receives text that fits its token budget.
    Offsets stay aligned to the original file.

    Callouts remain indivisible during this second pass. If a callout itself
    exceeds the budget, it stays whole and is marked oversized by the caller.
    """

    lines = piece_text.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0

    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    pieces: list[tuple[str, int, int]] = []
    callout_spans = find_callout_spans(piece_text)
    start_line = 0

    while start_line < len(lines):
        end_line = start_line
        length = 0

        while end_line < len(lines):
            if end_line > start_line and length + len(lines[end_line]) > chunk_size:
                break
            length += len(lines[end_line])
            end_line += 1

        boundary = line_starts[end_line] if end_line < len(lines) else len(piece_text)
        for span_start, span_end in callout_spans:
            if span_start < boundary < span_end:
                while end_line < len(lines) and line_starts[end_line] < span_end:
                    end_line += 1
                break

        raw_piece = "".join(lines[start_line:end_line])
        text, start, end = _trim_offsets(
            raw_piece,
            piece_start + line_starts[start_line],
        )
        if text:
            pieces.append((text, start, end))

        if end_line >= len(lines):
            break

        back_line = end_line - 1
        covered = 0
        while back_line > start_line and covered < overlap:
            covered += len(lines[back_line])
            back_line -= 1

        next_start_line = back_line if back_line > start_line else start_line + 1
        next_start = line_starts[next_start_line]
        current_end = (
            line_starts[end_line] if end_line < len(lines) else len(piece_text)
        )
        if any(
            next_start < span_end and span_start < current_end
            for span_start, span_end in callout_spans
        ):
            start_line = end_line
        else:
            start_line = next_start_line

    return pieces


def _merge_small_pieces(
    source_text: str,
    base_offset: int,
    pieces: list[ChunkPiece],
    minimum_chunk_size: int,
) -> list[ChunkPiece]:
    """Merge undersized pieces without duplicating their overlap text."""

    if len(pieces) < 2:
        return pieces

    merged: list[ChunkPiece] = []
    for piece in pieces:
        if not merged:
            merged.append(piece)
            continue

        previous = merged[-1]
        first_piece_is_small = (
            len(merged) == 1 and len(previous[0]) < minimum_chunk_size
        )
        if len(piece[0]) >= minimum_chunk_size and not first_piece_is_small:
            merged.append(piece)
            continue

        merged_start = previous[1]
        merged_end = max(previous[2], piece[2])
        relative_start = merged_start - base_offset
        relative_end = merged_end - base_offset
        merged[-1] = _trim_offsets(
            source_text[relative_start:relative_end],
            merged_start,
        )

    return merged


def _build_chunk_metadata(
    piece_text: str,
    metadata_defaults: Mapping[str, MetadataValue],
) -> dict[str, MetadataValue]:
    metadata: dict[str, MetadataValue] = {
        **metadata_defaults,
        "see_also": extract_see_also(piece_text),
        "related_commands": extract_related_commands(piece_text),
        "alarm_codes": extract_alarm_codes(piece_text),
        "warnings": extract_warnings(piece_text),
    }
    if len(piece_text) > OVERSIZED_CHUNK_THRESHOLD:
        metadata["oversized"] = True

    return {
        key: value for key, value in metadata.items() if value not in (None, [], "")
    }


def _merge_small_chunks(
    chunks: list[Chunk],
    source_text: str,
    minimum_chunk_size: int,
) -> list[Chunk]:
    """Merge residual undersized chunks across sections of one document."""

    if len(chunks) < 2:
        return chunks

    merged: list[Chunk] = []
    for chunk in chunks:
        if not merged:
            merged.append(chunk)
            continue

        previous = merged[-1]
        first_chunk_is_small = (
            len(merged) == 1
            and previous.end_offset - previous.start_offset < minimum_chunk_size
        )
        chunk_is_small = chunk.end_offset - chunk.start_offset < minimum_chunk_size
        if not chunk_is_small and not first_chunk_is_small:
            merged.append(chunk)
            continue

        merged_start = previous.start_offset
        merged_end = max(previous.end_offset, chunk.end_offset)
        merged_text = source_text[merged_start:merged_end]
        metadata_defaults = {
            key: value
            for key, value in {**previous.metadata, **chunk.metadata}.items()
            if key not in DERIVED_METADATA_KEYS
        }
        section = previous.section
        if chunk.section != previous.section:
            section = f"{previous.section} | {chunk.section}"

        merged[-1] = Chunk(
            chunk_id=previous.chunk_id,
            embedding=[],
            document_id=previous.document_id,
            source_path=list(previous.source_path),
            section=section,
            start_offset=merged_start,
            end_offset=merged_end,
            text=merged_text,
            metadata=_build_chunk_metadata(merged_text, metadata_defaults),
        )

    for index, chunk in enumerate(merged):
        chunk.chunk_id = f"{chunk.document_id}#{index}"

    return merged


def _first_h1_title(text: str) -> str | None:
    """Return the text of the first level-1 heading, when present."""

    for line in text.splitlines():
        if get_heading_level(line) == 1:
            return get_heading_text(line)

    return None


def chunk_markdown_file(
    file_path: str | Path,
    docs_root: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    document_metadata: Mapping[str, MetadataValue] | None = None,
    minimum_chunk_size: int = DEFAULT_MINIMUM_CHUNK_SIZE,
) -> list[Chunk]:
    """Read one Markdown file and split it into structure-aware chunks."""

    _validate_chunk_settings(chunk_size, overlap)
    _validate_minimum_chunk_size(minimum_chunk_size)

    docs_path = Path(docs_root).resolve()
    source_file = Path(file_path).resolve()
    if not source_file.is_relative_to(docs_path):
        raise ValueError("Source file is outside the documentation directory")

    relative_path = source_file.relative_to(docs_path)
    document_id = relative_path.as_posix()
    source_path = list(relative_path.parts)
    raw_bytes = source_file.read_bytes()
    text = raw_bytes.decode("utf-8")

    metadata_defaults = dict(document_metadata or {})
    supplied_title = metadata_defaults.get("document_title")
    document_title = (
        supplied_title.strip()
        if isinstance(supplied_title, str) and supplied_title.strip()
        else _first_h1_title(text) or source_file.stem.replace("-", " ").title()
    )
    metadata_defaults["document_title"] = document_title
    heading_stack: list[tuple[int, str]] = []
    chunks: list[Chunk] = []
    chunk_counter = 0
    section_start = 0
    section_has_heading = False
    line_offset = 0
    active_fence: FenceMarker | None = None

    def current_section_path() -> list[str]:
        return [title for _level, title in heading_stack] or [document_title]

    def flush_section(end_offset: int, has_heading: bool) -> None:
        nonlocal chunk_counter

        section_text = text[section_start:end_offset]
        stripped_section = section_text.strip()
        if not stripped_section:
            return

        if has_heading:
            first_line = section_text.splitlines()[0].strip()
            if stripped_section == first_line:
                return

        section_path = current_section_path()
        protected_spans = find_protected_spans(section_text)
        warning_spans = [
            (section_start + span_start, section_start + span_end)
            for span_start, span_end in _find_warning_spans(
                section_text,
                protected_spans,
            )
        ]

        pieces: list[tuple[str, int, int]] = []
        for piece_text, piece_start, piece_end in _split_with_offsets_protected(
            section_text,
            section_start,
            chunk_size,
            overlap,
            protected_spans,
        ):
            contains_warning = any(
                piece_start <= warning_start < piece_end
                for warning_start, warning_end in warning_spans
            )
            if len(piece_text) > OVERSIZED_CHUNK_THRESHOLD and not contains_warning:
                pieces.extend(
                    _split_oversized_piece(
                        piece_text,
                        piece_start,
                        chunk_size,
                        overlap,
                    )
                )
            else:
                pieces.append((piece_text, piece_start, piece_end))

        pieces = _merge_small_pieces(
            section_text,
            section_start,
            pieces,
            minimum_chunk_size,
        )

        for piece_text, piece_start, piece_end in pieces:
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}#{chunk_counter}",
                    embedding=[],
                    document_id=document_id,
                    source_path=source_path,
                    section=" > ".join(section_path),
                    start_offset=piece_start,
                    end_offset=piece_end,
                    text=piece_text,
                    metadata=_build_chunk_metadata(piece_text, metadata_defaults),
                )
            )
            chunk_counter += 1

    for line in text.splitlines(keepends=True):
        if active_fence is None:
            opener = _parse_fence_marker(line)
            if opener is None:
                level = get_heading_level(line)
            else:
                active_fence = opener
                level = None
        else:
            level = None
            if _closes_fence(line, active_fence):
                active_fence = None

        if level is not None:
            if line_offset > section_start:
                flush_section(line_offset, section_has_heading)

            title = get_heading_text(line) or ""
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

            section_start = line_offset
            section_has_heading = True

        line_offset += len(line)

    flush_section(len(text), section_has_heading)

    return _merge_small_chunks(chunks, text, minimum_chunk_size)


def chunk_documents(
    docs_root: str | Path = "examples/docs",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    metadata_by_document: Mapping[str, Mapping[str, MetadataValue]] | None = None,
    minimum_chunk_size: int = DEFAULT_MINIMUM_CHUNK_SIZE,
) -> list[Chunk]:
    """Chunk all Markdown documents in the documentation folder."""

    _validate_minimum_chunk_size(minimum_chunk_size)

    all_chunks: list[Chunk] = []
    docs_path = Path(docs_root).resolve()

    for file_path in find_markdown_files(docs_path):
        document_id = file_path.relative_to(docs_path).as_posix()
        document_metadata = (
            metadata_by_document.get(document_id) if metadata_by_document else None
        )
        all_chunks.extend(
            chunk_markdown_file(
                file_path=file_path,
                docs_root=docs_path,
                chunk_size=chunk_size,
                overlap=overlap,
                document_metadata=document_metadata,
                minimum_chunk_size=minimum_chunk_size,
            )
        )

    return all_chunks


def chunk_to_output(chunk: Chunk, docs_root: str | Path) -> dict[str, object]:
    """Return a complete JSON-compatible output, including the chunk text."""

    docs_path = Path(docs_root).resolve()
    source_file = docs_path.joinpath(*chunk.source_path).resolve()
    if not source_file.is_relative_to(docs_path):
        raise ValueError("Source file is outside the documentation directory")

    text = chunk.text
    if text is None:
        from .process_chunks import read_chunk_text

        text = read_chunk_text(
            docs_root=docs_path,
            source_path=chunk.source_path,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
        )

    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "text": text,
        "embedding": [float(value) for value in chunk.embedding],
        "source_path": list(chunk.source_path),
        "section": chunk.section,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "metadata": dict(chunk.metadata),
    }


if __name__ == "__main__":
    result = chunk_documents()
    print(f"Created {len(result)} chunks")
