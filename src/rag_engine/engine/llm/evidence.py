"""Evidence normalisation and formatting for the final-answer stage.

Maps stored chunk dicts onto stable ``C1..Cn`` evidence ids, renders them
as LLM-readable text, and builds document references from chunk metadata
(never from model output).
"""

from pathlib import Path
from typing import Any, Dict, List

from rag_engine.engine.vector_store.process_chunks import read_chunk_text

UNKNOWN = "I don't know"

# Human-readable document names, keyed by the top-level docs directory.
DOCUMENT_NAMES: dict[str, str] = {}


def normalise_chunks(
    chunks: List[Dict[str, Any]],
    docs_root: str | Path = "examples/docs",
) -> List[Dict[str, Any]]:
    """Map stored chunk dicts onto the LLM evidence format.

    Accepts dicts in the shape of an rag_engine ``Chunk`` (e.g. from
    ``dataclasses.asdict``) and assign stable evidence ids ``C1..Cn`` in
    supply order. Indexed text is available with each result.
    """

    normalised: List[Dict[str, Any]] = []

    for index, chunk in enumerate(chunks, start=1):
        source_path = [str(part) for part in chunk.get("source_path") or []]
        path = "/".join(source_path)
        text = chunk.get("text")
        if not text:
            text = read_chunk_text(
                docs_root=Path(docs_root),
                source_path=source_path,
                start_offset=int(chunk["start_offset"]),
                end_offset=int(chunk["end_offset"]),
            )

        top_dir = source_path[0] if source_path else ""
        normalised.append(
            {
                "llm_chunk_id": f"C{index}",
                "doc": DOCUMENT_NAMES.get(top_dir, top_dir or UNKNOWN.lower()),
                "path": path,
                "section": str(chunk.get("section", "")),
                "text": text,
                # Stitched evidence carries several indexed chunks; keeping the
                # list here means saved results still show what each passage was built from.
                "chunk_ids": [
                    str(chunk_id) for chunk_id in chunk.get("chunk_ids") or []
                ],
            }
        )

    return normalised


def format_chunks_for_llm(chunks: List[Dict[str, Any]]) -> str:
    """Render chunks as natural text labelled with their evidence ids."""

    sections = [
        f"[{chunk['llm_chunk_id']}] {chunk['doc']} — {chunk['section']}\n{chunk['text']}"
        for chunk in chunks
    ]
    return "\n\n".join(sections)


def build_doc_refs(
    supporting_chunk_ids: List[str],
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build document references directly from supplied chunk metadata.

    The LLM does not generate document names, paths, or sections itself.
    This prevents unsupported document references from appearing in the
    final answer.
    """

    chunk_lookup = {chunk["llm_chunk_id"]: chunk for chunk in chunks}

    references = []
    seen = set()

    for chunk_id in supporting_chunk_ids:
        chunk = chunk_lookup.get(chunk_id)

        if chunk is None:
            continue

        doc = str(chunk["doc"]) if chunk["doc"] else UNKNOWN

        path = str(chunk["path"]) if chunk["path"] else UNKNOWN

        section = str(chunk["section"]) if chunk["section"] else UNKNOWN

        ref_tuple = (doc, path, section)

        # Multiple chunks may originate from the same document section.
        if ref_tuple in seen:
            continue

        seen.add(ref_tuple)

        references.append({"doc": doc, "path": path, "section": section})

    return references
