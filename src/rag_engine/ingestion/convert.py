import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".docx", ".html", ".htm"}


class DocumentConverter:
    """Preserve text/Markdown and use Docling for structured document formats.

    One converter is retained for a batch. PDF pages stay in one document, with
    page spans recorded separately for citations. No reference answers influence
    conversion, document selection, or extraction settings.
    """

    def __init__(self) -> None:
        self._docling = None

    def convert(self, path: Path) -> tuple[str, list[dict], dict | None]:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported document format: {path}")
        if suffix in {".txt", ".md", ".markdown"}:
            text = path.read_text(encoding="utf-8-sig")
            return text, [], None

        if self._docling is None:
            from docling.document_converter import DocumentConverter as DoclingConverter

            self._docling = DoclingConverter()
        result = self._docling.convert(path)
        if result.status.value != "success":
            raise ValueError(f"Document conversion failed: {path}: {result.status}")
        document = result.document
        spans = []
        if suffix == ".pdf":

            # One md for one pdf
            parts = []
            offset = 0
            for page in sorted(document.pages):
                text = document.export_to_markdown(
                    page_no=page, traverse_pictures=True
                ).strip()
                if not text:
                    logger.warning("No extracted content: %s page=%s", path, page)
                    continue
                if parts:
                    parts.append("\n\n")
                    offset += 2
                spans.append({"start": offset, "end": offset + len(text), "page": page})
                parts.append(text)
                offset += len(text)
            markdown = "".join(parts)
        else:
            markdown = document.export_to_markdown(traverse_pictures=True)
        return markdown, spans, document.export_to_dict()


def ingest_documents(
    source_root: str | Path,
    output: str | Path,
    *,
    sources: dict[str, str] | None = None,
) -> dict:
    """Write normalized Markdown under documents/ and diagnostics under debug/.

    Source extensions remain in generated names (manual.pdf.md), so files with
    the same stem cannot overwrite each other. Unknown files are reported. A
    failed conversion never publishes a partially converted output directory.
    """
    source_root = Path(source_root).resolve()
    output = Path(output).resolve()
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if output.exists() and not output.is_dir():
        raise FileExistsError(
            f"{output} is not a directory; choose another output location"
        )
    for name in ("documents", "debug"):
        path = output / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"{path} already exists; choose a fresh output")
    if source_root.is_dir() and output.is_relative_to(source_root):
        raise ValueError("The ingestion output must be outside the source directory")
    files = sorted(source_root.rglob("*")) if source_root.is_dir() else [source_root]
    base = source_root if source_root.is_dir() else source_root.parent
    converter = DocumentConverter()
    manifest = {"version": 1, "documents": {}, "skipped": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared = output.is_dir()
    with tempfile.TemporaryDirectory(
        dir=output if prepared else output.parent
    ) as temporary:
        destination = Path(temporary) / "converted"
        debug = destination / "debug"
        debug.mkdir(parents=True)
        for path in files:
            if not path.is_file():
                continue
            if not path.resolve().is_relative_to(base):
                raise ValueError(f"Document symlink points outside the source: {path}")
            relative = path.relative_to(base).as_posix()
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                manifest["skipped"].append(relative)
                logger.warning("Unsupported document skipped: %s", relative)
                continue
            logger.info("Converting document %s", relative)
            text, spans, structured = converter.convert(path)
            if not text.strip():
                raise ValueError(f"Document produced no indexable text: {path}")
            name = relative + ".md"
            target = destination / "documents" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="\n")
            manifest["documents"][name] = {
                "source": relative,
                "location": (sources or {}).get(relative, str(path)),
                "converter": "direct" if structured is None else "docling",
                "page_spans": spans,
            }
            if structured is not None:
                structure_path = debug / "structure" / (relative + ".json")
                structure_path.parent.mkdir(parents=True, exist_ok=True)
                structure_path.write_text(
                    json.dumps(structured, ensure_ascii=False), encoding="utf-8"
                )
        if not manifest["documents"]:
            raise ValueError(f"No supported documents found in {source_root}")
        (debug / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if prepared:
            # Publish the documents last so an indexer can see them with provenance.
            published_debug = debug.rename(output / "debug")
            try:
                (destination / "documents").rename(output / "documents")
            except OSError:
                published_debug.rename(debug)
                raise
        else:
            destination.rename(output)
    return manifest


def attach_provenance(chunks: list, manifest_path: Path) -> None:
    """Attach original source and overlapping PDF pages before vector storage."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for chunk in chunks:
        source = manifest["documents"][chunk.document_id]
        chunk.metadata.update(
            source_document=source["source"], source_location=source["location"]
        )
        pages = sorted(
            {
                span["page"]
                for span in source["page_spans"]
                if span["start"] < chunk.end_offset and span["end"] > chunk.start_offset
            }
        )
        if pages:
            chunk.metadata["source_pages"] = pages
