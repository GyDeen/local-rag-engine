"""Prepare one public corpus and its separate evaluation labels."""

import argparse
import json
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import ZipFile

from .linuxcnc import LINUXCNC_CASES, LINUXCNC_SECTIONS

TECHQA_RELEASE = "0b5bbc84b7f07d6d09d063130e90b716d8d4a32a"
TECHQA_BASE = (
    f"https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval/resolve/{TECHQA_RELEASE}"
)
TECHMANUALQA_ARCHIVE = "https://zenodo.org/records/17410809/files/TechManualQA_700.zip"
LINUXCNC_BASE = "https://linuxcnc.org/docs/2.9/html/"
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_LINE_END_TAGS = _HEADINGS | {"p", "pre", "tr"}


def _download(url: str, path: Path) -> None:
    with urlopen(url, timeout=120) as response, path.open("wb") as output:
        shutil.copyfileobj(response, output)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _document(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(text.rstrip() + "\n")


def _write_sources(root: Path, sources: dict[str, str]) -> None:
    (root / "sources.json").write_text(
        json.dumps(sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _techqa(root: Path, scratch: Path) -> None:
    corpus = scratch / "corpus.zip"
    labels = scratch / "train.json"
    _download(f"{TECHQA_BASE}/corpus.zip", corpus)
    _download(f"{TECHQA_BASE}/train.json", labels)
    indexed = set()
    with ZipFile(corpus) as archive:
        for member in archive.infolist():
            source = PurePosixPath(member.filename)
            if source.suffix != ".txt":
                continue
            relative = source.with_suffix(".md").name
            raw = archive.read(member).decode("utf-8", errors="replace").strip()
            # Preserve the original text, including the title and paragraph boundaries.
            if raw:
                _document(root / "docs", relative, raw)
                indexed.add(relative)

    def cases() -> Iterator[dict]:
        for row in json.loads(labels.read_text(encoding="utf-8")):
            references = [
                {
                    "source": PurePosixPath(context["filename"])
                    .with_suffix(".md")
                    .name,
                    "text": context["text"].strip(),
                }
                for context in row["contexts"]
            ]
            missing = {ref["source"] for ref in references} - indexed
            if missing:
                raise ValueError(
                    f"TechQA evidence missing from the corpus: {sorted(missing)}"
                )
            yield {
                "id": row["id"],
                "question": row["question"],
                "answerable": not row["is_impossible"],
                "answer": None if row["is_impossible"] else row["answer"],
                "references": references,
            }

    _write_jsonl(root / "cases.jsonl", cases())
    _write_sources(
        root,
        {
            name: f"https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval/tree/{TECHQA_RELEASE}"
            for name in sorted(indexed)
        },
    )


def _manuals(root: Path, scratch: Path, pdf_dir: Path) -> None:
    # PDF documents are supplied separately, using the filenames in the release.
    from pypdf import PdfReader

    zip_path = scratch / "manual-qa.zip"
    _download(TECHMANUALQA_ARCHIVE, zip_path)
    with ZipFile(zip_path) as archive:
        with archive.open("TechManualQA_700/TechManualQA_700.jsonl") as source:
            labels = [json.loads(line) for line in source]

    manual_ids = {row["doc_id"] for row in labels}
    pdfs = sorted(pdf for pdf in pdf_dir.glob("*.pdf") if pdf.name in manual_ids)
    if not pdfs:
        raise ValueError(
            f"No matching manuals in {pdf_dir}. Expected: {', '.join(sorted(manual_ids))}"
        )
    sources = {}
    for pdf in pdfs:
        for page_number, page in enumerate(PdfReader(pdf).pages, 1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            relative = f"{pdf.stem}/page-{page_number:04d}.md"
            _document(
                root / "docs", relative, f"# {pdf.stem} — page {page_number}\n\n{text}"
            )
            sources[relative] = f"{pdf.name}#page={page_number}"

    def cases() -> Iterator[dict]:
        available = {pdf.name for pdf in pdfs}
        for row in labels:
            if row["doc_id"] not in available:
                continue
            answerable = row["category"] != "Unanswerable"
            references = []
            if answerable:
                page = int(row["gt_page_number"])
                relative = f"{Path(row['doc_id']).stem}/page-{page:04d}.md"
                if relative not in sources:
                    raise ValueError(
                        f"{row['question_id']} refers to page {page} of {row['doc_id']}, "
                        "which is missing or has no extractable text"
                    )
                references.append({"source": relative})
            yield {
                "id": row["question_id"],
                "question": row["question_text"],
                "document": row["doc_id"],
                "category": row["category"],
                "answerable": answerable,
                "answer": row["gt_answer_snippet"] if answerable else None,
                "required_steps": row["parsed_steps"] or [],
                "references": references,
            }

    _write_jsonl(root / "cases.jsonl", cases())
    _write_sources(root, sources)
    missing = sorted(manual_ids - {pdf.name for pdf in pdfs})
    if missing:
        print(f"TechManualQA imported {len(pdfs)} of {len(manual_ids)} manuals.")
        print(f"Remaining PDF filenames: {', '.join(missing)}")


class _LinuxCNCSections(HTMLParser):
    """Extract section headings and text from versioned AsciiDoc HTML."""

    def __init__(self, anchors: set[str]) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors = anchors
        self.active: list[str] | None = None
        self.sections: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        anchor = dict(attrs).get("id")
        if tag in _HEADINGS:
            self.active = (
                self.sections.setdefault(anchor, []) if anchor in self.anchors else None
            )
            if self.active is not None:
                self.active.append("# ")
        elif anchor == "footer":
            self.active = None
        elif self.active is not None:
            if tag in {"p", "pre", "ul", "ol", "table", "tr"}:
                self.active.append("\n")
            elif tag == "li":
                self.active.append("\n- ")
            elif tag in {"td", "th"}:
                self.active.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self.active is not None and tag in _LINE_END_TAGS:
            self.active.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self.active is not None:
            self.active.append(re.sub(r"\s+", " ", data))


def _linuxcnc(root: Path, scratch: Path) -> None:
    sources = {}
    for page, anchors in LINUXCNC_SECTIONS.items():
        url = LINUXCNC_BASE + page
        html = scratch / "manual.html"
        _download(url, html)
        parser = _LinuxCNCSections(set(anchors))
        parser.feed(html.read_text(encoding="utf-8"))
        parser.close()
        for anchor in anchors:
            if anchor not in parser.sections:
                raise ValueError(f"LinuxCNC section missing: {url}#{anchor}")
            slug = anchor.removeprefix("sec:").lstrip("_").replace("-", "_")
            relative = f"{PurePosixPath(page).stem}/{slug}.md"
            text = "\n".join(
                line.strip() for line in "".join(parser.sections[anchor]).splitlines()
            )
            _document(root / "docs", relative, re.sub(r"\n{3,}", "\n\n", text))
            sources[relative] = f"{url}#{anchor}"
    _write_jsonl(root / "cases.jsonl", LINUXCNC_CASES)
    _write_sources(root, sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("techqa", "techmanualqa", "linuxcnc"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks"))
    parser.add_argument(
        "--pdf-dir", type=Path, help="Directory of source PDFs for TechManualQA"
    )
    args = parser.parse_args()
    if args.dataset == "techmanualqa" and (
        not args.pdf_dir or not args.pdf_dir.is_dir()
    ):
        parser.error("TechManualQA requires --pdf-dir pointing to the source PDFs")
    destination = args.output / args.dataset
    if destination.exists():
        parser.error(
            f"{destination} already exists. Choose another --output directory."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output) as working:
        scratch = Path(working)
        root = scratch / "dataset"
        root.mkdir()
        if args.dataset == "techqa":
            _techqa(root, scratch)
        elif args.dataset == "techmanualqa":
            _manuals(root, scratch, args.pdf_dir)
        else:
            _linuxcnc(root, scratch)
        root.rename(destination)

    with (destination / "cases.jsonl").open(encoding="utf-8") as labels:
        cases = sum(1 for _ in labels)
    documents = len(
        json.loads((destination / "sources.json").read_text(encoding="utf-8"))
    )
    print(f"Prepared {documents:,} documents and {cases:,} cases in {destination}")


if __name__ == "__main__":
    main()
