"""Testing interface. Allow different format testing dataset and normalize into uniform format"""

import argparse
import json
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import ZipFile

from rag_engine.cli.errors import run_cli

from .linuxcnc import LINUXCNC_CASES, LINUXCNC_DOCUMENTS

TECHQA_RELEASE = "0b5bbc84b7f07d6d09d063130e90b716d8d4a32a"
TECHQA_BASE = (
    f"https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval/resolve/{TECHQA_RELEASE}"
)
TECHMANUALQA_ARCHIVE = "https://zenodo.org/records/17410809/files/TechManualQA_700.zip"


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response, path.open("wb") as output:
        shutil.copyfileobj(response, output)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


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
            relative = source.name
            with (root / "documents" / relative).open("xb") as output:
                output.write(archive.read(member))
            indexed.add(relative)

    def cases() -> Iterator[dict]:
        for row in json.loads(labels.read_text(encoding="utf-8")):
            references = [
                {
                    "source": PurePosixPath(context["filename"]).name,
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
    zip_path = scratch / "manual-qa.zip"
    _download(TECHMANUALQA_ARCHIVE, zip_path)
    with ZipFile(zip_path) as archive:
        with archive.open("TechManualQA_700/TechManualQA_700.jsonl") as source:
            labels = [json.loads(line) for line in source]

    manual_ids = {row["doc_id"] for row in labels}
    # Every supplied manual is eligible evidence, including distractors.
    pdfs = sorted(pdf for pdf in pdf_dir.iterdir() if pdf.suffix.lower() == ".pdf")
    if not pdfs:
        raise ValueError(
            f"No matching manuals in {pdf_dir}. Expected: {', '.join(sorted(manual_ids))}"
        )
    sources = {}
    for pdf in pdfs:
        shutil.copyfile(pdf, root / "documents" / pdf.name)
        sources[pdf.name] = str(pdf.resolve())

    def cases() -> Iterator[dict]:
        available = {pdf.name for pdf in pdfs}
        for row in labels:
            if row["doc_id"] not in available:
                continue
            answerable = row["category"] != "Unanswerable"
            references = []
            if answerable:
                page = int(row["gt_page_number"])
                references.append({"source": row["doc_id"], "page": page})
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


def _linuxcnc(root: Path, scratch: Path) -> None:
    # Download whole documents. Labelled anchors never guide extraction.
    for relative, url in LINUXCNC_DOCUMENTS.items():
        _download(url, root / "documents" / relative)
    _write_jsonl(root / "cases.jsonl", LINUXCNC_CASES)
    _write_sources(root, LINUXCNC_DOCUMENTS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("techqa", "techmanualqa", "linuxcnc"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/benchmarks"))
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
        (root / "documents").mkdir(parents=True)
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
    print(f"Prepared {documents:,} raw documents and {cases:,} cases in {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
