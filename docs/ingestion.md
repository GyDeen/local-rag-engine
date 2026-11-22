# Document ingestion

Ingestion takes a file or a directory of documents and produces Markdown for
indexing. It preserves relative paths and appends `.md` to each filename, so
`manual.pdf` becomes `manual.pdf.md` and `notes.md` becomes `notes.md.md`.

## Supported formats

| Input | What ingestion does |
| --- | --- |
| Plain text (`.txt`) and Markdown (`.md`, `.markdown`) | Preserves the source text. |
| PDF (`.pdf`) | Uses Docling to recover text, headings, and tables in page order. |
| Word (`.docx`) and HTML (`.html`, `.htm`) | Uses Docling to convert document structure to Markdown. |

## Output

- `documents/` contains one Markdown file per source. A PDF stays together in
  a single file, with Docling recovering the reading order within each page.
- `debug/manifest.json` maps each Markdown file to its original source. For
  PDFs, it records the Markdown character spans that belong to each page.
  An optional source map can supply original URLs or file paths.
- `debug/structure/` contains Docling's structured JSON for inspecting recovered
  elements, layout, and extraction issues. Complex tables and diagrams can
  lose information in the Markdown export.

Ingestion logs unsupported files and empty PDF pages. If a supported document
produces no usable text or fails to convert, the batch stops before publishing
partial output. Existing `documents/` and `debug/` directories are protected
from overwrites.
