# Local RAG Engine

A local retrieval-augmented generation engine for documentation-based support.
It indexes Markdown, retrieves and reranks evidence, and generates answers with
source citations. HTTP, CLI, reusable runtime, and evaluation interfaces are included.

## Features

- Structure-aware Markdown chunking and persistent Chroma storage.
- Vector retrieval with cross-encoder, LLM, or vector-order ranking.
- Section-based evidence stitching and source citations.
- Local answer generation through Ollama.
- A persistent FastAPI application with bounded generation concurrency.
- A retrieval-only endpoint for use with an external generator.
- Single-answer CLI caching and configurable evaluation judges.

## Start here

See [SETUP.md](SETUP.md) for installation, Docker, CLI, and shutdown instructions.
The included [example corpus and catalogue](examples/README.md) are fictional.

## Architecture

Markdown → chunking and embedding → Chroma index. A request is validated against
the catalogue, embedded, searched, reranked, and assembled into evidence.
Generation receives that evidence and returns text with source references.

| Layer | Responsibility |
| --- | --- |
| `src/rag_engine/engine/` | Input preparation, indexing, retrieval, reranking, and generation. |
| `src/rag_engine/runtime/` | Retained models, database handles, and reusable service operations. |
| `src/rag_engine/serving/` | HTTP contracts, response mapping, lifecycle, and scheduling. |
| `src/rag_engine/cli/` | Indexing, retrieval, and answer commands. |
| `src/rag_engine/evaluation/` | Batch generation and evaluation judges. |
| `src/rag_engine/settings.py` | Corpus paths, model defaults, and retrieval limits. |

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/alarms` | List catalogue entries. |
| `POST /api/v1/retrieve` | Return evidence for a catalogue code. Only `code` is required. |
| `POST /api/v1/resolve` | Generate an answer. Requires `code` and `env`. |
| `GET /health/live` | Process liveness. |
| `GET /health/ready` | Backend readiness snapshot. |

Interactive API documentation is available at `http://localhost:8000/docs`
after startup. The optional `X-RAG-Tier` header accepts `operator`, `technician`,
or `partner` and controls which response fields are included.

## Limitations

- Retrieval is catalogue-based. It is not an arbitrary-question search API.
- Environment metadata filtering is disabled.
- Citations identify selected evidence. Claim applicability, prerequisites, and
  step order are not validated by the response parser.
- There is no follow-up chat, confidence score, or HTTP answer cache.
- Tier selection is a response option and does not authenticate a caller.
- Browser clients need a same-origin proxy because CORS is not configured.
- Model downloads require network access initially. Latency depends on hardware
  and the selected models. No performance benchmark is supplied.

## Development

```bash
python -m ruff check src tests .github/scripts
python -m black --check src tests .github/scripts
python -m pytest -q
```

The [CI workflow](.github/workflows/ci.yml) defines Python checks, Compose
validation, and container smoke checks. It does not benchmark live-model quality.
See [LICENSE](LICENSE) for the copyright notice. This repository currently grants
no public open-source licence.
