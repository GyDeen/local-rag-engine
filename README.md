# Local RAG Engine

A local retrieval-augmented generation engine for technical support. It indexes
manuals and support articles, then serves retrieved evidence and cited answers
through FastAPI.

## What it supports

- Ingest plain text, Markdown, PDF, Word (`.docx`), and HTML.
- Search a Chroma index using embeddings and optional cross-encoder reranking.
- Retrieve documentation and resolve known alarms through the HTTP API.
- Generate answers with Ollama and cite the source document, section, or PDF page
  when available.

## Quick start

Use Python 3.14 and run these commands from the repository root. The index uses
the bundled example documents.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/requirements-service.txt
export PYTHONPATH=src

python -m rag_engine.cli.index
RAG_RETRIEVAL_ONLY=1 python -m uvicorn rag_engine.serving.app:create_app --factory
```

Open [Swagger UI](http://127.0.0.1:8000/docs) and try `POST /api/v1/retrieve` with
`{"code": "demo.svc.0001"}`. Retrieval-only mode returns the documented evidence
without starting Ollama.

- [Service setup](SETUP.md) covers Docker, Ollama, and runtime configuration.
- [Document ingestion](docs/ingestion.md) explains supported formats and converted files.
- [Benchmark guide](docs/benchmarks.md) covers evaluation.
