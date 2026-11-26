# FastAPI service setup

For labelled public corpus experiments, use the [benchmark setup](docs/benchmarks.md).

Run commands from the repository root. Use Python 3.14 as specified in
`.python-version`. Docker setups also require Docker and Compose.

## Local Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/requirements-dev.txt
```

Start Ollama separately if it is not already running, then prepare its model:

```bash
ollama serve
```

In another terminal with the environment activated:

```bash
ollama pull qwen3:4b
PYTHONPATH=src python -m rag_engine.cli.index
PYTHONPATH=src python -m uvicorn rag_engine.serving.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

Indexing also downloads the embedding model when it is not cached. Runtime
startup loads the reranker and retains models for repeated requests. Initial
startup may take time. Press Ctrl+C to stop the HTTP app. Stop a separately
launched Ollama process in its own terminal when finished.

## Serve your own documents

Choose a fresh output directory for the converted documents and index. See
[document ingestion](docs/ingestion.md) for supported formats and output files.

```bash
python -m pip install -r requirements/requirements-ingestion.txt -r requirements/requirements-service.txt
export PYTHONPATH=src

corpus=data/processed/my-corpus
embedding=sentence-transformers/all-MiniLM-L6-v2
python -m rag_engine.cli.ingest /path/to/documents --output "$corpus"
python -m rag_engine.cli.index --docs-root "$corpus/documents" \
  --database-path "$corpus/index" --collection my-documents \
  --embedding-model "$embedding"

RAG_DOCS_ROOT="$corpus/documents" RAG_DATABASE_PATH="$corpus/index" \
RAG_COLLECTION=my-documents RAG_EMBEDDING_MODEL="$embedding" \
RAG_RETRIEVAL_ONLY=1 python -m uvicorn rag_engine.serving.app:create_app --factory
```

Open [Swagger UI](http://127.0.0.1:8000/docs) or send a request from another terminal.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What should I check before replacing the motor?"}'
```

For cited answers, start Ollama with the configured `OLLAMA_MODEL` and start the
service with `RAG_RETRIEVAL_ONLY=0`. Send `"generate": false` to `/api/v1/ask` for
evidence only. `/api/v1/retrieve` accepts the same request and always returns
evidence only. Set `RAG_RERANKER=none` to use vector ranking.

## Index and filter document metadata

Pass `--metadata-fields fields.json` to `rag_engine.cli.index` to extract fields
from labelled Markdown lines such as `Model: North` and `Revision: 2`.

```json
{
  "product": {"labels": ["Product", "Model"]},
  "revision": {"labels": ["Revision"], "type": "integer"}
}
```

The indexer applies each extracted value to the document's chunks. Supported
types are `string` (the default), `integer`, `number`, and `boolean`. Fields with
missing, conflicting, or invalid values are omitted. Use `--metadata` with a
JSON map from relative document paths to metadata objects to supply or override
values. Start with a fresh index to apply new fields to existing documents.

Send explicit `filters` with `POST /api/v1/ask`. Each field must match the
indexed scalar value or an item in an indexed array. Multiple fields must all
match. Filter strings are case-sensitive.

```json
{"question": "How do I reset the controller?", "filters": {"product": "North", "revision": 2}}
```

## Docker

```bash
docker compose build
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3:4b
docker compose run --rm app python -m rag_engine.cli.index
docker compose up -d app
```

For later starts, use `docker compose up -d app`. View logs with
`docker compose logs -f app` and stop with `docker compose down`.
The database and model volumes survive a normal stop.

For development against host Ollama, use `compose.dev.yaml` instead:

```bash
docker compose -f compose.dev.yaml build
docker compose -f compose.dev.yaml run --rm dev python -m rag_engine.cli.index
docker compose -f compose.dev.yaml up -d dev
```

Stop with `docker compose -f compose.dev.yaml down`. Run one stack at a time
because both use port 8000.

## Try a request

Open `http://localhost:8000/docs`, or run:

```bash
curl http://localhost:8000/health/ready
curl -X POST http://localhost:8000/api/v1/ask -H 'Content-Type: application/json' -d '{"question":"What should I check when the demo worker stops responding?"}'
```

`/api/v1/ask` and `/api/v1/resolve` return `question`, `evidence`, `answer`, and
`status`. Answer text cites passages inline, for example `[E1]`. The status is
`answered` or `insufficient_evidence`. Evidence-only requests return `null` for
`answer` and `status`. All three endpoints use `doc`, `path`, and `pages` for evidence.
All three endpoints use the supplied `question` for retrieval and accept explicit
document filters. `/api/v1/resolve` can also pass optional `code`, `env`, `context`,
and `text` fields to answer generation. The service has no alarm catalogue lookup.
Use `filters` for environment details that have matching indexed metadata. Include
other relevant environment details in the question so they can inform text search.
Invalid input returns `422`, and a busy or unavailable backend returns `503`.

## Development CLI

```bash
PYTHONPATH=src python -m rag_engine.cli.retrieve "How do I restart the demo worker?"
PYTHONPATH=src python -m rag_engine.cli.answer "How do I restart the demo worker?"
```

Use `--help` on a command for options.

## Retrieval-only service

```bash
RAG_RETRIEVAL_ONLY=1 PYTHONPATH=src python -m uvicorn rag_engine.serving.app:create_app --factory --host 127.0.0.1 --port 8000
curl -X POST http://localhost:8000/api/v1/retrieve -H 'Content-Type: application/json' -d '{"question":"What should I check when the demo worker stops responding?"}'
```

This mode skips Ollama startup and answer generation. `/api/v1/ask` and
`/api/v1/resolve` return their evidence with `answer` and `status` set to `null`.

## Logging and errors

The HTTP service and executable commands write logs to stderr. Set
`RAG_LOG_LEVEL` to `DEBUG`, `INFO` (default), `WARNING`, `ERROR`, or
`CRITICAL`. Both Compose files pass this setting to the app. Use
`docker compose logs -f app` to follow service logs.

HTTP responses include `X-Request-ID`. The same ID appears in the request
summary and in retrieval and generation logs. Logs record stage timing and
failure details. Request summaries omit query strings and request bodies.

| Status | Failure |
| --- | --- |
| 422 | Invalid request |
| 502 | Invalid answer or reranker response |
| 503 | Busy service or unavailable dependency |
| 500 | Unexpected error with a request ID for diagnosis |

## Configuration and storage

Use `.env.example` as a template for deployment settings. Compose reads `.env`
by default. To switch configurations, pass `--env-file <file>` to `docker compose`.
For local Python, export settings in the terminal before starting the app.
Settings include `OLLAMA_MODEL`, `OLLAMA_HOST`, `RAG_RETRIEVAL_ONLY`,
`RAG_GENERATION_CONCURRENCY`, and `RAG_GENERATION_QUEUE_SIZE`.

Set `RAG_EVIDENCE_RANK_LIMIT` to a positive integer to limit stitching to the
first ranked chunks. Leaving it unset uses the full candidate pool. The development
retrieval and answer commands accept the same cutoff via `--evidence-rank-limit`.
If changing the model, pull its weights before startup.

The default corpus path is in `src/rag_engine/settings.py`.
The local vector index lives in `data/vector_db`. Docker stores it in the
`vector-db` volume. The default collection is `docs_bge_m3_c1000_o100`.
CLI answer-cache files live separately in `data/answer_cache`. Generated data,
outputs, local credentials, and virtual environments are excluded from Git.

Build the index once before starting the service. The builder rejects an
existing collection. After corpus changes, retain the previous index and build
a replacement in a fresh path or collection. Keep the consumer's database path,
collection, and embedding model consistent with the build. Indexing and CLI
commands support path and collection options. The HTTP runtime reads shared
defaults from `settings.py`. Set `RAG_DATABASE_PATH`, `RAG_COLLECTION`, and
`RAG_EMBEDDING_MODEL` to serve a different index. Use `RAG_DOCS_ROOT` for the
indexed document directory and `RAG_RERANKER` to choose `none` or `cross_encoder`.
With Compose, paths refer to files inside the container.
