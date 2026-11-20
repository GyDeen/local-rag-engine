# Catalogue API setup

For free-form questions and public corpus experiments, use the [benchmark setup](docs/benchmarks.md).

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
curl -X POST http://localhost:8000/api/v1/resolve -H 'Content-Type: application/json' -d '{"code":"demo.svc.0001","env":{"versions":{"service":"1.0.0"},"locale":"en-US"}}'
```

The answer contains `summary`, `steps`, and `citations` along with catalogue
metadata. Unknown codes return `404`, invalid input returns `422`, and a busy
or unavailable backend returns `503`.

## CLI and evaluation

```bash
PYTHONPATH=src python -m rag_engine.cli.retrieve examples/alarm-report.json
PYTHONPATH=src python -m rag_engine.cli.answer examples/alarm-report.json
PYTHONPATH=src python -m rag_engine.evaluation.run_all --codes demo.svc.0001
```

Use `--help` on a command for options. Evaluation configs point to the fictional
example catalogue and references. The default judge runs locally through Ollama.

## Retrieval-only service

```bash
RAG_RETRIEVAL_ONLY=1 PYTHONPATH=src python -m uvicorn rag_engine.serving.app:create_app --factory --host 127.0.0.1 --port 8000
curl -X POST http://localhost:8000/api/v1/retrieve -H 'Content-Type: application/json' -d '{"code":"demo.svc.0001"}'
```

This mode skips Ollama startup and answer generation. Resolve returns `503`.

## Configuration and storage

Copy `.env.example` to `.env` if needed. Compose reads `.env` automatically.
For local Python, export settings in the terminal before starting the app.
Settings include `OLLAMA_MODEL`, `OLLAMA_HOST`, `RAG_RETRIEVAL_ONLY`,
`RAG_GENERATION_CONCURRENCY`, and `RAG_GENERATION_QUEUE_SIZE`.
If changing the model, pull its weights before startup.

Default corpus and catalogue paths are in `src/rag_engine/settings.py`.
The local vector index lives in `data/vector_db`. Docker stores it in the
`vector-db` volume. The default collection is `docs_bge_m3_c1000_o100`.
CLI answer-cache files live separately in `data/answer_cache`. Generated data,
outputs, local credentials, and virtual environments are excluded from Git.

Build the index once before starting the service. The builder rejects an
existing collection. After corpus changes, retain the previous index and build
a replacement in a fresh path or collection. Keep the consumer's database path,
collection, and embedding model consistent with the build. Indexing and CLI
commands support path and collection options. The default HTTP runtime reads
shared defaults from `settings.py`.
