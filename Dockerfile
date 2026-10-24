# Pinned to the patch in .python-version so the image, CI and local tooling
# agree exactly. Bump all three together.
FROM python:3.14.7-slim AS base
WORKDIR /app
ENV PYTHONPATH=/app/src
ENV HF_HOME=/cache/huggingface

COPY requirements/ requirements/
RUN python -m pip install --no-cache-dir -r requirements/requirements.txt

FROM base AS test
COPY . /app
RUN python -m pip install --no-cache-dir pytest==9.1.1
CMD ["python", "-m", "pytest", "-q"]

FROM base AS runtime_cli
COPY . /app
CMD ["python", "-m", "rag_engine.cli.retrieve", "--help"]

# The final stage is the default for a plain docker build: serve the HTTP app.
FROM base AS runtime
RUN python -m pip install --no-cache-dir -r requirements/requirements-service.txt
COPY . /app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "rag_engine.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
