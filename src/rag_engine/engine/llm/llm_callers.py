"""Ollama HTTP chat backend for local generation."""

import json
import os
import urllib.request

from rag_engine.settings import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT,
)

OLLAMA_MODEL = DEFAULT_OLLAMA_MODEL
OLLAMA_HOST = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)

# Grounded generation wants the answer only, never an analysis prelude, so
# thinking is switched off here, once, at the single entry point to the model
# runtime, instead of inside individual prompts: changing OLLAMA_MODEL cannot
# silently re-enable it. Two independent switches are applied together because
# neither covers every model:
#   - the request field ``think: false`` is authoritative for runtimes that
#     expose thinking control, and is ignored by models without a thinking
#     mode (only ``think: true`` is rejected by those);
#   - the ``/no_think`` soft switch is part of the Qwen3 prompt contract, so it
#     still applies when a runtime ignores the request field.
NO_THINK_SWITCH = "/no_think"


def _without_thinking(system_prompt: str) -> str:
    """Return ``system_prompt`` carrying the no-thinking soft switch."""

    if NO_THINK_SWITCH in system_prompt:
        return system_prompt
    return f"{system_prompt.rstrip()}\n\n{NO_THINK_SWITCH}"


def call_ollama_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    temperature: float = 0.0,
    timeout: int = DEFAULT_OLLAMA_TIMEOUT,
    context_length: int | None = None,
    response_schema: dict | None = None,
    max_new_tokens: int | None = None,
    keep_alive: str | int | None = None,
) -> str:
    """Call a local Ollama model via its HTTP chat API (stdlib only).

    Greedy-equivalent (temperature 0) and non-thinking output are requested
    because this is a factual RAG stage and deterministic output is preferred.
    Thinking is disabled for every model, by the API request field and by the
    ``/no_think`` soft switch (see ``_without_thinking``). Structured output
    (format json) is required because reasoning models otherwise emit an
    analysis prelude instead of JSON-only text.
    """

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _without_thinking(system_prompt)},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "format": response_schema if response_schema is not None else "json",
        "options": {"temperature": temperature},
    }
    if context_length is not None:
        if context_length < 1:
            raise ValueError("context_length must be positive")
        payload["options"]["num_ctx"] = context_length
    if max_new_tokens is not None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        payload["options"]["num_predict"] = max_new_tokens
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["message"]["content"]).strip()


def preload_ollama_model(
    *,
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    keep_alive: str | int = -1,
    timeout: int = DEFAULT_OLLAMA_TIMEOUT,
) -> None:
    """Load a model before accepting requests and retain it across idle periods."""
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(
            {"model": model, "keep_alive": keep_alive, "stream": False}
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if body.get("error"):
        raise RuntimeError(f"Ollama model loading failed: {body['error']}")
