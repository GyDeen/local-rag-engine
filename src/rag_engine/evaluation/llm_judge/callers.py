import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from functools import partial
from typing import Any

from rag_engine.engine.llm.llm_callers import OLLAMA_HOST, call_ollama_llm

DEFAULT_JUDGE_MODEL = "qwen3:4b"
DEFAULT_API_BASE_URL = OLLAMA_HOST


class JudgeAPIError(RuntimeError):
    """A judge API request failed or did not produce a complete answer."""


def make_judge_caller(
    *,
    provider: str = "ollama",
    model: str = DEFAULT_JUDGE_MODEL,
    base_url: str = DEFAULT_API_BASE_URL,
    api_key_env: str = "OPENAI_API_KEY",
    api: str = "chat_completions",
    reasoning_effort: str | None = None,
    max_output_tokens: int = 8192,
    timeout: float = 300,
) -> Callable[[str, str], str]:
    """Build a caller using any configured model ID, without model-name guessing.

    openai and openai-compatible use Bearer authentication. The API protocol is
    explicit because model names alone cannot determine endpoint capabilities.
    ollama is the default local backend and needs no API key. No retries or
    schema validation occur.
    """
    if provider == "ollama":
        return partial(call_ollama_llm, model=model, host=base_url, timeout=timeout)
    if provider not in {"openai", "openai-compatible"}:
        raise ValueError(f"Unknown judge provider: {provider}")
    if api not in {"responses", "chat_completions"}:
        raise ValueError(f"Unknown judge API: {api}")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(
            f"Set the {api_key_env} environment variable for the LLM judge"
        )

    def caller(system_prompt: str, user_prompt: str) -> str:
        payload: dict[str, Any] = {"model": model, "store": False}
        if api == "responses":
            endpoint = "responses"
            payload.update(
                instructions=system_prompt,
                input=user_prompt,
                text={"format": {"type": "json_object"}},
                max_output_tokens=max_output_tokens,
                tools=[],
            )
            if reasoning_effort is not None:
                payload["reasoning"] = {"effort": reasoning_effort}
        else:
            endpoint = "chat/completions"
            payload.update(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=max_output_tokens,
            )
            if reasoning_effort is not None:
                payload["reasoning_effort"] = reasoning_effort
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            # Do not log request headers, credentials, or potentially private bodies.
            raise JudgeAPIError(f"Judge API returned HTTP {error.code}") from None
        except urllib.error.URLError:
            raise JudgeAPIError("Could not connect to the judge API") from None
        if api == "responses":
            if body["status"] != "completed":
                raise JudgeAPIError(f"Judge API response status: {body['status']}")
            return "".join(
                content["text"]
                for item in body["output"]
                if item["type"] == "message"
                for content in item["content"]
                if content["type"] == "output_text"
            )
        choice = body["choices"][0]
        if choice["finish_reason"] != "stop":
            raise JudgeAPIError(f"Judge API finish reason: {choice['finish_reason']}")
        return choice["message"]["content"]

    return caller
