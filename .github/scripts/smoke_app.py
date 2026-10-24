"""Exercise the HTTP image offline without inference models or test dependencies."""

import asyncio
import json
from typing import Any

from uvicorn import Config

from rag_engine.serving.app import create_app
from rag_engine.serving.backend import unconfigured_backend


async def request(app, method: str, path: str, body: Any = None) -> tuple[int, Any]:
    """Send an in-process ASGI request through the app's real middleware and routes."""
    payload = b"" if body is None else json.dumps(body).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }
    messages = []

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    response = b"".join(
        m.get("body", b"") for m in messages if m["type"] == "http.response.body"
    )
    return status, json.loads(response)


async def main() -> None:
    """Check startup, validation, health, service errors, and resource release."""
    config = Config("rag_engine.serving.app:create_app", factory=True)
    config.load()
    assert config.loaded
    app = create_app(unconfigured_backend)
    async with app.router.lifespan_context(app):
        assert app.state.phase == "running"
        status, data = await request(app, "GET", "/health/live")
        assert status == 200 and data == {"status": "alive", "phase": "running"}
        status, data = await request(app, "GET", "/health/ready")
        assert status == 503 and data["ready"] is False
        status, schema = await request(app, "GET", "/openapi.json")
        assert status == 200 and "/api/v1/resolve" in schema["paths"]
        status, _ = await request(app, "POST", "/api/v1/resolve", {})
        assert status == 422
        status, data = await request(
            app,
            "POST",
            "/api/v1/resolve",
            {"code": "demo.fb.0002", "env": {"versions": {}, "locale": "en-US"}},
        )
        assert status == 503 and "not connected" in data["detail"]
    assert app.state.phase == "stopped" and app.state.backend is None
    print("HTTP app startup, routes, validation, and shutdown passed.")


if __name__ == "__main__":
    asyncio.run(main())
