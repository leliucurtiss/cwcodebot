import json
import asyncio
import httpx
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_ai_harness.app import _translate_tool_call, create_app
from local_ai_harness.config import AppConfig, ContextConfig, MaasConfig, ServerConfig, StorageConfig, WorkspaceConfig


class FakeGateway:
    async def list_models(self) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "devstral", "object": "model"}]}

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "test",
            "object": "chat.completion",
            "model": payload["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }

    async def stream_chat_completion(self, payload: dict[str, Any]):
        # For non-streaming requests, the app.py logic still calls stream_chat_completion 
        # if req.stream is True. But for non-streaming, it calls chat_completion.
        # Wait, looking at app.py:
        # if req.stream:
        #     ...
        #     async for chunk in maas.stream_chat_completion(payload):
        # response = await maas.chat_completion(payload)
        
        # The test_chat_completion_route_uses_gateway uses req.stream=False (default).
        # So it should call chat_completion.
        # However, the error shows it's getting SSE data. 
        # This means the app.py might be defaulting to stream or the test is hitting the stream path.
        # Actually, the error is: s = 'data: {"choices": [{"delta": {"content": "ok"}}]}\n\ndata: [DONE]\n'
        # This is exactly what I added to FakeGateway.stream_chat_completion.
        # Why is it being called? 
        # In app.py:
        # @app.post("/v1/chat/completions")
        # async def chat_completions(req: ChatCompletionRequest, ...):
        #     ...
        #     if req.stream:
        #         ...
        #         return StreamingResponse(stream(), ...)
        #     response = await maas.chat_completion(payload)
        #     ...
        #     return response
        
        # If req.stream is False, it calls chat_completion.
        # My FakeGateway.chat_completion returns a dict.
        # But the error shows it's receiving SSE.
        # This suggests that in the test, req.stream might be True or the app is behaving differently.
        # Let's check the test:
        # response = client.post(
        #     "/v1/chat/completions",
        #     json={"model": "devstral", "messages": [{"role": "user", "content": "hello"}]},
        # )
        # Here stream is not provided, so it defaults to True in ChatCompletionRequest!
        # class ChatCompletionRequest(BaseModel):
        #     ...
        #     stream: bool = True
        
        # That's the problem. The test expects a JSON response but the request defaults to streaming.
        
        yield f"data: {json.dumps({'choices': [{'delta': {'content': 'ok'}}]})}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"


class StreamingGateway(FakeGateway):
    async def stream_chat_completion(self, payload: dict[str, Any]):
        events = [
            {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
            {"choices": [{"delta": {"content": "TOOL_NA"}}]},
            {"choices": [{"delta": {"content": 'ME: read_file\nBEGIN_ARG: filePath\n"/tmp/a.txt"\nEND_ARG'}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        for offset in range(0, len(stream), 19):
            yield stream[offset:offset + 19].encode("utf-8")


class ToolCaptureGateway(FakeGateway):
    def __init__(self, response_text="") -> None:
        self.payload: dict[str, Any] | None = None
        self.response_text = response_text

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payload = payload
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": self.response_text},
                    "finish_reason": "stop",
                }
            ]
        }


class VerifyingGateway(FakeGateway):
    async def verify_upstream(self) -> dict[str, Any]:
        return {"response_id": "gen-1", "content_present": True}


def test_gateway_retries_transient_rate_limit(monkeypatch) -> None:
    from local_ai_harness.gateway import MaasGateway

    attempts = 0

    async def fake_post(self, url, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(
                429,
                request=httpx.Request("POST", url),
                json={"error": {"message": "temporary"}},
            )
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "OK"}}]},
        )

    async def no_delay(self, response, attempt):
        return None

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr(MaasGateway, "_retry_delay", no_delay)
    gateway = MaasGateway(
        MaasConfig(
            base_url="https://provider.test/v1",
            api_key="test",
            default_model="test",
        )
    )

    response = asyncio.run(gateway.chat_completion({"model": "test", "messages": []}))

    assert attempts == 3
    assert response["choices"][0]["message"]["content"] == "OK"


def test_upstream_error_includes_provider_metadata() -> None:
    from local_ai_harness.gateway import MaasGateway, MaasUpstreamError

    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://provider.test"),
        json={
            "error": {
                "message": "Provider returned error",
                "metadata": {"raw": "temporary shared-pool rate limit"},
            }
        },
    )

    with pytest.raises(MaasUpstreamError, match="shared-pool rate limit"):
        MaasGateway._raise_for_status(response)


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        server=ServerConfig(),
        maas=MaasConfig(base_url="http://maas.test/v1", api_key="local", default_model="devstral"),
        context=ContextConfig(safe_input_budget=1000),
        workspace=WorkspaceConfig(root=tmp_path),
        storage=StorageConfig(sqlite_path=tmp_path / "harness.db"),
    )


def test_chat_completion_route_uses_gateway(tmp_path: Path) -> None:
    app = create_app(config=make_config(tmp_path), gateway=FakeGateway())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "devstral", "stream": False, "messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"


def test_text_tool_call_is_translated_in_stream_and_reasoning_is_preserved(tmp_path: Path) -> None:
    app = create_app(config=make_config(tmp_path), gateway=StreamingGateway())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "devstral", "stream": True, "messages": [{"role": "user", "content": "open file"}]},
        )

    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    tool_calls = [
        call
        for event in events
        for choice in event.get("choices", [])
        for call in choice.get("delta", {}).get("tool_calls", [])
    ]

    assert response.status_code == 200
    assert "thinking" in response.text
    assert "TOOL_NAME:" not in response.text
    assert tool_calls[0]["function"]["name"] == "read_file"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"filePath": "/tmp/a.txt"}


def test_translate_tool_call_rejects_incomplete_arguments() -> None:
    assert _translate_tool_call("TOOL_NAME: read_file\nBEGIN_ARG: filePath\n/path")[0] is None


def test_openai_tool_history_and_definitions_are_forwarded(tmp_path: Path) -> None:
    gateway = ToolCaptureGateway()
    app = create_app(config=make_config(tmp_path), gateway=gateway)  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "devstral",
                "stream": False,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "result"},
                ],
            },
        )

    assert response.status_code == 200
    assert gateway.payload is not None
    assert gateway.payload["tools"][0]["function"]["name"] == "read_file"
    assert gateway.payload["messages"][0]["tool_calls"][0]["id"] == "call_1"
    assert gateway.payload["messages"][1]["tool_call_id"] == "call_1"


def test_upstream_health_verifies_chat_authentication(tmp_path: Path) -> None:
    app = create_app(config=make_config(tmp_path), gateway=VerifyingGateway())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.get("/health/upstream")

    assert response.status_code == 200
    assert response.json()["authentication_verified"] is True
    assert response.json()["response_id"] == "gen-1"


def test_non_stream_text_tool_call_is_translated(tmp_path: Path) -> None:
    gateway = ToolCaptureGateway(
        '<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>'
    )
    app = create_app(config=make_config(tmp_path), gateway=gateway)  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "devstral",
                "stream": False,
                "messages": [{"role": "user", "content": "read README"}],
            },
        )

    message = response.json()["choices"][0]["message"]
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "path": "README.md"
    }


def test_debug_context_route(tmp_path: Path) -> None:
    app = create_app(config=make_config(tmp_path), gateway=FakeGateway())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post(
            "/harness/debug/context",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    assert response.json()["estimated_tokens"] > 0
