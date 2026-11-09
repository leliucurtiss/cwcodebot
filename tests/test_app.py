import json
from pathlib import Path
from typing import Any

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
            json={"model": "devstral", "messages": [{"role": "user", "content": "hello"}]},
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
    assert _translate_tool_call("TOOL_NAME: read_file\nBEGIN_ARG: filePath\n/path") is None


def test_debug_context_route(tmp_path: Path) -> None:
    app = create_app(config=make_config(tmp_path), gateway=FakeGateway())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post(
            "/harness/debug/context",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    assert response.json()["estimated_tokens"] > 0
