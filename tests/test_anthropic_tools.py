import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from local_ai_harness.app import create_app
from local_ai_harness.config import (
    AppConfig,
    ContextConfig,
    MaasConfig,
    ServerConfig,
    StorageConfig,
    WorkspaceConfig,
)


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        server=ServerConfig(),
        maas=MaasConfig(base_url="http://maas.test/v1", api_key="local", default_model="gemma"),
        context=ContextConfig(),
        workspace=WorkspaceConfig(root=tmp_path),
        storage=StorageConfig(sqlite_path=tmp_path / "harness.db"),
    )


class CaptureGateway:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def list_models(self) -> dict[str, Any]:
        return {"data": []}

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    async def stream_chat_completion(self, payload: dict[str, Any]):
        yield b""


class StreamingGateway(CaptureGateway):
    async def stream_chat_completion(self, payload: dict[str, Any]):
        self.payloads.append(payload)
        stream = (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123",'
            '"type":"function","function":{"name":"read_file","arguments":""}}]}}]}\n\n'
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
            '{"arguments":"{\\"path\\":\\"README.md\\"}"}}]}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            "data: [DONE]\n\n"
        )
        for size in range(0, len(stream), 11):
            yield stream[size : size + 11].encode()


def test_anthropic_tools_messages_and_response_round_trip(tmp_path: Path) -> None:
    gateway = CaptureGateway()
    app = create_app(config=make_config(tmp_path), gateway=gateway)  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "gemma",
                "max_tokens": 512,
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "input_schema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
                "tool_choice": {"type": "tool", "name": "read_file"},
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool_old",
                                "name": "list_dir",
                                "input": {"path": "."},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool_old",
                                "content": "README.md",
                            }
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200
    payload = gateway.payloads[0]
    assert payload["tools"][0]["function"]["name"] == "read_file"
    assert payload["tools"][0]["function"]["parameters"]["required"] == ["path"]
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "read_file"},
    }
    assert payload["messages"][1]["tool_calls"][0]["id"] == "tool_old"
    assert payload["messages"][2] == {
        "role": "tool",
        "tool_call_id": "tool_old",
        "content": "README.md",
    }
    block = response.json()["content"][0]
    assert block == {
        "type": "tool_use",
        "id": "call_123",
        "name": "read_file",
        "input": {"path": "README.md"},
    }
    assert response.json()["stop_reason"] == "tool_use"


def test_anthropic_stream_translates_fragmented_native_tool_call(tmp_path: Path) -> None:
    gateway = StreamingGateway()
    app = create_app(config=make_config(tmp_path), gateway=gateway)  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "max_tokens": 128,
                "stream": True,
                "messages": [{"role": "user", "content": "read README"}],
            },
        )

    events = []
    for block in response.text.split("\n\n"):
        event_type = ""
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if data is not None:
            events.append((event_type, data))

    starts = [data for event, data in events if event == "content_block_start"]
    deltas = [data for event, data in events if event == "content_block_delta"]
    message_deltas = [data for event, data in events if event == "message_delta"]
    assert starts[0]["content_block"] == {
        "type": "tool_use",
        "id": "call_123",
        "name": "read_file",
        "input": {},
    }
    assert deltas[0]["delta"]["type"] == "input_json_delta"
    assert json.loads(deltas[0]["delta"]["partial_json"]) == {"path": "README.md"}
    assert message_deltas[0]["delta"]["stop_reason"] == "tool_use"
