from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from local_ai_harness.app import create_app
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


def test_debug_context_route(tmp_path: Path) -> None:
    app = create_app(config=make_config(tmp_path), gateway=FakeGateway())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post(
            "/harness/debug/context",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    assert response.json()["estimated_tokens"] > 0
