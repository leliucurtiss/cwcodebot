import json
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
        yield f"data: {json.dumps({'choices': [{'delta': {'content': 'ok'}}]})}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

class FragmentedGateway(FakeGateway):
    async def stream_chat_completion(self, payload: dict[str, Any]):
        # Simulate a fragmented tool call that should trigger the warning
        # "tool\nrun_in_terminal{...}"
        chunks = [
            "data: {\"choices\": [{\"delta\": {\"content\": \"tool\\n\"}}]}\n\n",
            "data: {\"choices\": [{\"delta\": {\"content\": \"run_in_terminal{\"}}]}\n\n",
            "data: {\"choices\": [{\"delta\": {\"content\": \"command: 'ls'\"}}]}\n\n",
            "data: {\"choices\": [{\"delta\": {\"content\": \"}\"}}]}\n\n",
            "data: [DONE]\n\n"
        ]
        for c in chunks:
            yield c.encode("utf-8")

def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        server=ServerConfig(),
        maas=MaasConfig(base_url="http://maas.test/v1", api_key="local", default_model="devstral"),
        context=ContextConfig(safe_input_budget=1000),
        workspace=WorkspaceConfig(root=tmp_path),
        storage=StorageConfig(sqlite_path=tmp_path / "harness.db"),
    )

def test_fragmented_tool_call_warning(tmp_path: Path, capsys):
    app = create_app(config=make_config(tmp_path), gateway=FragmentedGateway())
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "devstral", "stream": True, "messages": [{"role": "user", "content": "test fragment"}]},
        )
    
    captured = capsys.readouterr()
    assert "WARNING: Potential fragmented tool call detected" in captured.out
    assert "run_in_terminal" in captured.out
    assert response.status_code == 200

def test_translate_tool_call_brace_format():
    # Test standard JSON in braces
    res, rem = _translate_tool_call("read_file{\"filePath\": \"/tmp/test.txt\"} trailing")
    assert res is not None
    assert "read_file" in res
    assert "filePath" in res
    assert rem == " trailing"

    # Test pseudo-JSON (unquoted keys)
    res, rem = _translate_tool_call("grep_search{query: \"test\", includePattern: \"*.py\"} trailing")
    assert res is not None
    assert "grep_search" in res
    assert "query" in res
    assert "includePattern" in res
    assert rem == " trailing"

def test_translate_tool_call_rejects_incomplete_arguments():
    # Should return None for the translation part
    res, rem = _translate_tool_call("TOOL_NAME: read_file\nBEGIN_ARG: filePath\n/path")
    assert res is None
    assert rem == "TOOL_NAME: read_file\nBEGIN_ARG: filePath\n/path"
