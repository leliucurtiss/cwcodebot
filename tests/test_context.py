from pathlib import Path

from local_ai_harness.config import ContextConfig
from local_ai_harness.context import ContextBuilder
from local_ai_harness.memory import HierarchicalMemory
from local_ai_harness.models import ChatMessage
from local_ai_harness.storage import Project, StoredMessage
from local_ai_harness.workspace import WorkspaceReader


def test_context_builder_keeps_latest_user_even_with_tiny_budget(tmp_path: Path) -> None:
    project = Project("p", str(tmp_path), "memory", "now", "now")
    builder = ContextBuilder(
        ContextConfig(safe_input_budget=120, token_char_ratio=3.0, token_safety_multiplier=1.0),
        WorkspaceReader(tmp_path, max_file_bytes=1000),
    )
    stored = [
        StoredMessage(i, "c", "user", "old message " * 100, "now")
        for i in range(1, 5)
    ]
    incoming = [ChatMessage(role="user", content="current request")]

    built = builder.build(project, HierarchicalMemory("memory", ""), stored, incoming, [])

    assert built.messages[-1]["content"] == "current request"
    assert "recent_history" in built.discarded


def test_context_builder_includes_explicit_file(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("VALUE = 42", encoding="utf-8")
    project = Project("p", str(tmp_path), "", "now", "now")
    builder = ContextBuilder(ContextConfig(), WorkspaceReader(tmp_path, max_file_bytes=1000))

    built = builder.build(project, HierarchicalMemory("", ""), [], [ChatMessage(role="user", content="read file")], ["a.py"])

    joined = "\n".join(msg["content"] for msg in built.messages)
    assert "VALUE = 42" in joined
