from dataclasses import dataclass, field
from typing import Any
from collections import OrderedDict

from .config import ContextConfig
from .memory import HierarchicalMemory
from .models import ChatMessage
from .storage import Project, StoredMessage
from .workspace import WorkspaceError, WorkspaceReader


def message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)

@dataclass
class ContextBuildResult:
    messages: list[dict[str, str]]
    estimated_tokens: int
    discarded: list[str] = field(default_factory=list)
    token_breakdown: dict[str, int] = field(default_factory=dict)

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class ContextBuilder:
    def __init__(self, config: ContextConfig, workspace: WorkspaceReader):
        self.config = config
        self.workspace = workspace

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        raw = len(text) / max(self.config.token_char_ratio, 1.0)
        return int(raw * self.config.token_safety_multiplier) + 1

    def _message_tokens(self, message: dict[str, str]) -> int:
        return self.estimate_tokens(message.get("role", "")) + self.estimate_tokens(message.get("content", ""))

    def compress_context(self, content: str) -> str:
        # Implement your compression logic here
        # For example, you can truncate the content or use a summarization technique
        max_length = self.config.max_compressed_length
        if len(content) > max_length:
            return content[:max_length]
        return content

    def build(
        self,
        project: Project,
        memory: HierarchicalMemory | None,
        stored_messages: list[StoredMessage],
        incoming_messages: list[ChatMessage],
        context_files: list[str],
    ) -> ContextBuildResult:
        budget = self.config.safe_input_budget
        discarded: list[str] = []
        breakdown: dict[str, int] = {}
        used = 0

        # Separate buckets to maintain priority and eventual order
        system_bucket: list[dict[str, str]] = []
        incoming_bucket: list[dict[str, str]] = []
        file_bucket: list[dict[str, str]] = []
        history_bucket: list[dict[str, str]] = []

        def add(bucket: list[dict[str, str]], label: str, role: str, content: str, required: bool = False) -> bool:
            nonlocal used
            msg = {"role": role, "content": content}
            cost = self._message_tokens(msg)
            
            if not required and used + cost > budget:
                discarded.append(label)
                return False
            
            if required and used + cost > budget:
                # Compress the context instead of raising an error
                compressed_content = self.compress_context(content)
                msg = {"role": role, "content": compressed_content}
                cost = self._message_tokens(msg)
                if used + cost <= budget:
                    bucket.append(msg)
                    used += cost
                    breakdown[label] = breakdown.get(label, 0) + cost
                    return True
                else:
                    discarded.append(label)
                    return False
            
            bucket.append(msg)
            used += cost
            breakdown[label] = breakdown.get(label, 0) + cost
            return True

        # 1. System Prompt (Highest Priority)
        system_text = (
            "You are a software engineering assistant.\n"
            "Always try best to use OpenAI JSON tool call format.\n"
            "Never apply files, use either edit or wait user to edit and/o apply.\n"
            "When information is missing, call tools to read file to get detail needed.\n\n"
            f"{(memory or HierarchicalMemory(project.memory_summary, '')).render()}"
        )
        add(system_bucket, "system", "system", system_text, required=True)

        # 2. Incoming Messages (High Priority - Current Turn)
        incoming = [
            {"role": msg.role, "content": message_content_to_text(msg.content)}
            for msg in incoming_messages
            if msg.role in {"system", "user", "assistant", "tool"}
        ]
        
        # Identify the last user message to ensure it's required
        latest_user_index = None
        for i in range(len(incoming) - 1, -1, -1):
            if incoming[i]["role"] == "user":
                latest_user_index = i
                break

        for idx, msg in enumerate(incoming):
            # All incoming messages are treated as required to avoid losing the current turn's context
            is_required = (latest_user_index is not None and idx == latest_user_index) or True 
            add(incoming_bucket, "incoming", msg["role"], msg["content"], required=is_required)

        # Fallback if no user message was provided
        if not any(m["role"] == "user" for m in incoming):
            add(incoming_bucket, "incoming", "user", "", required=True)

        # 3. Explicit Files (Medium Priority)
        file_context_parts: list[str] = []
        for rel in context_files:
            try:
                text, truncated = self.workspace.read_text(rel)
                marker = " truncated" if truncated else ""
                file_context_parts.append(f"--- FILE: {rel}{marker} ---\n{text}")
            except WorkspaceError as exc:
                discarded.append(f"file:{rel}:{exc}")
        
        if file_context_parts:
            add(file_bucket, "explicit_files", "system", "\n\n".join(file_context_parts))

        # 4. Recent History (Lowest Priority - fills remaining budget)
        history_candidates = [
            {"role": msg.role, "content": msg.content}
            for msg in stored_messages
            if msg.role in {"user", "assistant", "system", "tool"}
        ]
        
        # Add history from newest to oldest
        for msg in reversed(history_candidates):
            if not add(history_bucket, "recent_history", msg["role"], msg["content"]):
                break
        
        # Assemble in chronological order: System -> History (oldest first) -> Files -> Incoming
        result = system_bucket + list(reversed(history_bucket)) + file_bucket + incoming_bucket
        
        # Final token recalculation for the combined result
        final_used = sum(self._message_tokens(msg) for msg in result)

        return ContextBuildResult(
            messages=result,
            estimated_tokens=final_used,
            discarded=discarded,
            token_breakdown=breakdown,
        )

ContextPacker = ContextBuilder
