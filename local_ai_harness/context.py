from dataclasses import dataclass, field
import json
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

def detect_repetition_details(messages: list[dict[str, Any]], threshold: float = 0.7) -> tuple[bool, str | None]:
    """
    Circuit Breaker: Detects if the assistant is repeating itself.
    Checks for both turn-level repetition and internal message looping.
    Returns (is_repeating, reason).
    """
    assistant_msgs = [
        str(m.get("content", ""))
        for m in messages
        if m.get("role") == "assistant" and m.get("content")
    ]
    if not assistant_msgs:
        return False, None
    last_msg = assistant_msgs[-1].lower()

    # 1. Internal Repetition Detection (Intra-message loop)
    
    # A. Character-level repetition (e.g., "thoughtthoughtthought" or "aaaaaaaa")
    import re
    match = re.search(r'(.{2,})\1{4,}', last_msg)
    if match and match.group(1).strip():
        return True, f"Character-level repetition detected: '{match.group(1)}' repeats"

    # B. Word-level repetition (n-grams)
    words = last_msg.split()
    if len(words) > 10:
        n = 5  # Sequence length to track
        sequences = []
        for i in range(len(words) - n + 1):
            sequences.append(" ".join(words[i:i+n]))
        
        for seq in set(sequences):
            if sequences.count(seq) >= 3:
                return True, f"Word-level repetition detected: sequence '{seq}' repeats {sequences.count(seq)} times"

    # 2. Turn-level Repetition Detection (Inter-message loop)
    if len(assistant_msgs) < 2:
        return False, None
    
    m1 = last_msg
    m2 = assistant_msgs[-2].lower()
    
    set1 = set(m1.split())
    set2 = set(m2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    similarity = intersection / union if union > 0 else 0
    if similarity > threshold:
        return True, f"Turn-level similarity too high: {similarity:.2f} (threshold: {threshold})"
    
    return False, None


def detect_repetition(messages: list[dict[str, Any]], threshold: float = 0.7) -> bool:
    return detect_repetition_details(messages, threshold)[0]


@dataclass
class ContextBuildResult:
    messages: list[dict[str, Any]]
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
        raw = len(text) / max(self.config.token_char_ratio, 3.0)
        return int(raw * self.config.token_safety_multiplier) + 1

    def _message_tokens(self, message: dict[str, Any]) -> int:
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        return self.estimate_tokens(message.get("role", "")) + self.estimate_tokens(content)

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
        result = []

        incoming_messages_dicts: list[dict[str, Any]] = []
        for msg in incoming_messages:
            msg_dict = msg.model_dump(exclude_none=True)
            incoming_messages_dicts.append(msg_dict)
        incoming_cost = sum(self._message_tokens(message) for message in incoming_messages_dicts)

        recent_history: list[dict[str, Any]] = []
        dropped_history = 0
        remaining = max(budget - incoming_cost, 0)
        for stored_message in reversed(stored_messages):
            history_message = {
                "role": stored_message.role,
                "content": stored_message.content,
            }
            cost = self._message_tokens(history_message)
            if cost <= remaining:
                recent_history.append(history_message)
                remaining -= cost
                breakdown["recent_history"] = breakdown.get("recent_history", 0) + cost
            else:
                dropped_history += 1
        if dropped_history:
            discarded.append("recent_history")
        result.extend(reversed(recent_history))

        file_context_parts: list[str] = []
        for rel in context_files:
            try:
                text, truncated = self.workspace.read_text(rel)
                marker = " (truncated)" if truncated else ""
                file_context_parts.append(f"--- FILE: {rel}{marker} ---\n{text}")
            except WorkspaceError as exc:
                discarded.append(f"file:{rel}:{exc}")
        if file_context_parts:
            file_message = {"role": "system", "content": "\n\n".join(file_context_parts)}
            file_cost = self._message_tokens(file_message)
            if file_cost <= remaining:
                result.append(file_message)
                remaining -= file_cost
                breakdown["explicit_files"] = file_cost
            else:
                discarded.append("explicit_files")

        # Incoming messages carry the active protocol state and are never evicted.
        result.extend(incoming_messages_dicts)
        for role_message in incoming_messages_dicts:
            role = str(role_message.get("role", ""))
            cost = self._message_tokens(role_message)
            breakdown[role] = breakdown.get(role, 0) + cost

        # Final token recalculation
        final_used = sum(self._message_tokens(msg) for msg in result)
        used = final_used

        return ContextBuildResult(
            messages=result,
            estimated_tokens=used,
            discarded=discarded,
            token_breakdown=breakdown,
        )

ContextPacker = ContextBuilder
