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

def detect_repetition(messages: list[dict[str, str]], threshold: float = 0.7) -> bool:
    """
    Circuit Breaker: Detects if the assistant is repeating itself.
    Checks for both turn-level repetition and internal message looping.
    """
    assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    if not assistant_msgs:
        return False
    
    last_msg = assistant_msgs[-1].lower()

    # 1. Internal Repetition Detection (Intra-message loop)
    
    # A. Character-level repetition (e.g., "thoughtthoughtthought" or "aaaaaaaa")
    import re
    # Matches any sequence of 3+ characters that repeats 3 or more times consecutively
    if re.search(r'(.{2,})\1{2,}', last_msg):
        return True

    # B. Word-level repetition (n-grams)
    words = last_msg.split()
    if len(words) > 10:
        n = 5  # Sequence length to track
        sequences = []
        for i in range(len(words) - n + 1):
            sequences.append(" ".join(words[i:i+n]))
        
        for seq in set(sequences):
            if sequences.count(seq) >= 3:
                return True

    # 2. Turn-level Repetition Detection (Inter-message loop)
    if len(assistant_msgs) < 2:
        return False
    
    m1 = last_msg
    m2 = assistant_msgs[-2].lower()
    
    set1 = set(m1.split())
    set2 = set(m2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    similarity = intersection / union if union > 0 else 0
    return similarity > threshold

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
        raw = len(text) / max(self.config.token_char_ratio, 3.0)
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
        result = []

        # Transparently pass through all incoming messages (System, User, Assistant)
        for msg in incoming_messages:
            content = message_content_to_text(msg.content)
            msg_dict = {"role": msg.role, "content": content}
            cost = self._message_tokens(msg_dict)
            result.append(msg_dict)
            used += cost
            breakdown[msg.role] = breakdown.get(msg.role, 0) + cost

        # 5. Explicit Files (Commented out for transparency/future RAG)
        # file_context_parts: list[str] = []
        # for rel in context_files:
        #     try:
        #         text, truncated = self.workspace.read_text(rel)
        #         marker = " truncated" if truncated else ""
        #         file_context_parts.append(f"--- FILE: {rel}{marker} ---\n{text}")
        #     except WorkspaceError as exc:
        #         discarded.append(f"file:{rel}:{exc}")
        # 
        # if file_context_parts:
        #     file_msg = {"role": "system", "content": "\n\n".join(file_context_parts)}
        #     cost = self._message_tokens(file_msg)
        #     if used + cost <= budget:
        #         result.append(file_msg)
        #         used += cost
        #         breakdown["explicit_files"] = cost
        #     else:
        #         discarded.append("explicit_files")

        # SHould manage budget but comment out for a while 
        # Debug check  
        if used > budget:
            print(f"Message {used} is beyond budget {budget}")
            # discarded.append("budget_exceeded")

        # Final token recalculation
        final_used = sum(self._message_tokens(msg) for msg in result)
        # print(f"INFO: Incoming message TOKENS {final_used} token")

        return ContextBuildResult(
            messages=result,
            estimated_tokens=used,
            discarded=discarded,
            token_breakdown=breakdown,
        )

ContextPacker = ContextBuilder
