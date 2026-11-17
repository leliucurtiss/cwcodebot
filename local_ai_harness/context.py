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
    Compares the last two assistant messages using Jaccard similarity.
    """
    assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    if len(assistant_msgs) < 2:
        return False
    
    m1 = assistant_msgs[-1].lower()
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

        # 1. System Prompt (Highest Priority)
        system_text = (
            "You are a software engineering assistant.\n"
            "Always try best to use OpenAI JSON tool call format.\n"
            "CRITICAL FORMATTING RULE: When calling edit_existing_file or "
            "create_new_file, you MUST provide the raw code only. "
            "NEVER wrap the content in Markdown code blocks (e.g., no ```python ... ```). "
            "CRITICAL: Do not repeat yourself. Avoid echoing the user's phrasing. "
            "Do not repeat information that has already been provided in the conversation history. "
            "If you find yourself repeating a pattern, break it and provide the information in a new way."
            "NEVER include language identifiers or introductory text (e.g., \"Here is the code\") "
            "inside the tool parameters. The tool input must be exactly what should appear in the file."
            "When information is missing, call tools to read file to get detail needed.\n\n"
            f"{(memory or HierarchicalMemory(project.memory_summary, '')).render()}"
        )
        
        # We'll use a list for the final result
        result: list[dict[str, str]] = [{"role": "system", "content": system_text}]
        used += self._message_tokens(result[0])
        breakdown["system"] = self._message_tokens(result[0])

        # 2. Merge and Deduplicate History
        # Combine stored and incoming, keeping chronological order
        all_msgs = []
        for msg in stored_messages:
            if msg.role != "system":
                all_msgs.append({"role": msg.role, "content": msg.content})
        
        for msg in incoming_messages:
            if msg.role != "system":
                all_msgs.append({"role": msg.role, "content": message_content_to_text(msg.content)})
        
        # Deduplicate while preserving order (keep the most recent version of a message)
        seen = set()
        deduped = []
        for msg in reversed(all_msgs):
            key = (msg["role"], msg["content"])
            if key not in seen:
                deduped.append(msg)
                seen.add(key)
        full_history = list(reversed(deduped))

        # 3. Identify the "Current Turn"
        # The current turn starts at the last user message
        last_user_idx = -1
        for i in range(len(full_history) - 1, -1, -1):
            if full_history[i]["role"] == "user":
                last_user_idx = i
                break
        
        history_part = full_history[:last_user_idx] if last_user_idx != -1 else full_history
        current_turn_part = full_history[last_user_idx:] if last_user_idx != -1 else []

        # 4. Add History (Lowest Priority - fills remaining budget)
        # We add history from the end (most recent) backwards until budget is hit
        history_to_add = []
        for msg in reversed(history_part):
            cost = self._message_tokens(msg)
            if used + cost <= budget:
                history_to_add.append(msg)
                used += cost
                breakdown["history"] = breakdown.get("history", 0) + cost
            else:
                discarded.append("history_truncated")
                break
        
        result.extend(reversed(history_to_add))

        # 5. Explicit Files (Medium Priority - placed before the current turn)
        file_context_parts: list[str] = []
        for rel in context_files:
            try:
                text, truncated = self.workspace.read_text(rel)
                marker = " truncated" if truncated else ""
                file_context_parts.append(f"--- FILE: {rel}{marker} ---\n{text}")
            except WorkspaceError as exc:
                discarded.append(f"file:{rel}:{exc}")
        
        if file_context_parts:
            file_msg = {"role": "system", "content": "\n\n".join(file_context_parts)}
            cost = self._message_tokens(file_msg)
            if used + cost <= budget:
                result.append(file_msg)
                used += cost
                breakdown["explicit_files"] = cost
            else:
                discarded.append("explicit_files")

        # 6. Current Turn (High Priority - always include if possible)
        for msg in current_turn_part:
            cost = self._message_tokens(msg)
            # We try to include the current turn even if it exceeds budget slightly, 
            # but we still track it.
            result.append(msg)
            used += cost
            breakdown["current_turn"] = breakdown.get("current_turn", 0) + cost

        # Final token recalculation
        final_used = sum(self._message_tokens(msg) for msg in result)

        return ContextBuildResult(
            messages=result,
            estimated_tokens=final_used,
            discarded=discarded,
            token_breakdown=breakdown,
        )

ContextPacker = ContextBuilder
