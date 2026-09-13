from __future__ import annotations

from dataclasses import dataclass

from .storage import Conversation, Project, Storage


@dataclass(frozen=True)
class HierarchicalMemory:
    project_summary: str
    conversation_summary: str

    def render(self) -> str:
        return (
            "PROJECT MEMORY:\n"
            f"{self.project_summary or '(empty)'}\n\n"
            "CONVERSATION MEMORY:\n"
            f"{self.conversation_summary or '(empty)'}"
        )


class MemoryManager:
    def __init__(self, storage: Storage):
        self.storage = storage

    def load(self, project: Project, conversation: Conversation) -> HierarchicalMemory:
        return HierarchicalMemory(
            project_summary=project.memory_summary,
            conversation_summary=conversation.memory_summary,
        )

    def update_project_memory(self, project_id: str, summary: str) -> Project:
        return self.storage.update_project_memory(project_id, summary)

    def update_conversation_memory(self, conversation_id: str, summary: str) -> Conversation:
        return self.storage.update_conversation_memory(conversation_id, summary)

