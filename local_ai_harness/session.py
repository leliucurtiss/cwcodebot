from __future__ import annotations

from dataclasses import dataclass

from .storage import Conversation, Project, Storage


DEFAULT_PROJECT_ID = "default"
DEFAULT_CONVERSATION_ID = "default"


@dataclass(frozen=True)
class SessionState:
    project: Project
    conversation: Conversation


class SessionManager:
    def __init__(self, storage: Storage, default_root: str):
        self.storage = storage
        self.default_root = default_root

    def resolve(self, project_id: str | None, conversation_id: str | None) -> SessionState:
        pid = project_id or DEFAULT_PROJECT_ID
        cid = conversation_id or DEFAULT_CONVERSATION_ID
        project = self.storage.get_or_create_project(pid, self.default_root, "")
        conversation = self.storage.get_or_create_conversation(cid, project.id)
        return SessionState(project=project, conversation=conversation)

