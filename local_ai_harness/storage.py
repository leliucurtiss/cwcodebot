from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Project:
    id: str
    root_path: str
    memory_summary: str
    created_at: str
    updated_at: str


@dataclass
class Conversation:
    id: str
    project_id: str
    memory_summary: str
    created_at: str
    updated_at: str


@dataclass
class StoredMessage:
    id: int
    conversation_id: str
    role: str
    content: str
    created_at: str


class Storage:
    def __init__(self, sqlite_path: Path):
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def session(self):
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.session() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    root_path TEXT NOT NULL,
                    memory_summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    memory_summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                );
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "memory_summary" not in columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN memory_summary TEXT NOT NULL DEFAULT ''")

    def get_or_create_project(self, project_id: str, root_path: str, memory_summary: str = "") -> Project:
        with self.session() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            if row:
                return Project(**dict(row))
            now = utc_now()
            conn.execute(
                "INSERT INTO projects (id, root_path, memory_summary, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (project_id, root_path, memory_summary, now, now),
            )
            return Project(project_id, root_path, memory_summary, now, now)

    def get_project(self, project_id: str) -> Project | None:
        with self.session() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            return Project(**dict(row)) if row else None

    def update_project_memory(self, project_id: str, memory_summary: str) -> Project:
        now = utc_now()
        with self.session() as conn:
            conn.execute(
                "UPDATE projects SET memory_summary = ?, updated_at = ? WHERE id = ?",
                (memory_summary, now, project_id),
            )
        project = self.get_project(project_id)
        if not project:
            raise KeyError(project_id)
        return project

    def get_or_create_conversation(self, conversation_id: str | None, project_id: str) -> Conversation:
        cid = conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
        with self.session() as conn:
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
            if row:
                return Conversation(**dict(row))
            now = utc_now()
            conn.execute(
                "INSERT INTO conversations (id, project_id, memory_summary, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (cid, project_id, "", now, now),
            )
            return Conversation(cid, project_id, "", now, now)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self.session() as conn:
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            return Conversation(**dict(row)) if row else None

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        now = utc_now()
        with self.session() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )

    def update_conversation_memory(self, conversation_id: str, memory_summary: str) -> Conversation:
        now = utc_now()
        with self.session() as conn:
            conn.execute(
                "UPDATE conversations SET memory_summary = ?, updated_at = ? WHERE id = ?",
                (memory_summary, now, conversation_id),
            )
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            raise KeyError(conversation_id)
        return conversation

    def list_messages(self, conversation_id: str, limit: int = 200) -> list[StoredMessage]:
        with self.session() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
            return [StoredMessage(**dict(row)) for row in reversed(rows)]
