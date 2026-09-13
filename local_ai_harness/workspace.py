from __future__ import annotations

from pathlib import Path

from .document_reader import DocumentReadError, DocumentReader


class WorkspaceError(ValueError):
    pass


class WorkspaceReader:
    def __init__(self, root: Path, max_file_bytes: int):
        self.root = root.expanduser().resolve()
        self.max_file_bytes = max_file_bytes
        self.documents = DocumentReader()

    def resolve(self, relative_path: str) -> Path:
        if not relative_path or relative_path.strip() in {".", ""}:
            raise WorkspaceError("File path is required")
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError("Path escapes workspace root") from exc
        if not candidate.is_file():
            raise WorkspaceError("File does not exist")
        return candidate

    def read_text(self, relative_path: str) -> tuple[str, bool]:
        path = self.resolve(relative_path)
        if path.suffix.lower() in {".pdf", ".docx", ".doc"}:
            try:
                text = self.documents.read(path)
            except DocumentReadError as exc:
                raise WorkspaceError(str(exc)) from exc
            max_chars = self.max_file_bytes
            truncated = len(text.encode("utf-8", errors="replace")) > self.max_file_bytes
            if truncated:
                text = text[:max_chars]
        else:
            data = path.read_bytes()
            truncated = len(data) > self.max_file_bytes
            if truncated:
                data = data[: self.max_file_bytes]
            text = data.decode("utf-8", errors="replace")
        if truncated:
            text += "\n\n[TRUNCATED: file exceeded harness max_file_bytes]\n"
        return text, truncated
