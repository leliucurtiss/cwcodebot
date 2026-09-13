from pathlib import Path

import pytest

from local_ai_harness.workspace import WorkspaceError, WorkspaceReader


def test_workspace_reads_file_inside_root(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('ok')", encoding="utf-8")
    reader = WorkspaceReader(tmp_path, max_file_bytes=1000)

    text, truncated = reader.read_text("src/main.py")

    assert "print('ok')" in text
    assert truncated is False


def test_workspace_rejects_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    reader = WorkspaceReader(tmp_path, max_file_bytes=1000)

    with pytest.raises(WorkspaceError):
        reader.read_text("../outside.txt")


def test_workspace_reads_docx(tmp_path: Path) -> None:
    from docx import Document

    doc = Document()
    doc.add_paragraph("Word requirement text")
    doc.save(tmp_path / "raw.docx")
    reader = WorkspaceReader(tmp_path, max_file_bytes=10000)

    text, truncated = reader.read_text("raw.docx")

    assert "Word requirement text" in text
    assert truncated is False


def test_workspace_reads_pdf(tmp_path: Path) -> None:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 72 200 Td (PDF requirement text) Tj ET")
    page[NameObject("/Contents")] = stream
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({
            NameObject("/F1"): DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            })
        })
    })
    path = tmp_path / "raw.pdf"
    with path.open("wb") as f:
        writer.write(f)
    reader = WorkspaceReader(tmp_path, max_file_bytes=10000)

    text, truncated = reader.read_text("raw.pdf")

    assert "PDF requirement text" in text
    assert truncated is False
