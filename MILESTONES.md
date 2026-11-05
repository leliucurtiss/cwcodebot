# Local Harness Milestones

This file is the implementation dashboard for the Local Harness. Update it when a milestone changes status or its acceptance criteria change.

Last updated: 2026-09-11

## Status Legend

- `NOT STARTED`: no implementation work has been verified
- `IN PROGRESS`: implementation or validation is underway
- `BLOCKED`: work cannot continue without a recorded decision or dependency
- `DONE`: acceptance criteria are met and an executable check has passed

## Current Status

**Current milestone:** M0 - Project foundations

**Overall status:** IN PROGRESS

**Next engineering action:** Bootstrap the Python project, configuration, logging, and existing CW LLM interface.

## Milestone Table

| ID | Milestone | Status | Exit evidence |
| --- | --- | --- | --- |
| M0 | Project foundations | IN PROGRESS | Guidance, `./raw`, reference survey, Python project configuration, and a repeatable test command |
| M1 | Model gateway and schemas | NOT STARTED | Mock gateway, health checks, retries, streaming, and OpenAI-compatible request/response tests |
| M2 | SQLite FTS5 knowledge base | NOT STARTED | Schema, migrations/rebuild, chunk storage, FTS5 search, and provenance tests |
| M3 | P0 PDF and Word ingestion | NOT STARTED | PDF/DOCX conversion, legacy `.doc` fallback, OCR capability checks, idempotency, and fixture tests |
| M4 | Indexing CLI and diagnostics | NOT STARTED | Scan, status, retry, rebuild, and health commands with visible failures |
| M5 | Retrieval-augmented chat | NOT STARTED | Retrieved context, citations, prompt construction, and mock-LLM integration tests |
| M6 | Devstral intent routing | NOT STARTED | Structured intent schema, confidence handling, clarification questions, and routing tests |
| M7 | Codestral code assistance | NOT STARTED | Coding request path, repository context, response streaming, and model routing tests |
| M8 | Safe tools and bounded ReAct | NOT STARTED | Read-only tools, approvals, step/time limits, cancellation, loop detection, and safety tests |
| M9 | Public API compatibility | NOT STARTED | OpenAI-compatible streaming endpoint and Continue-compatible request/response fixtures |
| M10 | Optional web interface | NOT STARTED | Thin client for chat, sources, tool activity, cancellation, and model status |

## M0 - Project Foundations

Status: `IN PROGRESS`

Completed:

- Project guidance in `AGENTS.md` and `.github/copilot-instructions.md`
- Immutable `./raw` input directory
- Open-source reference survey
- P0 PDF and Word ingestion requirements

Remaining:

- Create Python package structure
- Add `pyproject.toml` and dependency groups
- Add configuration loading and environment validation
- Add structured logging and request IDs
- Add a test runner and first smoke test
- Document the existing CW LLM interface

## M3 - P0 Document Ingestion Contract

This milestone must preserve the source-of-truth rule:

```text
./raw original -> derived conversion -> chunks -> SQLite FTS5
```

Required P0 capabilities:

- Text PDF extraction through `pymupdf`
- DOCX extraction through `python-docx`
- Legacy `.doc` conversion through optional LibreOffice
- Scanned or low-text PDF fallback through optional OCRmyPDF/Tesseract capability
- Derived Markdown and metadata stored outside `./raw`
- SHA-256 identity and incremental processing
- Page, paragraph, table, and derived line provenance
- Retryable and observable conversion failures
- No LLM analysis during deterministic conversion

## Updating This File

For each milestone update:

1. Change the status in the table.
2. Add or revise the exit evidence.
3. Record blockers under the milestone if needed.
4. Link or name the test command that proves completion.
5. Update `Last updated`.

Do not mark a milestone `DONE` based only on files existing. It requires a focused executable validation check.
