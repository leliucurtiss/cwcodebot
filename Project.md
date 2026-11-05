# Local Harness Project Guidance

## Project Goal

Build a Python-based local coding harness that uses Devstral for intent capture and planning, and Codestral for code assistance. The harness provides local RAG, a knowledge base, bounded ReAct tooling, and an API usable by Continue and other OpenAI-compatible IDE clients.

## MVP Scope

- Use `./raw` as the immutable upload area; preserve every uploaded file byte-for-byte.
- Local workspace indexing and retrieval
- P0 PDF and Word document conversion to searchable Markdown where useful
- SQLite with FTS5 for text retrieval and metadata
- Incremental indexing with hashes, ignore rules, deletion detection, and provenance
- Devstral intent classification with structured output and clarification questions
- Codestral chat and code assistance
- Read-only tools first: search, list files, read files, inspect symbols, and run tests when explicitly approved
- OpenAI-compatible streaming API, especially chat-completions
- CLI for indexing, searching, health checks, and diagnostics

## Architecture Rules

- Keep model access behind a model gateway with health checks, timeouts, retries, streaming, and configurable local endpoints.
- Route intent and planning to Devstral; route coding and implementation responses to Codestral.
- Store document IDs, chunks, source paths, hashes, timestamps, language, and line ranges so responses can cite sources.
- Treat `./raw` as read-only input and generated Markdown as a derived index representation; never modify, rename, or delete originals.
- Store converted documents, SQLite databases, and processing metadata outside `./raw`.
- Use established document libraries instead of implementing PDF or Word parsers from scratch.
- Keep conversation state separate from the indexed knowledge base.
- Keep the web interface as a later thin client of the API, not a second backend.
- Defer embeddings and vector databases until FTS5 retrieval quality has been measured.

## Safety and Operations

- Restrict filesystem access to approved workspace roots.
- Require confirmation for file edits, shell commands, destructive actions, and broad changes.
- Bound ReAct loops with step limits, timeouts, cancellation, and loop detection.
- Defend against prompt injection in repository content.
- Redact secrets from prompts and logs where possible.
- Provide structured logs, request IDs, health/readiness endpoints, configuration, and database rebuild commands.

## Quality Bar

Add focused unit tests for PDF/DOCX conversion, chunking, FTS5 retrieval, model routing, and protocol serialization. Add integration tests with exisiting CW LLM server, retrieval evaluation fixtures, and compatibility tests for streaming clients. Prefer small, testable modules and explicit schemas over hidden behavior.

## P0 Document Ingestion

- Scan `./raw` as immutable input and identify files by content hash.
- Support text-based PDF with PyMuPDF (`pymupdf`).
- Support DOCX with `python-docx`, preserving headings, paragraphs, lists, tables, and source locations.
- Support legacy `.doc` through LibreOffice headless conversion when available; keep the converted file outside `./raw`.
- Detect low-quality or scanned PDF pages and provide an OCR fallback using OCRmyPDF/Tesseract or a later approved layout-aware converter.
- Write derived Markdown, metadata, conversion errors, and processing state outside `./raw`.
- Preserve page, paragraph, table, and derived line provenance for citations.
- Make conversion idempotent, retryable, versioned, and observable.
- Do not run LLM analysis during basic conversion; use Devstral later for optional enrichment.

### P0 Dependencies

Python dependencies:

- `pymupdf`
- `python-docx`

Optional system dependencies:

- LibreOffice for legacy `.doc`
- OCRmyPDF and Tesseract for scanned PDFs

Add these to the project dependency configuration when the Python project is initialized. Keep optional converters behind capability checks so missing system tools produce a visible, retryable status rather than a startup failure.

## Implementation Order

1. Configuration, logging, project structure, and existing CW LLM interface.
2. Model gateway and OpenAI-compatible data models.
3. SQLite schema and FTS5 search.
4. P0 PDF/DOCX conversion, file discovery, incremental indexing, and provenance.
5. CLI indexing/search commands and ingestion diagnostics.
6. Retrieval-augmented chat pipeline.
7. Devstral intent routing and Codestral assistance.
8. Safe tools and bounded ReAct orchestration.
9. Streaming API and client compatibility tests.
10. Optional web UI after the core path is reliable.
