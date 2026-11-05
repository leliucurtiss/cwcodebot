# Open Source Reference Survey

This document records open-source projects and patterns that may inform the Local Harness. It is a living research document and should be updated as implementation decisions become clearer.

Research date: 2026-09-11

## Executive Summary

No single project matches the Local Harness goals exactly. The best approach is to reuse proven patterns and focused libraries rather than fork a complete product.

Recommended combination:

1. Continue for the OpenAI-compatible API contract and streaming behavior.
2. Aider for symbol-aware repository maps and read-only context handling.
3. Tabby for self-hosted model-service separation and Codestral serving ideas.
4. Cline for Plan/Act behavior, approvals, tool events, and safe execution policy.
5. Docling or selected LlamaIndex readers for document conversion.
6. Haystack and LlamaIndex for pipeline and retrieval-evaluation ideas.
7. Direct Python SQLite FTS5 for the initial knowledge base.

## Project Comparisons

### Continue

Repository: https://github.com/continuedev/continue

Best reference area: API compatibility and model adapters.

Relevant patterns:

- OpenAI-compatible provider abstraction
- Streaming and non-streaming chat completions
- Abort and timeout propagation
- Tool-call conversion
- Embedding and reranking interfaces
- Flexible local API base configuration
- Model capability metadata such as context length and supported operations

Recommended use:

- Treat OpenAI-compatible behavior as the contract for the Local Harness API.
- Implement `/v1/chat/completions` first.
- Add `/v1/models`, `/v1/embeddings`, and legacy `/v1/completions` only when needed.
- Mirror the protocol carefully rather than forking Continue's full codebase.

### Aider

Repository: https://github.com/Aider-AI/aider

Best reference area: repository understanding.

Aider's repository map uses Tree-sitter and related parsing tools to extract and rank:

- Classes
- Functions
- Definitions
- References
- File relationships
- Important lines of code

The result is a compact codebase map that is more useful for code assistance than blindly retrieving arbitrary text chunks.

Aider also separates editable files from read-only files, which maps well to the Local Harness `./raw` policy.

Recommended use:

- Reuse the repository-map concept.
- Use Tree-sitter or an existing Python Tree-sitter package.
- Store symbol metadata in SQLite:
  - file path
  - symbol name
  - symbol type
  - line range
  - language
  - references
- Study Aider's repository-map tests for evaluation cases.

### Tabby

Repository: https://github.com/TabbyML/tabby

Best reference area: self-hosted model-server architecture.

Tabby separates:

- Completion model
- Chat model
- Embedding model
- Code search
- Document search
- HTTP model connections
- Local `llama.cpp` model processes
- Health endpoints
- Model configuration

Tabby also has a Codestral integration, making it especially relevant to this project.

Recommended use:

- Use Tabby as an architectural reference.
- Keep model serving behind a provider interface.
- Support separate models for intent, chat, completion, and embeddings when needed.
- Expose health and model endpoints.
- Keep indexing and retrieval independent from model inference.
- Keep Python orchestration separate from a future C++ or Rust model runtime.

Do not fork Tabby unless the project later decides to adopt its Rust-based server and broader product scope.

### Cline

Repository: https://github.com/cline/cline

Best reference area: agent interaction and approval design.

Relevant patterns:

- Explicit Plan mode and Act mode
- Human approval for file edits and terminal commands
- Structured tool activity events
- Streaming lifecycle events
- Session cancellation
- Context compaction
- Provider capability declarations
- Fail-closed behavior when a provider does not support a capability
- Project-specific instruction files

Recommended use:

- Separate planning from execution.
- Implement an approval state machine:
  - proposed
  - awaiting confirmation
  - approved
  - rejected
  - executed
  - failed
- Keep read-only tools available during planning.
- Block mutations in planning mode at the execution layer, not only through prompts.
- Use structured tool events in the API and future web interface.

Do not copy Cline's full TypeScript runtime for the initial Python MVP.

### Open WebUI

Repository: https://github.com/open-webui/open-webui

Best reference area: web UI and OpenAI-compatible proxy behavior.

Relevant patterns:

- OpenAI-compatible chat, completion, embeddings, and model endpoints
- Streaming response proxying
- Multiple upstream model base URLs
- Model catalog and model aliases
- Provider-specific payload conversion
- Configurable chunk sizes and overlap
- Document ingestion settings
- Knowledge-search status events
- Authentication and model-access checks

Recommended use:

- Study it when building the optional web client.
- Reuse UX ideas for source citations, retrieval status, tool activity, model selection, and cancellation.
- Keep the web UI as a thin client of the Local Harness API.

Do not adopt its full backend or vector-oriented knowledge system for the initial MVP.

### LlamaIndex

Repository: https://github.com/run-llama/llama_index

Best reference area: document readers and ingestion concepts.

Relevant patterns:

- Pluggable document readers
- Markdown conversion through Docling
- Sentence and token splitters
- Cached ingestion transformations
- Metadata-preserving nodes
- Parallel ingestion
- Retrieval evaluation with hit rate and MRR
- Local model integration

Recommended use:

- Consider selected reader packages, especially Docling-based readers, for PDF, DOCX, HTML, and similar formats.
- Borrow the ingestion pipeline concepts:
  - source document
  - transformation
  - chunk
  - metadata
  - cache
- Avoid adopting the full framework initially because it adds substantial dependency and vector-index complexity.

### Haystack

Repository: https://github.com/deepset-ai/haystack

Best reference area: pipeline composition and RAG evaluation.

A useful pipeline shape is:

```text
retriever -> prompt builder -> LLM -> answer builder
```

Haystack also demonstrates keyword and BM25 retrieval, which is compatible with the Local Harness FTS5 direction.

Recommended use:

- Use its pipeline concepts as design inspiration.
- Study its retrieval evaluation patterns.
- Keep the first pipeline explicit and small:
  - intent classifier
  - FTS5 retriever
  - context formatter
  - Codestral response generator
  - citation builder

Do not add Haystack as a core dependency until the project has a clear need for its broader component system.

### OpenHands

Repository: https://github.com/All-Hands-AI/OpenHands

The repository lookup was unavailable during the initial survey, so no reuse or fork decision is recorded yet.

Potential future research areas:

- Sandboxed command execution
- Agent runtime boundaries
- Workspace management
- Confirmation policies
- Long-running task recovery

## Reuse and Fork Decision

### Prefer focused reuse

- Continue protocol concepts and schemas
- Aider repository-map concepts
- Tree-Sitter parsing packages
- Docling or selected document readers
- Existing OpenAI-compatible client libraries
- Existing local model servers such as `llama.cpp`, Ollama, or vLLM

### Prefer architectural reference

- Tabby server decomposition
- Cline approval and event semantics
- Open WebUI web experience
- Haystack pipeline composition
- LlamaIndex ingestion and evaluation patterns

### Do not fork initially

- Continue: broad IDE and provider surface
- Aider: different CLI and editing workflow
- Tabby: Rust architecture and larger product scope
- Cline: large TypeScript agent runtime
- Open WebUI: full application backend and authentication system
- LlamaIndex or Haystack: framework overhead for the initial FTS5 MVP

## Mapping To Local Harness

| Local Harness area | Primary reference | Initial decision |
|---|---|---|
| OpenAI-compatible API | Continue, Open WebUI | Implement directly in Python |
| Streaming protocol | Continue | Support Server-Sent Events |
| Repository map | Aider | Add Tree-sitter-based symbol metadata later |
| Read-only source policy | Aider | Treat `./raw` as immutable input |
| Model gateway | Tabby | Use configurable local HTTP endpoints |
| Codestral serving | Tabby | Keep model runtime behind an adapter |
| Intent routing | Project-specific | Route intent and planning to Devstral |
| Code assistance | Project-specific | Route coding responses to Codestral |
| Plan/Act approvals | Cline | Add before mutating tools |
| Document conversion | Docling/LlamaIndex | Convert to derived Markdown outside `./raw` |
| Retrieval | Haystack/LlamaIndex concepts | Implement SQLite FTS5 directly |
| Web interface | Open WebUI | Build later as a thin API client |

## Important Constraints

- `./raw` remains immutable source data. Uploaded files must be preserved byte-for-byte.
- Converted Markdown, SQLite databases, caches, and processing metadata must live outside `./raw`.
- No embedded vector database is required for the initial MVP.
- Read-only tools should come before file edits or shell execution.
- Model providers should remain behind a gateway with health checks, timeouts, retries, and streaming.
- Repository content is untrusted input and must not override system safety rules.
- Tool execution requires step limits, timeouts, cancellation, loop detection, and confirmation policies.

## Licensing Follow-Up

Before copying code or adding a dependency:

1. Verify the repository's current license.
2. Check licenses of transitive dependencies and document converters.
3. Record attribution requirements.
4. Check model-specific terms separately from software licenses.
5. Verify Codestral licensing before any commercial deployment. Tabby's Codestral documentation notes restrictions for commercial use.

## Next Research Tasks

1. Evaluate OpenHands for sandbox and tool-execution patterns.
2. Compare `llama.cpp`, Ollama, and vLLM for Devstral/Codestral serving.
3. Compare Docling, MarkItDown, and format-specific Python readers.
4. Confirm Continue's minimum protocol surface for chat, autocomplete, and tool use.
5. Build a small FTS5 retrieval benchmark before considering embeddings.
6. Review repository licenses before reusing implementation code.
