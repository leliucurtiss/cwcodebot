# Local Code Assistant Harness — MVP Implementation Plan

## 1. Purpose

Build the smallest usable **local-first Code Assistant Harness** that sits between VS Code clients and one or more LAN-hosted MaaS endpoints.

The MVP exists to validate one core idea:

> A local Harness can own project context and persistent working memory, while the LLM inference servers remain stateless and replaceable.

The first implementation should be deliberately small and easy to delete, rewrite, or refactor.

---

## 2. MVP Scope

### 2.1 Required capabilities

The MVP MUST provide only these capabilities:

1. Accept chat/completion requests from VS Code clients through an OpenAI-compatible API.
2. Forward model requests to a configurable LAN MaaS endpoint such as LiteLLM, llama.cpp, vLLM, or another OpenAI-compatible service.
3. Read files from one configured local workspace.
4. Maintain minimal persistent project/session memory locally.
5. Rebuild a bounded prompt for every request instead of continuously appending conversation history until the MaaS 128K context window is exhausted.
6. Allow a user to start a new client/session without losing important project context.

### 2.2 Explicit non-goals

The MVP MUST NOT implement the following unless needed for basic operation:

- autonomous agent loops
- shell/tool execution
- file modification by the Harness
- vector database
- embeddings
- semantic RAG
- knowledge graph
- correlation engine
- AST dependency graph
- LSP integration
- multi-agent orchestration
- intent model
- planner model
- web search
- MCP server
- sophisticated permissions system
- custom VS Code extension
- distributed storage
- cloud services

These are future layers, not MVP requirements.

---

## 3. Target Topology

```text
                        VS Code
                  +----------------+
                  |                |
              Continue       Copilot/BYOK*
                  |                |
                  +-------+--------+
                          |
                          | OpenAI-compatible API
                          v
              +-------------------------+
              |   LocalAIHarness MVP    |
              |-------------------------|
              | API Gateway             |
              | Context Manager         |
              | Workspace Reader        |
              | Session/Memory Store    |
              +------------+------------+
                           |
                           | OpenAI-compatible API
                           v
                  +------------------+
                  | LAN MaaS Gateway |
                  | e.g. LiteLLM     |
                  +---------+--------+
                            |
                  +---------+----------+
                  |                    |
               Devstral           Other MaaS
               Code Model         Models
```

`*` Continue is the primary MVP client. GitHub Copilot/BYOK compatibility is desirable where the installed client supports an OpenAI-compatible endpoint, but it must not complicate the MVP.

The Harness should expose a standard OpenAI-compatible endpoint so the VS Code client sees it as another model provider.

---

## 4. Core Design Principle

The MaaS server is treated as **stateless inference**.

The Harness owns:

- workspace identity
- project memory
- conversation history
- context budgeting
- context reconstruction
- model routing configuration

The Harness does NOT depend on a MaaS-side session.

Each inference call should conceptually be:

```text
current request
    +
selected recent conversation
    +
project memory summary
    +
selected local files
    +
system prompt
    -> bounded request
    -> MaaS
```

The logical project history may grow indefinitely while each MaaS request stays safely below the 128K model context limit.

---

## 5. MVP Context Model

### 5.1 Keep only three context classes

Do not create a complex memory hierarchy in V0.

Use only:

```text
A. Current conversation window
B. Project memory summary
C. Explicitly selected workspace files
```

### A. Current conversation window

Store raw conversation turns locally.

For each request, include only the newest turns that fit the configured token budget.

Older turns remain stored but are not automatically sent.

### B. Project memory summary

Maintain one Markdown/text summary per project.

Example contents:

```text
Project purpose
Current architecture
Important implementation decisions
Current task
Important constraints
Known problems
Recent completed work
Next likely steps
```

The project memory is persistent across client sessions.

For MVP, the summary may be updated manually through an API or generated automatically when a threshold is reached.

### C. Explicit workspace files

The MVP should NOT attempt intelligent repository-wide retrieval.

Support explicit file inclusion only.

Examples:

```text
@file src/main.cpp
@file include/session.h
```

or a request field such as:

```json
{
  "context_files": [
    "src/main.cpp",
    "include/session.h"
  ]
}
```

If Continue already sends file context in the prompt, the Harness should preserve it and avoid duplicating files.

Future versions can add automatic repository retrieval.

---

## 6. Context Budget

Assume the MaaS model has a maximum context of 128K tokens.

The Harness MUST never intentionally fill the entire window.

Initial configuration:

```text
MODEL_CONTEXT_MAX       = 128000
SAFE_INPUT_BUDGET       = 90000
OUTPUT_RESERVE          = 16000
SAFETY_MARGIN           = 22000
```

The exact values must be configuration parameters.

### Packing priority

Highest priority first:

```text
1. system prompt
2. current user request
3. explicitly supplied/current editor context
4. project memory summary
5. recent conversation turns
6. additional explicitly requested files
```

When over budget, discard from the bottom of the priority list first.

Never truncate the current user request.

Large files should be rejected or truncated with a clear marker rather than silently overflowing the model context.

---

## 7. Minimal Session Strategy

Do not expose MaaS sessions to the client.

The Harness creates its own identifiers:

```text
project_id
conversation_id
```

Suggested behavior:

```text
VS Code request
    -> project_id identifies workspace
    -> conversation_id identifies current chat
    -> Harness loads project memory
    -> Harness loads recent conversation turns
    -> Harness builds fresh MaaS request
```

Starting a new conversation creates a new `conversation_id` but reuses the same `project_id` and project memory.

Therefore:

```text
new chat != lost project context
```

This is the main MVP feature.

---

## 8. Storage

Use SQLite only.

No external database.

Suggested schema:

```sql
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    root_path TEXT NOT NULL,
    memory_summary TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(id)
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
);
```

Do not normalize further in MVP.

---

## 9. Minimal Code Structure

Keep the implementation small.

Suggested layout:

```text
local-ai-harness/
|
|-- app.py
|-- config.py
|-- gateway.py
|-- context.py
|-- memory.py
|-- workspace.py
|-- storage.py
|-- models.py
|-- requirements.txt
|-- config.yaml
|-- README.md
`-- data/
    `-- harness.db
```

Approximate responsibilities:

### `app.py`

FastAPI application and HTTP routes.

### `gateway.py`

Calls the LAN MaaS OpenAI-compatible endpoint.

No model-specific logic unless absolutely required.

### `context.py`

Builds the bounded message list sent to MaaS.

This is the most important MVP module.

### `memory.py`

Reads/updates project memory summary.

### `workspace.py`

Safely reads files below the configured project root.

Must reject path traversal outside the workspace.

### `storage.py`

SQLite access.

### `models.py`

Pydantic request/response structures.

---

## 10. External API

### 10.1 OpenAI-compatible API

Minimum required routes:

```text
GET  /v1/models
POST /v1/chat/completions
```

Streaming support is preferred because Continue and coding clients work much better with streaming output.

If implementing streaming significantly increases the first-pass complexity, implement non-streaming first, then add streaming as the next task.

### 10.2 Harness-native API

Add only the minimum management endpoints:

```text
POST /harness/projects
GET  /harness/projects/{project_id}
PUT  /harness/projects/{project_id}/memory
POST /harness/conversations
GET  /harness/conversations/{conversation_id}
```

Optional debugging endpoint:

```text
POST /harness/debug/context
```

It should return the context that WOULD be sent to MaaS, token counts, and discarded sections, without invoking the model.

This endpoint will be extremely useful during development.

---

## 11. Request Metadata

OpenAI Chat Completions does not naturally contain `project_id` and `conversation_id` semantics.

For MVP, support them through optional HTTP headers:

```text
X-Harness-Project-ID
X-Harness-Conversation-ID
```

If absent:

- use one configured default project for development
- create or use a default conversation

This keeps Continue compatibility simple.

Later, a custom VS Code extension can use the native Harness API.

---

## 12. MaaS Configuration

Example `config.yaml`:

```yaml
server:
  host: 127.0.0.1
  port: 8000

maas:
  base_url: http://10.0.0.206:4000/v1
  api_key: local
  default_model: devstral
  timeout_seconds: 300

context:
  model_context_max: 128000
  safe_input_budget: 90000
  output_reserve: 16000

workspace:
  root: /home/user/projects/demo

storage:
  sqlite_path: ./data/harness.db
```

All hard-coded network addresses must be avoided.

---

## 13. Token Counting

Exact model tokenization is useful but not required to prove the MVP.

Implementation order:

### First pass

Use a conservative estimate:

```text
estimated_tokens = characters / 3
```

Use a configurable safety multiplier.

### Second pass

Add the tokenizer matching the deployed model if convenient.

The MVP requirement is preventing context exhaustion, not perfect token accounting.

---

## 14. Minimal Memory Compaction

Do NOT build a sophisticated memory subsystem yet.

Use this simple rule:

```text
if stored conversation becomes larger than threshold:
    summarize older messages
    append/update project memory summary
    retain the newest N turns raw
```

Example threshold:

```text
COMPACTION_TRIGGER = 60000 estimated tokens
RECENT_HISTORY_KEEP = 20000 estimated tokens
```

The summarization request can use the same MaaS initially.

Later it can be routed to a smaller Mistral intent/memory model.

Important rule:

> Raw history remains in SQLite even after compaction.

Compaction changes the active working context, not the historical record.

---

## 15. Initial System Prompt

Keep the Harness system prompt small.

Example:

```text
You are a software engineering assistant working on the local project described below.

Use the supplied project memory and file context as working context.
Do not assume files or APIs that are not present in the supplied context.
When information is missing, state what additional file or information is needed.

PROJECT MEMORY:
{project_memory}
```

Do not put repository-wide documentation into the system prompt.

---

## 16. Continue Integration

The first real client should be Continue.

Configure Continue to point at:

```text
http://127.0.0.1:8000/v1
```

The Harness then forwards to the configured LAN MaaS endpoint.

Expected path:

```text
Continue
   -> localhost:8000/v1/chat/completions
   -> LocalAIHarness
   -> LAN LiteLLM/MaaS
   -> Devstral
```

The MVP is successful if Continue can use the Harness exactly as if it were an OpenAI-compatible model server.

---

## 17. Copilot Compatibility

Do not make GitHub Copilot integration a blocker for V0.

Current Copilot products increasingly support BYOK/OpenAI-compatible providers in some clients/configurations, but behavior depends on the specific Copilot client and account/environment.

Architecturally, the Harness should already expose the required OpenAI-compatible endpoint.

Therefore the implementation rule is:

```text
Build standard API compatibility once.
Test Continue first.
Test Copilot/BYOK second without changing Harness architecture.
```

If Copilot requires client-specific behavior, implement it only after the core Harness works.

---

## 18. Security Boundaries

Even for MVP:

1. Workspace reader must only access files under the configured project root.
2. Resolve paths before reading.
3. Reject `../` traversal and symlink escapes outside the workspace.
4. Do not expose arbitrary filesystem read APIs.
5. Do not implement shell execution in V0.
6. Do not allow MaaS output to directly perform local actions.

This keeps the first version low-risk and easy to understand.

---

## 19. Logging

Keep development logging useful but simple.

For every request log:

```text
request_id
project_id
conversation_id
model
estimated input tokens
project-memory tokens
history tokens
file-context tokens
number of discarded messages/files
MaaS latency
response size
```

Do not log API keys.

A debug mode may log the assembled prompt to a local file.

---

## 20. MVP Request Flow

```text
1. Receive /v1/chat/completions

2. Resolve:
      project_id
      conversation_id

3. Save newest user message locally.

4. Load:
      project memory
      recent conversation
      explicitly available file context

5. Context Manager packs everything under SAFE_INPUT_BUDGET.

6. Send a fresh stateless request to LAN MaaS.

7. Stream/return response to VS Code.

8. Save assistant response locally.

9. If history exceeds compaction threshold:
      create/update project summary asynchronously only if execution model permits;
      otherwise compact synchronously after the response or on the next request.
```

For the first implementation, prefer deterministic synchronous behavior over background infrastructure.

---

## 21. Context Packing Pseudocode

```python
def build_context(project, conversation, incoming_messages, file_context):
    budget = config.safe_input_budget

    sections = []

    sections.append(system_prompt(project.memory_summary))
    sections.append(latest_user_request(incoming_messages))

    add_if_fits(sections, current_editor_context(incoming_messages), budget)
    add_if_fits(sections, project.memory_summary, budget)

    for message in newest_to_oldest(conversation.messages):
        if not add_if_fits(sections, message, budget):
            break

    for file in explicitly_requested_files(file_context):
        add_if_fits_or_truncate(sections, file, budget)

    return restore_required_message_order(sections)
```

The production implementation can improve this later.

Do not optimize prematurely.

---

## 22. Acceptance Tests

The MVP is complete when all of the following pass.

### Test 1 — Basic proxy

From `curl`:

```text
POST /v1/chat/completions
```

returns a valid Devstral response through the Harness.

### Test 2 — Continue

Continue can connect to the Harness and complete a normal coding conversation.

### Test 3 — Local file context

A selected workspace file can be supplied to the model and the response correctly references its contents.

### Test 4 — Persistence

1. Start conversation A.
2. Save/update project memory.
3. Stop VS Code or start conversation B.
4. Ask about the project.
5. The Harness supplies the previously stored project memory.

### Test 5 — Context overflow prevention

Generate/store enough conversation history to exceed the configured safe budget.

The Harness must:

- continue responding
- drop/compact older active context
- not send more than the configured safe input budget
- retain raw history in SQLite

### Test 6 — MaaS restart independence

Restart the MaaS server.

The Harness must still retain project memory and conversation history.

### Test 7 — Harness restart persistence

Restart the Harness.

Project memory and stored conversations must remain available.

### Test 8 — Workspace boundary

Attempt to read:

```text
../../etc/passwd
```

The Harness must reject it.

---

## 23. Suggested Implementation Order for Codex

Implement in this exact order.

### Phase 1 — Proxy

- FastAPI server
- `/v1/models`
- `/v1/chat/completions`
- forward to MaaS
- basic config

Do not implement persistence yet.

### Phase 2 — Persistence

- SQLite
- projects
- conversations
- messages
- save every request/response

### Phase 3 — Context Manager

- safe token estimation
- recent-history selection
- project memory injection
- context budget enforcement
- `/harness/debug/context`

### Phase 4 — Workspace Reader

- configured root
- safe file reads
- optional explicit file context

### Phase 5 — Compaction

- compaction threshold
- summary request
- update project memory
- preserve raw messages

### Phase 6 — Continue validation

- configure Continue
- test streaming if needed
- fix OpenAI compatibility issues only

### Phase 7 — Copilot/BYOK validation

- test supported client configuration
- add compatibility fixes only if they do not distort the core API design

---

## 24. Hard Rules for the First Implementation

Codex should follow these constraints:

1. Prefer readable code over abstractions.
2. No framework beyond FastAPI, HTTP client, Pydantic, SQLite, YAML/config support, and optional tokenizer.
3. No LangChain.
4. No LlamaIndex.
5. No vector database.
6. No Redis.
7. No message queue.
8. No Docker requirement for initial development.
9. No microservices.
10. No plugin architecture.
11. No generic agent framework.
12. No speculative future abstractions.
13. Keep modules small enough to rewrite.
14. Every MaaS request must be reconstructable from local state.
15. MaaS must remain replaceable through configuration.

Target size:

```text
roughly 1,000-2,500 lines of Python for the useful MVP
```

Smaller is better if all acceptance tests pass.

---

## 25. Definition of Success

The MVP is successful if this statement becomes true:

> I can work on the same local code project for days or weeks through Continue, start new chats whenever necessary, restart the MaaS or Harness, and still retain useful project context without ever depending on a single continuously growing 128K model session.

Nothing beyond that is required for V0.

---

## 26. What Comes After MVP

Only after the MVP works, evaluate the next layer.

Likely sequence:

```text
V0  Context/session Harness
 |
V1  Repository indexing and deterministic retrieval
 |
V2  Intent capture model
 |
V3  Local correlation engine
 |
V4  Structured project knowledge
 |
V5  Tool execution / controlled agent loop
 |
V6  General Local AI Harness beyond coding
```

This preserves the larger architecture direction while keeping the first implementation disposable.

---

## 27. Relationship to the Larger Local AI Architecture

The Code Assistant should be treated as the first narrow application of a more general Local AI Harness.

Future general architecture:

```text
                  Local Knowledge / Documents
                           |
                           v
                     Correlation Layer
                           |
                 +---------+---------+
                 |                   |
                 v                   v
          Intent Capture         Retrieval
                 |                   |
                 +---------+---------+
                           |
                           v
                    Local AI Harness
                           |
                           v
                     Stateless MaaS
```

The MVP intentionally implements only the thin vertical slice needed for coding:

```text
VS Code
   -> LocalAIHarness
       -> Local project context/memory
       -> Stateless Devstral MaaS
```

The future correlation/knowledge/intent layers should therefore be added behind stable Harness interfaces rather than baked into the first implementation.

---

## 28. Codex Handoff Prompt

The following can be given directly to Codex together with this document:

```text
Implement the MVP described in this specification.

Prioritize a working, minimal, readable implementation over architectural completeness.
Do not add features listed under non-goals.
Do not introduce LangChain, LlamaIndex, a vector database, Redis, message queues,
or an agent framework.

Implement phases in the specified order and keep the application runnable after each phase.

Use Python, FastAPI, SQLite, and an async HTTP client.
Expose an OpenAI-compatible `/v1/chat/completions` endpoint and proxy requests to the configured OpenAI-compatible LAN MaaS endpoint.

The central requirement is that the Harness—not the MaaS—owns conversation state, project memory, and context budgeting.

Add tests for the acceptance criteria, especially context overflow prevention, persistence across restarts, and workspace path isolation.

Before implementing optional functionality, prefer leaving a small TODO rather than creating speculative abstractions.
```

---

## 29. Implementation Note

The initial MVP implementation now includes explicit modules for:

- Session Manager
- Hierarchical Memory
- Context Packer
- read-only `./raw` original-file folder
- PDF and DOCX explicit context extraction

The MaaS endpoint remains stateless; the harness reconstructs each request from local SQLite state and explicitly selected files.
