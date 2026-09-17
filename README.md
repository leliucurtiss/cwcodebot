# Local Code Assistant Harness MVP

Minimal local-first OpenAI-compatible harness for VS Code Continue.

Continue talks to this local server:

```text
For local: http://127.0.0.1:8000/v1
For LAN:   http://serverip:8000/v1
```

The harness rebuilds a bounded prompt for every request, stores project memory locally, and forwards each request to a stateless OpenAI-compatible Devstral MaaS endpoint.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Edit `config.yaml`:

- `maas.base_url`: your Devstral OpenAI-compatible `/v1` URL
- `maas.api_key`: preferably `${DEVSTRAL_API_KEY}`
- `maas.default_model`: your Devstral model name
- `workspace.root`: local project path to expose read-only to the harness

Then run:

```bash
export DEVSTRAL_API_KEY="..."
export DEVSTRAL_MODEL="devstral"
bash scripts/run.sh
```

## Continue Config

Use `continue/config.yaml.template` as the Continue config starting point.

Set Continue's model API base to:

```text
http://127.0.0.1:8000/v1 
http://serverip:8000/v1
```

Use any placeholder API key for Continue, for example `local`, because the harness owns the MaaS key.

## Native Harness Endpoints

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /harness/projects
GET  /harness/projects/{project_id}
PUT  /harness/projects/{project_id}/memory
POST /harness/conversations
GET  /harness/conversations/{conversation_id}
PUT  /harness/conversations/{conversation_id}/memory
POST /harness/debug/context
```

Project/session metadata can be supplied through:

```text
X-Harness-Project-ID
X-Harness-Conversation-ID
```

If headers are absent, the harness uses a default project and conversation for development.

## MVP Feature Check

This MVP includes the three core pieces:

- `SessionManager`: resolves `project_id` and `conversation_id`, using defaults when Continue/Copilot Chat do not send harness headers.
- `MemoryManager`: loads hierarchical memory for the request.
- `ContextPacker`: rebuilds a fresh bounded prompt for every stateless MaaS call.

Memory hierarchy:

```text
project memory
  -> conversation memory
      -> recent raw messages
          -> explicit files from workspace/raw
```

## Raw Source Files

Put original read-only materials in:

```text
./raw
```

The harness can include PDF and Word files as explicit context:

```json
{
  "messages": [{"role": "user", "content": "Summarize the requirements."}],
  "context_files": ["raw/requirements.pdf", "raw/spec.docx"]
}
```

The MVP supports `.pdf` and `.docx`. Legacy `.doc` is intentionally not supported; convert it to `.docx`.

## Tests

```bash
pytest
```

Without pytest installed, a basic route smoke test can be run after the server starts:

```bash
bash scripts/smoke.sh
```

The first tests focus on:

- context overflow prevention
- workspace path isolation
- OpenAI-compatible route behavior with a mocked MaaS gateway
