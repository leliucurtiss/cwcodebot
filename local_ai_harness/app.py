from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from .config import AppConfig, load_config
from .context import ContextPacker, message_content_to_text
from .gateway import MaasGateway
from .memory import MemoryManager
from .models import (
    ChatCompletionRequest,
    ConversationCreateRequest,
    ConversationMemoryUpdate,
    DebugContextRequest,
    ProjectCreateRequest,
    ProjectMemoryUpdate,
)
from .session import SessionManager
from .storage import Storage
from .workspace import WorkspaceReader

# Example function for printing messages
def print_messages(messages, label):
    total_char_size = 0
    for i, message in enumerate(messages):
        # Handle the case where content is a list or string
        if isinstance(message.content, str):
            message_size = len(message.content)
        elif isinstance(message.content, list):
            message_size = sum(len(str(item)) for item in message.content)
        else:
            message_size = 0

        total_char_size += message_size

        # Print message with newline handling
        print(f"{label} {i+1}: (Size: {message_size} chars)\n{message}\n")

    # Print total character size
    print(f"Total {label}s: {len(messages)}")
    print(f"Total character size: {total_char_size} chars\n")

def create_app(config_path: str | None = None, config: AppConfig | None = None, gateway: MaasGateway | None = None) -> FastAPI:
    cfg = config or load_config(config_path)
    storage = Storage(cfg.storage.sqlite_path)
    workspace = WorkspaceReader(cfg.workspace.root, cfg.context.max_file_bytes)
    context_builder = ContextPacker(cfg.context, workspace)
    session_manager = SessionManager(storage, str(cfg.workspace.root))
    memory_manager = MemoryManager(storage)
    maas = gateway or MaasGateway(cfg.maas)

    app = FastAPI(title="Local Code Assistant Harness", version="0.1.0")
    app.state.config = cfg
    app.state.storage = storage
    app.state.workspace = workspace
    app.state.context_builder = context_builder
    app.state.session_manager = session_manager
    app.state.memory_manager = memory_manager
    app.state.gateway = maas

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "workspace_root": str(cfg.workspace.root),
            "model": cfg.maas.default_model,
            "safe_input_budget": cfg.context.safe_input_budget,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return await maas.list_models()

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        x_harness_project_id: str | None = Header(default=None),
        x_harness_conversation_id: str | None = Header(default=None),
    ) -> Any:
        # print(f" !!!!! Model: {req.model or cfg.maas.default_model}")

        # Print the messages and their length
        print_messages(req.messages,"request")

        session = session_manager.resolve(x_harness_project_id, x_harness_conversation_id)
        memory = memory_manager.load(session.project, session.conversation)
        stored = storage.list_messages(session.conversation.id)
        context_files = req.context_files or []

        try:
            built = context_builder.build(session.project, memory, stored, req.messages, context_files)
        except ValueError as e:
            print(f"Context building error: {e}")
            raise

        for msg in req.messages:
            if msg.role in {"user", "assistant"}:
                storage.add_message(session.conversation.id, msg.role, message_content_to_text(msg.content))

        payload = req.model_dump(exclude_none=True)
        payload["model"] = req.model or cfg.maas.default_model
        payload["messages"] = built.messages
        payload.pop("context_files", None)

        if req.stream:
            async def stream() -> Any:
                first_chunk = True
                accumulated_data = ""
                async for chunk in maas.stream_chat_completion(payload):
                    # Decode the bytes and strip any "data: " prefix
                    chunk_str = chunk.decode('utf-8').strip()
                    if chunk_str.startswith("data: "):
                        chunk_str = chunk_str[6:]  # Remove "data: " prefix

                    accumulated_data += chunk_str

                    # Check if we have a complete JSON object
                    try:
                        chunk_json = json.loads(accumulated_data)
                        if first_chunk:
                            print(f"Streaming chunk: {chunk_json}")  # Print the full message once
                            first_chunk = False
                        else:
                            print(f"{chunk_json.get('choices')[0].get('delta', {}).get('content', '')}",end="")  
                            # Print only the content, without the \n

                        # Reset accumulated data after successful parse
                        accumulated_data = ""
                    except json.JSONDecodeError:
                        # If we have an incomplete JSON, continue accumulating
                        pass

                    yield chunk
                storage.add_message(session.conversation.id, "assistant", "[streamed response not captured by MVP]")

            return StreamingResponse(stream(), media_type="text/event-stream")

        response = await maas.chat_completion(payload)
        print(f"Response: {response}")  # Print the non-streaming response
        assistant_text = _extract_assistant_text(response)
        if assistant_text:
            storage.add_message(session.conversation.id, "assistant", assistant_text)
        return response

    @app.post("/harness/projects")
    async def create_project(req: ProjectCreateRequest) -> dict[str, Any]:
        project_id = req.project_id or f"proj-{uuid.uuid4().hex[:12]}"
        root_path = str(Path(req.root_path).expanduser().resolve()) if req.root_path else str(cfg.workspace.root)
        project = storage.get_or_create_project(project_id, root_path, req.memory_summary)
        return project.__dict__

    @app.get("/harness/projects/{project_id}")
    async def get_project(project_id: str) -> dict[str, Any]:
        project = storage.get_project(project_id)
        if not project:
            raise HTTPException(404, "Project not found")
        return project.__dict__

    @app.put("/harness/projects/{project_id}/memory")
    async def update_project_memory(project_id: str, req: ProjectMemoryUpdate) -> dict[str, Any]:
        if not storage.get_project(project_id):
            raise HTTPException(404, "Project not found")
        project = memory_manager.update_project_memory(project_id, req.memory_summary)
        return project.__dict__

    @app.post("/harness/conversations")
    async def create_conversation(req: ConversationCreateRequest) -> dict[str, Any]:
        if not storage.get_project(req.project_id):
            raise HTTPException(404, "Project not found")
        conv = storage.get_or_create_conversation(req.conversation_id, req.project_id)
        return conv.__dict__

    @app.get("/harness/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str) -> dict[str, Any]:
        conv = storage.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(404, "Conversation not found")
        messages = [msg.__dict__ for msg in storage.list_messages(conversation_id, limit=500)]
        return {"conversation": conv.__dict__, "messages": messages}

    @app.put("/harness/conversations/{conversation_id}/memory")
    async def update_conversation_memory(conversation_id: str, req: ConversationMemoryUpdate) -> dict[str, Any]:
        if not storage.get_conversation(conversation_id):
            raise HTTPException(404, "Conversation not found")
        conv = memory_manager.update_conversation_memory(conversation_id, req.memory_summary)
        return conv.__dict__

    @app.post("/harness/debug/context")
    async def debug_context(req: DebugContextRequest) -> dict[str, Any]:
        session = session_manager.resolve(req.project_id, req.conversation_id)
        memory = memory_manager.load(session.project, session.conversation)
        stored = storage.list_messages(session.conversation.id)
        built = context_builder.build(session.project, memory, stored, req.messages, req.context_files)
        return {
            "project_id": session.project.id,
            "conversation_id": session.conversation.id,
            "messages": built.messages,
            "estimated_tokens": built.estimated_tokens,
            "discarded": built.discarded,
            "token_breakdown": built.token_breakdown,
        }

    return app

def _extract_assistant_text(response: dict[str, Any]) -> str:
    try:
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return ""