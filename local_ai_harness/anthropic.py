from fastapi import APIRouter, Response, Request, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from .models import (
    ConversationCreateRequest,
    ConversationMemoryUpdate,
    DebugContextRequest,
    ProjectCreateRequest,
    ProjectMemoryUpdate,
    MessagesRequest,
    ChatMessage,
)

from .app_utils import prepare_chat_context
from .tools_helpers import (
    translate_openai_tool_to_anthropic,
    translate_anthropic_to_openai_tools,
)
anthropic_router = APIRouter()

@anthropic_router.get("/v1/oauth/hello")
@anthropic_router.get("/oauth/hello")
async def oauth_hello():
    """Handshake endpoint for Claude Code OAuth check."""
    now = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
    return JSONResponse(
        content={"status": "ok"},
        headers={
            "x-amzn-request-id": uuid.uuid4().hex,
            "server": "cloudflare",
            "date": now,
            "content-type": "application/json",
            "connection": "keep-alive",
            "cache-control": "no-cache, no-store, must-revalidate",
            "pragma": "no-cache",
            "expires": "0",
            "access-control-allow-origin": "*",
            "access-control-allow-methods": "GET, POST, HEAD, OPTIONS",
            "access-control-allow-headers": "*"
        }
    )

@anthropic_router.post("/api/event_logging/v2/batch")
async def event_logging_batch():
    """Telemetry endpoint to silence logging errors."""
    return {"status": "ok"}

@anthropic_router.post("/api/eval/{path:path}")
async def eval_telemetry(path: str):
    """Telemetry endpoint for SDK evaluations."""
    return {"status": "ok"}

@anthropic_router.get("/mcp-registry/v0/servers")
async def mcp_servers():
    """Mock MCP registry endpoint."""
    return {"servers": []}

@anthropic_router.get("/api/claude_code_penguin_mode")
async def penguin_mode():
    """Mock penguin mode feature flag."""
    return {"enabled": False}

@anthropic_router.get("/api/claude_cli/bootstrap")
async def cli_bootstrap():
    """Mock CLI bootstrap endpoint."""
    return {"status": "ok", "version": "latest"}

@anthropic_router.get("/api/directory/plugins")
async def plugin_directory():
    """Mock plugin directory endpoint."""
    return {"plugins": []}

@anthropic_router.get("/v1/api/hello")
@anthropic_router.get("/api/hello")
async def api_hello():
    """Handshake endpoint for Claude Code / Anthropic compatibility check."""
    now = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
    return JSONResponse(
        content={"status": "ok"},
        headers={
            "x-amzn-request-id": uuid.uuid4().hex,
            "server": "cloudflare",
            "date": now,
            "content-type": "application/json",
            "connection": "keep-alive",
            "cache-control": "no-cache, no-store, must-revalidate",
            "pragma": "no-cache",
            "expires": "0",
            "access-control-allow-origin": "*",
            "access-control-allow-methods": "GET, POST, HEAD, OPTIONS",
            "access-control-allow-headers": "*"
        }
    )

@anthropic_router.head("/v1/api/hello")
@anthropic_router.head("/api/hello")
async def api_hello_head():
    """HEAD request handler — returns only headers, no body."""
    now = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
    return Response(
        status_code=200,
        headers={
            "x-amzn-request-id": uuid.uuid4().hex,
            "server": "cloudflare",
            "date": now,
            "content-type": "application/json",
            "connection": "keep-alive",
            "cache-control": "no-cache, no-store, must-revalidate",
            "pragma": "no-cache",
            "expires": "0",
            "access-control-allow-origin": "*",
            "access-control-allow-methods": "GET, POST, HEAD, OPTIONS",
            "access-control-allow-headers": "*"
        }
    )

@anthropic_router.post("/v1/messages")
@anthropic_router.post("/messages")
@anthropic_router.post("/v1/v1/messages")
async def messages(
    request: Request,
    req: MessagesRequest,
    x_harness_project_id: str | None = Header(default=None),
    x_harness_conversation_id: str | None = Header(default=None),
) -> Any:
    """
    Anthropic Messages API wrapper for Claudecode compatibility.
    """
    # Retrieve managers from app state
    storage = request.app.state.storage
    logger = request.app.state.logger
    maas = request.app.state.gateway
    cfg = request.app.state.config
    session_manager = request.app.state.session_manager
    memory_manager = request.app.state.memory_manager
    context_builder = request.app.state.context_builder
      
    # 1. Translate Anthropic request to internal format
    logger.debug(f"This is message from Claude {req.messages}")
    internal_messages = list(req.messages)
    if req.system:
        internal_messages.insert(0, ChatMessage(role="system", content=req.system))

    # 2. Prepare context and save messages
    session, built_messages, _ = await prepare_chat_context(
        session_manager, memory_manager, storage, context_builder, logger,
        internal_messages, x_harness_project_id, x_harness_conversation_id
    )

    # 3. Build payload for MAAS (OpenAI format)
    # We use a selective reconstruction to avoid 400 Bad Request from MAAS
    # while preserving the 'tools' and other critical parameters.
    payload = {
        "model": cfg.maas.default_model,
        "messages": built_messages,
        "max_tokens": req.max_tokens or cfg.context.model_context_max,
        "temperature": req.temperature,
        "stream": req.stream,
    }
    
    # Extract tools from model_extra if present (Claude Code sends them here)
    model_extra = getattr(req, 'model_extra', {})
    if model_extra and 'tools' in model_extra:
        translated_tools = translate_anthropic_to_openai_tools(model_extra['tools'])
        if translated_tools:
            payload["tools"] = translated_tools


    if req.stream:
        async def anthropic_stream() -> Any:
            msg_id = f"msg_{uuid.uuid4().hex[:12]}"
            block_index = 0
            current_block_type = "text"
            active_tools = {} # Map OpenAI tool index -> Anthropic block_index

            # 1. message_start
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'role': 'assistant', 'model': payload['model']}})}\n\n".encode("utf-8")
            
            # 2. Initial text block
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode("utf-8")

            async for chunk in maas.stream_chat_completion(payload):
                if not chunk.startswith(b"data: "):
                    continue
                
                line = chunk.decode("utf-8").strip()
                if line == "data: [DONE]":
                    # Close all remaining blocks
                    for idx in sorted(active_tools.keys()):
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': active_tools[idx]})}\n\n".encode("utf-8")
                    if current_block_type == "text":
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n".encode("utf-8")
                    
                    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}})}\n\n".encode("utf-8")
                    yield b"event: message_stop\ndata: {}\n\n"
                    return
                
                try:
                    data = json.loads(line[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    
                    # --- Handle Text or Reasoning ---
                    reasoning = delta.get("reasoning_content")
                    content = delta.get("content")
                    
                    if reasoning or content:
                        if current_block_type != "text":
                            # Close all active tools before starting a new text block
                            for idx in sorted(active_tools.keys()):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': active_tools[idx]})}\n\n".encode("utf-8")
                            active_tools.clear()
                            
                            block_index += 1
                            current_block_type = "text"
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode("utf-8")
                        
                        text_delta = reasoning or content
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': text_delta}})}\n\n".encode("utf-8")

                    # --- Handle Native Tool Calls ---
                    tool_calls = delta.get("tool_calls")
                    if tool_calls:
                        for tc in tool_calls:
                            idx = tc.get("index")
                            if idx not in active_tools:
                                if current_block_type == "text":
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n".encode("utf-8")
                                    current_block_type = "none"
                                
                                block_index += 1
                                active_tools[idx] = block_index
                                current_block_type = "tool_use"
                                
                                name = tc.get("function", {}).get("name", "unknown")
                                tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:12]}")
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tc_id, 'name': name, 'input': {}}})}\n\n".encode("utf-8")
                            
                            args = tc.get("function", {}).get("arguments", "")
                            if args:
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': active_tools[idx], 'delta': {'type': 'input_json_delta', 'partial_json': args}})}\n\n".encode("utf-8")
                                
                except Exception as e:
                    print(f"DEBUG: SSE Translation Error: {e}")
                    continue
            return

        return StreamingResponse(anthropic_stream(), media_type="text/event-stream")

    # Non-streaming response
    response = await maas.chat_completion(payload)
    
    message = response.get("choices", [{}])[0].get("message", {})
    assistant_text = message.get("content") or ""
    
    content_blocks = []
    if assistant_text:
        content_blocks.append({"type": "text", "text": assistant_text})
        storage.add_message(session.conversation.id, "assistant", assistant_text)
    
    tool_calls = message.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            content_blocks.append(translate_openai_tool_to_anthropic(tc))

    return {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": payload["model"],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}
    }

@anthropic_router.post("/v1/projects")
async def create_project(
        request: Request, 
        req: ProjectCreateRequest) -> dict[str, Any]:
    storage = request.app.state.storage
    cfg = request.app.state.config
    
    project_id = req.project_id or f"proj-{uuid.uuid4().hex[:12]}"
    root_path = str(Path(req.root_path).expanduser().resolve()) if req.root_path else str(cfg.workspace.root)
    project = storage.get_or_create_project(project_id, root_path, req.memory_summary)
    return project.__dict__

@anthropic_router.get("/v1/projects/{project_id}")
async def get_project(
        request: Request, 
        project_id: str) -> dict[str, Any]:
    storage = request.app.state.storage
    project = storage.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return project.__dict__

@anthropic_router.put("/v1/projects/{project_id}/memory")
async def update_project_memory(
        request: Request,
        project_id: str, req: ProjectMemoryUpdate) -> dict[str, Any]:
    storage = request.app.state.storage
    memory_manager = request.app.state.memory_maager
    if not storage.get_project(project_id):
        raise HTTPException(404, "Project not found")
    project = memory_manager.update_project_memory(project_id, req.memory_summary)
    return project.__dict__

@anthropic_router.post("/v1/conversations")
async def create_conversation(
        request: Request,
        req: ConversationCreateRequest) -> dict[str, Any]:
    storage = request.app.state.storage
    if not storage.get_project(req.project_id):
        raise HTTPException(404, "Project not found")
    conv = storage.get_or_create_conversation(req.conversation_id, req.project_id)
    return conv.__dict__

@anthropic_router.get("/v1/conversations/{conversation_id}")
async def get_conversation(
        request: Request,
        conversation_id: str) -> dict[str, Any]:
    storage = request.app.state.storage
    conv = storage.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    messages = [msg.__dict__ for msg in storage.list_messages(conversation_id, limit=500)]
    return {"conversation": conv.__dict__, "messages": messages}

@anthropic_router.put("/v1/conversations/{conversation_id}/memory")
async def update_conversation_memory(
        request: Request,
        conversation_id: str, req: ConversationMemoryUpdate) -> dict[str, Any]:
    storage = request.app.state.storage
    memory_manager = request.app.state.memory_manager    
    if not storage.get_conversation(conversation_id):
        raise HTTPException(404, "Conversation not found")
    conv = memory_manager.update_conversation_memory(conversation_id, req.memory_summary)
    return conv.__dict__

@anthropic_router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all(request: Request, path: str):
    """Catch-all route to log any unexpected requests from the client."""
    print(f"DEBUG: [Catch-all] {request.method} /{path}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content={"error": "Endpoint not implemented in harness", "path": path},
        status_code=404
    )

@anthropic_router.post("/v1/debug/context")
async def debug_context(
        request: Request,
        req: DebugContextRequest) -> dict[str, Any]:
    storage = request.app.state.storage
    memory_manager = request.app.state.memory_manager
    session_manager = request.app.state.session_manager
    context_builder = request.app.state.context_builder
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