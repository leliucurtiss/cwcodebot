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
from .gateway import MaasUpstreamError
from .tools_helpers import (
    translate_openai_tool_to_anthropic,
    translate_anthropic_to_openai_tools,
    anthropic_start_tool_block,
    anthropic_tool_delta,
    anthropic_stop_block,
    translate_text_to_anthropic_tools,
    translate_anthropic_messages,
    OpenAISSEDecoder,
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
      
    # 1. Translate Anthropic request to internal OpenAI messages.
    logger.debug(f"This is message from Claude {req.messages}")
    internal_messages = [
        ChatMessage(**message)
        for message in translate_anthropic_messages(req.messages)
    ]
    
    # Augment system prompt to force agentic tool use and prevent "chatty" JSON output
    agent_steering = (
        "\n\nIMPORTANT: You are a professional coding agent. When you need to perform an action "
        "(like listing files, reading code, or running commands), you MUST use the provided tools. "
        "Do not describe your thought process or output JSON in markdown; use the tool-calling API directly."
    )
    
    if req.system:
        if isinstance(req.system, str):
            content = req.system + agent_steering
        else:
            system_text = " ".join([b.get("text", str(b)) if isinstance(b, dict) else str(b) for b in req.system])
            content = system_text + agent_steering
        internal_messages.insert(0, ChatMessage(role="system", content=content))
    else:
        internal_messages.insert(0, ChatMessage(role="system", content=agent_steering))

    # 2. Prepare context and save messages
    session, built_messages, _ = await prepare_chat_context(
        session_manager, memory_manager, storage, context_builder, logger,
        internal_messages, x_harness_project_id, x_harness_conversation_id
    )

    # 3. Build payload for MAAS (OpenAI format).
    payload = {
        "model": cfg.maas.default_model,
        "messages": built_messages,
        "max_tokens": req.max_tokens or cfg.context.model_context_max,
        "stream": req.stream,
    }
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.top_p is not None:
        payload["top_p"] = req.top_p
    if req.stop_sequences is not None:
        payload["stop"] = req.stop_sequences

    model_extra = req.model_extra or {}
    if isinstance(model_extra.get("tools"), list):
        translated_tools = translate_anthropic_to_openai_tools(model_extra["tools"])
        if translated_tools:
            payload["tools"] = translated_tools
            requested_choice = model_extra.get("tool_choice")
            if isinstance(requested_choice, dict):
                choice_type = requested_choice.get("type")
                if choice_type == "tool" and requested_choice.get("name"):
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": requested_choice["name"]},
                    }
                elif choice_type == "any":
                    payload["tool_choice"] = "required"
                elif choice_type == "none":
                    payload["tool_choice"] = "none"
                else:
                    payload["tool_choice"] = "auto"
            elif requested_choice in {"auto", "required", "none"}:
                payload["tool_choice"] = requested_choice
            else:
                payload["tool_choice"] = "auto"

    if req.stream:
        async def anthropic_stream() -> Any:
            msg_id = f"msg_{uuid.uuid4().hex[:12]}"
            decoder = OpenAISSEDecoder()
            text_buffer = ""
            native_tools: dict[int, dict[str, str]] = {}

            yield (
                "event: message_start\n"
                f"data: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': payload['model'], 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
            ).encode()

            try:
                async for chunk in maas.stream_chat_completion(payload):
                    for event in decoder.feed(chunk):
                        if event == "[DONE]":
                            continue
                        try:
                            choice = json.loads(event).get("choices", [{}])[0]
                        except (json.JSONDecodeError, IndexError):
                            logger.error("Ignoring invalid OpenAI SSE event")
                            continue
                        delta = choice.get("delta", {})
                        if isinstance(delta.get("reasoning_content"), str):
                            text_buffer += delta["reasoning_content"]
                        if isinstance(delta.get("content"), str):
                            text_buffer += delta["content"]
                        for tool_call in delta.get("tool_calls") or []:
                            index = int(tool_call.get("index", len(native_tools)))
                            active = native_tools.setdefault(index, {"id": "", "name": "", "arguments": ""})
                            if tool_call.get("id"):
                                active["id"] += str(tool_call["id"])
                            function = tool_call.get("function") or {}
                            if function.get("name"):
                                active["name"] += str(function["name"])
                            if function.get("arguments"):
                                active["arguments"] += str(function["arguments"])
            except MaasUpstreamError as exc:
                logger.error("MaaS streaming error HTTP %s: %s", exc.status_code, exc.detail)
                error = {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"MaaS upstream returned HTTP {exc.status_code}: {exc.detail}",
                    },
                }
                yield f"event: error\ndata: {json.dumps(error)}\n\n".encode()
                return

            tools: list[dict[str, Any]] = []
            cleaned_text = text_buffer
            if native_tools:
                for index in sorted(native_tools):
                    state = native_tools[index]
                    arguments = state["arguments"] or "{}"
                    try:
                        parsed = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise HTTPException(502, f"Upstream returned invalid tool arguments: {exc}") from exc
                    tools.append(
                        {
                            "type": "tool_use",
                            "id": state["id"] or f"call_{uuid.uuid4().hex[:24]}",
                            "name": state["name"] or "unknown_tool",
                            "input": parsed,
                        }
                    )
            else:
                tools, cleaned_text = translate_text_to_anthropic_tools(text_buffer)

            block_index = 0
            if cleaned_text:
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode()
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': cleaned_text}})}\n\n".encode()
                yield anthropic_stop_block(block_index)
                block_index += 1
            for tool in tools:
                yield anthropic_start_tool_block(block_index, tool["id"], tool["name"])
                yield anthropic_tool_delta(block_index, json.dumps(tool["input"], ensure_ascii=False))
                yield anthropic_stop_block(block_index)
                block_index += 1

            yield (
                "event: message_delta\n"
                f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'tool_use' if tools else 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
            ).encode()
            yield b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

        return StreamingResponse(anthropic_stream(), media_type="text/event-stream")

    # Non-streaming response
    response = await maas.chat_completion(payload)
    
    message = response.get("choices", [{}])[0].get("message", {})
    assistant_text = message.get("content") or ""
    
    content_blocks = []
    if assistant_text:
        storage.add_message(session.conversation.id, "assistant", assistant_text)

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        if assistant_text:
            content_blocks.append({"type": "text", "text": assistant_text})
        content_blocks.extend(translate_openai_tool_to_anthropic(call) for call in tool_calls)
    elif assistant_text:
        text_tools, cleaned = translate_text_to_anthropic_tools(assistant_text)
        if cleaned:
            content_blocks.append({"type": "text", "text": cleaned})
        content_blocks.extend(text_tools)
    elif assistant_text:
        content_blocks.append({"type": "text", "text": assistant_text})

    return {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": payload["model"],
        "stop_reason": "tool_use" if any(block["type"] == "tool_use" for block in content_blocks) else "end_turn",
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

@anthropic_router.post("/v1/debug/context")
@anthropic_router.post("/harness/debug/context")
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


@anthropic_router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all(request: Request, path: str):
    """Catch-all route to log any unexpected requests from the client."""
    print(f"TRACE: [Catch-all] {request.method} /{path}")
    return JSONResponse(
        content={"error": "Endpoint not implemented in harness", "path": path},
        status_code=404,
    )