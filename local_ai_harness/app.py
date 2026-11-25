from __future__ import annotations

import json
import uuid
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Response, Request
from fastapi.responses import StreamingResponse

from .config import AppConfig, load_config
from .context import ContextPacker, message_content_to_text
from .gateway import MaasGateway
from .memory import MemoryManager
from .logger import setup_logging
from .models import (
    ChatCompletionRequest,
    CompletionRequest,
    ConversationCreateRequest,
    ConversationMemoryUpdate,
    DebugContextRequest,
    ProjectCreateRequest,
    ProjectMemoryUpdate,
    MessagesRequest,
    ChatMessage,
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
        # print(f"{label} {i+1}: (Size: {message_size} chars)\n{message[:50]}\n")

    # Print Total requests total character size
    # print(f"INFO: Total {label}s: {len(messages)} with {total_char_size} chars")

class LoggingMiddleware:
    def __init__(self, app, logger):
        self.app = app
        self.logger = logger

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        path = scope["path"]
        
        status_code = [None]
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code[0] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            self.logger.error(f"Request failed: {method} {path} - Error: {e}")
            raise e
        finally:
            if status_code[0] and status_code[0] >= 400:
                self.logger.error(f"Error Response: {method} {path} - Status: {status_code[0]}")

def create_app(config_path: str | None = None, config: AppConfig | None = None, gateway: MaasGateway | None = None) -> FastAPI:
    cfg = config or load_config(config_path)
    logger = setup_logging(cfg.log_level)
    storage = Storage(cfg.storage.sqlite_path)
    workspace = WorkspaceReader(cfg.workspace.root, cfg.context.max_file_bytes)
    context_builder = ContextPacker(cfg.context, workspace)
    session_manager = SessionManager(storage, str(cfg.workspace.root))
    memory_manager = MemoryManager(storage)
    maas = gateway or MaasGateway(cfg.maas)

    app = FastAPI(title="Local Code Assistant Harness", version="0.1.0")
    app.add_middleware(LoggingMiddleware, logger=logger)
    app.state.config = cfg
    app.state.storage = storage
    app.state.workspace = workspace
    app.state.context_builder = context_builder
    app.state.session_manager = session_manager
    app.state.memory_manager = memory_manager
    app.state.gateway = maas

    @app.get("/v1/oauth/hello")
    @app.get("/oauth/hello")
    async def oauth_hello():
        """Handshake endpoint for Claude Code OAuth check."""
        from datetime import datetime, timezone
        from fastapi.responses import JSONResponse
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

    @app.post("/api/event_logging/v2/batch")
    async def event_logging_batch():
        """Telemetry endpoint to silence logging errors."""
        return {"status": "ok"}

    @app.post("/api/eval/{path:path}")
    async def eval_telemetry(path: str):
        """Telemetry endpoint for SDK evaluations."""
        return {"status": "ok"}

    @app.get("/mcp-registry/v0/servers")
    async def mcp_servers():
        """Mock MCP registry endpoint."""
        return {"servers": []}

    @app.get("/api/claude_code_penguin_mode")
    async def penguin_mode():
        """Mock penguin mode feature flag."""
        return {"enabled": False}

    @app.get("/api/claude_cli/bootstrap")
    async def cli_bootstrap():
        """Mock CLI bootstrap endpoint."""
        return {"status": "ok", "version": "latest"}

    @app.get("/api/directory/plugins")
    async def plugin_directory():
        """Mock plugin directory endpoint."""
        return {"plugins": []}

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

    @app.head("/v1/models")
    async def models_head():
        """
        HEAD request handler for /v1/models.
        """
        return Response(status_code=200)

    @app.get("/v1/api/hello")
    @app.get("/api/hello")
    async def api_hello():
        """Handshake endpoint for Claude Code / Anthropic compatibility check."""
        from datetime import datetime, timezone
        from fastapi.responses import JSONResponse
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
                "connection": "keep-alive",
                "cache-control": "no-cache, no-store, must-revalidate",
                "pragma": "no-cache",
                "access-control-allow-origin": "*",
                "access-control-allow-methods": "GET, POST, HEAD, OPTIONS",
                "access-control-allow-headers": "*"
            }
        )

    async def _prepare_chat_context(
        messages: list[ChatMessage],
        project_id: str | None,
        conversation_id: str | None,
        context_files: list[str] | None = None,
    ) -> tuple[Any, Any, list[dict[str, str]]]:
        """
        Common logic to resolve session, load memory, build context, and save messages.
        Returns (session, payload_messages, original_messages).
        """
        session = session_manager.resolve(project_id, conversation_id)
        memory = memory_manager.load(session.project, session.conversation)
        stored = storage.list_messages(session.conversation.id)
        
        try:
            built = context_builder.build(
                session.project, 
                memory, 
                stored, 
                messages, 
                context_files or []
            )
        except ValueError as e:
            logger.error(f"Context building error: {e}")
            raise

        # Circuit Breaker: Detect repetition in the built context
        from .context import detect_repetition
        is_repeated, reason = detect_repetition(built.messages)
        if is_repeated:
            logger.debug(f"[Circuit Breaker] Repetition detected! Reason: {reason}. Injecting warning.")
            built.messages.append({
                "role": "system", 
                "content": "Warning: You are repeating yourself. Stop performing the same action and try a different approach or summarize your findings."
            })

        for msg in messages:
            if msg.role in {"user", "assistant"}:
                content = message_content_to_text(msg.content)
                # Avoid duplicating messages already in storage
                if not any(m.role == msg.role and m.content == content for m in stored):
                    storage.add_message(session.conversation.id, msg.role, content)
        
        return session, built.messages, messages

    # HEAD handler
    @app.head("/v1/api/hello")
    @app.head("/api/hello")
    async def api_hello_head():
        """
        HEAD request handler — returns only headers, no body.
        """
        from datetime import datetime, timezone
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

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> dict[str, Any]:
        # ...existing code... (I will use a larger block to be safe)

        # print(f"DEBUG: [Route] Received completion request: {req.prompt[:100]}")
        # Wrap the completion prompt into a chat request
        # We use a system prompt to force the model into 'completion mode'
        # SAFETY VALVE: Limit prompt size to prevent "bloat" from the client.
        # We keep the SUFFIX (the end of the prompt) because that is where the cursor is.
        MAX_COMPLETION_PROMPT_LEN = 1024 
        prompt = req.prompt

        # Return immediately for Ghost empty autocomp
        if not prompt or not prompt.strip():
            logger.info(f"[Autocomp] Return immediately without Calling MAAS: with request (len:{len(prompt)} prompt: {prompt[:100]}")
            return {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": 0,
                "model": req.model or cfg.maas.default_model,
                "choices": [
                    {
                        "text": "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ]
            }

        if prompt and len(prompt) > MAX_COMPLETION_PROMPT_LEN:
            # print(f"INFO: [Autocomp] Prompt too large ({len(prompt)} chars). Truncating to last {MAX_COMPLETION_PROMPT_LEN} chars.")
            prompt = prompt[-MAX_COMPLETION_PROMPT_LEN:]

        # Wrap the completion prompt into a chat request
        chat_payload = {
            "model": req.model or cfg.maas.default_model,
            "messages": [
                # {"role": "system", "content": "You are a code completion engine. Your task is to provide the immediate next tokens of code that logically follow the provided prompt. Do not explain, do not apologize, do not wrap in markdown blocks. Provide ONLY the code completion."},
                {"role": "user", "content": prompt}
            ],
            "temperature": req.temperature if req.temperature is not None else 0.0,
            "max_tokens": req.max_tokens,
        }
        
      
        logger.info(f"[Autocomp] Calling MAAS with request (len:{len(prompt)} prompt: {prompt[:50]}")
        try:
            response = await maas.chat_completion(chat_payload)
        except Exception as e:
            logger.error(f"[Route] ERROR calling MAAS: {str(e)}")
            raise HTTPException(status_code=500, detail=f"MAAS Gateway Error: {str(e)}")
        
        # Extract the content
        assistant_text = _extract_assistant_text(response)
        logger.info(f"[Autocomp] Received response from MAAS: {assistant_text[:50]}")
        
        # Return in OpenAI completions format
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": 0, # Simplified
            "model": chat_payload["model"],
            "choices": [
                {
                    "text": assistant_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        x_harness_project_id: str | None = Header(default=None),
        x_harness_conversation_id: str | None = Header(default=None),
    ) -> Any:
        # # Print the messages and their length
        print_messages(req.messages,"request")

        session, built_messages, _ = await _prepare_chat_context(
            req.messages, x_harness_project_id, x_harness_conversation_id, req.context_files
        )

        payload = req.model_dump(exclude_none=True)
        payload["model"] = req.model or cfg.maas.default_model
        payload["max_tokens"] = cfg.context.model_context_max
        payload["messages"] = built_messages
        # print(f"DEBUG: Payload keys: {list(payload.keys())}")
      
        if req.stream:
            # --- DIAGNOSTIC LOGGING START ---
            # print(f"Model: {payload['model']}")
            params = {k: v for k, v in payload.items() if k != 'messages' and k != 'model'}
            paramSize = sum(len(str(v)) for v in params.values())
            # print(f"INFO: Param total:{paramSize} chars")

            total_payload_size = len(json.dumps(payload))
            logger.info(f"Total payload size: {total_payload_size} characters")

            # print(f"      SYS: {str(payload['messages'])[4800:4880]}")
            # This prints hyperparameters like temperature, top_p, etc.
            # print("="*80 + "\n")
            # --- DIAGNOSTIC LOGGING END ---
            async def stream() -> Any:
                thought_header_printed = False
                buffer = b""
                pending_content = ""
                text_tool_call = False
                pending_reasoning = ""
                text_tool_call_reasoning = False

                # Markers for tool detection
                tool_markers = ["TOOL_NAME:", "```tool", "tool_code", "tool_call", "tool\n", "Action:", "Action Input:"]
                # Regex for tool_name{...} format
                tool_name_brace_pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\{'
                # Regex for tool_name(...) format
                tool_name_paren_pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\('
                # tool_markers = ["TOOL_NAME:", "```tool", "<tool_code>", "<function_name>"]

                async for chunk in maas.stream_chat_completion(payload):
                    buffer += chunk
                    while b"\n" in buffer:
                        raw_line, buffer = buffer.split(b"\n", 1)
                        line = raw_line.decode("utf-8").strip()

                        if not line or not line.startswith("data: "):
                            continue

                        if line == "data: [DONE]":
                            # Final Flush: Handle remaining buffer
                            # 1. Handle text_tool_call in reasoning
                            if text_tool_call_reasoning:
                                # FIX: Unpack the tuple (translated, remaining)
                                translated_r, remaining_r = _translate_tool_call(pending_reasoning)
                                if translated_r:
                                    yield f"data: {translated_r}\n\n".encode("utf-8")
                                    if remaining_r:
                                        yield _reason_sse(remaining_r)
                                elif pending_reasoning:
                                    yield _reason_sse(pending_reasoning)
                            elif pending_reasoning:
                                yield _reason_sse(pending_reasoning)
                            pending_reasoning = ""
                            text_tool_call_reasoning = False

                            # 2. Handle text_tool_call in Contents
                            if text_tool_call:
                                # FIX: Unpack the tuple (translated, remaining)
                                translated, remaining_c = _translate_tool_call(pending_content)
                                if translated:
                                    yield f"data: {translated}\n\n".encode("utf-8")
                                    if remaining_c:
                                        yield _content_sse(remaining_c)
                                elif pending_content:
                                    yield _content_sse(pending_content)
                            elif pending_content:
                                yield _content_sse(pending_content)
                            pending_content = ""
                            text_tool_call = False
                        try:
                            chunk_json = json.loads(line[6:].strip())
                        except json.JSONDecodeError:
                            continue

                        choices = chunk_json.get("choices", [])
                        for choice in choices:
                            delta = choice.get("delta", {})

                            # 1. Handle Reasoning (Thinking)
                            reasoning = delta.get("reasoning_content")
                            if reasoning:
                                pending_reasoning += reasoning
                                
                                # --- REPETITION DETECTION START ---
                                if len(pending_reasoning) > 100:
                                    from .context import detect_repetition
                                    is_repeated, reason = detect_repetition([{"role": "assistant", "content": pending_reasoning}])
                                    if is_repeated:
                                        logger.critical(f"[Circuit Breaker] Repetition detected in reasoning! Breaking stream.")
                                        # We can't easily "reset" the MAAS call from here, but we can stop the stream
                                        # and send a final warning to the user/client.
                                        yield f"data: {json.dumps({'choices': [{'delta': {'content': '\\n\\n[Circuit Breaker: Repetition detected. Stopping generation.]'}}]})}\n\n".encode("utf-8")
                                        return 
                                # --- REPETITION DETECTION END ---

                                if not text_tool_call_reasoning:
                                    stripped_r = pending_reasoning.lstrip()
                                    if any(stripped_r.startswith(m) for m in tool_markers):
                                        text_tool_call_reasoning = True
                                    elif not (stripped_r.startswith('T') or stripped_r.startswith('`') or stripped_r.startswith('<')):
                                        yield _reason_sse(pending_reasoning)
                                        pending_reasoning = ""
                                    elif len(pending_reasoning) > max(len(m) for m in tool_markers) + 10:
                                        yield _reason_sse(pending_reasoning)
                                        pending_reasoning = ""
                                
                                if not thought_header_printed:
                                    # print(f"--- Thinking --- len {len(reasoning)} of {reasoning[0:50]} ")
                                    # logger.info(f"--- Thinking --- len {len(reasoning)} of {reasoning[0:50]} ")
                                    thought_header_printed = True

                            # 2. Handle Native Tool Calls (Pass-through)
                            tool_calls = delta.get("tool_calls")
                            if tool_calls:
                                yield _tool_calls_sse(tool_calls)

                            # 3. Handle Content (and Text-to-Tool Translation)
                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                pending_content += content

                                # --- REPETITION DETECTION START ---
                                if len(pending_content) > 100:
                                    from .context import detect_repetition
                                    is_repeated, reason =  detect_repetition([{"role": "assistant", "content": pending_content}])
                                    if is_repeated:
                                        logger.critical(f"[Circuit Breaker] Repetition detected in content! Breaking stream.")
                                        yield f"data: {json.dumps({'choices': [{'delta': {'content': '\\n\\n[Circuit Breaker: Repetition detected. Stopping generation.]'}}]})}\n\n".encode("utf-8")
                                        return
                                # --- REPETITION DETECTION END ---

                                if not text_tool_call:
                                    # 1. Create a stripped version for detection only
                                    stripped = pending_content.lstrip()

                                    # 2. Check if the stripped content starts with any of our markers
                                    if any(stripped.startswith(m) for m in tool_markers) or re.match(tool_name_brace_pattern, stripped):
                                        text_tool_call = True
                                    # 3. Buffer if it starts with any character that COULD be a marker (T, `, or <)
                                    elif not (stripped.startswith('T') or stripped.startswith('`') or stripped.startswith('<')):
                                        # Definitely not a marker, flush immediately
                                        yield _content_sse(pending_content)
                                        pending_content = ""
                                    elif len(pending_content) > max(len(m) for m in tool_markers) + 10:
                                        # Buffered enough to know it's not a marker (added padding for whitespace)
                                        yield _content_sse(pending_content)
                                        pending_content = ""
                                else:
                                    # We are in text_tool_call translating, just accumulate
                                    print(content, end="", flush=True)
                                    pass

                            # 3. Handle Finish Reason
                            if choice.get("finish_reason") is not None:
                                if text_tool_call: 
                                    logger.debug(f"We are finishing with tool call {text_tool_call}")

                                # 3.1 Handle text_tool_call in reasoning
                                if text_tool_call_reasoning:
                                    translated_r, remaining_r = _translate_tool_call(pending_reasoning)
                                    if translated_r:
                                        yield f"data: {translated_r}\n\n".encode("utf-8")
                                        if remaining_r:
                                            yield _reason_sse(remaining_r)
                                    elif pending_reasoning:
                                        yield _reason_sse(pending_reasoning)
                                elif pending_reasoning:
                                    yield _reason_sse(pending_reasoning)
                                pending_reasoning = ""
                                text_tool_call_reasoning = False                                
                                # 3.2 Handle text_tool_call in content
                                if text_tool_call:
                                    translated, remaining_c = _translate_tool_call(pending_content)
                                    if translated:
                                        yield f"data: {translated}\n\n".encode("utf-8")
                                        if remaining_c:
                                            yield _content_sse(remaining_c)
                                    elif pending_content:
                                        yield _content_sse(pending_content)
                                elif pending_content:
                                    yield _content_sse(pending_content)

                                yield f"data: {json.dumps({'choices': [{'finish_reason': choice.get('finish_reason')}]})}\n\n".encode("utf-8")
                                pending_content = ""
                                text_tool_call = False

                # Final Cleanup for any trailing buffer
                if pending_reasoning:
                    if text_tool_call_reasoning:
                        translated_r, remaining_r = _translate_tool_call(pending_reasoning)
                        if translated_r:
                            yield f"data: {translated_r}\n\n".encode("utf-8")
                            if remaining_r:
                                yield _reason_sse(remaining_r)
                        else:
                            yield _reason_sse(pending_reasoning)
                    else:
                        yield _reason_sse(pending_reasoning)

                if pending_content:
                    if text_tool_call:
                        translated, remaining_c = _translate_tool_call(pending_content)
                        if translated:
                            yield f"data: {translated}\n\n".encode("utf-8")
                            if remaining_c:
                                yield _content_sse(remaining_c)
                        else:
                            yield _content_sse(pending_content)
                    else:
                        yield _content_sse(pending_content)
                return
            
            return StreamingResponse(stream(), media_type="text/event-stream")

        response = await maas.chat_completion(payload)
        logger.debug(f"Raw MAAS response: {json.dumps(response)}")
        # print(f"Response: {response}")  # Print the non-streaming response
        assistant_text = _extract_assistant_text(response)
        if assistant_text:
            storage.add_message(session.conversation.id, "assistant", assistant_text)
        return response

    @app.post("/v1/messages")
    @app.post("/messages")
    @app.post("/v1/v1/messages")
    async def messages(
        req: MessagesRequest,
        x_harness_project_id: str | None = Header(default=None),
        x_harness_conversation_id: str | None = Header(default=None),
    ) -> Any:
        """
        Anthropic Messages API wrapper for Claudecode compatibility.
        """
        # 1. Translate Anthropic request to internal format
        # Anthropic separates system prompt from messages
        logger.debug(f"This is message from Claude {req.messages}")
        internal_messages = list(req.messages)
        if req.system:
            # We prepend the system prompt to the messages list for our ContextBuilder
            internal_messages.insert(0, ChatMessage(role="system", content=req.system))

        # 2. Prepare context and save messages
        session, built_messages, _ = await _prepare_chat_context(
            internal_messages, x_harness_project_id, x_harness_conversation_id
        )

        # 3. Build payload for MAAS (OpenAI format)
        payload = {
            "model": req.model or cfg.maas.default_model,
            "messages": built_messages,
            "max_tokens": req.max_tokens or cfg.context.model_context_max,
            "temperature": req.temperature,
            "stream": req.stream,
        }

        if req.stream:
            async def anthropic_stream() -> Any:
                # We wrap the OpenAI stream into Anthropic SSE events
                msg_id = f"msg_{uuid.uuid4().hex[:12]}"
                
                # 1. message_start
                yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'role': 'assistant', 'model': payload['model']}})}\n\n".encode("utf-8")
                
                # 2. content_block_start (assuming first block is text)
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode("utf-8")

                block_index = 0
                current_block_type = "text"
                
                async for chunk in maas.stream_chat_completion(payload):
                    if not chunk.startswith(b"data: "):
                        continue
                    
                    line = chunk.decode("utf-8").strip()
                    if line == "data: [DONE]":
                        # Finalize blocks and message
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n".encode("utf-8")
                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}})}\n\n".encode("utf-8")
                        yield b"event: message_stop\ndata: {}\n\n"
                        return
                    
                    try:
                        data = json.loads(line[6:])
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        
                        # Handle Reasoning (Thinking)
                        reasoning = delta.get("reasoning_content")
                        if reasoning:
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': reasoning}})}\n\n".encode("utf-8")

                        # Handle Content
                        content = delta.get("content")
                        if content:
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': content}})}\n\n".encode("utf-8")
                        
                        # Handle Tool Calls (Native OpenAI format)
                        tool_calls = delta.get("tool_calls")
                        if tool_calls:
                            if current_block_type == "text":
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n".encode("utf-8")
                                block_index += 1
                                current_block_type = "tool_use"
                            
                            for tc in tool_calls:
                                tc_id = tc.get("id")
                                if tc_id:
                                    name = tc.get("function", {}).get("name", "unknown")
                                    print(f"DEBUG: Sending content_block_start for tool {name} (id: {tc_id}) at index {block_index}")
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tc_id, 'name': name, 'input': {}}})}\n\n".encode("utf-8")
                                    block_index += 1
                                
                                args = tc.get("function", {}).get("arguments", "")
                                if args:
                                    # The block index for this tool call is the one just started, or the current one if it was started in a previous chunk
                                    current_tool_idx = block_index - 1 if tc_id else block_index
                                    print(f"DEBUG: Sending content_block_delta for tool at index {current_tool_idx}: {args[:50]}...")
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': current_tool_idx, 'delta': {'type': 'input_json_delta', 'partial_json': args}})}\n\n".encode("utf-8")
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
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id"),
                    "name": tc.get("function", {}).get("name"),
                    "input": json.loads(tc.get("function", {}).get("arguments", "{}")) if isinstance(tc.get("function", {}).get("arguments"), str) else tc.get("function", {}).get("arguments", {})
                })

        # Translate OpenAI response to Anthropic format
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

    @app.post("/v1/projects")
    async def create_project(req: ProjectCreateRequest) -> dict[str, Any]:
        project_id = req.project_id or f"proj-{uuid.uuid4().hex[:12]}"
        root_path = str(Path(req.root_path).expanduser().resolve()) if req.root_path else str(cfg.workspace.root)
        project = storage.get_or_create_project(project_id, root_path, req.memory_summary)
        return project.__dict__

    @app.get("/v1/projects/{project_id}")
    async def get_project(project_id: str) -> dict[str, Any]:
        project = storage.get_project(project_id)
        if not project:
            raise HTTPException(404, "Project not found")
        return project.__dict__

    @app.put("/v1/projects/{project_id}/memory")
    async def update_project_memory(project_id: str, req: ProjectMemoryUpdate) -> dict[str, Any]:
        if not storage.get_project(project_id):
            raise HTTPException(404, "Project not found")
        project = memory_manager.update_project_memory(project_id, req.memory_summary)
        return project.__dict__

    @app.post("/v1/conversations")
    async def create_conversation(req: ConversationCreateRequest) -> dict[str, Any]:
        if not storage.get_project(req.project_id):
            raise HTTPException(404, "Project not found")
        conv = storage.get_or_create_conversation(req.conversation_id, req.project_id)
        return conv.__dict__

    @app.get("/v1/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str) -> dict[str, Any]:
        conv = storage.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(404, "Conversation not found")
        messages = [msg.__dict__ for msg in storage.list_messages(conversation_id, limit=500)]
        return {"conversation": conv.__dict__, "messages": messages}

    @app.put("/v1/conversations/{conversation_id}/memory")
    async def update_conversation_memory(conversation_id: str, req: ConversationMemoryUpdate) -> dict[str, Any]:
        if not storage.get_conversation(conversation_id):
            raise HTTPException(404, "Conversation not found")
        conv = memory_manager.update_conversation_memory(conversation_id, req.memory_summary)
        return conv.__dict__

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
    async def catch_all(request: Request, path: str):
        """Catch-all route to log any unexpected requests from the client."""
        print(f"DEBUG: [Catch-all] {request.method} /{path}")
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content={"error": "Endpoint not implemented in harness", "path": path},
            status_code=404
        )

    @app.post("/v1/debug/context")
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

def _reason_sse(content: str) -> bytes:
    event = {"choices": [{"delta": {"reasoning_content": content}}]}
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")

def _content_sse(content: str) -> bytes:
    # print({content})
    event = {"choices": [{"delta": {"content": content}}]}
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")

def _tool_calls_sse(tool_calls: Any) -> bytes:
    # print(f"INFO: Native {tool_calls[:50]}")
    event = {"choices": [{"delta": {"tool_calls": tool_calls}}]}
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")

def _translate_tool_call(raw_text: str) -> tuple[str | None, str]:
    """
    Translates MaaS text format into OpenAI tool call SSE.
    Returns (translated_json_or_none, remaining_text).
    """
    print(f"DEBUG: translating tool calls: \n{raw_text[:50]}")

    # --- Strategy 0: Action/Action Input format (ReAct style) ---
    if "Action:" in raw_text and "Action Input:" in raw_text:
        try:
            action_match = re.search(r"Action:\s*(.*?)\n", raw_text)
            input_match = re.search(r"Action Input:\s*(.*?)(?:\n\n|\Z)", raw_text, re.DOTALL)
            if action_match and input_match:
                tool_name = action_match.group(1).strip()
                args_str = input_match.group(1).strip()
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {"input": args_str}
                
                end_pos = input_match.end()
                return _build_openai_tool_call(tool_name, args), raw_text[end_pos:]
        except Exception:
            pass

    # --- Strategy 1: XML-like format (<tool_code> or <tool_call>) ---
    if any(m in raw_text for m in ["<tool_code>", "<tool_call>", "<function_name>"]):
        # First, try the specific <function_name>/<parameters> format
        try:
            name_match = re.search(r"<function_name>\s*(.*?)\s*</function_name>", raw_text)
            params_match = re.search(r"<parameters>\s*(.*?)\s*</parameters>", raw_text, re.DOTALL)
            
            if name_match and params_match:
                tool_name = name_match.group(1).strip()
                args_str = params_match.group(1).strip()
                try:
                    args = json.loads(args_str)
                    # Find where the tool call ends to return remaining text
                    end_pos = raw_text.find("</parameters>") + len("</parameters>")
                    # If there's a closing </tool_call> or </tool_code>, skip that too
                    for tag in ["tool_call", "tool_code"]:
                        end_tag = f"</{tag}>"
                        if end_tag in raw_text[end_pos:]:
                            end_pos += raw_text[end_pos:].find(end_tag) + len(end_tag)
                    
                    return _build_openai_tool_call(tool_name, args), raw_text[end_pos:]
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass

        # Second, try JSON inside <tool_call> or <tool_code> tags
        for tag in ["tool_call", "tool_code"]:
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            if start_tag in raw_text and end_tag in raw_text:
                start_idx = raw_text.find(start_tag) + len(start_tag)
                end_idx = raw_text.find(end_tag)
                json_str = raw_text[start_idx:end_idx].strip()
                try:
                    data = json.loads(json_str)
                    tool_name = data.get("command") or data.get("name")
                    args = data.get("parameters") or data.get("arguments") or {}
                    if tool_name:
                        return _build_openai_tool_call(tool_name, args), raw_text[end_idx + len(end_tag):]
                except json.JSONDecodeError:
                    pass

    # --- Strategy 1: Try to extract JSON (Handles Format 2 & 3) ---
    if any(m in raw_text for m in ["```tool"]) or raw_text.strip().startswith("{"):
        try:
            json_content = raw_text
            # Remove leading markers
            while json_content.startswith("```tool"):
                json_content = json_content[len("```tool"):].lstrip()
            if json_content.startswith("```"):
                json_content = json_content[3:].lstrip()
            
            # Find the end of the JSON block
            if json_content.endswith("```"):
                json_content = json_content[:-3].rstrip()
            
            # We need to find where the JSON actually ends if there's trailing text
            # A simple way is to try and parse it, and if it fails, try to find the last '}'
            try:
                data = json.loads(json_content)
                # If it parsed, we need to know how much of the original raw_text was used
                # This is tricky because of the stripping. 
                # For simplicity, if it's a ```tool block, we assume it ends at the next ```
                if "```tool" in raw_text:
                    end_pos = raw_text.find("```", raw_text.find("```tool") + 7)
                    if end_pos != -1:
                        end_pos += 3
                        return _build_openai_tool_call(data.get("command") or data.get("name"), data.get("parameters") or data.get("arguments") or {}), raw_text[end_pos:]
                
                # If it just started with '{', we find the matching '}'
                if raw_text.strip().startswith("{"):
                    # Very basic brace matching
                    count = 0
                    for i, char in enumerate(raw_text):
                        if char == '{': count += 1
                        elif char == '}': count -= 1
                        if count == 0 and i > 0:
                            return _build_openai_tool_call(data.get("command") or data.get("name"), data.get("parameters") or data.get("arguments") or {}), raw_text[i+1:]
                
                return _build_openai_tool_call(data.get("command") or data.get("name"), data.get("parameters") or data.get("arguments") or {}), ""
            except json.JSONDecodeError:
                pass
        except Exception:
            pass

    # --- Strategy 2: Line-by-Line Parsing (Handles Format 1) ---
    tool_name = ""
    args: dict[str, Any] = {}
    current_arg_key: str | None = None
    value_lines: list[str] = []
    last_line_end = 0

    lines = raw_text.splitlines(keepends=True)
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("TOOL_NAME:"):
            tool_name = stripped.removeprefix("TOOL_NAME:").strip()
        elif stripped.startswith("BEGIN_ARG:"):
            if current_arg_key is not None: break
            current_arg_key = stripped.removeprefix("BEGIN_ARG:").strip()
            if not current_arg_key: break
            value_lines = []
        elif stripped == "END_ARG":
            if current_arg_key is None: break
            raw_value = "\n".join(value_lines).strip()
            try:
                args[current_arg_key] = json.loads(raw_value)
            except json.JSONDecodeError:
                args[current_arg_key] = raw_value.strip('"')
            current_arg_key = None
            # We found a complete tool call if we have a name and args
            if tool_name and args:
                # Calculate where this ended
                # This is approximate since we splitlines
                # We'll just return the rest of the text from the current line
                # But we need to know the index.
                pass
        elif current_arg_key is not None:
            value_lines.append(line.strip())
        
        last_line_end += len(line)

    if tool_name and not current_arg_key and args:
        # For Strategy 2, we'll just return the remaining text as empty for now 
        # or try to calculate it.
        return _build_openai_tool_call(tool_name, args), raw_text[last_line_end:]

    # --- Strategy 4: JSON Array of Tool Calls ---
    # This matches: json [ { "tool": "...", "parameters": { ... } } ]
    if raw_text.strip().startswith("json [") or raw_text.strip().startswith("["):
        try:
            # Remove "json " prefix if present
            json_str = raw_text.strip()
            if json_str.startswith("json "):
                json_str = json_str[5:].strip()
            
            data = json.loads(json_str)
            if isinstance(data, list) and len(data) > 0:
                first_tool = data[0]
                tool_name = first_tool.get("tool") or first_tool.get("name")
                args = first_tool.get("parameters") or first_tool.get("arguments") or {}
                if tool_name:
                    # For simplicity, we return the first tool call and the rest of the text
                    # In a real scenario, we might want to handle multiple tool calls
                    return _build_openai_tool_call(tool_name, args), ""
        except json.JSONDecodeError:
            pass

    # --- Strategy 3: tool_name{json_args} ---
    # This matches: tool_name{...} or tool\ntool_name{...}
    match = re.match(r'^(?:\s*tool\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*(\{)', raw_text, re.IGNORECASE)
    if match:
        tool_name = match.group(1)
        brace_start = match.start(2)
        count = 0
        for i in range(brace_start, len(raw_text)):
            if raw_text[i] == '{':
                count += 1
            elif raw_text[i] == '}':
                count -= 1
                if count == 0:
                    json_str = raw_text[brace_start : i+1]
                    try:
                        # Try standard JSON first
                        args = json.loads(json_str)
                        return _build_openai_tool_call(tool_name, args), raw_text[i+1:]
                    except json.JSONDecodeError:
                        # Fallback: Try to parse pseudo-JSON (unquoted keys)
                        try:
                            # Very basic pseudo-JSON parser: split by comma, then by colon
                            # This is fragile but handles simple cases like {key:value, key2:value2}
                            inner = json_str[1:-1].strip()
                            parsed_args = {}
                            # This regex finds key:value pairs where value can be a string, bool, or number
                            kv_pairs = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|true|false|null|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[^{},]+)', inner)
                            for k, v in kv_pairs:
                                # Clean up value
                                v = v.strip()
                                if v.startswith('"') and v.endswith('"'): v = v[1:-1]
                                elif v.startswith("'") and v.endswith("'"): v = v[1:-1]
                                elif v == 'true': v = True
                                elif v == 'false': v = False
                                elif v == 'null': v = None
                                parsed_args[k] = v
                            
                            if parsed_args:
                                return _build_openai_tool_call(tool_name, parsed_args), raw_text[i+1:]
                        except Exception:
                            pass
                    break
    
    print(f"DEBUG: text_tool_translate failed")
    return None, raw_text

def _build_openai_tool_call(name: str, args: dict) -> str:
    """Helper to wrap a tool name and args into the OpenAI SSE format."""
    tool_call = {
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }]
            }
        }]
    }
    return json.dumps(tool_call, ensure_ascii=False)

