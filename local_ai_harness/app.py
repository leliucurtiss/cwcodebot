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

from .middleware import LoggingMiddleware
from .app_utils import print_messages_length
from .tools_helpers import (
    _extract_assistant_text,
    _reason_sse,
    _content_sse,
    _tool_calls_sse,
    _translate_tool_call,
)
from .anthropic import anthropic_router
from .app_utils import prepare_chat_context

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
 
    # Specific routes are defined below.


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
        prompt = req.prompt
        # Wrap the completion prompt into a chat request
        chat_payload = {
            "model": req.model or cfg.maas.default_model,
            "messages": [
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
        print_messages_length(req.messages,"request")

        session, built_messages, _ = await prepare_chat_context(
            session_manager, memory_manager, storage, context_builder, logger,
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
            print(f"INFO: Param total:{paramSize} chars")
            # print(f" ___ DEBUG: IDE Param total:{params}")

            total_payload_size = len(json.dumps(payload))
            print(f"INFO:   Total payload size: {total_payload_size} characters")

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
                tool_markers = ["TOOL_NAME:", "```tool", "tool_code"]
                # Regex for tool_name{...} format
                # tool_name_brace_pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\{'

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
                                        logger.critical(f"Repetition detected in reasoning! Breaking stream.")
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
                                    if any(stripped.startswith(m) for m in tool_markers):
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

    # Include the anthropic_router last so its catch-all doesn't intercept other routes
    app.include_router(anthropic_router)

    return app