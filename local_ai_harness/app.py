from __future__ import annotations

import json
import uuid
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Response, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import AppConfig, load_config
from .context import ContextPacker, message_content_to_text
from .gateway import MaasGateway, MaasUpstreamError
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
    OpenAISSEDecoder,
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
    app.state.logger = logger
    @app.exception_handler(MaasUpstreamError)
    async def maas_upstream_error_handler(request: Request, exc: MaasUpstreamError) -> JSONResponse:
        logger.error(
            "MaaS upstream error %s for %s %s: %s",
            exc.status_code,
            request.method,
            request.url.path,
            exc.detail,
        )
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"MaaS upstream returned HTTP {exc.status_code}: {exc.detail}",
                },
            },
            headers={"x-harness-upstream-status": str(exc.status_code)},
        )

    # Specific routes are defined below.


    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "workspace_root": str(cfg.workspace.root),
            "model": cfg.maas.default_model,
            "maas_api_key_configured": bool(cfg.maas.api_key),
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
        # print(f"TRACE: Payload keys: {list(payload.keys())}")
      
        if req.stream:
            # --- DIAGNOSTIC LOGGING START ---
            # print(f"Model: {payload['model']}")
            params = {k: v for k, v in payload.items() if k != 'messages' and k != 'model'}
            paramSize = sum(len(str(v)) for v in params.values())
            print(f"INFO: Param total:{paramSize} chars")
            # print(f" ___ TRACE: IDE Param total:{params}")

            total_payload_size = len(json.dumps(payload))
            print(f"INFO:   Total payload size: {total_payload_size} characters")

            # print(f"      SYS: {str(payload['messages'])[4800:4880]}")
            # This prints hyperparameters like temperature, top_p, etc.
            # print("="*80 + "\n")
            # --- DIAGNOSTIC LOGGING END ---
            async def stream() -> Any:
                pending_content = ""
                pending_reasoning = ""
                saw_native_tool = False
                translated_tool = False
                decoder = OpenAISSEDecoder()

                def maybe_tool_text(value: str) -> bool:
                    value = value.lstrip()
                    if not value:
                        return True
                    lower = value.lower()
                    return (
                        any(marker.startswith(lower) or lower.startswith(marker) for marker in
                            ("tool_name:", "```tool", "```json", "<tool_call", "<tool_code",
                             "<function_name", "action:", "tool"))
                        or lower.startswith("{")
                        or lower.startswith("[")
                        or re.fullmatch(r"[A-Za-z_][\w.-]*(?:\s+)?(?:\{[^}]*)?", value) is not None
                    )

                async def flush_pending() -> Any:
                    nonlocal pending_content, pending_reasoning, translated_tool
                    for reasoning in (True, False):
                        pending = pending_reasoning if reasoning else pending_content
                        if not pending:
                            continue
                        while pending:
                            translated, remaining = _translate_tool_call(pending)
                            if translated is None:
                                if maybe_tool_text(pending):
                                    logger.warning("Potential fragmented tool call detected: %s", pending[:200])
                                    print(f"WARNING: Potential fragmented tool call detected: {pending[:200]}")
                                yield _reason_sse(pending) if reasoning else _content_sse(pending)
                                break
                            if pending.lstrip().lower().startswith("tool"):
                                logger.warning("Potential fragmented tool call translated: %s", pending[:200])
                                print(f"WARNING: Potential fragmented tool call detected: {pending[:200]}")
                            translated_tool = True
                            yield f"data: {translated}\n\n".encode()
                            pending = remaining
                    pending_reasoning = ""
                    pending_content = ""

                async for chunk in maas.stream_chat_completion(payload):
                    for event in decoder.feed(chunk):
                        if event == "[DONE]":
                            async for output in flush_pending():
                                yield output
                            finish_reason = "tool_calls" if saw_native_tool or translated_tool else "stop"
                            yield f"data: {json.dumps({'choices': [{'finish_reason': finish_reason}]})}\n\n".encode()
                            yield b"data: [DONE]\n\n"
                            return
                        try:
                            chunk_json = json.loads(event)
                        except json.JSONDecodeError:
                            continue

                        for choice in chunk_json.get("choices", []):
                            delta = choice.get("delta", {})
                            reasoning = delta.get("reasoning_content")
                            if isinstance(reasoning, str) and reasoning:
                                pending_reasoning += reasoning
                            tool_calls = delta.get("tool_calls")
                            if tool_calls:
                                saw_native_tool = True
                                yield _tool_calls_sse(tool_calls)
                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                pending_content += content
                                while True:
                                    translated, remaining = _translate_tool_call(pending_content)
                                    if translated is None:
                                        if not maybe_tool_text(pending_content):
                                            yield _content_sse(pending_content)
                                            pending_content = ""
                                        break
                                    if pending_content.lstrip().lower().startswith("tool"):
                                        print(
                                            "WARNING: Potential fragmented tool call detected: "
                                            f"{pending_content[:200]}"
                                        )
                                    translated_tool = True
                                    yield f"data: {translated}\n\n".encode()
                                    pending_content = remaining

                async for output in flush_pending():
                    yield output
                yield f"data: {json.dumps({'choices': [{'finish_reason': 'tool_calls' if saw_native_tool or translated_tool else 'stop'}]})}\n\n".encode()
                yield b"data: [DONE]\n\n"
            
            return StreamingResponse(stream(), media_type="text/event-stream")

        response = await maas.chat_completion(payload)
        logger.debug(f"Raw MAAS response: {json.dumps(response)}")
        assistant_message = response.get("choices", [{}])[0].get("message", {})
        assistant_text = _extract_assistant_text(response)
        if assistant_text and not assistant_message.get("tool_calls"):
            translated, remaining = _translate_tool_call(assistant_text)
            if translated:
                tool_event = json.loads(translated)
                assistant_message["tool_calls"] = tool_event["choices"][0]["delta"]["tool_calls"]
                assistant_message["content"] = remaining or None
                assistant_message.pop("function_call", None)
        if assistant_text:
            storage.add_message(session.conversation.id, "assistant", assistant_text)
        return response

    # Include the anthropic_router last so its catch-all doesn't intercept other routes
    app.include_router(anthropic_router)

    return app