from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import uuid
from typing import Any

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
    """
    HEAD request handler — returns only headers, no body.
    """
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
