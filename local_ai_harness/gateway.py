from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from .config import MaasConfig


class MaasGateway:
    def __init__(self, config: MaasConfig):
        self.config = config

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def list_models(self) -> dict[str, Any]:
        url = f"{self.config.base_url}/models"
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            try:
                response = await client.get(url, headers=self._headers())
                response.raise_for_status()
                return response.json()
            except Exception:
                return {
                    "object": "list",
                    "data": [
                        {
                            "id": self.config.default_model,
                            "object": "model",
                            "owned_by": "local-harness",
                        }
                    ],
                }

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(url, headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()

    async def stream_chat_completion(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        url = f"{self.config.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, headers=self._headers(), json=payload, timeout=self.config.timeout_seconds) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk

