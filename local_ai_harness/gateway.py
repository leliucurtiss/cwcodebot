from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from .config import MaasConfig


class MaasUpstreamError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


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
                data = response.json()
                if "data" in data and isinstance(data["data"], list):
                    for model in data["data"]:
                        if "id" in model:
                            model["id"] = f"{model['id']}-cwcode"
                return data
            except Exception:
                return {
                    "object": "list",
                    "data": [
                        {
                            "id": f"{self.config.default_model}-cwcode",
                            "object": "model",
                            "owned_by": "local-harness",
                        }
                    ],
                }

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(url, headers=self._headers(), json=payload)
            self._raise_for_status(response)
            return response.json()

    async def stream_chat_completion(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        url = f"{self.config.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, headers=self._headers(), json=payload, timeout=self.config.timeout_seconds) as response:
                self._raise_for_status(response)
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            body: Any = response.json()
            if isinstance(body, dict) and isinstance(body.get("error"), dict):
                detail = body["error"].get("message") or body["error"]
            else:
                detail = body
        except ValueError:
            detail = response.text
        message = str(detail).replace("\n", " ")[:1000]
        raise MaasUpstreamError(response.status_code, message or response.reason_phrase)


