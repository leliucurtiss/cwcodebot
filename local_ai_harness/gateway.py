from __future__ import annotations

import asyncio
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
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "cwcodebot",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def list_models(self) -> dict[str, Any]:
        url = f"{self.config.base_url}/models"
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.get(url, headers=self._headers())
            self._raise_for_status(response)
            data = response.json()
            if "data" in data and isinstance(data["data"], list):
                for model in data["data"]:
                    if "id" in model:
                        model["id"] = f"{model['id']}-cwcode"
            return data

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            for attempt in range(3):
                response = await client.post(url, headers=self._headers(), json=payload)
                if not self._should_retry(response.status_code, attempt):
                    self._raise_for_status(response)
                    return response.json()
                await self._retry_delay(response, attempt)
        raise RuntimeError("Unreachable retry state")

    async def verify_upstream(self) -> dict[str, Any]:
        response = await self.chat_completion(
            {
                "model": self.config.default_model,
                "messages": [{"role": "user", "content": "Reply exactly OK."}],
                "max_tokens": 1,
                "stream": False,
            }
        )
        content = response.get("choices", [{}])[0].get("message", {}).get("content")
        return {
            "response_id": response.get("id"),
            "content_present": bool(content),
        }

    async def stream_chat_completion(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        url = f"{self.config.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=None) as client:
            for attempt in range(3):
                async with client.stream("POST", url, headers=self._headers(), json=payload, timeout=self.config.timeout_seconds) as response:
                    if self._should_retry(response.status_code, attempt):
                        await response.aread()
                        await self._retry_delay(response, attempt)
                        continue
                    if not response.is_success:
                        await response.aread()
                        self._raise_for_status(response)
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk
                    return

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            body: Any = response.json()
            if isinstance(body, dict) and isinstance(body.get("error"), dict):
                error = body["error"]
                detail = error.get("message") or error
                metadata = error.get("metadata")
                if isinstance(metadata, dict) and metadata.get("raw"):
                    detail = f"{detail}: {metadata['raw']}"
            else:
                detail = body
        except ValueError:
            detail = response.text
        message = str(detail).replace("\n", " ")[:1000]
        raise MaasUpstreamError(response.status_code, message or response.reason_phrase)

    @staticmethod
    def _should_retry(status_code: int, attempt: int) -> bool:
        return attempt < 2 and status_code in {429, 502, 503, 504}

    @staticmethod
    async def _retry_delay(response: httpx.Response, attempt: int) -> None:
        retry_after = response.headers.get("Retry-After")
        try:
            delay = min(float(retry_after), 10.0) if retry_after else 1.0 * (2**attempt)
        except ValueError:
            delay = 1.0 * (2**attempt)
        await asyncio.sleep(delay)


