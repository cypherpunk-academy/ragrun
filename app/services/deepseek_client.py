"""Minimal DeepSeek chat client for server-side calls."""
from __future__ import annotations

from typing import Iterable, List, Mapping, Optional

import httpx


class DeepSeekClient:
    """Thin wrapper around DeepSeek chat completions."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "deepseek-chat",
        timeout: float = 30.0,
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        if not api_key:
            raise ValueError("DeepSeek API key is required")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")

    async def chat(
        self,
        messages: Iterable[Mapping[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 300,
    ) -> str:
        """Call DeepSeek chat completions."""

        payload: dict[str, object] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            response.raise_for_status()
            data = response.json()
            choices: Optional[List[Mapping[str, object]]] = data.get("choices")  # type: ignore[arg-type]
            if not choices:
                raise RuntimeError("DeepSeek returned no choices")
            message = choices[0].get("message", {})
            content = message.get("content") if isinstance(message, dict) else None
            if not content or not isinstance(content, str):
                raise RuntimeError("DeepSeek returned empty content")
            return content.strip()
