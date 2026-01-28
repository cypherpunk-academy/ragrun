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
        timeout: float = 120.0,
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        if not api_key:
            raise ValueError("DeepSeek API key is required")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = str(base_url).rstrip("/")

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

        timeout_obj = httpx.Timeout(self.timeout, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout_obj) as client:
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

    async def list_models(self) -> list[str]:
        """Best-effort probe for available models (if endpoint is exposed)."""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        timeout_obj = httpx.Timeout(self.timeout, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout_obj) as client:
            response = await client.get(f"{self.base_url}/models", headers=headers)
            response.raise_for_status()
            data = response.json()
            models = data.get("data") or data.get("models") or []
            names: list[str] = []
            if isinstance(models, list):
                for m in models:
                    name = m.get("id") if isinstance(m, Mapping) else None
                    if isinstance(name, str):
                        names.append(name)
            return names
