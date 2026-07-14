"""Minimal DeepSeek chat client for server-side calls."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional

import httpx

from app.debug_agent_log import agent_log


@dataclass
class ChatResult:
    """Return value of DeepSeekClient.chat() — content + raw usage dict."""

    content: str
    usage: dict = field(default_factory=dict)


class DeepSeekClient:
    """Thin wrapper around DeepSeek chat completions."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "deepseek-v4-flash",
        timeout: float = 120.0,
        base_url: str = "https://api.deepseek.com",
        thinking: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not api_key:
            raise ValueError("DeepSeek API key is required")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = str(base_url).rstrip("/")
        # Explicit thinking mode (deepseek-v4-flash defaults to "enabled" server-side
        # if omitted — pass {"type": "disabled"} to match the old non-reasoning
        # deepseek-chat behavior). None = let the API use its own default.
        self.thinking = dict(thinking) if thinking is not None else None

    async def chat(
        self,
        messages: Iterable[Mapping[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 300,
        _debug_caller: str = "unknown",
    ) -> ChatResult:
        """Call DeepSeek chat completions. Returns content + token usage."""

        payload: dict[str, object] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.thinking is not None:
            payload["thinking"] = self.thinking

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        target_url = f"{self.base_url}/chat/completions"
        # region agent log
        agent_log(
            location="deepseek_client.py:chat:before_post",
            message="DeepSeek chat request",
            data={"target_url": target_url, "caller": _debug_caller, "max_tokens": max_tokens},
            hypothesis_id="A-B",
        )
        # endregion
        timeout_obj = httpx.Timeout(self.timeout, connect=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout_obj) as client:
                response = await client.post(target_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectTimeout as exc:
            # region agent log
            agent_log(
                location="deepseek_client.py:chat:connect_timeout",
                message="DeepSeek ConnectTimeout",
                data={"target_url": target_url, "caller": _debug_caller, "exc": str(exc)},
                hypothesis_id="A-B",
            )
            # endregion
            raise
        except httpx.HTTPError as exc:
            # region agent log
            agent_log(
                location="deepseek_client.py:chat:http_error",
                message="DeepSeek HTTP error",
                data={
                    "target_url": target_url,
                    "caller": _debug_caller,
                    "exc_type": type(exc).__name__,
                    "exc": str(exc),
                },
                hypothesis_id="A-B",
            )
            # endregion
            raise
        choices: Optional[List[Mapping[str, object]]] = data.get("choices")  # type: ignore[arg-type]
        if not choices:
            raise RuntimeError("DeepSeek returned no choices")
        message = choices[0].get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        if not content or not isinstance(content, str):
            raise RuntimeError("DeepSeek returned empty content")
        usage: dict = data.get("usage") or {}
        return ChatResult(content=content.strip(), usage=usage)

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
