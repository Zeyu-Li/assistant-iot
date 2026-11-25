"""LanguageModel over any OpenAI-compatible chat endpoint (llama-server, Ollama, vLLM, ...)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Mapping, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


class OpenAICompatSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    """Up to and including /v1, for example http://127.0.0.1:11434/v1."""
    model: str
    api_key_env: Optional[str] = None
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 400
    extra_body: dict[str, Any] = Field(default_factory=dict)
    """Server-specific request fields, merged into every request."""


class OpenAICompatLanguageModel:
    def __init__(self, client: httpx.AsyncClient, settings: OpenAICompatSettings) -> None:
        self._client = client
        self._settings = settings

    async def generate_json(self, system: str, user: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        content = await self._complete(
            system,
            user,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "arguments", "strict": True, "schema": dict(schema)},
            },
        )
        try:
            parsed = json.loads(_strip_thinking(content))
        except json.JSONDecodeError:
            log.warning("model returned invalid JSON: %r", content[:200])
            return {}
        return parsed if isinstance(parsed, dict) else {}

    async def generate_text(self, system: str, user: str) -> str:
        return _strip_thinking(await self._complete(system, user))

    async def _complete(self, system: str, user: str, **extra: Any) -> str:
        body = {
            "model": self._settings.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
            **self._settings.extra_body,
            **extra,
        }
        response = await self._client.post("/chat/completions", json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"].get("content") or ""


def _strip_thinking(text: str) -> str:
    return _THINKING.sub("", text).strip()
