"""Ollama: the OpenAI-compatible adapter, plus keeping the model loaded between requests.

Ollama unloads an idle model after five minutes, and reloading a 12B model costs about
4 s, which a voice reply cannot hide. Its OpenAI endpoint ignores ``keep_alive``, so the
model is loaded once through the native API; later OpenAI-style requests keep that setting.
"""

from __future__ import annotations

import logging
from typing import Union

import httpx

from assistant.adapters.llm.openai_compat import OpenAICompatLanguageModel, OpenAICompatSettings

log = logging.getLogger(__name__)


class OllamaSettings(OpenAICompatSettings):
    keep_alive: Union[int, str] = -1
    """How long the model stays loaded: -1 forever, 0 unload at once, or a duration like "30m"."""


class OllamaLanguageModel(OpenAICompatLanguageModel):
    def __init__(self, client: httpx.AsyncClient, settings: OllamaSettings) -> None:
        super().__init__(client, settings)
        self._keep_alive = settings.keep_alive
        self._model = settings.model
        self._generate_url = settings.base_url.rstrip("/").removesuffix("/v1") + "/api/generate"

    async def preload(self) -> None:
        # A request with no prompt only loads the model.
        response = await self._client.post(
            self._generate_url, json={"model": self._model, "keep_alive": self._keep_alive}
        )
        response.raise_for_status()
        log.info("%s loaded (keep_alive=%s)", self._model, self._keep_alive)
