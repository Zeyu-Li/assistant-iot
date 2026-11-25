from __future__ import annotations

from assistant.ports.flow import FlowContext
from assistant.ports.language import LanguageModel

CHAT_SYSTEM = (
    "You are a home voice assistant. Answer in at most two short spoken sentences, with no markdown. "
    "You have no internet access and no access to the user's accounts. If the answer needs current "
    "information such as news, prices or scores, say you can't look that up."
)


class ChatFlow:
    def __init__(self, llm: LanguageModel) -> None:
        self._llm = llm

    async def run(self, context: FlowContext) -> str:
        return (await self._llm.generate_text(CHAT_SYSTEM, context.request)).strip()
