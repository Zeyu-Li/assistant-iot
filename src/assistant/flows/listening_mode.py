from __future__ import annotations

from assistant.core.listening import ListeningController, ListeningMode
from assistant.ports.flow import FlowContext

_MODES = {
    "mute": (ListeningMode.MUTED, "Okay, I've stopped listening. Turn me back on from the app or the button."),
    "wake_word": (ListeningMode.WAKE_WORD, "Okay, I'll only listen for the wake word."),
    "conversation": (ListeningMode.OPEN_MIC, "Conversation mode is on. I'll answer when you talk to me."),
}


class ListeningModeFlow:
    def __init__(self, controller: ListeningController) -> None:
        self._controller = controller

    async def run(self, context: FlowContext) -> str:
        mode, reply = _MODES[context.arguments["mode"]]
        self._controller.set_mode(mode)
        return reply
