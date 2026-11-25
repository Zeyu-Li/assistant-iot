"""Entry point for one transcribed utterance, shared by the voice loop and the text REPL."""

from __future__ import annotations

from typing import Optional

from assistant.core.dialogue import DialogueManager, Reply
from assistant.core.listening import ListeningController, ListeningMode


class Assistant:
    def __init__(self, dialogue: DialogueManager, listening: ListeningController) -> None:
        self._dialogue = dialogue
        self.listening = listening
        listening.on_change(self._on_mode_change)

    def _on_mode_change(self, mode: ListeningMode) -> None:
        if mode is ListeningMode.MUTED:
            self._dialogue.reset()

    async def handle_utterance(self, text: str, *, wake_word_detected: bool) -> Optional[Reply]:
        if not self.listening.accepts_speech(wake_word_detected=wake_word_detected):
            return None
        # Answers to the assistant's own question never need the "is this for me" check.
        require_addressed = not self._dialogue.in_progress and self.listening.needs_addressed_check(
            wake_word_detected=wake_word_detected
        )
        # In open mic, only the wake phrase earns an "I can't help with that"; anything else
        # that matches no capability was probably conversation, so stay quiet.
        quiet = self.listening.mode is ListeningMode.OPEN_MIC and not wake_word_detected
        reply = await self._dialogue.handle(text, require_addressed=require_addressed, quiet_if_unmatched=quiet)
        if reply is not None:
            self.listening.open_follow_up()
        return reply
