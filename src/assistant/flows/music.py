from __future__ import annotations

from assistant.ports.flow import FlowContext
from assistant.ports.music import MusicPlayer

VOLUME_STEP = 10


class PlayMusicFlow:
    def __init__(self, player: MusicPlayer) -> None:
        self._player = player

    async def run(self, context: FlowContext) -> str:
        query = str(context.arguments["query"])
        title = await self._player.play(query)
        return f"Playing {title}." if title else f"I couldn't find anything for {query}."


class MusicControlFlow:
    def __init__(self, player: MusicPlayer) -> None:
        self._player = player

    async def run(self, context: FlowContext) -> str:
        action = context.arguments["action"]
        if action == "pause":
            await self._player.pause()
            return "Paused."
        if action == "resume":
            await self._player.resume()
            return "Resuming."
        if action == "stop":
            await self._player.stop()
            return "Stopped."
        if action == "set_volume":
            level = context.arguments.get("level")
            if level is None:
                return "Tell me a volume between 0 and 100."
            return f"Volume {await self._player.set_volume(int(level))}."
        delta = VOLUME_STEP if action == "volume_up" else -VOLUME_STEP
        return f"Volume {await self._player.adjust_volume(delta)}."
