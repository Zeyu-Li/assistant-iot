"""Local control channel: switch listening modes from outside the voice loop.

Muted closes the microphone, so voice cannot unmute it; this socket (or a GPIO button
script talking to it) can. It is a Unix-domain socket readable only by its owner, so it
is reachable from this machine and this user only. Protocol, one line each way:

    "mode"            -> "ok wake_word"
    "mode muted"      -> "ok muted"
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from assistant.core.listening import ListeningController, ListeningMode

log = logging.getLogger(__name__)

DEFAULT_SOCKET = Path("~/.local/state/assistant-iot/control.sock")


class ControlServer:
    def __init__(self, path: Path, listening: ListeningController) -> None:
        self._path = path.expanduser()
        self._listening = listening

    async def serve(self) -> asyncio.AbstractServer:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            self._path.unlink()
        old_umask = os.umask(0o177)  # socket created as 0600
        try:
            server = await asyncio.start_unix_server(self._handle, path=str(self._path))
        finally:
            os.umask(old_umask)
        log.info("control socket at %s", self._path)
        return server

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = (await asyncio.wait_for(reader.readline(), timeout=5)).decode().strip()
            writer.write((self.respond(line) + "\n").encode())
            await writer.drain()
        finally:
            writer.close()

    def respond(self, line: str) -> str:
        parts = line.split()
        if parts[:1] != ["mode"] or len(parts) > 2:
            return "error usage: mode [muted|wake_word|open_mic]"
        if len(parts) == 2:
            try:
                self._listening.set_mode(ListeningMode(parts[1]))
            except ValueError:
                return f"error unknown mode {parts[1]}"
        return f"ok {self._listening.mode.value}"


async def send(path: Path, line: str) -> str:
    reader, writer = await asyncio.open_unix_connection(str(path.expanduser()))
    writer.write((line + "\n").encode())
    await writer.drain()
    reply = (await reader.readline()).decode().strip()
    writer.close()
    return reply
