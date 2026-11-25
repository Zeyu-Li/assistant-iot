"""Music playback port."""

from __future__ import annotations

from typing import Optional, Protocol


class MusicPlayer(Protocol):
    async def play(self, query: str) -> Optional[str]:
        """Start playing the best match for ``query``; return its title, or None if nothing matched."""
        ...

    async def pause(self) -> None: ...

    async def resume(self) -> None: ...

    async def stop(self) -> None: ...

    async def set_volume(self, level: int) -> int:
        """Set volume to ``level`` percent (clamped to 0-100); return the new level."""
        ...

    async def adjust_volume(self, delta: int) -> int:
        """Change volume by ``delta`` percentage points; return the new level."""
        ...
