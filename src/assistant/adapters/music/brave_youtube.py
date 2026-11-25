"""MusicPlayer that drives Brave: search YouTube, play the first real (non-ad) video result.

The search results are read from ``window.ytInitialData``, the JSON YouTube embeds in the
results page, rather than from the rendered DOM: it is available long before the page
finishes rendering, and ads are their own entry types there (``adSlotRenderer``,
``searchPyvRenderer``, ...) instead of a styling detail.

Brave runs from a dedicated profile directory, never the user's own, and every request
it makes is checked against YouTube's hosts so the browser cannot be steered elsewhere.
Brave Shields handles in-video ads.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import quote_plus, urlsplit

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

_DEFAULT_BRAVE = {
    "darwin": "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "linux": "/usr/bin/brave-browser",
}
ALLOWED_HOST_SUFFIXES = ("youtube.com", "youtube-nocookie.com", "googlevideo.com", "ytimg.com", "ggpht.com", "gstatic.com")

_RESULTS_JS = """
() => {
  const data = window.ytInitialData;
  const sections = data?.contents?.twoColumnSearchResultsRenderer?.primaryContents?.sectionListRenderer?.contents || [];
  const out = [];
  for (const section of sections) {
    for (const item of (section.itemSectionRenderer?.contents || [])) {
      const kind = Object.keys(item)[0];
      const v = item[kind] || {};
      out.push({
        kind,
        video_id: v.videoId || null,
        title: (v.title?.runs || []).map(r => r.text).join('') || v.title?.simpleText || null,
        short: !!v.navigationEndpoint?.reelWatchEndpoint,
      });
    }
  }
  return out;
}
"""
_PLAYER_READY_JS = "() => typeof document.getElementById('movie_player')?.playVideo === 'function'"
# YouTube's player overrides direct changes to the <video> element, so go through its own API.
_PLAYER_CALL_JS = """
([method, args]) => {
  const player = document.getElementById('movie_player');
  if (!player || typeof player[method] !== 'function') return false;
  player[method](...args);
  return true;
}
"""


class BraveYouTubeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brave_path: str = Field(default_factory=lambda: _DEFAULT_BRAVE.get(sys.platform, "brave-browser"))
    profile_dir: Path = Path("~/.local/share/assistant-iot/brave-profile")
    headless: bool = False
    initial_volume: int = Field(60, ge=0, le=100)
    extra_args: tuple[str, ...] = ()
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class SearchResult:
    kind: str
    video_id: Optional[str]
    title: Optional[str]
    short: bool = False


def pick_first_video(results: Sequence[SearchResult]) -> Optional[SearchResult]:
    """First organic video: ads, promoted items, Shorts, playlists, mixes and channel cards are skipped."""
    for result in results:
        if result.kind == "videoRenderer" and result.video_id and not result.short:
            return result
    return None


def host_allowed(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == suffix or host.endswith("." + suffix) for suffix in ALLOWED_HOST_SUFFIXES)


class BraveYouTubePlayer:
    def __init__(self, settings: BraveYouTubeSettings) -> None:
        self._settings = settings
        self._volume = settings.initial_volume
        self._playwright: Any = None
        self._context: Any = None
        self._page: Any = None
        self._starting: Optional[asyncio.Task[None]] = None

    async def play(self, query: str) -> Optional[str]:
        """Pick the video and return its title; playback starts in the background so the reply is not held up."""
        page = await self._ensure_page()
        self._cancel_start()
        await page.goto(f"https://www.youtube.com/results?search_query={quote_plus(query)}", wait_until="commit")
        await page.wait_for_function("() => !!window.ytInitialData", polling=250)
        choice = pick_first_video([SearchResult(**item) for item in await page.evaluate(_RESULTS_JS)])
        if choice is None or choice.video_id is None:
            log.info("no playable result for %r", query)
            return None
        self._starting = asyncio.create_task(self._start_playback(choice.video_id))
        return choice.title

    async def pause(self) -> None:
        self._cancel_start()
        await self._player_call("pauseVideo")

    async def resume(self) -> None:
        await self._player_call("playVideo")

    async def stop(self) -> None:
        self._cancel_start()
        await self._player_call("stopVideo")
        if self._page is not None:
            await self._page.goto("about:blank")

    async def set_volume(self, level: int) -> int:
        self._volume = max(0, min(100, level))
        await self._apply_volume()
        return self._volume

    async def adjust_volume(self, delta: int) -> int:
        return await self.set_volume(self._volume + delta)

    async def close(self) -> None:
        self._cancel_start()
        if self._context is not None:
            await self._context.close()
        if self._playwright is not None:
            await self._playwright.stop()
        self._context = self._page = self._playwright = None

    async def wait_until_started(self) -> None:
        """For tests and scripts: block until the background playback start has finished."""
        if self._starting is not None:
            await self._starting

    async def _start_playback(self, video_id: str) -> None:
        try:
            await self._page.goto(f"https://www.youtube.com/watch?v={video_id}", wait_until="commit")
            await self._page.wait_for_function(_PLAYER_READY_JS, polling=250)
            await self._apply_volume()
            await self._player_call("playVideo")
        except asyncio.CancelledError:
            raise
        except Exception:  # background task: nobody awaits it, so log rather than lose the error
            log.exception("could not start playback of %s", video_id)

    def _cancel_start(self) -> None:
        if self._starting is not None and not self._starting.done():
            self._starting.cancel()
        self._starting = None

    async def _apply_volume(self) -> None:
        await self._player_call("unMute")
        await self._player_call("setVolume", self._volume)

    async def _player_call(self, method: str, *args: Any) -> bool:
        if self._page is None or self._page.is_closed():
            return False
        return bool(await self._page.evaluate(_PLAYER_CALL_JS, [method, list(args)]))

    async def _ensure_page(self) -> Any:
        if self._page is not None and not self._page.is_closed():
            return self._page
        from playwright.async_api import async_playwright  # heavy import, only when music is used

        if self._playwright is None:
            self._playwright = await async_playwright().start()
        profile_dir = self._settings.profile_dir.expanduser()
        profile_dir.mkdir(parents=True, exist_ok=True)
        self._context = await self._playwright.chromium.launch_persistent_context(
            str(profile_dir),
            executable_path=self._settings.brave_path,
            headless=self._settings.headless,
            args=["--autoplay-policy=no-user-gesture-required", *self._settings.extra_args],
        )
        self._context.set_default_timeout(self._settings.timeout_seconds * 1000)
        await self._context.route("**/*", _enforce_allowlist)
        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        return self._page


async def _enforce_allowlist(route: Any) -> None:
    url = route.request.url
    if url.startswith(("data:", "blob:", "about:")) or host_allowed(url):
        await route.continue_()
    else:
        log.debug("blocked browser request to %s", urlsplit(url).hostname)
        await route.abort()
