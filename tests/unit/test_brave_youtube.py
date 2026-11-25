from __future__ import annotations

from assistant.adapters.music.brave_youtube import SearchResult, host_allowed, pick_first_video


def test_skips_ads_shorts_playlists_and_cards() -> None:
    # Entry kinds as seen in a live ytInitialData for "Mr Brightside The Killers" (2026-11-25).
    results = [
        SearchResult("searchPyvRenderer", "promo1", "Sponsored promo"),
        SearchResult("officialCardViewModel", None, None),
        SearchResult("adSlotRenderer", None, None),
        SearchResult("reelShelfRenderer", None, None),
        SearchResult("videoRenderer", "short1", "A short", short=True),
        SearchResult("playlistRenderer", None, "Playlist"),
        SearchResult("videoRenderer", "real", "The Killers - Mr Brightside"),
        SearchResult("videoRenderer", "second", "Second"),
    ]
    assert pick_first_video(results) == results[6]


def test_no_playable_result() -> None:
    assert pick_first_video([SearchResult("adSlotRenderer", None, None)]) is None


def test_browser_allowlist() -> None:
    assert host_allowed("https://www.youtube.com/watch?v=1")
    assert host_allowed("https://rr3---sn-abc.googlevideo.com/videoplayback")
    assert host_allowed("https://i.ytimg.com/vi/1/hq.jpg")
    assert not host_allowed("https://evil-youtube.com/")
    assert not host_allowed("https://googleads.g.doubleclick.net/pagead")
    assert not host_allowed("https://example.com/")
