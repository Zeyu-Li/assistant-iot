"""The real registry and profiles load and wire up."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from conftest import REPO_ROOT

from assistant.app import build_assistant
from assistant.config import load_profile
from assistant.core.alerts import Alert
from assistant.core.listening import ListeningMode
from assistant.core.registry import Capability, Registry


def test_registry_loads_and_disables(capabilities_path: Path) -> None:
    full = Registry.load(capabilities_path)
    trimmed = Registry.load(capabilities_path, disabled=["chat"])
    assert len(trimmed) == len(full) - 1
    with pytest.raises(ValueError, match="unknown"):
        Registry.load(capabilities_path, disabled=["nope"])


def test_capability_rejects_reserved_id_and_bad_binding() -> None:
    with pytest.raises(ValueError):
        Capability.model_validate({"id": "none", "kind": "flow", "description": "x"})
    with pytest.raises(ValueError):
        Capability.model_validate({"id": "x", "kind": "mcp", "description": "x"})


async def test_fake_profile_end_to_end(capabilities_path: Path) -> None:
    assistant = await build_assistant(load_profile(REPO_ROOT / "config/profiles/fake.yaml"), capabilities_path)

    weather = await assistant.handle_utterance("what is the weather forecast", wake_word_detected=True)
    assert weather is not None and weather.text.startswith("Today in Mountain View")

    assert await assistant.handle_utterance("overheard", wake_word_detected=False) is not None  # follow-up window

    question = await assistant.handle_utterance("play a song", wake_word_detected=True)
    assert question is not None and question.expects_response

    assistant.listening.set_mode(ListeningMode.MUTED)
    assert await assistant.handle_utterance("what is the weather", wake_word_detected=True) is None

    assistant.listening.set_mode(ListeningMode.WAKE_WORD)  # the pending "what to play" was dropped
    weather_again = await assistant.handle_utterance("what is the weather forecast", wake_word_detected=True)
    assert weather_again is not None and weather_again.text.startswith("Today")


async def test_open_mic_stays_quiet_on_unmatched_speech(capabilities_path: Path) -> None:
    assistant = await build_assistant(load_profile(REPO_ROOT / "config/profiles/fake.yaml"), capabilities_path)
    assistant.listening.set_mode(ListeningMode.OPEN_MIC)
    assistant.listening.open_follow_up()  # right after a reply, when the "for me?" check is skipped

    assert await assistant.handle_utterance("the kids are watching a movie", wake_word_detected=False) is None
    woken = await assistant.handle_utterance("the kids are watching a movie", wake_word_detected=True)
    assert woken is not None and woken.text == "Sorry, I can't help with that."


async def test_startup_fails_when_pinned_tool_is_missing(capabilities_path: Path, tmp_path: Path) -> None:
    profile_text = (REPO_ROOT / "config/profiles/fake.yaml").read_text().replace("search_threads:", "other_tool:")
    profile_path = tmp_path / "broken.yaml"
    profile_path.write_text(profile_text)

    with pytest.raises(LookupError, match="search_threads"):
        await build_assistant(load_profile(profile_path), capabilities_path)


async def test_speech_while_an_alert_rings_dismisses_it(capabilities_path: Path) -> None:
    assistant = await build_assistant(load_profile(REPO_ROOT / "config/profiles/fake.yaml"), capabilities_path)
    assistant.alerts.ringing = Alert("t1", "timer", datetime(2026, 11, 25, tzinfo=timezone.utc))

    reply = await assistant.handle_utterance("stop", wake_word_detected=True)

    assert reply is not None and reply.text == "Okay."
    assistant.alerts.ringing = None
    again = await assistant.handle_utterance("tell me a joke", wake_word_detected=True)
    assert again is not None and again.text != "Okay."  # back to normal once it stopped ringing


async def test_fake_profile_keeps_data_in_memory(capabilities_path: Path) -> None:
    profile = load_profile(REPO_ROOT / "config/profiles/fake.yaml")
    assert profile.storage.adapter == "memory"
    assert load_profile(REPO_ROOT / "config/profiles/mac-dev.yaml").storage.adapter == "json_file"
