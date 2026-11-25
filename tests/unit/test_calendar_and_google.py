from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import httpx
import pytest
from conftest import TODAY

from assistant.adapters.fakes import FakeToolClient
from assistant.adapters.google.oauth import FileStore, GoogleTokenProvider, NotLoggedIn, StoredCredentials
from assistant.adapters.net.allowlist import HttpClientFactory
from assistant.flows.calendar import CreateEventFlow, UpdateEventFlow
from assistant.flows.speech import time_phrase, when_phrase
from assistant.ports.flow import DescribesAction, FlowContext
from assistant.ports.tools import ToolSpec


def calendar(list_response: str = '{"events": []}') -> FakeToolClient:
    tools = [ToolSpec(n, "") for n in ("list_events", "create_event", "update_event")]
    return FakeToolClient(tools, {"list_events": list_response})


def test_spoken_times() -> None:
    from datetime import time

    assert [time_phrase(time(12)), time_phrase(time(0)), time_phrase(time(15, 30)), time_phrase(time(9))] == [
        "noon", "midnight", "3:30 PM", "9 AM"
    ]
    assert when_phrase("2026-11-26T18:00:00", TODAY) == "tomorrow at 6 PM"
    assert when_phrase("2026-11-28", TODAY) == "on Saturday"


async def test_create_defaults_to_one_hour_and_describes_itself() -> None:
    cal = calendar()
    flow = CreateEventFlow(cal, lambda: TODAY)
    args = {"summary": "Lunch with Sam", "startTime": "2026-11-27T12:00:00"}

    assert isinstance(flow, DescribesAction)
    assert flow.describe(args) == "Add Lunch with Sam on Friday at noon?"
    reply = await flow.run(FlowContext("add lunch", args))

    assert reply == "Done. Lunch with Sam is on your calendar on Friday at noon."
    assert cal.calls == [("create_event", {"summary": "Lunch with Sam", "startTime": "2026-11-27T12:00:00", "endTime": "2026-11-27T13:00:00"})]


async def test_update_needs_exactly_one_match() -> None:
    args = {"event": "dentist", "newStartTime": "2026-11-28T15:00:00"}

    none = await UpdateEventFlow(calendar(), lambda: TODAY).run(FlowContext("", args))
    many_cal = calendar(json.dumps({"events": [{"id": "a", "summary": "Dentist"}, {"id": "b", "summary": "Dentist follow-up"}]}))
    many = await UpdateEventFlow(many_cal, lambda: TODAY).run(FlowContext("", args))
    one_cal = calendar(json.dumps({"events": [{"id": "evt1", "summary": "Dentist"}]}))
    one_flow = UpdateEventFlow(one_cal, lambda: TODAY)
    one = await one_flow.run(FlowContext("", args))

    assert none == "I couldn't find an upcoming event matching dentist."
    assert many.startswith("I found 2 upcoming events") and [c[0] for c in many_cal.calls] == ["list_events"]
    assert one == "Done, I've updated it."
    assert one_cal.calls[-1] == ("update_event", {"eventId": "evt1", "startTime": "2026-11-28T15:00:00"})
    assert one_flow.describe(args) == "Should I move your dentist event to Saturday at 3 PM?"
    assert one_flow.describe({"event": "sync", "newSummary": "planning"}) == "Should I rename your sync event to planning?"


async def test_update_never_touches_an_event_the_user_did_not_name() -> None:
    # A search that returns an unrelated event (as a misrouted "add lunch with Sam" once did).
    cal = calendar(json.dumps({"events": [{"id": "evt1", "summary": "Dentist"}]}))
    reply = await UpdateEventFlow(cal, lambda: TODAY).run(
        FlowContext("", {"event": "lunch with Sam", "newStartTime": "2026-11-27T12:00:00"})
    )
    assert reply == "I couldn't find an upcoming event matching lunch with Sam."
    assert [c[0] for c in cal.calls] == ["list_events"]


def test_restated_name_is_not_a_rename() -> None:
    flow = UpdateEventFlow(calendar(), lambda: TODAY)
    args = {"event": "dentist", "newStartTime": "2026-11-28T15:00:00", "newSummary": "dentist appointment"}
    assert flow.describe(args) == "Should I move your dentist event to Saturday at 3 PM?"


class FakeTokenEndpoint:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "invalid_grant"})
        return httpx.Response(200, json={"access_token": f"access-{self.calls}", "expires_in": 3600})


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def provider(tmp_path: Path, endpoint: FakeTokenEndpoint, clock: Clock, stored: Optional[StoredCredentials]) -> GoogleTokenProvider:
    store = FileStore(tmp_path / "token.json")
    if stored is not None:
        store.save(stored)
    client = HttpClientFactory(("oauth2.googleapis.com",)).create(timeout=5, transport=httpx.MockTransport(endpoint))
    return GoogleTokenProvider(store, client, clock)


CREDS = StoredCredentials(client_id="id", client_secret="secret", refresh_token="refresh", scopes=("s",))


async def test_token_is_cached_until_near_expiry(tmp_path: Path) -> None:
    endpoint, clock = FakeTokenEndpoint(), Clock()
    tokens = provider(tmp_path, endpoint, clock, CREDS)

    first = await tokens()
    clock.now = 3000
    cached = await tokens()
    clock.now = 3550  # inside the 60 s safety margin
    refreshed = await tokens()

    assert (first, cached, refreshed) == ("access-1", "access-1", "access-2")
    assert (tmp_path / "token.json").stat().st_mode & 0o777 == 0o600


async def test_missing_or_revoked_login_says_how_to_fix(tmp_path: Path) -> None:
    with pytest.raises(NotLoggedIn, match="google-login"):
        await provider(tmp_path, FakeTokenEndpoint(), Clock(), None)()
    with pytest.raises(NotLoggedIn, match="invalid_grant"):
        await provider(tmp_path / "b", FakeTokenEndpoint(status=400), Clock(), CREDS)()
