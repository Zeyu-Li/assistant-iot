from __future__ import annotations

from conftest import TODAY

from assistant.adapters.fakes import FakeMusicPlayer, FakeWeatherProvider
from assistant.flows.music import MusicControlFlow, PlayMusicFlow
from assistant.flows.weather import WeatherFlow
from assistant.ports.flow import FlowContext


def weather_flow(provider: FakeWeatherProvider) -> WeatherFlow:
    return WeatherFlow(provider, "Mountain View", lambda: TODAY)


async def test_weather_defaults_to_home_and_today() -> None:
    provider = FakeWeatherProvider()
    text = await weather_flow(provider).run(FlowContext("weather?", {}))
    assert text == (
        "Today in Mountain View: partly cloudy, with a high of 61 and a low of 48 degrees, "
        "and a 20 percent chance of rain."
    )
    assert provider.requests == [("Mountain View", TODAY)]


async def test_weather_day_phrases() -> None:
    flow = weather_flow(FakeWeatherProvider())
    saturday = await flow.run(FlowContext("", {"date": "2026-11-28"}))
    far = await flow.run(FlowContext("", {"date": "2026-12-10"}))
    assert saturday.startswith("On Saturday in")
    assert far.startswith("On December 10 in")


async def test_weather_unknown_place() -> None:
    flow = weather_flow(FakeWeatherProvider(unknown_locations=["Atlantis"]))
    assert await flow.run(FlowContext("", {"location": "Atlantis"})) == "I couldn't find a place called Atlantis."


async def test_music_play_and_miss() -> None:
    player = FakeMusicPlayer(catalogue_miss=["zzz"])
    flow = PlayMusicFlow(player)
    assert await flow.run(FlowContext("", {"query": "jazz"})) == "Playing jazz."
    assert await flow.run(FlowContext("", {"query": "zzz"})) == "I couldn't find anything for zzz."


async def test_music_volume() -> None:
    player = FakeMusicPlayer()
    flow = MusicControlFlow(player)
    assert await flow.run(FlowContext("", {"action": "volume_up"})) == "Volume 60."
    assert await flow.run(FlowContext("", {"action": "set_volume", "level": 150})) == "Volume 100."
    assert await flow.run(FlowContext("", {"action": "set_volume"})) == "Tell me a volume between 0 and 100."
