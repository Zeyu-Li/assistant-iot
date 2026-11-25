from __future__ import annotations

from datetime import date

import httpx
import pytest

from assistant.adapters.net.allowlist import HttpClientFactory
from assistant.adapters.weather.open_meteo import HOSTS, OpenMeteoWeatherProvider
from assistant.ports.weather import ForecastUnavailable, LocationNotFound

DAY = date(2026, 11, 28)


def provider(handler: httpx.MockTransport) -> OpenMeteoWeatherProvider:
    return OpenMeteoWeatherProvider(HttpClientFactory(HOSTS).create(timeout=5, transport=handler), "fahrenheit")


def open_meteo(forecast_status: int = 200, places: list[dict] | None = None) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "geocoding-api.open-meteo.com":
            default = [{"name": "South Lake Tahoe", "latitude": 38.9, "longitude": -119.98}]
            return httpx.Response(200, json={"results": default if places is None else places})
        if forecast_status != 200:
            return httpx.Response(forecast_status, json={"error": True, "reason": "out of range"})
        return httpx.Response(200, json={"daily": {
            "time": ["2026-11-28"], "weather_code": [73], "temperature_2m_max": [34.6],
            "temperature_2m_min": [21.2], "precipitation_probability_max": [80],
        }})

    return httpx.MockTransport(handle), seen


async def test_geocodes_then_forecasts_in_profile_units() -> None:
    transport, seen = open_meteo()
    weather = provider(transport)

    forecast = await weather.daily_forecast("Tahoe", DAY)
    await weather.daily_forecast("tahoe ", DAY)  # geocoding is cached

    assert (forecast.location, forecast.summary, forecast.high, forecast.precipitation_chance) == ("South Lake Tahoe", "snow", 34.6, 80)
    hosts = [r.url.host for r in seen]
    assert hosts.count("geocoding-api.open-meteo.com") == 1
    params = seen[1].url.params
    assert params["temperature_unit"] == "fahrenheit" and params["start_date"] == params["end_date"] == "2026-11-28"


async def test_picks_most_populous_match_and_honours_pinned_places() -> None:
    transport, seen = open_meteo(places=[
        {"name": "Paris", "latitude": 33.66, "longitude": -95.55, "population": 25000},
        {"name": "Paris", "latitude": 48.85, "longitude": 2.35, "population": 2138551},
    ])
    weather = OpenMeteoWeatherProvider(
        HttpClientFactory(HOSTS).create(timeout=5, transport=transport), "celsius", {"Home Town": (1.0, 2.0)}
    )

    await weather.daily_forecast("Paris", DAY)
    await weather.daily_forecast("home town", DAY)

    forecasts = [r for r in seen if r.url.host == "api.open-meteo.com"]
    assert forecasts[0].url.params["latitude"] == "48.85"
    assert forecasts[1].url.params["latitude"] == "1.0"
    assert sum(r.url.host == "geocoding-api.open-meteo.com" for r in seen) == 1


async def test_unknown_place_and_out_of_range_day() -> None:
    with pytest.raises(LocationNotFound):
        await provider(open_meteo(places=[])[0]).daily_forecast("Atlantis", DAY)
    with pytest.raises(ForecastUnavailable):
        await provider(open_meteo(forecast_status=400)[0]).daily_forecast("Tahoe", date(2027, 6, 1))
