"""WeatherProvider backed by Open-Meteo (no API key; geocoding + daily forecast)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Mapping, Optional

import httpx

from assistant.ports.weather import DailyForecast, ForecastUnavailable, LocationNotFound

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOSTS = ("geocoding-api.open-meteo.com", "api.open-meteo.com")
FORECAST_DAYS = 16

# WMO weather interpretation codes, as documented by Open-Meteo.
_WMO = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy with frost",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "rain showers", 81: "rain showers", 82: "heavy rain showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}


@dataclass(frozen=True)
class _Place:
    name: str
    latitude: float
    longitude: float


class OpenMeteoWeatherProvider:
    def __init__(
        self,
        client: httpx.AsyncClient,
        units: Literal["celsius", "fahrenheit"],
        pinned_places: Optional[Mapping[str, tuple[float, float]]] = None,
    ) -> None:
        """``pinned_places`` maps a spoken name to coordinates, skipping geocoding (use it for home)."""
        self._client = client
        self._units = units
        self._places: dict[str, _Place] = {
            name.strip().lower(): _Place(name, lat, lon) for name, (lat, lon) in (pinned_places or {}).items()
        }

    async def daily_forecast(self, location: str, day: date) -> DailyForecast:
        place = await self._geocode(location)
        response = await self._client.get(
            FORECAST_URL,
            params={
                "latitude": place.latitude,
                "longitude": place.longitude,
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "temperature_unit": self._units,
                "timezone": "auto",
                "start_date": day.isoformat(),
                "end_date": day.isoformat(),
            },
        )
        if response.status_code == 400:  # Open-Meteo's answer for dates outside its window
            raise ForecastUnavailable(response.json().get("reason", "date out of range"))
        response.raise_for_status()
        daily: dict[str, list[Any]] = response.json()["daily"]
        if not daily.get("time"):
            raise ForecastUnavailable(day.isoformat())
        chance = daily.get("precipitation_probability_max", [None])[0]
        return DailyForecast(
            location=place.name,
            day=day,
            summary=_WMO.get(daily["weather_code"][0], "mixed conditions"),
            high=float(daily["temperature_2m_max"][0]),
            low=float(daily["temperature_2m_min"][0]),
            precipitation_chance=int(chance) if chance is not None else None,
        )

    async def _geocode(self, location: str) -> _Place:
        key = location.strip().lower()
        if key not in self._places:
            response = await self._client.get(
                GEOCODING_URL, params={"name": location, "count": 10, "language": "en", "format": "json"}
            )
            response.raise_for_status()
            results = response.json().get("results") or []
            if not results:
                raise LocationNotFound(location)
            # A bare spoken name almost always means the best-known place of that name.
            top = max(results, key=lambda r: r.get("population") or 0)
            self._places[key] = _Place(top["name"], float(top["latitude"]), float(top["longitude"]))
        return self._places[key]
