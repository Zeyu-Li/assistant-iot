"""Weather forecast port."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Protocol


@dataclass(frozen=True)
class DailyForecast:
    location: str
    day: date
    summary: str
    high: float
    low: float
    precipitation_chance: Optional[int] = None


class LocationNotFound(Exception):
    pass


class ForecastUnavailable(Exception):
    """The provider has no forecast for the requested day (for example, too far ahead)."""


class WeatherProvider(Protocol):
    async def daily_forecast(self, location: str, day: date) -> DailyForecast: ...
