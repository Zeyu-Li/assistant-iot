from __future__ import annotations

from datetime import date
from typing import Any, Callable, Optional

from assistant.flows.speech import day_phrase
from assistant.ports.flow import FlowContext
from assistant.ports.weather import ForecastUnavailable, LocationNotFound, WeatherProvider


class WeatherFlow:
    def __init__(self, provider: WeatherProvider, home_location: str, today: Callable[[], date]) -> None:
        self._provider = provider
        self._home_location = home_location
        self._today = today

    async def run(self, context: FlowContext) -> str:
        today = self._today()
        location = context.arguments.get("location") or self._home_location
        day = _parse_day(context.arguments.get("date")) or today
        when = day_phrase(day, today)
        try:
            forecast = await self._provider.daily_forecast(location, day)
        except LocationNotFound:
            return f"I couldn't find a place called {location}."
        except ForecastUnavailable:
            return f"I don't have a forecast for {when} yet."

        sentence = (
            f"{when[0].upper()}{when[1:]} in {forecast.location}: {forecast.summary}, "
            f"with a high of {round(forecast.high)} and a low of {round(forecast.low)} degrees"
        )
        if forecast.precipitation_chance:
            chance = forecast.precipitation_chance
            article = "an" if chance in (8, 11, 18) or str(chance).startswith("8") else "a"
            sentence += f", and {article} {chance} percent chance of rain"
        return sentence + "."


def _parse_day(value: Any) -> Optional[date]:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


