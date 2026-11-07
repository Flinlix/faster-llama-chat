"""A tool written in a module that postpones its annotations, as many do."""

from __future__ import annotations


def weather(city: str, days: int = 1) -> str:
    """Weather for a city."""
    return f"sunny in {city} for {days} day(s)"
