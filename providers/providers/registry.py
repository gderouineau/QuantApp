from __future__ import annotations
from django.conf import settings
from .yahoo import YahooProvider
from .alphavantage import AlphaVantageProvider

_REGISTRY = {
    "yahoo": YahooProvider(),
    "alphavantage": AlphaVantageProvider(),
}

def get_provider(name: str):
    return _REGISTRY[name]

def provider_chain() -> list[str]:
    return list(getattr(settings, "MARKET_DATA_PROVIDERS", ["yahoo"]))
