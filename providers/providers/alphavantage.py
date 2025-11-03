from __future__ import annotations
from datetime import date
from typing import Iterable, Optional, Dict
import httpx
import pandas as pd
from django.conf import settings

from .base import Provider, RateLimitError, ProviderError

class AlphaVantageProvider(Provider):
    name = "alphavantage"
    BASE = "https://www.alphavantage.co/query"

    def _get(self, params: dict) -> dict:
        if not settings.ALPHAVANTAGE_API_KEY:
            raise ValueError("Missing ALPHAVANTAGE_API_KEY")
        params = {**params, "apikey": settings.ALPHAVANTAGE_API_KEY}
        r = httpx.get(self.BASE, params=params, timeout=30)
        if r.status_code == 429:
            raise RateLimitError("429 Too Many Requests")
        r.raise_for_status()
        js = r.json()
        if js.get("Error Message"):
            raise ProviderError(js["Error Message"])
        if js.get("Information"):
            raise RateLimitError(js["Information"])
        if js.get("Note"):
            raise RateLimitError(js["Note"])
        return js

    def fetch_bars(self, symbols, timeframe, start, end):
        out = {}
        function = "TIME_SERIES_DAILY_ADJUSTED" if timeframe == "1D" else "TIME_SERIES_WEEKLY_ADJUSTED"
        key = "Time Series (Daily)" if timeframe == "1D" else "Weekly Adjusted Time Series"

        # Heuristique simple: si start est fourni -> on demande l'historique complet
        outputsize = "full" if start else "compact"

        for sym in symbols:
            js = self._get({"function": function, "symbol": sym, "datatype": "json", "outputsize": outputsize})
            ser = js.get(key) or {}
            if not ser:
                continue
            rows = []
            for d, val in ser.items():
                rows.append({
                    "date": pd.to_datetime(d),
                    "open": float(val.get("1. open", 0)),
                    "high": float(val.get("2. high", 0)),
                    "low": float(val.get("3. low", 0)),
                    "close": float(val.get("4. close", 0)),
                    "adj_close": float(val.get("5. adjusted close", val.get("4. close", 0))),
                    "volume": float(val.get("6. volume", 0)),
                })
            df = pd.DataFrame(rows).sort_values("date").set_index("date")
            if start:
                df = df[df.index.date >= start]
            if end:
                df = df[df.index.date <= end]
            out[sym] = df
        return out

    def search_symbol(self, query: str) -> pd.DataFrame:
        js = self._get({"function": "SYMBOL_SEARCH", "keywords": query})
        best = js.get("bestMatches", []) or []
        rows = []
        for m in best:
            rows.append({
                "symbol": m.get("1. symbol"),
                "name": m.get("2. name"),
                "type": m.get("3. type"),
                "region": m.get("4. region"),
                "currency": m.get("8. currency"),
            })
        return pd.DataFrame(rows)

    def fetch_indicator(self, symbol: str, name: str, **params) -> pd.DataFrame:
        av_params = {"function": name.upper(), "symbol": symbol, **params}
        js = self._get(av_params)
        data_key = next((k for k in js.keys() if "Technical Analysis" in k), None)
        if not data_key:
            return pd.DataFrame()
        ser = js[data_key]
        rows = []
        for d, val in ser.items():
            val = {k.lower().replace(" ", "_"): float(v) for k, v in val.items()}
            rows.append({"date": pd.to_datetime(d), **val})
        return pd.DataFrame(rows).sort_values("date").set_index("date")
