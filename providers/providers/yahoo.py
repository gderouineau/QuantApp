# providers/providers/yahoo.py
from __future__ import annotations

from datetime import date
from typing import Iterable, Optional, Dict
import pandas as pd
import yfinance as yf

from .base import Provider, RateLimitError

_SPECS = {"1D": "1d", "1W": "1wk"}


def _clean_bars_df(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage robuste: typage, tri, drop NaN OHLC, volume NaN->0, index propre."""
    if df is None or df.empty:
        return df

    # Toujours travailler sur une copie pour éviter SettingWithCopyWarning
    df = df.copy()

    # S'assurer que l'index est un DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    # Si timezone-aware -> convertir UTC puis rendre naïf
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    # Forcer numérique (tolérant)
    for col in ["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

    # Fallback adj_close si absent/NaN
    if "adj_close" not in df.columns and "close" in df.columns:
        df.loc[:, "adj_close"] = df["close"]
    elif "adj_close" in df.columns and "close" in df.columns:
        # Remplace adj_close NaN par close si besoin
        mask = df["adj_close"].isna()
        if mask.any():
            df.loc[mask, "adj_close"] = df.loc[mask, "close"]

    # Drop lignes invalides si un OHLC est NaN
    subset = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if subset:
        df = df.dropna(subset=subset, how="any")

    # Volume NaN -> 0 (puis int64)
    if "volume" in df.columns:
        df.loc[:, "volume"] = df["volume"].fillna(0).astype("int64")

    # Typage OHLC en float64
    for col in ["open", "high", "low", "close", "adj_close"]:
        if col in df.columns:
            df.loc[:, col] = df[col].astype("float64")

    # Tri + dédup + nom d'index
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "date"
    return df


class YahooProvider(Provider):
    name = "yahoo"

    def _download(
        self,
        symbols: list[str],
        interval: str,
        start: Optional[date],
        end: Optional[date],
    ) -> Dict[str, pd.DataFrame]:
        if not symbols:
            return {}
        try:
            data = yf.download(
                tickers=" ".join(symbols),
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                actions=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            raise RateLimitError(str(e))

        out: Dict[str, pd.DataFrame] = {}
        cols = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "splits",
        }

        if isinstance(data.columns, pd.MultiIndex):
            for sym in symbols:
                if sym not in data.columns.get_level_values(0):
                    continue
                df = data[sym].rename(columns=cols).copy()
                df = _clean_bars_df(df)
                if not df.empty:
                    out[sym] = df
        else:
            df = data.rename(columns=cols).copy()
            df = _clean_bars_df(df)
            if symbols and not df.empty:
                out[symbols[0]] = df

        return out

    def fetch_bars(
        self,
        symbols: Iterable[str],
        timeframe: str,
        start: Optional[date],
        end: Optional[date],
    ) -> Dict[str, pd.DataFrame]:
        return self._download(list(symbols), _SPECS[timeframe], start, end)
