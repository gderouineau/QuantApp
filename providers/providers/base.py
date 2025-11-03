from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date
from typing import Iterable, Optional, Dict
import pandas as pd


class ProviderError(Exception):
    pass


class RateLimitError(ProviderError):
    pass


class Provider(ABC):
    name: str

    @abstractmethod
    def fetch_bars(
            self,
            symbols: Iterable[str],
            timeframe: str,  # "1D" | "1W"
            start: Optional[date],
            end: Optional[date],
    ) -> Dict[str, pd.DataFrame]:
        ...

    def search_symbol(self, query: str) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_indicator(self, symbol: str, name: str, **params) -> pd.DataFrame:
        raise NotImplementedError
