# strategies/golden_cross.py

import numpy as np
import pandas as pd
from .base import BaseStrategy


class GoldenCrossStrategy(BaseStrategy):
    """
    Golden Cross : SMA50 croise au-dessus de SMA200.
    Score:
      +70 si croisement aujourd'hui (SMA50[-2] < SMA200[-2] et SMA50[-1] > SMA200[-1])
      +15 si close > SMA50
      +15 si volume du jour > 1.5 * vol_sma20
    Seuil signal: score >= 50
    """

    def __init__(self, parameters=None):
        super().__init__(parameters or {})
        self.p = {
            "vol_mult": 1.5,
            "require_liquidity": True,
            "min_vol_sma20": 100_000,
            "min_price": 1.0,
        }
        self.p.update(self.parameters)

    def _ensure_ma(self, indicators: pd.DataFrame, close: pd.Series, window: int) -> pd.Series:
        col = f"sma_{window}"
        if col in indicators.columns and indicators[col].notna().any():
            return indicators[col].astype(float)
        return close.rolling(window).mean()

    def evaluate(self, asset_data: pd.DataFrame, indicators: pd.DataFrame) -> dict:
        need = {"close", "volume"}
        if not need.issubset(asset_data.columns):
            return {"signal": False, "score": 0.0, "strength": "WEAK",
                    "details": {"error": f"missing columns: {sorted(need - set(asset_data.columns))}"}}

        close = asset_data["close"].astype(float)
        volume = asset_data["volume"].astype(float)
        if len(close) < 205:  # un peu de marge
            return {"signal": False, "score": 0.0, "strength": "WEAK", "details": {"reason": "not_enough_data"}}

        last = float(close.iloc[-1])
        if last < float(self.p["min_price"]):
            return {"signal": False, "score": 0.0, "strength": "WEAK", "details": {"reason": "min_price"}}

        if self.p["require_liquidity"]:
            vol_sma20 = float(volume.rolling(20).mean().iloc[-1])
            if np.isnan(vol_sma20) or vol_sma20 < float(self.p["min_vol_sma20"]):
                return {"signal": False, "score": 0.0, "strength": "WEAK",
                        "details": {"reason": "liquidity", "vol_sma20": vol_sma20}}
        else:
            vol_sma20 = float(volume.rolling(20).mean().iloc[-1])

        sma50 = self._ensure_ma(indicators, close, 50)
        sma200 = self._ensure_ma(indicators, close, 200)
        if sma50.isna().iloc[-2] or sma200.isna().iloc[-2]:
            return {"signal": False, "score": 0.0, "strength": "WEAK", "details": {"reason": "ma_nan"}}

        prev_cross = float(sma50.iloc[-2]) < float(sma200.iloc[-2])
        curr_cross = float(sma50.iloc[-1]) > float(sma200.iloc[-1])
        cross_up = bool(prev_cross and curr_cross)

        score = 70.0 if cross_up else 0.0
        details = {
            "sma50_prev": float(sma50.iloc[-2]),
            "sma200_prev": float(sma200.iloc[-2]),
            "sma50": float(sma50.iloc[-1]),
            "sma200": float(sma200.iloc[-1]),
            "cross_up": cross_up,
            "vol_sma20": vol_sma20,
        }

        if last > float(sma50.iloc[-1]):
            score += 15.0
            details["close_above_sma50"] = True

        vol_ok = float(volume.iloc[-1]) > vol_sma20 * float(self.p["vol_mult"])
        if vol_ok:
            score += 15.0
            details["volume_spike"] = True
            details["last_vol"] = float(volume.iloc[-1])

        strength = "STRONG" if score >= 70 else ("MEDIUM" if score >= 50 else "WEAK")
        return {"signal": score >= 50.0, "score": float(score), "strength": strength, "details": details}
