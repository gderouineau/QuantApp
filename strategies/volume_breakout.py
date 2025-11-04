# strategies/volume_breakout.py

import numpy as np
import pandas as pd
from .base import BaseStrategy


class VolumeBreakoutStrategy(BaseStrategy):
    """
    Cassure du plus haut N jours avec volume confirmé.
    Conditions:
      close[-1] > max(high[-N:-1])  (cassure)
      volume[-1] > k * mean(volume[-M:])
    Score:
      +50 si cassure
      +30 si volume >= k * vol_smaM (linéaire jusqu'à 2*k)
      +20 si close > close[-2] (confirmation)
    Seuil signal: score >= 50
    """

    def __init__(self, parameters=None):
        super().__init__(parameters or {})
        self.p = {
            "lookback_high": 20,   # N
            "vol_window": 20,      # M
            "vol_mult": 2.0,       # k
            "min_price": 1.0,
            "require_liquidity": True,
            "min_vol_sma20": 100_000,
        }
        self.p.update(self.parameters)

    def evaluate(self, asset_data: pd.DataFrame, indicators: pd.DataFrame) -> dict:
        need = {"close", "high", "volume"}
        if not need.issubset(asset_data.columns):
            return {"signal": False, "score": 0.0, "strength": "WEAK",
                    "details": {"error": f"missing columns: {sorted(need - set(asset_data.columns))}"}}

        close = asset_data["close"].astype(float)
        high = asset_data["high"].astype(float)
        volume = asset_data["volume"].astype(float)

        n = len(close)
        N = int(self.p["lookback_high"])
        M = int(self.p["vol_window"])
        if n < max(N + 2, M + 1):
            return {"signal": False, "score": 0.0, "strength": "WEAK", "details": {"reason": "not_enough_data"}}

        last = float(close.iloc[-1])
        if last < float(self.p["min_price"]):
            return {"signal": False, "score": 0.0, "strength": "WEAK", "details": {"reason": "min_price"}}

        vol_sma = float(volume.rolling(M).mean().iloc[-1])
        if self.p["require_liquidity"]:
            if np.isnan(vol_sma) or vol_sma < float(self.p["min_vol_sma20"]):
                return {"signal": False, "score": 0.0, "strength": "WEAK",
                        "details": {"reason": "liquidity", "vol_sma": vol_sma}}

        prev_max = float(high.iloc[-(N+1):-1].max())
        breakout = last > prev_max
        details = {"prev_max_N": prev_max, "vol_sma": vol_sma, "breakout": breakout}

        score = 0.0
        if breakout:
            score += 50.0

            vol_ratio = float(volume.iloc[-1]) / vol_sma if vol_sma == vol_sma and vol_sma > 0 else 0.0
            # 0..30 points supplémentaires de  k -> 2k
            k = float(self.p["vol_mult"])
            if vol_ratio >= k:
                bonus = min(30.0, 30.0 * (vol_ratio - k) / k)
                score += max(0.0, bonus)
                details["vol_ratio"] = vol_ratio

            if last > float(close.iloc[-2]):
                score += 20.0
                details["close_confirms"] = True

        strength = "STRONG" if score >= 70 else ("MEDIUM" if score >= 50 else "WEAK")
        return {"signal": score >= 50.0, "score": float(score), "strength": strength, "details": details}
