# strategies/relative_strength.py

import numpy as np
import pandas as pd
from .base import BaseStrategy


class RelativeStrengthStrategy(BaseStrategy):
    """
    Détecte les actions fortes : momentum, alignement MAs et proximité du plus haut 52W.
    Paramètres possibles :
        lookback_days: int = 63         # ~3 mois
        hi_52w_days: int = 252          # ~1 an boursier
        near_high_max_drawdown: float = 0.15  # tolérance sous le high (15%)
        require_liquidity: bool = True
        min_price: float = 1.0
        min_vol_sma20: float = 100_000
    Le score additionne 3 critères (0..70 dans cette version simple).
    """

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.p = {
            "lookback_days": 63,
            "hi_52w_days": 252,
            "near_high_max_drawdown": 0.15,
            "require_liquidity": True,
            "min_price": 1.0,
            "min_vol_sma20": 100_000,
        }
        if parameters:
            self.p.update(parameters)

    def _ensure_ma(self, indicators: pd.DataFrame, asset_close: pd.Series, window: int) -> pd.Series:
        col = f"sma_{window}"
        if col in indicators.columns and indicators[col].notna().any():
            return indicators[col].astype(float)
        # fallback si absent dans indicators
        return asset_close.rolling(window).mean()

    def evaluate(self, asset_data: pd.DataFrame, indicators: pd.DataFrame) -> dict:
        # Garde-fous de base
        needed_cols = {"close", "high", "volume"}
        if not needed_cols.issubset(asset_data.columns):
            return {'signal': False, 'score': 0.0, 'strength': 'WEAK',
                    'details': {'error': f"missing columns: {sorted(needed_cols - set(asset_data.columns))}"}}

        close = asset_data["close"].astype(float)
        high = asset_data["high"].astype(float)
        volume = asset_data["volume"].astype(float)

        n = len(asset_data)
        lb = int(self.p["lookback_days"])
        hi_days = int(self.p["hi_52w_days"])

        if n < max(hi_days, lb) + 5:
            return {'signal': False, 'score': 0.0, 'strength': 'WEAK',
                    'details': {'reason': 'not_enough_data', 'len': n}}

        last = float(close.iloc[-1])

        # Filtres simples prix/liquidité
        if last < float(self.p["min_price"]):
            return {'signal': False, 'score': 0.0, 'strength': 'WEAK',
                    'details': {'reason': 'min_price', 'last': last}}

        if self.p["require_liquidity"]:
            vol_sma20 = float(volume.rolling(20).mean().iloc[-1])
            if np.isnan(vol_sma20) or vol_sma20 < float(self.p["min_vol_sma20"]):
                return {'signal': False, 'score': 0.0, 'strength': 'WEAK',
                        'details': {'reason': 'liquidity', 'vol_sma20': vol_sma20}}

        # Critère 1 : Performance sur la période (0..20 pts)
        perf_asset = float(last / float(close.iloc[-lb]) - 1.0)
        score = 0.0
        details = {'performance': perf_asset}

        if perf_asset > 0:
            score += 20.0

        # Critère 3 : Alignement des moyennes + close > SMA50 (0..25 pts)
        sma50 = self._ensure_ma(indicators, close, 50)
        sma200 = self._ensure_ma(indicators, close, 200)

        try:
            sma_aligned = bool(sma50.iloc[-1] > sma200.iloc[-1] and last > float(sma50.iloc[-1]))
        except Exception:
            sma_aligned = False

        if sma_aligned:
            score += 25.0
            details['sma_aligned'] = True
            details['sma50'] = float(sma50.iloc[-1])
            details['sma200'] = float(sma200.iloc[-1])

        # Critère 4 : Proche du plus haut 52W (0..25 pts)
        hi_window = high.iloc[-hi_days:] if n >= hi_days else high
        hi_52w = float(hi_window.max())
        if hi_52w > 0:
            distance_from_high = float(last / hi_52w - 1.0)  # négatif si sous le high
            details['distance_from_high'] = distance_from_high
            if distance_from_high >= -float(self.p["near_high_max_drawdown"]):
                score += 25.0
                details['near_52w_high'] = True
                details['hi_52w'] = hi_52w

        # Force & signal
        strength = self.strength_from_score(score)
        return {
            'signal': score >= 50.0,
            'score': float(score),
            'strength': strength,
            'details': details
        }
