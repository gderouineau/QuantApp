# indicators/patterns.py (simple)

import numpy as np
import pandas as pd

def hammer_inverted(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = [df[x].astype(float) for x in ("open","high","low","close")]
    body = (c - o).abs()
    upper = h - np.maximum(c, o)
    lower = np.minimum(c, o) - l
    eps = 1e-9
    cond = (upper > body * 2) & (lower < body) & (body > eps)
    return cond.astype(int)

def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = [df[x].astype(float) for x in ("open","high","low","close")]
    prev_o, prev_c = o.shift(1), c.shift(1)
    cond = (prev_c < prev_o) & (c > o) & (o <= prev_c) & (c >= prev_o)
    return cond.astype(int)
