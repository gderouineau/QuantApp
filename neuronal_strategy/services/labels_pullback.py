from __future__ import annotations
import numpy as np
import pandas as pd

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def _bb(close: pd.Series, n=20, k=2.0):
    mid = _sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    up = mid + k * std
    lo = mid - k * std
    return mid, up, lo

def _rolling_min(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).min()

def _rolling_max(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).max()

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.copy()
    if "date" in out.columns:
        out.index = pd.to_datetime(out["date"], errors="coerce", utc=True)
    else:
        out.index = pd.to_datetime(out.index, errors="coerce", utc=True)
    if out.index.isna().all():
        out.index = df.index
    return out

def compute_pullback_labels(
    df: pd.DataFrame,
    horizon_bars: int = 30,
    tp_Rs=(1.5, 2.0, 3.0),
    bb_k=2.0,
    stop_lookback=3,
) -> pd.DataFrame:
    """
    Labels 'first-touch' pour stratégie pullback:
      - Biais long si SMA20 > SMA50 (sinon short).
      - Setup si (biais long et low < BB_lower20_k) OU (biais short et high > BB_upper20_k).
      - Entrée si break confirmé: long: close[t] > high[t-1]; short: close[t] < low[t-1].
      - Stop: min/max des N dernières barres (hors barre courante).
      - Labels y_{m}R True si le TP (m*R) est touché AVANT le SL dans (t+1..t+horizon), en scannant barre par barre.

    Sortie: DataFrame indexé comme df, colonnes:
      ['trend_up','is_setup','is_entry','entry','stop','R','y_1p5R','y_2R','y_3R']
    """
    df = _ensure_dt_index(df)
    idx = df.index

    # Prix (numpy-friendly + Series alignées)
    o = pd.to_numeric(df.get("open"),  errors="coerce")
    h = pd.to_numeric(df.get("high"),  errors="coerce")
    l = pd.to_numeric(df.get("low"),   errors="coerce")
    c = pd.to_numeric(df.get("close"), errors="coerce")

    # Indicateurs
    sma20 = _sma(c, 20)
    sma50 = _sma(c, 50)
    bb_mid, bb_up, bb_lo = _bb(c, 20, bb_k)

    trend_up = (sma20 > sma50)

    # Setup pullback
    setup_long = trend_up & (l < bb_lo)
    setup_shrt = (~trend_up) & (h > bb_up)

    # Entrées (break)
    prev_high = h.shift(1)
    prev_low  = l.shift(1)
    entry_long = setup_long & (c > prev_high)
    entry_shrt = setup_shrt & (c < prev_low)
    is_entry = (entry_long | entry_shrt)

    # Prix d'entrée et stop
    entry_price = pd.Series(np.nan, index=idx, dtype="float64")
    entry_price.loc[entry_long] = c.loc[entry_long]
    entry_price.loc[entry_shrt] = c.loc[entry_shrt]

    stop_long = _rolling_min(l.shift(1), stop_lookback)
    stop_shrt = _rolling_max(h.shift(1), stop_lookback)
    stop_raw = pd.Series(np.nan, index=idx, dtype="float64")
    stop_raw.loc[entry_long] = stop_long.loc[entry_long]
    stop_raw.loc[entry_shrt] = stop_shrt.loc[entry_shrt]

    R = (entry_price - stop_raw).abs()

    # Prépare arrays pour scan séquentiel
    highA = h.values
    lowA  = l.values
    n = len(df)

    # Positions d’entrée (indices entiers)
    entry_pos = np.flatnonzero(is_entry.values)

    # Containers résultats (False par défaut)
    y_map = {}
    # On veut toujours ces 3 noms dans la sortie, même si tp_Rs changé
    wanted = {1.5: "y_1p5R", 2.0: "y_2R", 3.0: "y_3R"}
    for r, name in wanted.items():
        y_map[name] = pd.Series(False, index=idx)

    # Scan bar-by-bar "first touch"
    for i in entry_pos:
        e = entry_price.iat[i]
        st = stop_raw.iat[i]
        r = R.iat[i]
        if not np.isfinite(e) or not np.isfinite(st) or not np.isfinite(r) or r <= 0:
            continue

        long_side = bool(entry_long.iat[i])

        # Pour chaque multiple demandé, on teste indépendamment si TP avant SL
        for mult in tp_Rs:
            # nom de colonne
            col = wanted.get(mult, f"y_{str(mult).replace('.','p')}R")
            if col not in y_map:
                y_map[col] = pd.Series(False, index=idx)

            if long_side:
                tp = e + mult * r
                # scanner j = i+1 .. i+horizon
                hit = False
                for j in range(i + 1, min(i + 1 + horizon_bars, n)):
                    # SL long si low <= stop
                    if lowA[j] <= st:
                        hit = False
                        break
                    # TP long si high >= tp
                    if highA[j] >= tp:
                        hit = True
                        break
                if hit:
                    y_map[col].iat[i] = True
            else:
                tp = e - mult * r
                hit = False
                for j in range(i + 1, min(i + 1 + horizon_bars, n)):
                    # SL short si high >= stop
                    if highA[j] >= st:
                        hit = False
                        break
                    # TP short si low <= tp
                    if lowA[j] <= tp:
                        hit = True
                        break
                if hit:
                    y_map[col].iat[i] = True

    # DataFrame de sortie
    out = pd.DataFrame(index=idx)
    out["trend_up"] = trend_up.fillna(False).astype(bool)
    out["is_setup"] = (setup_long | setup_shrt).fillna(False).astype(bool)
    out["is_entry"] = is_entry.fillna(False).astype(bool)
    out["entry"]    = entry_price.astype("float64")
    out["stop"]     = stop_raw.astype("float64")
    out["R"]        = R.astype("float64")
    # y_*R
    for _, name in sorted(((k, v) for k, v in wanted.items()), key=lambda t: t[0]):
        out[name] = y_map[name].fillna(False).astype(bool)
    # Ajoute colonnes supplémentaires si tp_Rs non standard
    for mult in tp_Rs:
        name = wanted.get(mult, f"y_{str(mult).replace('.','p')}R")
        if name not in out.columns:
            out[name] = y_map[name].fillna(False).astype(bool)

    return out
