from __future__ import annotations
import pandas as pd
import numpy as np

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def _prep_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    ren = {
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "adj_close", "Adj_Close": "adj_close", "adj close": "adj_close",
        "Volume": "volume",
    }
    out = df.copy()
    out = out.rename(columns=ren)
    # colonnes minimales
    for c in ("open", "high", "low", "close"):
        if c not in out.columns:
            raise ValueError(f"Colonne manquante: {c}")
    # to numeric
    for c in ("open", "high", "low", "close"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    # index datetime utc
    if not isinstance(out.index, pd.DatetimeIndex):
        if "date" in out.columns:
            out = out.set_index("date")
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out.sort_index()
    out = out[~out.index.isna()]
    return out

def compute_trend_everybar_labels(
    df: pd.DataFrame,
    ma_fast: int = 20,
    ma_slow: int = 50,
    lookback_stop: int = 5,
    horizon_bars: int = 30,
    tps: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    """
    Par barre t:
      - Trend: is_long = SMA(ma_fast) > SMA(ma_slow) (sur close)
      - Entry: open[t+1]
      - SL:    LONG -> min(low[t-lookback+1 .. t]); SHORT -> max(high[...])
      - R:     LONG -> entry - sl;  SHORT -> sl - entry  (si R<=0 -> pas de setup)
      - Horizon: on parcourt [t+1 .. t+horizon] et on marque le 1er évènement atteint (SL / TP1 / TP2 / TP3).
      - y = (TP1 atteint avant SL)
    Colonnes renvoyées (index = date t):
      ['is_long','y','y_tp1','y_tp2','y_tp3','is_entry','entry_price','stop_price','r_value','first_event','bars_to_event']
    """
    px = _prep_ohlcv(df)
    close = px["close"]
    high = px["high"]
    low = px["low"]
    open_ = px["open"]

    sma_f = _sma(close, ma_fast)
    sma_s = _sma(close, ma_slow)
    is_long = (sma_f > sma_s)

    # SL côté trend
    sl_long = low.rolling(lookback_stop, min_periods=lookback_stop).min()
    sl_short = high.rolling(lookback_stop, min_periods=lookback_stop).max()

    # entry = open[t+1]
    entry_next = open_.shift(-1)

    # Pré-allocation
    idx = px.index
    y_tp1 = pd.Series(False, index=idx)
    y_tp2 = pd.Series(False, index=idx)
    y_tp3 = pd.Series(False, index=idx)
    is_entry = pd.Series(False, index=idx)
    entry_price = pd.Series(np.nan, index=idx, dtype="float64")
    stop_price = pd.Series(np.nan, index=idx, dtype="float64")
    r_value = pd.Series(np.nan, index=idx, dtype="float64")
    first_event = pd.Series("none", index=idx, dtype="object")
    bars_to_event = pd.Series(np.nan, index=idx, dtype="float64")

    # boucle (simple, robuste)
    # bornes: s'arrêter assez tôt pour avoir entry_next et horizon
    n = len(idx)
    for i, t in enumerate(idx):
        if i >= n - 1:
            break  # pas d'open suivant
        horizon_end = min(n - 1, i + horizon_bars)
        if horizon_end <= i:
            continue

        long_side = bool(is_long.iat[i])
        entry = float(entry_next.iat[i]) if pd.notna(entry_next.iat[i]) else np.nan
        if not np.isfinite(entry):
            continue

        if long_side:
            sl = float(sl_long.iat[i]) if pd.notna(sl_long.iat[i]) else np.nan
            if not np.isfinite(sl):
                continue
            R = entry - sl
            if R <= 0:
                continue
            tp_levels = [entry + k * R for k in tps]
        else:
            sl = float(sl_short.iat[i]) if pd.notna(sl_short.iat[i]) else np.nan
            if not np.isfinite(sl):
                continue
            R = sl - entry
            if R <= 0:
                continue
            tp_levels = [entry - k * R for k in tps]

        # fenêtre d'observation = barres futures i+1..horizon_end
        lows = low.iloc[i + 1 : horizon_end + 1]
        highs = high.iloc[i + 1 : horizon_end + 1]

        hit_tp = [False] * len(tps)
        hit_sl = False
        first_hit_j = None
        first_hit_label = "none"

        for j, (h, l) in enumerate(zip(highs.values, lows.values), start=1):
            if long_side:
                # ordre: on teste collision du SL puis TPs le même bar (priorité au 1er évènement)
                if l <= sl:
                    hit_sl = True
                    first_hit_j = j
                    first_hit_label = "sl"
                    break
                for k, lvl in enumerate(tp_levels):
                    if h >= lvl:
                        hit_tp[k] = True
                        first_hit_j = j
                        first_hit_label = f"tp{k+1}"
                        break
                if first_hit_j is not None:
                    break
            else:
                if h >= sl:
                    hit_sl = True
                    first_hit_j = j
                    first_hit_label = "sl"
                    break
                for k, lvl in enumerate(tp_levels):
                    if l <= lvl:
                        hit_tp[k] = True
                        first_hit_j = j
                        first_hit_label = f"tp{k+1}"
                        break
                if first_hit_j is not None:
                    break

        # enregistrer
        is_entry.iat[i] = True
        entry_price.iat[i] = entry
        stop_price.iat[i] = sl
        r_value.iat[i] = R
        first_event.iat[i] = first_hit_label
        bars_to_event.iat[i] = float(first_hit_j) if first_hit_j is not None else np.nan
        if hit_tp:
            y_tp1.iat[i] = bool(hit_tp[0])
            if len(hit_tp) > 1:
                y_tp2.iat[i] = bool(hit_tp[1])
            if len(hit_tp) > 2:
                y_tp3.iat[i] = bool(hit_tp[2])

    # y par défaut = TP1 avant SL
    y = y_tp1.copy()

    out = pd.DataFrame(
        {
            "is_long": is_long.astype(bool),
            "y": y.astype(bool),
            "y_tp1": y_tp1.astype(bool),
            "y_tp2": y_tp2.astype(bool),
            "y_tp3": y_tp3.astype(bool),
            "is_entry": is_entry.astype(bool),
            "entry_price": entry_price.astype("float64"),
            "stop_price": stop_price.astype("float64"),
            "r_value": r_value.astype("float64"),
            "first_event": first_event.astype("object"),
            "bars_to_event": bars_to_event.astype("float64"),
        },
        index=idx,
    )
    return out
