from typing import Tuple
import pandas as pd
import numpy as np

def add_indicators(df: pd.DataFrame,
                   use_ma=True, ma_periods=None,
                   use_bb=True, bb_period=20, bb_k=2.0,
                   use_atr=True, atr_period=14) -> pd.DataFrame:
    df = df.copy()
    ma_periods = ma_periods or []

    if use_ma and ma_periods:
        for p in ma_periods:
            df[f"ma_{p}"] = df["close"].rolling(p).mean()

    if use_bb:
        mid = df["close"].rolling(bb_period).mean()
        std = df["close"].rolling(bb_period).std(ddof=0)
        df["bb_mid"] = mid
        df["bb_up"] = mid + bb_k * std
        df["bb_dn"] = mid - bb_k * std
        # %B pratique
        df["bb_pct_b"] = (df["close"] - df["bb_dn"]) / ((df["bb_up"] - df["bb_dn"]).replace(0, np.nan))

    if use_atr:
        tr1 = (df["high"] - df["low"]).abs()
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(atr_period).mean()

    return df


def grid_scale_row(row: pd.Series, y_resolution: int, include_cols: list[str]) -> dict:
    """
    Mappe les valeurs (OHLC + indicateurs choisis) sur [0, y_resolution].
    min = min(low, indicateurs sélectionnés)
    max = max(high, indicateurs sélectionnés)
    Renvoie un dict de positions normalisées (entiers).
    """
    vals = []
    if "low" in row and "high" in row:
        vals.extend([row["low"], row["high"]])
    for c in include_cols:
        if c in row and pd.notna(row[c]):
            vals.append(row[c])
    if not vals:
        return {}

    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return {}

    out = {}
    def to_grid(v):
        pos = (v - vmin) / (vmax - vmin)
        return int(round(pos * y_resolution))

    # OHLC positions
    out["pos_open"] = to_grid(row["open"]) if pd.notna(row.get("open")) else None
    out["pos_high"] = to_grid(row["high"]) if pd.notna(row.get("high")) else None
    out["pos_low"] = to_grid(row["low"]) if pd.notna(row.get("low")) else None
    out["pos_close"] = to_grid(row["close"]) if pd.notna(row.get("close")) else None

    # Indicateurs choisis
    for c in include_cols:
        if c in row and pd.notna(row[c]):
            out[f"pos_{c}"] = to_grid(row[c])
    return out


def build_features(df: pd.DataFrame,
                   y_resolution: int,
                   scaling_mode: str,
                   use_ma=True, ma_periods=None,
                   use_bb=True, bb_period=20, bb_k=2.0,
                   use_atr=True, atr_period=14) -> pd.DataFrame:
    """
    - grid_scaled: on renvoie des positions 0..Y pour open/high/low/close + indicateurs.
    - relative_feats: on renvoie des ratios invariants (body, wicks, vol_rel, dist_ma, %B, atr_rel).
    """
    ma_periods = ma_periods or []
    dfi = add_indicators(df, use_ma, ma_periods, use_bb, bb_period, bb_k, use_atr, atr_period)

    if scaling_mode == "grid_scaled":
        include_cols = []
        if use_ma:
            include_cols += [f"ma_{p}" for p in ma_periods]
        if use_bb:
            include_cols += ["bb_mid", "bb_up", "bb_dn"]
        if use_atr:
            include_cols += ["atr"]

        rows = []
        for idx, row in dfi.iterrows():
            rows.append(grid_scale_row(row, y_resolution, include_cols))
        feat = pd.DataFrame(rows, index=dfi.index)

        # Volume relatif (en feature annexe, pas sur la grille verticale)
        vol = dfi["volume"].copy()
        p95 = vol.rolling(200, min_periods=20).quantile(0.95)
        feat["vol_rel"] = (vol / (p95.replace(0, np.nan))).clip(0, 10)

        return feat

    elif scaling_mode == "relative_feats":
        feat = pd.DataFrame(index=dfi.index)
        o, h, l, c = dfi["open"], dfi["high"], dfi["low"], dfi["close"]
        # Invariants de forme
        feat["body"] = (c - o) / (o.replace(0, np.nan))
        feat["range"] = (h - l) / (o.replace(0, np.nan))
        feat["up_wick"] = (h - np.maximum(o, c)) / (o.replace(0, np.nan))
        feat["lo_wick"] = (np.minimum(o, c) - l) / (o.replace(0, np.nan))
        # Volume relatif
        vol = dfi["volume"]
        p95 = vol.rolling(200, min_periods=20).quantile(0.95)
        feat["vol_rel"] = (vol / (p95.replace(0, np.nan))).clip(0, 10)
        # MA distances
        for p in (ma_periods or []):
            feat[f"dist_ma_{p}"] = (c / dfi[f"ma_{p}"] - 1.0)
        # Bollinger %B
        if "bb_pct_b" in dfi:
            feat["bb_pct_b"] = dfi["bb_pct_b"]
        # ATR relatif
        if "atr" in dfi:
            feat["atr_rel"] = (dfi["atr"] / c.replace(0, np.nan))
        return feat

    else:
        raise ValueError(f"Unknown scaling_mode {scaling_mode}")
