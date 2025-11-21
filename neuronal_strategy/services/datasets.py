from __future__ import annotations
from typing import Tuple, List
import pandas as pd
import math


def _read_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # normalise index/colonnes temporelles
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df = df.set_index("date")
        elif "__ts__" in df.columns:
            df["__ts__"] = pd.to_datetime(df["__ts__"], utc=True, errors="coerce")
            df = df.set_index("__ts__")
        else:
            first = df.columns[0]
            df[first] = pd.to_datetime(df[first], utc=True, errors="coerce")
            df = df.set_index(first)
    else:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.sort_index()
    df = df[~df.index.isna()]
    return df


def _label_col_for_margin(labels_df: pd.DataFrame, tp_margin: float) -> str:
    """
    Nouveau schéma (préféré) : 'y' (TP1) ou 'y_tp1' / 'y_tp2' / 'y_tp3'.
    Ancien schéma (fallback) : colonnes dépendantes de tp_margin (ex: y_tp15, label_tp15...)
    """
    cols = set(labels_df.columns)

    # 1) Nouveau schéma — priorité à 'y' (TP1)
    if "y" in cols:
        return "y"
    if "y_tp1" in cols:
        return "y_tp1"

    # 2) Ancien schéma — mapping via pourcentage
    pct = int(round(tp_margin * 100))
    candidates: List[str] = [
        f"y_{pct}", f"y_tp{pct}", f"y_tp_{pct}",
        f"label_tp{pct}", f"label_tp_{pct}",
        "label", "y_true", "target",
    ]
    if math.isclose(tp_margin, 0.15, rel_tol=1e-9):
        candidates.insert(0, "y_1p5R")
    if math.isclose(tp_margin, 0.10, rel_tol=1e-9):
        candidates.insert(0, "y_1R")

    for c in candidates:
        if c in cols:
            return c

    raise ValueError(
        f"Label column missing for tp_margin={tp_margin}. "
        f"Tried {candidates}. Available: {sorted(list(cols))}"
    )


def _windowize_features(df: pd.DataFrame, x_window: int, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construit des features fenêtrées t-0..t-(x_window-1) pour les colonnes passées.
    Renvoie (df_windowized, feat_cols_resultants).
    """
    xw = max(1, int(x_window))
    if xw == 1:
        return df, feature_cols

    parts = []
    new_cols: List[str] = []
    for lag in range(0, xw):
        shifted = df[feature_cols].shift(lag)
        suf = f"_t-{lag}"  # suffixes t-0, t-1...
        shifted.columns = [f"{c}{suf}" for c in shifted.columns]
        parts.append(shifted)
        new_cols.extend(list(shifted.columns))
    out = pd.concat(parts, axis=1)
    out = out.dropna(how="any")  # drop lignes incomplètes (lags)
    return out, new_cols


def load_joined_table(
    features_path: str,
    labels_path: str,
    tp_margin: float,
    x_window: int = 1,
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Charge features + labels, joint par (instrument, date), applique fenêtrage des features
    **par instrument**, et renvoie (table, feat_cols, y_col_name par défaut).

    - features parquet : index DateTimeIndex, colonnes incluent 'instrument' et features (ex: pos_* , vol_rel ...)
    - labels   parquet : colonnes ['instrument','date', 'y', 'y_tp1', 'y_tp2', 'y_tp3', 'is_long', 'is_entry', ...]
                         (ou ancien schéma de labels)
    """
    # ---------- FEATURES ----------
    X = _read_parquet(features_path).copy()
    X = X.loc[:, ~X.columns.duplicated()]  # dédoublonnage sécurité
    if "instrument" not in X.columns:
        raise ValueError("Features parquet must include 'instrument' column.")
    if "date" in X.columns:
        X = X.drop(columns=["date"])
    X["__date__"] = X.index
    feat_cols_in_X = [c for c in X.columns if c not in ("instrument", "__date__")]
    left = X[["instrument", "__date__"] + feat_cols_in_X].copy().rename(columns={"__date__": "date"})

    # purge tout doublon de 'date' côté left
    if (left.columns.tolist().count("date")) != 1:
        keep = []
        seen_date = False
        for c in left.columns:
            if c == "date":
                if not seen_date:
                    keep.append(c); seen_date = True
            else:
                keep.append(c)
        left = left[keep]

    # ---------- LABELS ----------
    Y = pd.read_parquet(labels_path).copy()
    if "date" not in Y.columns:
        if not isinstance(Y.index, pd.DatetimeIndex):
            for cand in ("__ts__", "timestamp", "time"):
                if cand in Y.columns:
                    Y[cand] = pd.to_datetime(Y[cand], utc=True, errors="coerce")
                    Y = Y.rename(columns={cand: "date"})
                    break
        else:
            Y = Y.rename_axis("date").reset_index()

    Y["date"] = pd.to_datetime(Y["date"], utc=True, errors="coerce")
    Y = Y.dropna(subset=["date"])
    if "instrument" not in Y.columns:
        raise ValueError("Labels parquet must include 'instrument' column.")

    # Conserve variantes de labels — **SANS 'y'** pour éviter les doublons ; on ajoutera 'y' plus bas.
    keep_label_variants = [c for c in ("y_tp1", "y_tp2", "y_tp3", "y_true") if c in Y.columns]
    keep_extra = [c for c in ("is_long", "is_entry", "is_setup") if c in Y.columns]
    y_col = _label_col_for_margin(Y, float(tp_margin))

    right = Y[["instrument", "date", y_col] + keep_label_variants + keep_extra].copy()

    # ---------- MERGE ----------
    tbl = pd.merge(left, right, on=["instrument", "date"], how="inner", sort=True)
    tbl["date"] = pd.to_datetime(tbl["date"], utc=True, errors="coerce")
    tbl = tbl.dropna(subset=["date"]).sort_values(["instrument", "date"]).set_index("date")

    # colonnes de features candidates
    exclude = {"instrument", y_col, "is_long", "is_entry", "is_setup"} | set(keep_label_variants)
    base_feat_cols = [c for c in tbl.columns if c not in exclude]

    # ---------- FENÊTRAGE PAR INSTRUMENT ----------
    parts: List[pd.DataFrame] = []
    feat_cols: List[str] = base_feat_cols  # écrasé si x_window > 1

    for instr, g in tbl.groupby("instrument", sort=False):
        feats_df, feat_cols_g = _windowize_features(
            g[base_feat_cols], x_window=int(x_window), feature_cols=base_feat_cols
        )
        if feats_df.empty:
            continue
        if int(x_window) > 1:
            feat_cols = feat_cols_g

        common_idx = feats_df.index

        meta_cols = (
            ["instrument"]
            + [c for c in ("is_long", "is_entry", "is_setup") if c in g.columns]
            + [c for c in ("y_tp1", "y_tp2", "y_tp3", "y_true") if c in g.columns]  # PAS 'y' ici
        )
        meta_df = g.loc[common_idx, meta_cols].copy()
        y_series = g.loc[common_idx, y_col].astype(int)

        g_out = pd.concat([meta_df, feats_df], axis=1)
        # ajoute 'y' une seule fois
        g_out["y"] = y_series
        parts.append(g_out)

    if not parts:
        raise ValueError("After windowing, no data left (check x_window vs series lengths).")

    out = pd.concat(parts, axis=0)
    out["__date__"] = out.index
    out = out.sort_values(["instrument", "__date__"]).drop(columns="__date__")

    # dédoublonne définitivement les colonnes si besoin
    out = out.loc[:, ~out.columns.duplicated()]
    out.index.name = None

    return out, feat_cols, "y"


def split_masks_by_timeframe(idx: pd.DatetimeIndex, timeframe: str | None) -> Tuple[pd.Series, pd.Series]:
    """
    Split temporel pour train/val :
      - 1D / 1H : mois impairs = train, mois pairs = validation
      - 1W      : semaines ISO impaires = train, paires = validation
      - par défaut : jour du mois impair/pair
    """
    tf = (timeframe or "1D").upper().strip()
    if tf in ("1D", "D", "1H", "H"):
        months = idx.month
        train_mask = pd.Series((months % 2 == 1), index=idx)  # impairs
        val_mask = ~train_mask
        return train_mask, val_mask
    elif tf in ("1W", "W"):
        weeks = idx.isocalendar().week.astype(int)
        train_mask = pd.Series((weeks % 2 == 1), index=idx)
        val_mask = ~train_mask
        return train_mask, val_mask
    else:
        days = idx.day
        train_mask = pd.Series((days % 2 == 1), index=idx)
        val_mask = ~train_mask
        return train_mask, val_mask
