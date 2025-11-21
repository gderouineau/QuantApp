from typing import Dict, Any
import pandas as pd
import numpy as np

def compute_labels_for_df(df: pd.DataFrame,
                          horizon_bars: int,
                          tp_margins: list[float]) -> pd.DataFrame:
    """
    df: colonnes ['open','high','low','close','volume'] indexées par datetime trié.
    Retourne un DataFrame aligné sur df avec colonnes pour chaque marge:
      - label_tp_{pct} (bool)
      - bars_to_hit_tp_{pct}, bars_to_hit_sl (int ou NaN)
      - hit_type_{pct} in {'TP','SL','NONE'}
    Règles:
      - Entrée = close[t]
      - SL = low[t-1]
      - TP = close[t] * (1 + pct)
      - Priorité au SL si les deux touchent la même barre
      - Gaps:
          * si open[t+1] <= SL  -> SL (perte)
          * si open[t+1] >= TP  -> TP (gain mais cap au niveau TP conceptuellement)
    """
    df = df.copy()
    df = df.sort_index()
    labels = pd.DataFrame(index=df.index)
    prev_low = df["low"].shift(1)

    for pct in tp_margins:
        col_label = f"label_tp_{int(pct*100)}"
        col_hit = f"hit_type_{int(pct*100)}"
        col_btp = f"bars_to_hit_tp_{int(pct*100)}"
        col_bsl = "bars_to_hit_sl"

        # init
        labels[col_label] = False
        labels[col_hit] = "NONE"
        labels[col_btp] = np.nan
        labels[col_bsl] = np.nan

        tp_level = df["close"] * (1.0 + pct)
        sl_level = prev_low

        for i in range(1, len(df) - horizon_bars - 1):
            t = df.index[i]
            entry = df.at[t, "close"]
            sl = sl_level.iloc[i]
            if pd.isna(sl):
                continue
            tp = tp_level.iloc[i]

            # Fenêtre forward
            fwd = df.iloc[i+1 : i+1+horizon_bars]
            if fwd.empty:
                continue

            # GAPS (barre i+1 open)
            f_open = fwd["open"].iloc[0]
            if f_open <= sl:
                labels.at[t, col_label] = False
                labels.at[t, col_hit] = "SL"
                labels.at[t, col_bsl] = 1
                continue
            if f_open >= tp:
                labels.at[t, col_label] = True
                labels.at[t, col_hit] = "TP"
                labels.at[t, col_btp] = 1
                continue

            # Parcours barre par barre: priorité SL
            hit_sl_idx = None
            hit_tp_idx = None
            lows = fwd["low"].values
            highs = fwd["high"].values
            for k in range(len(fwd)):
                # priorité SL
                if lows[k] <= sl:
                    hit_sl_idx = k + 1
                    break
                if highs[k] >= tp:
                    hit_tp_idx = k + 1
                    # on ne break pas ici, car SL a priorité;
                    # mais comme SL n'a pas été touché à cette iteration,
                    # on peut s'arrêter.
                    break

            if hit_sl_idx is not None:
                labels.at[t, col_label] = False
                labels.at[t, col_hit] = "SL"
                labels.at[t, col_bsl] = hit_sl_idx
            elif hit_tp_idx is not None:
                labels.at[t, col_label] = True
                labels.at[t, col_hit] = "TP"
                labels.at[t, col_btp] = hit_tp_idx
            else:
                labels.at[t, col_label] = False
                labels.at[t, col_hit] = "NONE"

    return labels
