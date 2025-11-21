import pandas as pd
import numpy as np

def backtest_signals(df: pd.DataFrame,
                     signals: pd.Series,
                     tp_margin: float,
                     horizon_bars: int,
                     fees_bps: float = 2.0,
                     slippage_bps: float = 2.0) -> dict:
    """
    df: OHLCV, index temps
    signals: bool par t (True => on prend le trade long à close[t])
    Règle d’exé: SL = low[t-1], TP = close[t]*(1+tp_margin), priorité SL, gaps inclus.
    Retourne métriques simples + equity curve.
    """
    df = df.copy()
    entry_idx = signals[signals].index
    equity = [1.0]
    wins = 0
    losses = 0

    for t in entry_idx:
        if t not in df.index:
            continue
        i = df.index.get_loc(t)
        if i < 1 or i+1 >= len(df):
            continue

        entry = df.at[t, "close"]
        sl = df["low"].iloc[i-1]
        tp = entry * (1.0 + tp_margin)
        fwd = df.iloc[i+1 : i+1+horizon_bars]
        if fwd.empty:
            continue

        # Ajoute fees+slippage ~ entrée+sortie (approx)
        cost = (fees_bps + slippage_bps) * 1e-4

        # gaps open
        f_open = fwd["open"].iloc[0]
        if f_open <= sl:
            pnl = (sl / entry) - 1.0 - cost
            losses += 1
        elif f_open >= tp:
            pnl = (tp / entry) - 1.0 - cost
            wins += 1
        else:
            hit_sl = None
            hit_tp = None
            lows = fwd["low"].values
            highs = fwd["high"].values
            for k in range(len(fwd)):
                if lows[k] <= sl:
                    hit_sl = True
                    break
                if highs[k] >= tp:
                    hit_tp = True
                    break
            if hit_sl:
                pnl = (sl / entry) - 1.0 - cost
                losses += 1
            elif hit_tp:
                pnl = (tp / entry) - 1.0 - cost
                wins += 1
            else:
                # time-out flat (ni TP ni SL): 0 - cost?
                pnl = -cost

        equity.append(equity[-1] * (1.0 + pnl))

    if len(equity) == 1:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "pf": 0.0, "equity_end": 1.0}

    eq_arr = np.array(equity)
    trades = wins + losses
    win_rate = wins / trades if trades else 0.0
    # Profit factor approx: somme gains / somme pertes abs
    # (tu peux raffiner avec un enregistrement trade-par-trade)
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "equity_end": float(eq_arr[-1]),
    }
