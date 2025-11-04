# strategies/services/backtest.py

from __future__ import annotations
from dataclasses import dataclass
import math
import pandas as pd


@dataclass
class BTParams:
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01                 # 1% du capital
    atr_mult_stop: float = 2.0                   # stop = entry - k*ATR (long)
    take_profit_R: tuple[float, ...] = (1.0, 2.0)
    commission_bps: float = 1.0                  # 1 bp = 0.01%
    slippage_bps: float = 2.0


def _bps(price: float, bps: float) -> float:
    return price * (bps / 10_000.0)


def _next_open(dfp: pd.DataFrame, idx: int) -> float | None:
    if idx + 1 < len(dfp):
        return float(dfp["open"].iloc[idx + 1])
    return None


def simulate_one_symbol(
    df_prices: pd.DataFrame,
    df_ind: pd.DataFrame,
    strat,
    params: BTParams,
    warmup_bars: int = 252,
) -> dict:
    """
    Simule une stratégie sur 1 symbole.
    Hypothèse: evaluate() est appelé à la clôture de la barre i (df_prices.iloc[:i+1]),
    entrée à l'open de i+1. SL prioritaire, puis TP1/TP2. Sortie EOD si rien.
    """

    # --- ALIGNEMENT D'INDEX : tout en DatetimeIndex normalisé (date-only) ---
    df_prices = df_prices.copy()
    df_prices.index = pd.to_datetime(df_prices.index).normalize()
    df_prices.sort_index(inplace=True)

    df_ind = df_ind.copy()
    if not df_ind.empty:
        df_ind.index = pd.to_datetime(df_ind.index).normalize()
        df_ind.sort_index(inplace=True)

    closes = df_prices["close"].astype(float)
    highs = df_prices["high"].astype(float)
    lows = df_prices["low"].astype(float)

    equity = params.initial_capital
    trades: list[dict] = []

    # Boucle barres (on saute warmup pour indicateurs)
    for i in range(warmup_bars, len(df_prices) - 1):
        cut_ts = df_prices.index[i]
        dfp_sub = df_prices.iloc[: i + 1]
        dfi_sub = df_ind.loc[:cut_ts] if not df_ind.empty else df_ind

        res = strat.evaluate(dfp_sub, dfi_sub)
        if not res.get("signal"):
            continue

        last_close = float(closes.iloc[i])

        # Plan (si fourni par la stratégie)
        plan = res.get("details", {}).get("plan", {})
        entry = plan.get("entry_price") or last_close

        # Stop via ATR*mult (fallback %)
        atr = None
        if not dfi_sub.empty and "atr_14" in dfi_sub.columns:
            atr_series = dfi_sub["atr_14"].dropna()
            if not atr_series.empty:
                atr = float(atr_series.iloc[-1])
        stop = plan.get("stop_price")
        if stop is None:
            stop = entry - params.atr_mult_stop * atr if atr is not None else entry * 0.97

        risk_per_share = max(entry - stop, 1e-6)
        risk_cash = equity * params.risk_per_trade
        qty = max(int(risk_cash / risk_per_share), 1)

        # Entrée à l'open suivant + slippage/commission
        nxt_open = _next_open(df_prices, i)
        if nxt_open is None:
            break
        px_in = nxt_open + _bps(nxt_open, params.slippage_bps)
        fee_in = px_in * qty * (params.commission_bps / 10_000.0)

        # TP en prix (plan -> sinon 1R/2R)
        tps = plan.get("take_profits")
        if not tps:
            tps = [{"target": r, "price": px_in + r * (px_in - stop)} for r in params.take_profit_R]

        # Parcours des barres suivantes pour sortie
        exit_reason, exit_px, exit_date = None, None, None
        for j in range(i + 1, len(df_prices)):
            hi, lo = float(highs.iloc[j]), float(lows.iloc[j])

            # SL prioritaire
            if lo <= stop:
                exit_reason = "SL"
                exit_px = stop - _bps(stop, params.slippage_bps)
                exit_date = df_prices.index[j]
                break

            # TP
            hit = next((tp for tp in tps if hi >= tp["price"]), None)
            if hit:
                tgt = float(hit["target"])
                exit_reason = f"TP{int(tgt) if tgt.is_integer() else tgt}"
                exit_px = hit["price"] - _bps(hit["price"], params.slippage_bps)
                exit_date = df_prices.index[j]
                break

        if exit_px is None:
            # fin de série
            exit_reason = "EOD"
            exit_px = float(closes.iloc[-1])
            exit_date = df_prices.index[-1]

        fee_out = exit_px * qty * (params.commission_bps / 10_000.0)
        pnl = (exit_px - px_in) * qty - (fee_in + fee_out)
        r_mult = pnl / (risk_per_share * qty)

        equity += pnl
        trades.append({
            "entry_date": df_prices.index[i + 1],
            "entry_price": px_in,
            "qty": qty,
            "stop_price": stop,
            "tps": tps,
            "exit_date": exit_date,
            "exit_price": exit_px,
            "outcome": exit_reason,
            "r_multiple": r_mult,
            "pnl": pnl,
            "equity": equity,
        })

    wins = [t["r_multiple"] for t in trades if t["r_multiple"] > 0]
    losses = [t["r_multiple"] for t in trades if t["r_multiple"] <= 0]
    p_win = (len(wins) / len(trades)) if trades else 0.0
    p_loss = 1.0 - p_win if trades else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (-sum(losses) / len(losses)) if losses else 0.0  # valeur positive
    expectancy = p_win * avg_win - p_loss * avg_loss

    summary = {
        "trades": trades,
        "equity_final": equity,
        "n_trades": len(trades),
        "win_rate": p_win,
        "avg_R": (sum(t["r_multiple"] for t in trades) / len(trades)) if trades else 0.0,
        "expectancy_R": expectancy,
    }
    return summary
