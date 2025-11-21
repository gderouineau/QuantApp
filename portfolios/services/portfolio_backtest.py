# app: portfolios
# file: portfolios/services/portfolio_backtest.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import date

import pandas as pd

from market_data.services.store import read_parquet, bars_path
from indicators.models import Indicator
from strategies.models import Strategy

# Simulateur existant
from strategies.services.backtest import simulate_one_symbol, BTParams
# Simulateur SL/TP avancé
from strategies.services.sltp import simulate_one_symbol_sltp

from strategies.relative_strength import RelativeStrengthStrategy
from strategies.custom_strategy import CustomStrategy
from strategies.golden_cross import GoldenCrossStrategy
from strategies.volume_breakout import VolumeBreakoutStrategy

STRAT_MAP = {
    "CUSTOM": lambda s: CustomStrategy(code=s.code, parameters=s.parameters),
    "RS":     lambda s: RelativeStrengthStrategy(s.parameters),
    "GC":     lambda s: GoldenCrossStrategy(s.parameters),
    "VB":     lambda s: VolumeBreakoutStrategy(s.parameters),
}

@dataclass
class PortfolioSimParams:
    start: Optional[date] = None
    end:   Optional[date] = None
    warmup_bars: int = 252
    capital_override: Optional[float] = None   # sinon portfolio.initial_capital
    risk_override: Optional[float] = None      # fraction (ex: 0.01) sinon portfolio.risk_per_trade

def _build_strategy_instance(s: Strategy):
    ctor = STRAT_MAP.get(s.type)
    if not ctor:
        return None
    return ctor(s)

def _load_dataframes(symbol: str):
    dfp = read_parquet(bars_path(symbol, "1D"))
    if dfp is None or dfp.empty:
        return None, None
    dfi = pd.DataFrame(list(
        Indicator.objects.filter(asset__symbol=symbol).order_by("date").values()
    ))
    if not dfi.empty:
        dfi.set_index("date", inplace=True)
    return dfp, dfi

def _slice_by_dates(dfp: pd.DataFrame, dfi: pd.DataFrame, start: Optional[date], end: Optional[date]):
    if dfp is None or dfp.empty:
        return dfp, dfi
    # Normaliser les index (daily)
    dfp = dfp.copy()
    dfp.index = pd.to_datetime(dfp.index).normalize()
    if dfi is not None and not dfi.empty:
        dfi = dfi.copy()
        dfi.index = pd.to_datetime(dfi.index).normalize()
    if start:
        ts = pd.Timestamp(start)
        dfp = dfp.loc[dfp.index >= ts]
        if dfi is not None and not dfi.empty:
            dfi = dfi.loc[dfi.index >= ts]
    if end:
        te = pd.Timestamp(end)
        dfp = dfp.loc[dfp.index <= te]
        if dfi is not None and not dfi.empty:
            dfi = dfi.loc[dfi.index <= te]
    return dfp, dfi

def run_portfolio_backtest(portfolio, params: PortfolioSimParams) -> Dict[str, Any]:
    """
    Retourne un dict structuré avec les résultats agrégés.
    Hypothèses v1 (simple) :
      - Capital par allocation = initial * weight
      - Capital par symbole = alloc_capital / max_positions (fallback égalitaire)
      - Risk/trade = allocation.per_trade_risk (sinon portfolio.risk_per_trade)
      - On ne force pas le chevauchement de positions (v1 naïve).
    """
    initial_capital = params.capital_override if params.capital_override is not None else portfolio.initial_capital
    default_risk = params.risk_override if params.risk_override is not None else portfolio.risk_per_trade

    results = {
        "portfolio": {"name": portfolio.name, "initial_capital": float(initial_capital)},
        "params": {
            "start": params.start.isoformat() if params.start else None,
            "end": params.end.isoformat() if params.end else None,
            "warmup_bars": params.warmup_bars,
            "capital_override": params.capital_override,
            "risk_override": params.risk_override,
        },
        "allocations": [],
        "symbols": [],
        "totals": {"n_trades": 0, "sum_R": 0.0, "symbols": 0, "equity_final": 0.0},
    }

    equity_sum = 0.0
    for alloc in portfolio.allocations.select_related("strategy", "group").filter(is_active=True):
        if not alloc.group or not alloc.strategy:
            continue

        strat_inst = _build_strategy_instance(alloc.strategy)
        if not strat_inst:
            continue

        alloc_capital = initial_capital * float(alloc.weight or 0.0)
        per_trade_risk = float(alloc.per_trade_risk) if alloc.per_trade_risk is not None else float(default_risk or 0.0)

        symbols = list(alloc.group.assets.values_list("symbol", flat=True))
        if not symbols:
            continue

        max_pos = int(alloc.max_positions or len(symbols))
        per_symbol_capital = alloc_capital / max(1, max_pos)

        alloc_stats = {
            "name": getattr(alloc, "name", f"alloc#{alloc.id}"),
            "strategy": alloc.strategy.name,
            "group": alloc.group.name,
            "weight": float(alloc.weight or 0.0),
            "symbols": [],
            "n_trades": 0,
            "sum_R": 0.0,
            "equity_final": 0.0
        }

        # Config SLTP éventuelle au niveau stratégie
        sltp_cfg = (alloc.strategy.parameters or {}).get("sltp")

        for sym in symbols[:max_pos]:
            dfp, dfi = _load_dataframes(sym)
            if dfp is None or dfp.empty:
                continue

            # Découper à la fenêtre demandée
            dfp, dfi = _slice_by_dates(dfp, dfi, params.start, params.end)

            bt = BTParams(initial_capital=per_symbol_capital, risk_per_trade=per_trade_risk)

            if sltp_cfg:
                res = simulate_one_symbol_sltp(
                    df_prices=dfp,
                    df_ind=dfi,
                    strat=strat_inst,
                    initial_capital=per_symbol_capital,
                    risk_per_trade=per_trade_risk,
                    sltp_cfg=sltp_cfg,
                    warmup_bars=params.warmup_bars
                )
            else:
                res = simulate_one_symbol(dfp, dfi, strat_inst, bt, warmup_bars=params.warmup_bars)

            sym_row = {
                "symbol": sym,
                "n_trades": int(res.get("n_trades", 0)),
                "win_rate": float(res.get("win_rate", 0.0)),
                "avg_R": float(res.get("avg_R", 0.0)),
                "equity_final": float(res.get("equity_final", per_symbol_capital)),
            }
            alloc_stats["symbols"].append(sym_row)
            alloc_stats["n_trades"] += sym_row["n_trades"]
            alloc_stats["sum_R"] += float(sum(t.get("r_multiple", 0.0) for t in res.get("trades", [])))
            alloc_stats["equity_final"] += sym_row["equity_final"]

            results["symbols"].append({
                **sym_row,
                "allocation": alloc_stats["name"],
                "strategy": alloc_stats["strategy"],
                "group": alloc_stats["group"],
            })

        results["allocations"].append(alloc_stats)
        results["totals"]["n_trades"] += alloc_stats["n_trades"]
        results["totals"]["sum_R"] += alloc_stats["sum_R"]
        equity_sum += alloc_stats["equity_final"]

    results["totals"]["symbols"] = len(results["symbols"])
    results["totals"]["equity_final"] = equity_sum if equity_sum else initial_capital
    return results
