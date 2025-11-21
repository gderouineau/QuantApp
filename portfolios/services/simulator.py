# portfolios/services/simulator.py
from __future__ import annotations
import math
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd

from market_data.models import Asset
from indicators.models import Indicator
from market_data.services.store import read_parquet, bars_path
from strategies.models import Strategy
from strategies.custom_strategy import CustomStrategy
from strategies.relative_strength import RelativeStrengthStrategy
from strategies.golden_cross import GoldenCrossStrategy
from strategies.volume_breakout import VolumeBreakoutStrategy

STRAT_MAP = {
    "CUSTOM": lambda s: CustomStrategy(code=s.code, parameters=s.parameters),
    "RS":     lambda s: RelativeStrengthStrategy(s.parameters),
    "GC":     lambda s: GoldenCrossStrategy(s.parameters),
    "VB":     lambda s: VolumeBreakoutStrategy(s.parameters),
}


@dataclass
class SimParams:
    warmup_bars: int = 252
    commission_bps: float = 1.0
    slippage_bps: float = 2.0


def _bps(price: float, bps: float) -> float:
    return price * (bps / 10_000.0)


def load_asset_frames(symbols: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
    """Précharge les DataFrames prix/indicateurs alignés (DatetimeIndex normalisé)."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for sym in symbols:
        dfp = read_parquet(bars_path(sym, "1D"))
        if dfp is None or dfp.empty:
            continue
        dfp = dfp.copy()
        dfp.index = pd.to_datetime(dfp.index).normalize()
        dfp.sort_index(inplace=True)

        dfi_qs = Indicator.objects.filter(asset__symbol=sym).order_by("date").values()
        dfi = pd.DataFrame(list(dfi_qs))
        if not dfi.empty:
            dfi.set_index("date", inplace=True)
            dfi.index = pd.to_datetime(dfi.index).normalize()
            dfi.sort_index(inplace=True)

        out[sym] = {"prices": dfp, "ind": dfi}
    return out


def strategy_instance(s: Strategy):
    ctor = STRAT_MAP.get(s.type)
    if not ctor:
        return None
    return ctor(s)


def simulate_portfolio_range(
    portfolio,
    allocations,
    symbols_by_alloc: dict[int, list[str]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    params: SimParams,
):
    """
    MVP : chaque jour (timeline globale), pour chaque allocation :
      - évalue tous les actifs du groupe pour ce jour,
      - sélectionne les top 'max_positions' par score (signal=True),
      - entre à l'open J+1, sort par SL prioritaire sinon 1er TP sinon EOD.
    Limitations MVP :
      - pas de budget strict par allocation (on limite via max_positions),
      - pas de déduplication inter-allocations (à faire plus tard),
      - un seul trade par actif en même temps,
      - les sorties sont "résolues" immédiatement (pas de positions persistées dans la boucle).
    """
    # 1) Précharge toutes les datas nécessaires
    all_symbols = sorted({sym for lst in symbols_by_alloc.values() for sym in lst})
    frames = load_asset_frames(all_symbols)

    # 2) Timeline globale = union des dates disponibles
    all_dates = set()
    for sym in frames:
        idx = frames[sym]["prices"].index
        if len(idx) == 0:
            continue
        mask = (idx >= start) & (idx <= end)
        all_dates.update(idx[mask])
    if not all_dates:
        return {"trades": [], "equity_points": [], "exposure_points": [], "summary": {}}
    timeline = sorted(all_dates)

    equity = float(portfolio.initial_capital)
    open_assets = set()  # évite 2 positions simultanées sur même symbole (MVP)
    trades: list[dict] = []
    equity_points: list[dict] = []

    for t in timeline:
        # snapshot equity quotidien (simple)
        equity_points.append({"date": t.date(), "equity": equity})

        # Pour chaque allocation active
        for alloc in allocations:
            if not getattr(alloc, "is_active", True):
                continue
            sym_list = symbols_by_alloc.get(alloc.id, [])
            if not sym_list:
                continue

            strat = strategy_instance(alloc.strategy)
            if not strat:
                continue

            # 2.1 évalue les actifs éligibles pour ce jour t
            candidates = []
            for sym in sym_list:
                f = frames.get(sym)
                if not f:
                    continue
                dfp = f["prices"]
                if t not in dfp.index:
                    continue
                # Warmup et bornes
                idx = dfp.index.get_loc(t)
                if idx < params.warmup_bars or idx >= len(dfp) - 1:
                    continue
                dfp_sub = dfp.iloc[: idx + 1]
                dfi = f["ind"]
                dfi_sub = dfi.loc[:t] if dfi is not None and not dfi.empty else dfi

                res = strat.evaluate(dfp_sub, dfi_sub)
                if res.get("signal"):
                    score = float(res.get("score", 0.0))
                    candidates.append((sym, score, res))

            if not candidates:
                continue

            # 2.2 sélectionne top par score
            candidates.sort(key=lambda x: x[1], reverse=True)
            selected = []
            for sym, score, res in candidates:
                if len(selected) >= int(alloc.max_positions or 0):
                    break
                if sym in open_assets:
                    continue  # déjà une position ouverte (MVP)
                selected.append((sym, score, res))

            # 2.3 entre en position pour ceux sélectionnés
            for sym, score, res in selected:
                f = frames[sym]
                dfp = f["prices"]
                idx = dfp.index.get_loc(t)

                # entrée à l'open J+1 (+ slippage/commission)
                px_in_raw = float(dfp["open"].iloc[idx + 1])
                px_in = px_in_raw + _bps(px_in_raw, params.slippage_bps)
                fee_in = px_in * (params.commission_bps / 10_000.0)

                # sizing : risk = equity * (alloc.per_trade_risk or portfolio.risk_per_trade)
                r = alloc.per_trade_risk if alloc.per_trade_risk is not None else portfolio.risk_per_trade
                r = float(r or 0.0)

                # stop / plan
                plan = res.get("details", {}).get("plan", {})
                entry = plan.get("entry_price") or float(dfp["close"].iloc[idx])
                stop = plan.get("stop_price")
                if stop is None:
                    # fallback ATR via indicateurs du jour
                    dfi = f["ind"]
                    atr = None
                    if dfi is not None and not dfi.empty and "atr_14" in dfi.columns:
                        atr_series = dfi.loc[:t]["atr_14"].dropna()
                        if not atr_series.empty:
                            atr = float(atr_series.iloc[-1])
                    stop = entry - 2.0 * atr if atr is not None else entry * 0.97

                risk_per_share = max(px_in - float(stop), 1e-6)
                qty = max(int((equity * r) / risk_per_share), 1)

                # TPs (prix)
                tps = plan.get("take_profits")
                if not tps:
                    tps = [
                        {"target": 1.0, "price": px_in + (px_in - float(stop))},
                        {"target": 2.0, "price": px_in + 2 * (px_in - float(stop))},
                    ]

                # 2.4 simule la sortie (SL prioritaire, sinon 1er TP, sinon EOD)
                exit_reason, exit_px, exit_date = None, None, None
                highs = dfp["high"].astype(float)
                lows = dfp["low"].astype(float)

                for j in range(idx + 1, len(dfp)):
                    dtj = dfp.index[j]
                    if dtj > end:
                        break
                    hi, lo = float(highs.iloc[j]), float(lows.iloc[j])

                    if lo <= float(stop):
                        exit_reason = "SL"
                        exit_px = float(stop) - _bps(float(stop), params.slippage_bps)
                        exit_date = dtj
                        break

                    hit = next((tp for tp in tps if hi >= float(tp["price"])), None)
                    if hit:
                        tgt = float(hit["target"])
                        exit_reason = f"TP{int(tgt) if tgt.is_integer() else tgt}"
                        exit_px = float(hit["price"]) - _bps(float(hit["price"]), params.slippage_bps)
                        exit_date = dtj
                        break

                if exit_px is None:
                    # pas touché SL/TP dans la fenêtre -> sortie à la dernière clôture incluse
                    # borne droite = min(end, dernière barre)
                    last_idx = dfp.index.get_slice_bound(min(end.normalize(), dfp.index[-1]), "right") - 1
                    last_idx = max(last_idx, idx + 1)
                    exit_date = dfp.index[last_idx]
                    exit_px = float(dfp["close"].iloc[last_idx])
                    exit_reason = "EOD"

                fee_out = exit_px * (params.commission_bps / 10_000.0)
                pnl = (exit_px - px_in) * qty - (fee_in + fee_out)
                r_mult = pnl / (risk_per_share * qty)

                trades.append({
                    "allocation_id": alloc.id,
                    "strategy_id": alloc.strategy_id,
                    "symbol": sym,
                    "entry_date": dfp.index[idx + 1].date(),
                    "entry_price": px_in,
                    "qty": int(qty),
                    "stop_price": float(stop),
                    "exit_date": exit_date.date(),
                    "exit_price": float(exit_px),
                    "outcome": exit_reason,
                    "r_multiple": float(r_mult),
                    "pnl": float(pnl),
                    "score": float(score),
                })

                equity += pnl
                open_assets.add(sym)   # MVP: bloque nouvel ordre tant que "consommé"
                open_assets.discard(sym)

        # snapshot après traitements du jour
        equity_points[-1]["equity"] = float(equity)

    # --- métriques de performance
    wins = [t for t in trades if t["r_multiple"] > 0]
    p_win = (len(wins) / len(trades)) if trades else 0.0
    avg_R = (sum(t["r_multiple"] for t in trades) / len(trades)) if trades else 0.0
    losses = [t["r_multiple"] for t in trades if t["r_multiple"] <= 0]
    avg_win = (sum(w["r_multiple"] for w in wins) / len(wins)) if wins else 0.0
    avg_loss = (-sum(losses) / len(losses)) if losses else 0.0
    expectancy = p_win * avg_win - (1 - p_win) * avg_loss

    summary = {
        "n_trades": len(trades),
        "win_rate": float(p_win),
        "avg_R": float(avg_R),
        "expectancy_R": float(expectancy),
        "equity_final": float(equity),
    }

    # -------------------------------------------------------------------------
    # EXPO : post-traitement à partir des trades + séries de prix
    # On reconstruit l'exposition quotidienne : somme(|qty| * close) des positions
    # dont date ∈ [entry_date, exit_date], rapportée à l'equity du jour.
    # Cela évite de modifier le MVP (pas de positions persistées dans la boucle).
    # -------------------------------------------------------------------------
    exposure_points: list[dict] = []
    if equity_points and trades:
        # map equity par date (date -> equity)
        eq_map = {p["date"]: float(p["equity"]) for p in equity_points}

        # précompute close par symbole (index -> date, valeurs -> close) pour accès rapide
        close_by_symbol: dict[str, dict] = {}
        for sym, fr in frames.items():
            ser = fr["prices"]["close"].astype(float)
            # clé = date (datetime.date) pour lookup direct
            ser_index_dates = [ts.date() for ts in ser.index]
            close_by_symbol[sym] = dict(zip(ser_index_dates, ser.values.tolist()))

        # pour chaque jour observable, somme des valeurs investies
        for d in sorted(eq_map.keys()):
            equity_val = eq_map[d]
            invested_val = 0.0
            if equity_val > 0:
                for tr in trades:
                    # si la position est "ouverte" à la date d
                    if tr["entry_date"] <= d <= tr["exit_date"]:
                        sym = tr["symbol"]
                        qty = int(tr["qty"])
                        px_map = close_by_symbol.get(sym, {})
                        px = px_map.get(d)
                        if px is not None and qty != 0:
                            invested_val += abs(qty) * float(px)

                expo = invested_val / equity_val if equity_val > 0 else 0.0
                # si pas de levier, on borne à [0, 1]
                expo = max(0.0, min(1.0, expo))
            else:
                expo = 0.0

            exposure_points.append({"date": d, "exposure_pct": float(expo)})

        if exposure_points:
            vals = [p["exposure_pct"] for p in exposure_points]
            summary["avg_exposure_pct"] = float(sum(vals) / len(vals))
            summary["max_exposure_pct"] = float(max(vals))
        else:
            summary["avg_exposure_pct"] = 0.0
            summary["max_exposure_pct"] = 0.0
    else:
        summary["avg_exposure_pct"] = 0.0
        summary["max_exposure_pct"] = 0.0

    return {
        "trades": trades,
        "equity_points": equity_points,
        "exposure_points": exposure_points,
        "summary": summary,
    }
