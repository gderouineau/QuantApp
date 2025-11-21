# neuronal_strategy/management/commands/ns_backtest.py
from __future__ import annotations
import math
from typing import Any, Dict

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from neuronal_strategy.models import NSDataset, NSRun
from neuronal_strategy.selectors.prices import load_universe_ohlcv
from neuronal_strategy.services.backtest import backtest_signals


def _to_json_safe(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is pd.NA:
        return None
    if isinstance(v, (int, float, bool, str)) or v is None:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    if isinstance(v, dict):
        return {str(k): _to_json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_json_safe(x) for x in v]
    return str(v)


class Command(BaseCommand):
    help = (
        "Backtest simple (long-only, SL prioritaire, gaps inclus). "
        "Par défaut, utilise les labels comme proxy (optionnellement shift +1 barre via --use_labels_shifted)."
    )

    def add_arguments(self, parser):
        parser.add_argument("--dataset_id", type=int, required=True)
        parser.add_argument("--tp_margin", type=float, required=True)
        parser.add_argument("--threshold", type=float, default=0.85)  # réservé pour proba modèle plus tard
        parser.add_argument("--fees_bps", type=float, default=2.0)
        parser.add_argument("--slippage_bps", type=float, default=2.0)
        parser.add_argument(
            "--use_labels_shifted",
            action="store_true",
            help="Si présent: signal = label_tpXX.shift(+1) (évite look-ahead).",
        )

    def handle(self, *args, **opts):
        ds = NSDataset.objects.get(pk=opts["dataset_id"])
        tp_margin = float(opts["tp_margin"])
        _thr = float(opts["threshold"])
        fees_bps = float(opts["fees_bps"])
        slippage_bps = float(opts["slippage_bps"])
        use_labels_shifted = bool(opts.get("use_labels_shifted", False))

        run = NSRun.objects.create(dataset=ds, kind="backtest", status="pending")
        run.mark_running()

        try:
            if not ds.labels_path:
                raise CommandError("Labels path missing. Run ns_labels first.")

            labels = pd.read_parquet(ds.labels_path)
            if "instrument" not in labels.columns:
                raise CommandError("Labels parquet must contain 'instrument' column.")

            # Index temporel propre
            if not isinstance(labels.index, pd.DatetimeIndex):
                if "date" in labels.columns:
                    labels = labels.set_index("date")
                labels.index = pd.to_datetime(labels.index)

            col = f"label_tp_{int(tp_margin * 100)}"
            if col not in labels.columns:
                raise CommandError(f"Column {col} missing in labels parquet.")

            data = load_universe_ohlcv(ds.universe, ds.date_from, ds.date_to, timeframe=ds.timeframe)

            agg: Dict[str, Any] = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "equity_end": 1.0,  # sera mis à la géométrique moyenne inter-instruments
                "tp_margin": tp_margin,
                "fees_bps": fees_bps,
                "slippage_bps": slippage_bps,
                "horizon_bars": int(ds.horizon_bars),
                "y_resolution": int(ds.y_resolution),
                "signals": "labels_shift+1" if use_labels_shifted else "labels_same_bar (smoke-test ONLY)",
            }

            # On agrège l’equity via la MOYENNE des logs (géometric mean) pour éviter les overflows.
            log_equities = []

            for instr, df in data.items():
                if df.empty:
                    continue

                lab_i = labels.loc[labels["instrument"] == instr, [col]].copy()

                # Index datetime + dédoublonnage
                if not isinstance(lab_i.index, pd.DatetimeIndex):
                    if "date" in lab_i.columns:
                        lab_i = lab_i.set_index("date")
                    lab_i.index = pd.to_datetime(lab_i.index)
                lab_i = lab_i[~lab_i.index.duplicated(keep="last")]

                # SHIFT +1 pour éviter look-ahead si demandé
                if use_labels_shifted:
                    lab_i[col] = lab_i[col].shift(1)

                # Aligne sur l'index prix
                # Pour éviter le FutureWarning: on force un type numérique avant fillna
                s = lab_i[col].astype("float64")
                s_aligned = s.reindex(df.index)
                s_aligned = s_aligned.fillna(0.0).astype(bool)

                m = backtest_signals(
                    df=df,
                    signals=s_aligned,
                    tp_margin=tp_margin,
                    horizon_bars=ds.horizon_bars,
                    fees_bps=fees_bps,
                    slippage_bps=slippage_bps,
                )

                t = int(m.get("trades", 0))
                w = int(m.get("wins", 0))
                l = int(m.get("losses", 0))
                agg["trades"] += t
                agg["wins"] += w
                agg["losses"] += l

                eq_end = m.get("equity_end", 1.0)
                try:
                    eq_end_f = float(eq_end)
                except Exception:
                    eq_end_f = float("nan")

                # Stocke log(equity_end_instrument) si valide (>0)
                if eq_end_f > 0 and not (math.isnan(eq_end_f) or math.isinf(eq_end_f)):
                    log_equities.append(math.log(eq_end_f))

            # Equity globale = géometric mean des equity_end instruments (si présents)
            if log_equities:
                mean_log = float(np.mean(log_equities))
                # bornage de sécurité (anti-overflow/underflow extrême)
                mean_log = max(min(mean_log, 50.0), -50.0)
                eq_global = math.exp(mean_log)
            else:
                eq_global = 1.0
            agg["equity_end"] = eq_global

            trades = int(agg["trades"])
            wins = int(agg["wins"])
            agg["win_rate"] = (wins / trades) if trades > 0 else 0.0

            run.mark_done(metrics=_to_json_safe(agg))
            self.stdout.write(self.style.SUCCESS(f"Backtest done: {agg}"))

        except Exception as e:
            run.mark_failed(str(e))
            raise
