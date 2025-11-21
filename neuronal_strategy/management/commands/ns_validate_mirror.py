from __future__ import annotations
import os, math
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from neuronal_strategy.models import NSDataset, NSRun
from neuronal_strategy.services.io import default_data_dir
from neuronal_strategy.services.backtest import backtest_signals
from neuronal_strategy.selectors.prices import load_universe_ohlcv


def _metrics_from_probs(df: pd.DataFrame, thr: float):
    y = df["y_true"].astype(int).values
    p = (df["prob"].astype("float64").values >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


class Command(BaseCommand):
    help = "Validate predictions parquet: metrics (precision/recall/F1) + backtest sur période validation (mois pairs). Direction-aware via --direction."

    def add_arguments(self, parser):
        parser.add_argument("--dataset_id", type=int, required=True)
        parser.add_argument("--tp_margin", type=float, default=0.10)
        parser.add_argument("--preds_path", type=str, default=None)
        parser.add_argument("--thr", type=float, default=None)
        parser.add_argument("--fees_bps", type=float, default=2.0)
        parser.add_argument("--slippage_bps", type=float, default=2.0)
        parser.add_argument("--min_preds", type=int, default=0)
        parser.add_argument("--target_precision", type=float, default=0.0)
        parser.add_argument("--direction", type=str, default="long", help="long | both (backtest long-only si 'long')")

    def handle(self, *args, **opts):
        ds = NSDataset.objects.get(pk=opts["dataset_id"])
        tp = float(opts["tp_margin"])
        path = opts["preds_path"] or os.path.join(
            default_data_dir(), "preds", f"preds_ds{ds.id}_tp{int(tp * 100)}.parquet"
        )
        if not os.path.isfile(path):
            raise CommandError(f"Predictions parquet not found: {path}")

        run = NSRun.objects.create(dataset=ds, kind="backtest", status="pending")
        run.mark_running()

        try:
            # 1) Lire prédictions & ranger
            df = pd.read_parquet(path)
            if "is_long" not in df.columns:
                raise CommandError("Preds parquet missing 'is_long'. Re-run ns_infer (nouvelle version) pour l'ajouter.")
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.sort_values(["instrument", "date"]).reset_index(drop=True)

            # 2) Masque validation
            from neuronal_strategy.services.datasets import split_masks_by_timeframe
            idx = pd.DatetimeIndex(df["date"])
            _, val_mask = split_masks_by_timeframe(idx, ds.timeframe)
            val_df = df[val_mask.to_numpy()].copy()
            if val_df.empty:
                raise CommandError("Validation set is empty after mask; check dates/timeframe.")

            # 2bis) Direction filtering for metrics/backtest
            direction = str(opts["direction"]).lower().strip()
            if direction == "long":
                val_df = val_df[val_df["is_long"] == True]
            elif direction == "both":
                pass
            else:
                raise CommandError("Unsupported --direction. Use 'long' or 'both'.")

            if val_df.empty:
                raise CommandError("No validation rows after direction filter. Try --direction both.")

            # 3) Métriques vs seuil(s) (sur sous-ensemble directionnel)
            thrs = [float(opts["thr"])] if opts["thr"] is not None else list(np.linspace(0.6, 0.95, 15))
            metrics_list = []
            for thr in thrs:
                m = _metrics_from_probs(val_df, thr)
                m["thr"] = float(thr)
                m["preds"] = int(m["tp"] + m["fp"])
                metrics_list.append(m)

            # 4) Choix du meilleur seuil
            min_preds = int(opts["min_preds"])
            target_prec = float(opts["target_precision"])
            if opts["thr"] is not None:
                best_thr = float(opts["thr"])
            else:
                candidates = [
                    m for m in metrics_list if m["preds"] >= min_preds and m["precision"] >= target_prec
                ] if (min_preds > 0 or target_prec > 0.0) else metrics_list
                candidates.sort(key=lambda d: (d["precision"], d["recall"], d["preds"]), reverse=True)
                best_thr = float(candidates[0]["thr"]) if candidates else float(metrics_list[0]["thr"])

            # 5) Backtest sur période validation — direction-aware (long-only si --direction=long)
            data = load_universe_ohlcv(ds.universe, ds.date_from, ds.date_to, timeframe=ds.timeframe)

            # Signaux par instrument/date
            signals_map = {}
            for instr, g in val_df.groupby("instrument"):
                s = (g.set_index("date")["prob"].astype("float64") >= best_thr).astype(bool)
                # direction filter for signals
                if direction == "long":
                    s = s & (g.set_index("date")["is_long"] == True)
                s.index = pd.to_datetime(s.index, utc=True)
                signals_map[instr] = s

            log_equities = []
            trades = wins = losses = 0

            for instr, dfp in data.items():
                if instr not in signals_map or dfp is None or dfp.empty:
                    continue

                # Assurer DatetimeIndex UTC sur les prix
                if not isinstance(dfp.index, pd.DatetimeIndex):
                    if "date" in dfp.columns:
                        dfp = dfp.set_index("date")
                    dfp.index = pd.to_datetime(dfp.index, utc=True)
                else:
                    dfp.index = pd.to_datetime(dfp.index, utc=True)

                s_aligned = signals_map[instr].reindex(dfp.index, fill_value=False).astype(bool, copy=False)

                m = backtest_signals(
                    df=dfp,
                    signals=s_aligned,
                    tp_margin=tp,
                    horizon_bars=ds.horizon_bars,
                    fees_bps=float(opts["fees_bps"]),
                    slippage_bps=float(opts["slippage_bps"]),
                )

                trades += int(m.get("trades", 0))
                wins += int(m.get("wins", 0))
                losses += int(m.get("losses", 0))

                eq_end = m.get("equity_end", 1.0)
                try:
                    eq_end_f = float(eq_end)
                except Exception:
                    eq_end_f = float("nan")
                if eq_end_f > 0 and not (math.isnan(eq_end_f) or math.isinf(eq_end_f)):
                    log_equities.append(math.log(eq_end_f))

            if log_equities:
                mean_log = float(np.mean(log_equities))
                mean_log = max(min(mean_log, 50.0), -50.0)
                equity_end = math.exp(mean_log)
            else:
                equity_end = 1.0

            # 6) Récap
            metrics_list.sort(key=lambda d: (d["precision"], d["recall"], d["preds"]), reverse=True)
            out_metrics = {
                "best_thr": best_thr,
                "metrics": metrics_list[:5],
                "bt_val": {
                    "trades": trades,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": (wins / trades) if trades else 0.0,
                    "equity_end": float(equity_end),
                },
                "tp_margin": tp,
                "direction": direction,
            }

            run.mark_done(metrics=out_metrics)
            self.stdout.write(self.style.SUCCESS(f"Validate done: {out_metrics}"))

        except Exception as e:
            run.mark_failed(str(e))
            raise
