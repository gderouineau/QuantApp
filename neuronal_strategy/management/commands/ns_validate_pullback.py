# neuronal_strategy/management/commands/ns_validate_pullback.py
from __future__ import annotations
import os, math
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from neuronal_strategy.models import NSDataset, NSRun
from neuronal_strategy.services.io import default_data_dir
from neuronal_strategy.services.datasets import split_masks_by_timeframe

def _tp_col_for_r(tp_r: float) -> str:
    # 1.5 -> "y_1p5R", 2.0 -> "y_2R", 3.0 -> "y_3R"
    if abs(tp_r - 1.5) < 1e-9:
        return "y_1p5R"
    elif abs(tp_r - 2.0) < 1e-9:
        return "y_2R"
    elif abs(tp_r - 3.0) < 1e-9:
        return "y_3R"
    else:
        raise ValueError(f"Unsupported tp_r={tp_r}. Use 1.5, 2.0 or 3.0.")

def _build_default_preds_path(ds_id: int, tp_r: float) -> str:
    # On suit la convention de ns_infer: ds{ID}_tp{int(tp_r*10)}.pt / preds_ds{ID}_tp{int(tp_r*10)}.parquet
    # ex: 1.5R -> "tp15"
    tag = int(round(tp_r * 10))  # 1.5 -> 15 ; 2.0 -> 20 ; 3.0 -> 30
    return os.path.join(default_data_dir(), "preds", f"preds_ds{ds_id}_tp{tag}.parquet")

def _metrics_from_probs(df: pd.DataFrame, thr: float) -> dict:
    # df doit déjà contenir y_true (bool/int) et prob (float)
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
        "preds": int(p.sum()),
    }

class Command(BaseCommand):
    help = (
        "Validate predictions en mode PULLBACK (R-multiples): "
        "métriques (precision/recall/F1) vs y_1p5R/y_2R/y_3R et backtest R-mode "
        "avec equity = Π(1 + risk_per_trade * outcome_R)."
    )

    def add_arguments(self, parser):
        parser.add_argument("--dataset_id", type=int, required=True)
        parser.add_argument("--tp_r", type=float, default=1.5, help="R target: 1.5, 2.0, 3.0")
        parser.add_argument("--preds_path", type=str, default=None, help="Parquet des prédictions (sinon auto).")
        parser.add_argument("--thr", type=float, default=None, help="Seuil fixe; sinon recherche.")
        parser.add_argument("--thr_min", type=float, default=0.2)
        parser.add_argument("--thr_max", type=float, default=0.95)
        parser.add_argument("--n_thrs", type=int, default=25)
        parser.add_argument("--only_setups", action="store_true")
        parser.add_argument("--only_entries", action="store_true")
        parser.add_argument("--min_preds", type=int, default=0, help="Min #preds positives pour valider un seuil auto.")
        parser.add_argument("--target_precision", type=float, default=0.0, help="Précision cible min pour choix auto.")
        parser.add_argument("--risk_per_trade", type=float, default=0.01, help="Taille de risque par trade (1% = 0.01).")

    def handle(self, *args, **opts):
        ds = NSDataset.objects.get(pk=opts["dataset_id"])
        if not ds.labels_path:
            raise CommandError("labels_path manquant sur le dataset. Lance ns_labels_pullback d'abord.")

        if opts["only_setups"] and opts["only_entries"]:
            raise CommandError("Choisir au plus un filtre: --only_setups OU --only_entries.")

        tp_r = float(opts["tp_r"])
        y_col = _tp_col_for_r(tp_r)
        preds_path = opts["preds_path"] or _build_default_preds_path(ds.id, tp_r)
        if not os.path.isfile(preds_path):
            raise CommandError(f"Predictions parquet not found: {preds_path}")

        run = NSRun.objects.create(dataset=ds, kind="backtest_pullback", status="pending")
        run.mark_running()

        try:
            # 1) charger prédictions + labels
            preds = pd.read_parquet(preds_path)
            lab = pd.read_parquet(ds.labels_path)

            # dates & tri
            preds["date"] = pd.to_datetime(preds["date"], utc=True, errors="coerce")
            lab["date"] = pd.to_datetime(lab["date"], utc=True, errors="coerce")
            preds = preds.dropna(subset=["instrument", "date", "prob"]).sort_values(["instrument", "date"])
            lab = lab.dropna(subset=["instrument", "date"]).sort_values(["instrument", "date"])

            # 2) merge (inner) pour récupérer y_true (labels) aligné sur les prédictions
            keep_cols = ["instrument", "date", y_col, "is_setup", "is_entry"]
            # certaines colonnes peuvent manquer selon la version de labels
            for c in ["is_setup", "is_entry"]:
                if c not in lab.columns:
                    lab[c] = False
            merged = preds.merge(lab[keep_cols], on=["instrument", "date"], how="inner")

            if merged.empty:
                raise CommandError("Aucun recouvrement entre predictions et labels. Vérifie dates/timeframe.")

            # 3) filtrage optionnel (cohérent avec l'entraînement)
            if opts["only_setups"]:
                merged = merged[merged["is_setup"] == True]
            elif opts["only_entries"]:
                merged = merged[merged["is_entry"] == True]

            if merged.empty:
                raise CommandError("Après filtrage setups/entries, plus de données. Relaxe les filtres.")

            # 4) split temporel pair/impair aligné
            merged = merged.sort_values(["instrument", "date"]).reset_index(drop=True)
            idx = pd.DatetimeIndex(merged["date"])
            _, val_mask = split_masks_by_timeframe(idx, ds.timeframe)
            val_df = merged[val_mask.to_numpy()].copy()
            if val_df.empty:
                raise CommandError("Validation set vide après split. Vérifie dates/timeframe.")

            # 5) y_true et métriques
            val_df["y_true"] = val_df[y_col].astype(bool)

            if opts["thr"] is None:
                thrs = np.linspace(float(opts["thr_min"]), float(opts["thr_max"]), int(opts["n_thrs"]))
            else:
                thrs = [float(opts["thr"])]

            metrics_list = []
            for thr in thrs:
                m = _metrics_from_probs(val_df, thr)
                m["thr"] = float(thr)
                metrics_list.append(m)

            # choix du meilleur seuil (contraintes optionnelles)
            min_preds = int(opts["min_preds"])
            target_prec = float(opts["target_precision"])
            if opts["thr"] is not None:
                best_thr = float(opts["thr"])
            else:
                candidates = [
                    m for m in metrics_list if m["preds"] >= min_preds and m["precision"] >= target_prec
                ] if (min_preds > 0 or target_prec > 0.0) else metrics_list
                candidates.sort(key=lambda d: (d["precision"], d["recall"], d["preds"]), reverse=True)
                if not candidates:
                    raise CommandError("Aucun seuil ne satisfait min_preds/target_precision.")
                best_thr = float(candidates[0]["thr"])

            # 6) backtest R-mode : outcome_R = +tp_r si y_true=1 quand prob>=thr, sinon −1
            sel = val_df[val_df["prob"] >= best_thr]
            trades = int(len(sel))
            wins = int(sel["y_true"].sum())
            losses = trades - wins

            risk = float(opts["risk_per_trade"])
            equity_end = ((1.0 + risk * tp_r) ** wins) * ((1.0 - risk) ** losses)
            win_rate = (wins / trades) if trades else 0.0

            # 7) sortie
            metrics_list.sort(key=lambda d: (d["precision"], d["recall"], d["preds"]), reverse=True)
            out_metrics = {
                "mode": "pullback_R",
                "tp_r": float(tp_r),
                "best_thr": float(best_thr),
                "metrics": metrics_list[:5],
                "bt_val": {
                    "trades": trades,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": float(win_rate),
                    "equity_end": float(equity_end),
                    "risk_per_trade": risk,
                },
                "filters": {
                    "only_setups": bool(opts["only_setups"]),
                    "only_entries": bool(opts["only_entries"]),
                },
                "paths": {
                    "preds": preds_path,
                    "labels": ds.labels_path,
                },
            }

            run.mark_done(metrics=out_metrics)
            self.stdout.write(self.style.SUCCESS(f"Validate (pullback) done: {out_metrics}"))

        except Exception as e:
            run.mark_failed(str(e))
            raise
