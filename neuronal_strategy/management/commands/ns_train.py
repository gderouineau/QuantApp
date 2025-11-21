# neuronal_strategy/management/commands/ns_train.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from django.core.management.base import BaseCommand, CommandError

from neuronal_strategy.models import NSDataset, NSRun
from neuronal_strategy.services.datasets import load_joined_table, split_masks_by_timeframe
from neuronal_strategy.services.io import default_data_dir
from neuronal_strategy.services.torch_models import SimpleMLP


def _tensorize(df: pd.DataFrame, feat_cols: list[str]):
    X_np = (
        df[feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype("float32")
        .values
    )
    y_np = df["y"].astype("float32").values
    return torch.from_numpy(X_np), torch.from_numpy(y_np)


def _pr_curve_thresholds(
    probs: np.ndarray,
    y: np.ndarray,
    thr_min: float = 0.2,
    thr_max: float = 0.95,
    n: int = 25,
):
    thrs = np.linspace(thr_min, thr_max, n)
    out = []
    for thr in thrs:
        preds = probs >= thr
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        out.append({"thr": float(thr), "precision": precision, "recall": recall, "preds": int(preds.sum())})
    return out


class Command(BaseCommand):
    help = (
        "Train a simple MLP on features vs labels. "
        "Options: gating (is_entry/is_setup), seuils (thr_min/thr_max), "
        "normalisation automatique des colonnes pos_* par y_resolution."
    )

    def add_arguments(self, parser):
        parser.add_argument("--dataset_id", type=int, required=True)
        parser.add_argument("--tp_margin", type=float, default=0.15)
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--batch_size", type=int, default=1024)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--hidden", type=int, default=128)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--pos_weight", type=float, default=None, help="Override; sinon auto depuis le ratio de classe (clamp 1..20).")
        parser.add_argument("--early_stop_patience", type=int, default=4)
        parser.add_argument("--thr_min", type=float, default=0.2)
        parser.add_argument("--thr_max", type=float, default=0.95)
        parser.add_argument("--gate_entries", action="store_true", help="Garde uniquement les lignes où is_entry=True.")
        parser.add_argument("--gate_setups", action="store_true", help="Garde uniquement les lignes où is_setup=True.")

    def handle(self, *args, **opts):
        torch.set_num_threads(max(1, os.cpu_count() or 1))

        ds = NSDataset.objects.get(pk=opts["dataset_id"])
        if not ds.dataset_path or not ds.labels_path:
            raise CommandError("Missing dataset/labels parquet. Run ns_dataset_indicators ET ns_labels_pullback d'abord.")

        if opts["gate_entries"] and opts["gate_setups"]:
            raise CommandError("Choisir au plus un gating: --gate_entries OU --gate_setups.")

        run = NSRun.objects.create(dataset=ds, kind="train", status="pending")
        run.mark_running()

        try:
            tp_margin = float(opts["tp_margin"])

            # 1) charge table (features + labels), X_window déjà géré dans load_joined_table
            tbl, feat_cols, _ = load_joined_table(
                ds.dataset_path, ds.labels_path, tp_margin, x_window=int(ds.x_window or 1)
            )

            # 2) normalise toutes les features pos_* (y compris fenêtrées pos_*_t-*)
            y_res = int(ds.y_resolution or 0)
            if y_res > 0:
                pos_cols = [c for c in feat_cols if c.startswith("pos_")]
                if pos_cols:
                    tbl[pos_cols] = (
                        tbl[pos_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0) / float(y_res)
                    )

            # 3) gating optionnel sur labels propagés
            if opts["gate_entries"]:
                if "is_entry" in tbl.columns:
                    tbl = tbl[tbl["is_entry"] == True]
                else:
                    self.stdout.write(self.style.WARNING("is_entry absent des labels; gating ignoré."))
            elif opts["gate_setups"]:
                if "is_setup" in tbl.columns:
                    tbl = tbl[tbl["is_setup"] == True]
                else:
                    self.stdout.write(self.style.WARNING("is_setup absent des labels; gating ignoré."))

            if tbl.empty or tbl["y"].sum() == 0:
                raise CommandError(
                    "Après jointure/gating, aucune cible positive. "
                    "Relaxe tes labels (bb_k/horizon/stop) ou enlève le gating."
                )

            # 4) split temporel pair/impair
            train_mask, val_mask = split_masks_by_timeframe(tbl.index, ds.timeframe)
            train_df = tbl[train_mask].dropna(subset=["y"])
            val_df = tbl[val_mask].dropna(subset=["y"])
            if len(train_df) == 0 or len(val_df) == 0:
                raise CommandError("Split train/val vide. Vérifie les dates et la timeframe.")

            # stats de classe
            p = float(train_df["y"].mean())
            self.stdout.write(self.style.WARNING(
                f"train_rows={len(train_df)} val_rows={len(val_df)}  pos_rate_train={p:.6f}"
            ))

            # 5) tensors + loaders
            Xtr, ytr = _tensorize(train_df, feat_cols)
            Xva, yva = _tensorize(val_df, feat_cols)
            tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=opts["batch_size"], shuffle=True, drop_last=False)
            va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=4096, shuffle=False, drop_last=False)

            # 6) modèle + opti
            model = SimpleMLP(input_dim=len(feat_cols), hidden=opts["hidden"], dropout=opts["dropout"])
            opt = torch.optim.Adam(model.parameters(), lr=opts["lr"], weight_decay=opts["weight_decay"])

            # 7) BCE with logits + pos_weight auto (clamp 1..20) si non fourni
            if opts["pos_weight"] is not None:
                pos_weight = torch.tensor([float(opts["pos_weight"])], dtype=torch.float32)
            else:
                p_safe = max(p, 1e-9)
                w = (1 - p_safe) / p_safe
                w = float(min(max(w, 1.0), 20.0))  # clamp 1..20
                pos_weight = torch.tensor([w], dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            best_val_prec = -1.0
            best_state = None
            bad = 0

            # 8) boucle d'entraînement
            for epoch in range(int(opts["epochs"])):
                model.train()
                tr_loss = 0.0
                for xb, yb in tr_loader:
                    opt.zero_grad()
                    logits = model(xb).squeeze(-1)  # (B,)
                    loss = criterion(logits, yb)
                    loss.backward()
                    opt.step()
                    tr_loss += float(loss.item()) * len(xb)
                tr_loss /= max(1, len(train_df))

                # validation
                model.eval()
                with torch.no_grad():
                    logits_all = []
                    for xb, _ in va_loader:
                        logits_all.append(model(xb).squeeze(-1))
                    logits_all = torch.cat(logits_all, dim=0)
                    probs = torch.sigmoid(logits_all).cpu().numpy()
                    ytrue = yva.cpu().numpy()

                curve = _pr_curve_thresholds(
                    probs, ytrue,
                    thr_min=float(opts["thr_min"]),
                    thr_max=float(opts["thr_max"]),
                    n=25
                )
                curve.sort(key=lambda d: (d["precision"], d["recall"], d["preds"]), reverse=True)
                top = curve[0] if curve else {"precision": 0.0, "recall": 0.0, "thr": 0.9}
                val_prec = float(top["precision"])

                if val_prec > best_val_prec:
                    best_val_prec = val_prec
                    best_state = {
                        "model": model.state_dict(),
                        "feat_cols": feat_cols,
                        "input_dim": len(feat_cols),
                        "pos_weight": float(pos_weight.item()),
                        "best_thr": float(top["thr"]),
                        "curve": curve[:5],
                    }
                    bad = 0
                else:
                    bad += 1
                    if bad >= int(opts["early_stop_patience"]):
                        break

                self.stdout.write(self.style.SUCCESS(
                    f"epoch {epoch+1}/{opts['epochs']}  tr_loss={tr_loss:.4f}  "
                    f"val_prec@best={val_prec:.4f}  thr≈{top['thr']:.2f}"
                ))

            if best_state is None:
                raise CommandError("No best state recorded; training failed?")

            # 9) sauvegarde
            out_dir = os.path.join(default_data_dir(), "models")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"ds{ds.id}_tp{int(tp_margin*100)}.pt")
            torch.save(best_state, out_path)

            run.mark_done(metrics={
                "val_best_precision": float(best_val_prec),
                "model_path": out_path,
                "tp_margin": float(tp_margin),
                "feat_count": len(feat_cols),
                "pos_rate_train": float(p),
            })
            self.stdout.write(self.style.SUCCESS(f"Saved best model -> {out_path}"))

        except Exception as e:
            run.mark_failed(str(e))
            raise
