from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from django.core.management.base import BaseCommand, CommandError

from neuronal_strategy.models import NSDataset, NSRun, NSPrediction
from neuronal_strategy.services.datasets import load_joined_table
from neuronal_strategy.services.io import default_data_dir
from neuronal_strategy.services.torch_models import SimpleMLP


class Command(BaseCommand):
    help = (
        "Run inference with a saved .pt, write parquet of probs; option --to_db to upsert NSPrediction. "
        "Normalise pos_* par y_resolution et MIRROR si is_long=False (aligné train). "
        "Permet --target pour y_true (tp1/tp2/tp3/any/auto). Ajoute is_long dans le parquet."
    )

    def add_arguments(self, parser):
        parser.add_argument("--dataset_id", type=int, required=True)
        parser.add_argument("--tp_margin", type=float, default=0.10)
        parser.add_argument("--model_path", type=str, default=None)
        parser.add_argument("--batch_size", type=int, default=4096)
        parser.add_argument("--to_db", action="store_true")
        parser.add_argument("--target", type=str, default="auto",
                            help="tp1 | tp2 | tp3 | any | auto (pour y_true; le modèle reste le même)")

    def handle(self, *args, **opts):
        ds = NSDataset.objects.get(pk=opts["dataset_id"])
        if not ds.dataset_path or not ds.labels_path:
            raise CommandError("Missing dataset/labels parquet.")

        tp_margin = float(opts["tp_margin"])
        model_path = opts["model_path"] or os.path.join(
            default_data_dir(), "models", f"ds{ds.id}_tp{int(tp_margin * 100)}.pt"
        )
        if not os.path.isfile(model_path):
            raise CommandError(f"Model file not found: {model_path}")

        run = NSRun.objects.create(dataset=ds, kind="infer", status="pending")
        run.mark_running()

        try:
            state = torch.load(model_path, map_location="cpu")
            feat_cols = state["feat_cols"]
            trained_target = state.get("target", "auto")
            trained_y_res = int(state.get("y_resolution", int(ds.y_resolution or 0)))

            # table jointe (contient is_long + y_tp1/2/3 si dispo)
            tbl, feat_cols2, _ = load_joined_table(
                ds.dataset_path, ds.labels_path, tp_margin, x_window=int(ds.x_window or 1)
            )

            # aligne features avec le modèle
            if feat_cols2 != feat_cols:
                for c in feat_cols:
                    if c not in tbl.columns:
                        tbl[c] = 0.0
                tbl = tbl[["instrument"] + [c for c in tbl.columns if c not in ["instrument"]]]

            # normalisation pos_*
            pos_cols = [c for c in feat_cols if c.startswith("pos_")]
            if trained_y_res > 0 and pos_cols:
                tbl[pos_cols] = (tbl[pos_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0) / float(trained_y_res))

            # mirroring si SHORT
            if "is_long" in tbl.columns and pos_cols:
                short_mask = (tbl["is_long"] == False)
                if bool(short_mask.any()):
                    tbl.loc[short_mask, pos_cols] = 1.0 - tbl.loc[short_mask, pos_cols]

            # tenseur X
            feat_df = tbl[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
            X = torch.from_numpy(feat_df.values)

            loader = torch.utils.data.DataLoader(X, batch_size=opts["batch_size"], shuffle=False)

            model = SimpleMLP(input_dim=len(feat_cols))
            model.load_state_dict(state["model"])
            model.eval()

            probs_chunks = []
            with torch.no_grad():
                for xb in loader:
                    logits = model(xb)
                    probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
                    probs_chunks.append(probs)
            probs = np.concatenate(probs_chunks, axis=0)

            # y_true pour analyse (selon --target)
            target = str(opts["target"]).lower().strip()
            y_true = None
            if target in ("tp1", "tp_1") and "y_tp1" in tbl.columns:
                y_true = tbl["y_tp1"].astype(int)
            elif target in ("tp2", "tp_2") and "y_tp2" in tbl.columns:
                y_true = tbl["y_tp2"].astype(int)
            elif target in ("tp3", "tp_3") and "y_tp3" in tbl.columns:
                y_true = tbl["y_tp3"].astype(int)
            elif target in ("any", "tp_any"):
                cols = [c for c in ("y_tp1", "y_tp2", "y_tp3") if c in tbl.columns]
                if cols:
                    y_true = (tbl[cols].max(axis=1) >= 1).astype(int)
            if y_true is None:
                y_true = tbl["y"].astype(int)

            # >>> AJOUT is_long dans la sortie
            out = pd.DataFrame({
                "instrument": tbl["instrument"].values,
                "date": pd.to_datetime(tbl.index, utc=True),
                "prob": probs.astype("float32"),
                "y_true": y_true.values.astype(int),
                "is_long": tbl["is_long"].astype(bool).values if "is_long" in tbl.columns else np.ones(len(tbl), dtype=bool),
            })

            out_dir = os.path.join(default_data_dir(), "preds")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"preds_ds{ds.id}_tp{int(tp_margin * 100)}.parquet")
            out.to_parquet(out_path, index=False)

            metrics = {
                "preds_path": out_path,
                "rows": int(len(out)),
                "tp_margin": tp_margin,
                "model_best_thr": float(state.get("best_thr", 0.9)),
                "trained_target": trained_target,
            }

            if opts["to_db"]:
                recs = []
                batch_size_db = 50_000
                for _, row in out.iterrows():
                    recs.append(
                        NSPrediction(
                            dataset=ds,
                            instrument=row["instrument"],
                            ts=pd.to_datetime(row["date"]),
                            prob_tp10=float(row["prob"]) if abs(tp_margin - 0.10) < 1e-9 else None,
                            prob_tp15=float(row["prob"]) if abs(tp_margin - 0.15) < 1e-9 else None,
                            decision=False,
                            label_tp10=bool(row["y_true"]) if abs(tp_margin - 0.10) < 1e-9 else None,
                            label_tp15=bool(row["y_true"]) if abs(tp_margin - 0.15) < 1e-9 else None,
                            meta={"target": target or trained_target, "is_long": bool(row["is_long"])},
                        )
                    )
                    if len(recs) >= batch_size_db:
                        NSPrediction.objects.bulk_create(recs, ignore_conflicts=True)
                        recs.clear()
                if recs:
                    NSPrediction.objects.bulk_create(recs, ignore_conflicts=True)
                metrics["to_db"] = True

            run.mark_done(metrics=metrics)
            self.stdout.write(self.style.SUCCESS(f"Infer done → {out_path}"))

        except Exception as e:
            run.mark_failed(str(e))
            raise
