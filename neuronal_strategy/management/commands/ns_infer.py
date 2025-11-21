# neuronal_strategy/management/commands/ns_infer.py
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
    help = "Run inference with a saved .pt, write parquet of probs; option --to_db to upsert NSPrediction."

    def add_arguments(self, parser):
        parser.add_argument("--dataset_id", type=int, required=True)
        parser.add_argument("--tp_margin", type=float, default=0.10)
        parser.add_argument("--model_path", type=str, default=None)
        parser.add_argument("--batch_size", type=int, default=4096)
        parser.add_argument("--to_db", action="store_true")

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

            tbl, feat_cols2, ycol = load_joined_table(
                ds.dataset_path, ds.labels_path, tp_margin, x_window=int(ds.x_window or 1)
            )

            if feat_cols2 != feat_cols:
                for c in feat_cols:
                    if c not in tbl.columns:
                        tbl[c] = 0.0
                tbl = tbl[["instrument", "y"] + feat_cols]

            feat_df = (
                tbl[feat_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .astype("float32")
            )
            X = torch.from_numpy(feat_df.values)

            loader = torch.utils.data.DataLoader(X, batch_size=opts["batch_size"], shuffle=False)

            model = SimpleMLP(input_dim=len(feat_cols))
            model.load_state_dict(state["model"])
            model.eval()

            probs_chunks = []
            with torch.no_grad():
                for xb in loader:
                    logits = model(xb)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    probs_chunks.append(probs.reshape(-1))
            probs = np.concatenate(probs_chunks, axis=0)

            out = tbl[["instrument"]].copy()
            out["date"] = tbl.index
            out["prob"] = probs.astype("float32")
            out["y_true"] = tbl["y"].astype(int)

            out_dir = os.path.join(default_data_dir(), "preds")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"preds_ds{ds.id}_tp{int(tp_margin * 100)}.parquet")
            out.to_parquet(out_path, index=False)

            metrics = {
                "preds_path": out_path,
                "rows": int(len(out)),
                "tp_margin": tp_margin,
                "model_best_thr": float(state.get("best_thr", 0.9)),
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
                            prob_tp10=float(row["prob"]) if tp_margin == 0.10 else None,
                            prob_tp15=float(row["prob"]) if tp_margin == 0.15 else None,
                            decision=False,
                            label_tp10=bool(row["y_true"]) if tp_margin == 0.10 else None,
                            label_tp15=bool(row["y_true"]) if tp_margin == 0.15 else None,
                            meta={},
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
