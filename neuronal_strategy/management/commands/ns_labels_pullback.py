from __future__ import annotations
import os
import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from neuronal_strategy.models import NSDataset, NSRun
from neuronal_strategy.services.io import default_data_dir
from neuronal_strategy.selectors.prices import load_universe_ohlcv
from neuronal_strategy.services.labels_pullback import compute_pullback_labels

class Command(BaseCommand):
    help = "Labels pullback MM20/50 + BB + break, stop=lookback-N, TP en R-multiples. Écrit un parquet FLAT (instrument,date, ...)."

    def add_arguments(self, parser):
        parser.add_argument("--dataset_id", type=int, required=True)
        parser.add_argument("--bb_k", type=float, default=2.0)
        parser.add_argument("--horizon", type=int, default=30)
        parser.add_argument("--stop_lookback", type=int, default=3)

    def handle(self, *args, **opts):
        ds = NSDataset.objects.get(pk=opts["dataset_id"])
        tf = (ds.timeframe or getattr(ds.universe, "timeframe", None) or "1D").upper().strip()

        run = NSRun.objects.create(dataset=ds, kind="labels_pullback", status="pending")
        run.mark_running()

        try:
            data = load_universe_ohlcv(ds.universe, ds.date_from, ds.date_to, timeframe=tf)
            if not data:
                raise CommandError(f"Aucune donnée chargée (univers/TF={tf}). Vérifie DataFile.kind et timeframe).")

            frames = []
            for instr, df in data.items():
                if df is None or df.empty:
                    self.stdout.write(self.style.WARNING(f"[skip] {instr}: dataframe vide"))
                    continue

                lab = compute_pullback_labels(
                    df,
                    horizon_bars=int(opts["horizon"]),
                    bb_k=float(opts["bb_k"]),
                    stop_lookback=int(opts["stop_lookback"]),
                )
                if lab is None or lab.empty:
                    self.stdout.write(self.style.WARNING(f"[skip] {instr}: labels vides"))
                    continue

                # Index -> colonne 'date', + 'instrument'
                lab = lab.reset_index().rename(columns={"index": "date"})
                if "date" not in lab.columns:
                    lab["date"] = pd.to_datetime(df.index, utc=True, errors="coerce")
                else:
                    lab["date"] = pd.to_datetime(lab["date"], utc=True, errors="coerce")
                lab["instrument"] = instr

                keep = [
                    "instrument", "date",
                    "y_1p5R", "y_2R", "y_3R",
                    "is_entry", "trend_up", "is_setup",
                    "entry", "stop", "R",
                ]
                lab = lab[[c for c in keep if c in lab.columns]].copy()

                for c in ["y_1p5R", "y_2R", "y_3R", "is_entry", "trend_up", "is_setup"]:
                    if c in lab.columns:
                        lab[c] = lab[c].fillna(False).astype(bool)
                for c in ["entry", "stop", "R"]:
                    if c in lab.columns:
                        lab[c] = pd.to_numeric(lab[c], errors="coerce")

                frames.append(lab)

            if not frames:
                raise CommandError("Aucun instrument avec labels valides.")

            big = pd.concat(frames, axis=0, ignore_index=True)
            big = big.dropna(subset=["date"])
            big = big.sort_values(["instrument", "date"]).reset_index(drop=True)

            out_dir = os.path.join(default_data_dir(), "labels")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"pullback_ds_{ds.id}.parquet")
            big.to_parquet(out_path, index=False)

            ds.labels_path = out_path
            ds.save(update_fields=["labels_path"])

            run.mark_done(metrics={
                "rows": int(len(big)),
                "instruments": int(big["instrument"].nunique()),
                "timeframe": tf,
                "path": out_path,
            })
            self.stdout.write(self.style.SUCCESS(f"Pullback labels saved -> {out_path}"))

        except Exception as e:
            run.mark_failed(str(e))
            raise

