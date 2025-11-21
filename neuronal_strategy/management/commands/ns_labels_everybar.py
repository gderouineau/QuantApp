from __future__ import annotations
import os
import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from neuronal_strategy.models import NSDataset, NSRun
from neuronal_strategy.services.io import default_data_dir
from neuronal_strategy.selectors.prices import load_universe_ohlcv
from neuronal_strategy.services.labels_everybar import compute_trend_everybar_labels

class Command(BaseCommand):
    help = "Labels every-bar (trend SMA20/50, SL lookback, TP 1/2/3×R, entry at next open)."

    def add_arguments(self, parser):
        parser.add_argument("--dataset_id", type=int, required=True)
        parser.add_argument("--ma_fast", type=int, default=20)
        parser.add_argument("--ma_slow", type=int, default=50)
        parser.add_argument("--lookback_stop", type=int, default=5)
        parser.add_argument("--horizon", type=int, default=30)
        parser.add_argument("--tps", type=str, default="1.0,2.0,3.0")

    def handle(self, *args, **opts):
        ds = NSDataset.objects.get(pk=opts["dataset_id"])
        tf = (ds.timeframe or getattr(ds.universe, "timeframe", None) or "1D").upper().strip()

        run = NSRun.objects.create(dataset=ds, kind="labels_everybar", status="pending")
        run.mark_running()

        try:
            tps = tuple(float(x.strip()) for x in str(opts["tps"]).split(",") if x.strip())
            data = load_universe_ohlcv(ds.universe, ds.date_from, ds.date_to, timeframe=tf)
            if not data:
                raise CommandError(f"Aucune donnée chargée (univers/TF={tf}).")

            frames = []
            for instr, df in data.items():
                if df is None or df.empty:
                    self.stdout.write(self.style.WARNING(f"[skip] {instr}: df vide"))
                    continue

                lab = compute_trend_everybar_labels(
                    df=df,
                    ma_fast=int(opts["ma_fast"]),
                    ma_slow=int(opts["ma_slow"]),
                    lookback_stop=int(opts["lookback_stop"]),
                    horizon_bars=int(opts["horizon"]),
                    tps=tps,
                )
                if lab is None or lab.empty:
                    self.stdout.write(self.style.WARNING(f"[skip] {instr}: labels vides"))
                    continue

                out = lab.reset_index().rename(columns={"index": "date"})
                out["instrument"] = instr
                out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
                out = out.dropna(subset=["date"]).sort_values("date")
                frames.append(out)

            if not frames:
                raise CommandError("Aucun instrument labellisé.")

            big = pd.concat(frames, axis=0, ignore_index=True)
            big = big.sort_values(["instrument", "date"]).reset_index(drop=True)

            out_dir = os.path.join(default_data_dir(), "labels")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"everybar_ds_{ds.id}.parquet")
            big.to_parquet(out_path, index=False)

            ds.labels_path = out_path
            ds.save(update_fields=["labels_path"])

            run.mark_done(metrics={
                "rows": int(len(big)),
                "instruments": int(big["instrument"].nunique()),
                "timeframe": tf,
                "path": out_path,
            })
            self.stdout.write(self.style.SUCCESS(f"Every-bar labels saved -> {out_path}"))

        except Exception as e:
            run.mark_failed(str(e))
            raise
