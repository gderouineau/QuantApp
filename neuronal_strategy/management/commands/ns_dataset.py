# neuronal_strategy/management/commands/ns_dataset.py
from django.core.management.base import BaseCommand, CommandError
from neuronal_strategy.models import NSDataset, NSRun
from neuronal_strategy.selectors.prices import load_universe_ohlcv
from neuronal_strategy.services.features import build_features
from neuronal_strategy.services.io import save_parquet
import pandas as pd

class Command(BaseCommand):
    help = "Build feature dataset for training/inference (grid_scaled or relative_feats)."

    def add_arguments(self, parser):
        parser.add_argument("--dataset_id", type=int, required=True)
        parser.add_argument("--scaling_mode", type=str, default=None)

    def handle(self, *args, **opts):
        ds_id = opts["dataset_id"]
        scaling_override = opts.get("scaling_mode")

        ds = NSDataset.objects.get(pk=ds_id)
        run = NSRun.objects.create(dataset=ds, kind="dataset", status="pending")
        run.mark_running()

        try:
            universe = ds.universe
            data = load_universe_ohlcv(ds.universe, ds.date_from, ds.date_to)

            frames = []
            for instr, df in data.items():
                feat = build_features(
                    df=df,
                    y_resolution=ds.y_resolution,
                    scaling_mode=scaling_override or ds.scaling_mode,
                    use_ma=ds.use_ma, ma_periods=ds.ma_periods,
                    use_bb=ds.use_bb, bb_period=ds.bb_period, bb_k=ds.bb_k,
                    use_atr=ds.use_atr, atr_period=ds.atr_period
                )
                feat["instrument"] = instr
                frames.append(feat)

            if not frames:
                raise CommandError("No features built (empty OHLCV?).")

            big = pd.concat(frames).sort_index()
            path = save_parquet(big, f"features/dataset_{ds.id}_{ds.scaling_mode}.parquet")
            ds.dataset_path = path
            ds.last_built_at = run.started_at
            ds.save(update_fields=["dataset_path", "last_built_at"])

            run.mark_done(metrics={"rows": int(len(big))})
            self.stdout.write(self.style.SUCCESS(f"Features saved -> {path}"))
        except Exception as e:
            run.mark_failed(str(e))
            raise
