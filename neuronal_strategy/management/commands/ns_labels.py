#neuronal_strategy/management/commands/ns_labels.py
from django.core.management.base import BaseCommand, CommandError
from neuronal_strategy.models import NSDataset, NSRun
from neuronal_strategy.selectors.prices import load_universe_ohlcv
from neuronal_strategy.services.labels import compute_labels_for_df
from neuronal_strategy.services.io import save_parquet
import pandas as pd

class Command(BaseCommand):
    help = "Build TP/SL labels for a dataset (long-only, SL before TP, gaps included)."

    def add_arguments(self, parser):
        parser.add_argument("--dataset_id", type=int, required=True)
        parser.add_argument("--date_from", type=str, default=None)
        parser.add_argument("--date_to", type=str, default=None)

    def handle(self, *args, **opts):
        ds_id = opts["dataset_id"]
        date_from = opts.get("date_from")
        date_to = opts.get("date_to")

        ds = NSDataset.objects.get(pk=ds_id)
        run = NSRun.objects.create(dataset=ds, kind="labels", status="pending")
        run.mark_running()

        try:
            universe = ds.universe
            data = load_universe_ohlcv(
                ds.universe,
                date_from or ds.date_from,
                date_to or ds.date_to,
                timeframe=ds.timeframe
            )

            frames = []
            for instr, df in data.items():
                labels = compute_labels_for_df(df, ds.horizon_bars, ds.tp_margins)
                out = labels.copy()
                out["instrument"] = instr
                out = out.reset_index().rename(columns={"index": "date"})  # si ton index s'appelle "index"
                out = out.set_index(["instrument", "date"]).sort_index()
                frames.append(out)

            if not frames:
                raise CommandError("No OHLCV data loaded.")

            big = pd.concat(frames).sort_index()
            path = save_parquet(big, f"labels/dataset_{ds.id}.parquet")
            ds.labels_path = path
            ds.last_built_at = run.started_at
            ds.save(update_fields=["labels_path", "last_built_at"])

            run.mark_done(metrics={"rows": int(len(big))})
            self.stdout.write(self.style.SUCCESS(f"Labels saved -> {path}"))
        except Exception as e:
            run.mark_failed(str(e))
            raise
