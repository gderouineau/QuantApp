# market_data/management/commands/ingest_daily.py

from __future__ import annotations
import time
from datetime import date
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.conf import settings

from providers.providers.registry import get_provider, provider_chain
from market_data.models import Asset, IngestionRun
from market_data.services.store import bars_path, write_parquet, upsert_datafile


class Command(BaseCommand):
    help = "Ingestion daily (1D) via providers chain; écrit Parquet + DB DataFile/Run."

    def add_arguments(self, parser):
        parser.add_argument("--symbols", help="Comma list; sinon prend tous les assets actifs", default=None)
        parser.add_argument("--start", default=None)  # YYYY-MM-DD
        parser.add_argument("--end", default=None)

    def handle(self, *args, **opts):
        symbols = [s.strip() for s in opts["symbols"].split(",")] if opts["symbols"] else None
        start = date.fromisoformat(opts["start"]) if opts["start"] else None
        end = date.fromisoformat(opts["end"]) if opts["end"] else None

        assets = list(Asset.objects.filter(is_active=True).order_by("symbol"))
        if symbols:
            symset = set(symbols)
            assets = [a for a in assets if a.symbol in symset]

        providers = provider_chain()
        run = IngestionRun.objects.create(source=providers[0], timeframe="1D", started_at=timezone.now())

        ok, ko, anomalies, errors = 0, 0, [], []
        t0 = time.time()

        remaining = assets
        for pname in providers:
            prov = get_provider(pname)
            sym_map = {a.for_provider(pname): a for a in remaining}
            batch = list(sym_map.keys())
            if not batch:
                break

            try:
                data = prov.fetch_bars(batch, "1D", start, end)
            except Exception as e:
                errors.append(f"{pname}: batch error: {e}")
                continue

            done = set()
            for p_sym, df in data.items():
                a = sym_map.get(p_sym)
                if not a or df is None or df.empty:
                    continue
                cols = [c for c in ["open","high","low","close","adj_close","volume"] if c in df.columns]
                df = df[cols].copy(); df.index.name = "date"
                path = bars_path(a.symbol, "1D")
                n, last = write_parquet(df, path)
                upsert_datafile(a, "bars_1D", path, n, last)
                ok += 1; done.add(p_sym)

            remaining = [sym_map[k] for k in batch if k not in done]
            if not remaining:
                break

        ko = len(remaining)
        if ko:
            for a in remaining:
                errors.append(f"no_data:{a.symbol}")

        run.ok_count = ok
        run.fail_count = ko
        run.anomalies = anomalies
        run.errors = errors
        run.duration_ms = int((time.time() - t0) * 1000)
        run.finished_at = timezone.now()
        run.save()

        self.stdout.write(self.style.SUCCESS(f"Done. ok={ok} ko={ko} ms={run.duration_ms}"))
