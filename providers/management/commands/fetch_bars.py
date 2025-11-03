from __future__ import annotations
from datetime import date
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from providers.providers.registry import get_provider

SPECS = {"1D": "1d", "1W": "1w"}

def _safe_symbol(sym: str) -> str:
    return sym.replace("/", "_").replace(":", "_").replace("=", "_")

def _out_path(symbol: str, timeframe: str) -> Path:
    outdir = Path(settings.MARKET_DATA_DIR) / SPECS[timeframe]
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"{_safe_symbol(symbol)}.parquet"

class Command(BaseCommand):
    help = "Fetch OHLCV bars from a provider and write Parquet files under MARKET_DATA_DIR."

    def add_arguments(self, parser):
        parser.add_argument("--provider", choices=["yahoo", "alphavantage"], default="yahoo")
        parser.add_argument("--symbols", required=True, help="Comma-separated list, e.g. AAPL,MSFT,HO.PA")
        parser.add_argument("--tf", choices=["1D", "1W"], default="1D")
        parser.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
        parser.add_argument("--end", default=None, help="YYYY-MM-DD (optional)")

    def handle(self, *args, **opts):
        provider_name = opts["provider"]
        symbols = [s.strip() for s in opts["symbols"].split(",") if s.strip()]
        tf = opts["tf"]

        start = date.fromisoformat(opts["start"]) if opts["start"] else None
        end = date.fromisoformat(opts["end"]) if opts["end"] else None

        provider = get_provider(provider_name)
        self.stdout.write(self.style.NOTICE(f"Provider={provider_name} tf={tf} symbols={symbols}"))

        data = provider.fetch_bars(symbols, tf, start, end)

        wrote = 0
        for sym, df in data.items():
            if df is None or df.empty:
                self.stdout.write(self.style.WARNING(f"{sym}: empty dataframe"))
                continue
            df = df[[c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]].copy()
            df.index.name = "date"
            out = _out_path(sym, tf)
            df.to_parquet(out, compression="snappy")
            wrote += 1
            self.stdout.write(self.style.SUCCESS(f"Wrote {sym} -> {out} ({len(df)} rows)"))

        if wrote == 0:
            raise CommandError("No files written")
        self.stdout.write(self.style.SUCCESS(f"Done. {wrote}/{len(symbols)} symbols written."))
