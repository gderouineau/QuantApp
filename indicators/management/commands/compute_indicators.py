# indicators/management/commands/compute_indicators.py
from django.core.management.base import BaseCommand
from django.db import transaction
from pathlib import Path
from market_data.models import Asset, DataFile
from indicators.models import Indicator
from market_data.services.store import read_parquet
from indicators.services.indicators import compute_all_indicators

class Command(BaseCommand):
    help = "Calcule et stocke les indicateurs techniques pour tous les assets"

    def add_arguments(self, parser):
        parser.add_argument("--symbols", help="Liste séparée par des virgules; sinon tous", default=None)

    def handle(self, *args, **opts):
        symbols = [s.strip() for s in opts["symbols"].split(",")] if opts["symbols"] else None
        assets = Asset.objects.filter(is_active=True)
        if symbols:
            assets = assets.filter(symbol__in=symbols)

        total = 0
        for asset in assets:
            try:
                df_file = DataFile.objects.filter(asset=asset, kind="bars_1D").first()
                if not df_file:
                    self.stdout.write(self.style.WARNING(f"No data for {asset.symbol}"))
                    continue

                df = read_parquet(Path(df_file.path))
                if df is None or df.empty:
                    continue

                indicators_df = compute_all_indicators(df)
                if indicators_df is None or indicators_df.empty:
                    continue

                with transaction.atomic():
                    Indicator.objects.filter(asset=asset).delete()

                    batch = []
                    for ts, row in indicators_df.iterrows():
                        # ts peut être un Timestamp: force -> date
                        d = getattr(ts, "date", lambda: ts)()
                        batch.append(Indicator(
                            asset=asset,
                            date=d,
                            sma_20=row.get('sma_20'),
                            sma_50=row.get('sma_50'),
                            sma_200=row.get('sma_200'),
                            ema_12=row.get('ema_12'),
                            ema_26=row.get('ema_26'),
                            rsi_14=row.get('rsi_14'),
                            macd=row.get('macd'),
                            macd_signal=row.get('macd_signal'),
                            macd_hist=row.get('macd_hist'),
                            bb_upper=row.get('bb_upper'),
                            bb_middle=row.get('bb_middle'),
                            bb_lower=row.get('bb_lower'),
                            atr_14=row.get('atr_14'),
                            volume_sma_20=row.get('volume_sma_20'),
                        ))

                    if batch:
                        Indicator.objects.bulk_create(batch, batch_size=1000)
                        total += len(batch)
                        self.stdout.write(self.style.SUCCESS(f"✓ {asset.symbol}: {len(batch)} indicators"))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"✗ {asset.symbol}: {e}"))

        self.stdout.write(self.style.SUCCESS(f"\nTotal: {total} indicators computed"))
