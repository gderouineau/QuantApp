from __future__ import annotations
from django.core.management.base import BaseCommand
from market_data.models import Asset

SETS = {
    "sample_cac40": [
        # symbol (canonique), y_symbol (Yahoo), type
        ("AIR.PA", "AIR.PA", "equity"),   # Airbus
        ("AI.PA",  "AI.PA",  "equity"),   # Air Liquide
        ("BNP.PA", "BNP.PA", "equity"),   # BNP Paribas
        ("CAP.PA", "CAP.PA", "equity"),   # Capgemini
        ("CA.PA",  "CA.PA",  "equity"),   # Carrefour
        ("DG.PA",  "DG.PA",  "equity"),   # Vinci
        ("DSY.PA", "DSY.PA", "equity"),   # Dassault Systèmes
        ("ENGI.PA","ENGI.PA","equity"),   # Engie
        ("KER.PA", "KER.PA", "equity"),   # Kering
        ("LR.PA",  "LR.PA",  "equity"),   # Legrand
        ("MC.PA",  "MC.PA",  "equity"),   # LVMH
        ("ORA.PA", "ORA.PA", "equity"),   # Orange
        ("OR.PA",  "OR.PA",  "equity"),   # L'Oréal
        ("RMS.PA", "RMS.PA", "equity"),   # Hermès
        ("RI.PA",  "RI.PA",  "equity"),   # Pernod Ricard
        ("SAF.PA", "SAF.PA", "equity"),   # Safran
        ("SAN.PA", "SAN.PA", "equity"),   # Sanofi
        ("SGO.PA", "SGO.PA", "equity"),   # Saint-Gobain
        ("SU.PA",  "SU.PA",  "equity"),   # Schneider Electric
        ("TTE.PA", "TTE.PA", "equity"),   # TotalEnergies
        # Index CAC (optionnel)
        ("^FCHI",  "^FCHI",  "index"),    # CAC 40 index
    ],
    "us_mega": [
        ("AAPL", "AAPL", "equity"),
        ("MSFT", "MSFT", "equity"),
        ("GOOGL","GOOGL","equity"),
        ("AMZN", "AMZN", "equity"),
        ("META", "META", "equity"),
        ("^GSPC","^GSPC","index"),  # S&P 500
        ("^NDX", "^NDX", "index"),  # Nasdaq 100
        ("^DJI", "^DJI", "index"),  # Dow Jones
    ],
    "fx_majors": [
        ("EURUSD", "EURUSD=X", "fx"),
        ("GBPUSD", "GBPUSD=X", "fx"),
        ("USDJPY", "USDJPY=X", "fx"),
        ("USDCHF", "USDCHF=X", "fx"),
        ("AUDUSD", "AUDUSD=X", "fx"),
        ("USDCAD", "USDCAD=X", "fx"),
    ],
}

class Command(BaseCommand):
    help = "Seed default assets (e.g. CAC40/US mega/FX majors) without using admin."

    def add_arguments(self, parser):
        parser.add_argument(
            "--set",
            required=True,
            help="Comma-separated sets: sample_cac40,us_mega,fx_majors",
        )

    def handle(self, *args, **opts):
        names = [s.strip() for s in opts["set"].split(",") if s.strip()]
        created, skipped = 0, 0

        for name in names:
            rows = SETS.get(name, [])
            for symbol, y_symbol, atype in rows:
                obj, was_created = Asset.objects.get_or_create(
                    symbol=symbol,
                    defaults={
                        "type": atype,
                        "exchange": "",
                        "currency": "EUR" if symbol.endswith(".PA") else "",
                        "timezone": "Europe/Paris" if symbol.endswith(".PA") else "UTC",
                        "y_symbol": y_symbol,
                        "av_symbol": "",
                        "is_active": True,
                    },
                )
                if was_created:
                    created += 1
                else:
                    skipped += 1

        self.stdout.write(self.style.SUCCESS(f"Assets created={created}, already there={skipped}"))
