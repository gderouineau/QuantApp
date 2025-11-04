# strategies/management/commands/scan_signals.py

from django.core.management.base import BaseCommand
import pandas as pd
from datetime import date

from market_data.models import Asset
from indicators.models import Indicator
from market_data.services.store import read_parquet, bars_path
from strategies.models import Strategy, Signal

from strategies.relative_strength import RelativeStrengthStrategy
from strategies.custom_strategy import CustomStrategy
from strategies.golden_cross import GoldenCrossStrategy
from strategies.volume_breakout import VolumeBreakoutStrategy


class Command(BaseCommand):
    help = "Scanne les actions actives avec les stratégies actives et enregistre les signaux du jour."

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbols",
            help="Liste de tickers séparés par des virgules (ex: AAPL,MSFT). Si omis, scanne tout l'univers equity actif.",
            default=None,
        )

    def handle(self, *args, **opts):
        # Filtre optionnel par symboles
        symbols = [s.strip() for s in (opts.get("symbols") or "").split(",") if s.strip()]

        strategies = Strategy.objects.filter(is_active=True)
        assets = Asset.objects.filter(is_active=True, type="equity")
        if symbols:
            assets = assets.filter(symbol__in=symbols)

        today = date.today()
        total_signals = 0

        for asset in assets:
            try:
                # 1) Prix (Parquet 1D)
                path = bars_path(asset.symbol, "1D")
                df_prices = read_parquet(path)
                if df_prices is None or df_prices.empty:
                    continue

                # 2) Indicateurs (DB → DataFrame)
                qs = Indicator.objects.filter(asset=asset).order_by("date").values()
                df_indicators = pd.DataFrame(list(qs))
                if df_indicators.empty:
                    continue
                df_indicators.set_index("date", inplace=True)

                # 3) Évaluation des stratégies actives
                for strategy in strategies:
                    if strategy.type == "CUSTOM":
                        strat_instance = CustomStrategy(strategy.code, strategy.parameters)
                    elif strategy.type == "RS":
                        strat_instance = RelativeStrengthStrategy(strategy.parameters)
                    elif strategy.type == "GC":
                        strat_instance = GoldenCrossStrategy(strategy.parameters)
                    elif strategy.type == "VB":
                        strat_instance = VolumeBreakoutStrategy(strategy.parameters)
                    else:
                        # Types non implémentés ici
                        continue

                    # Contrat: evaluate(asset_data_df, indicators_df) -> dict
                    result = strat_instance.evaluate(df_prices, df_indicators)

                    if not result or not result.get("signal"):
                        continue

                    # Plan de trade (optionnel) : on le range dans details.plan
                    plan = {
                        "direction": result.get("direction") or result.get("details", {}).get("direction", "LONG"),
                        "entry_price": result.get("entry_price") or result.get("details", {}).get("entry_price"),
                        "stop_price": result.get("stop_price") or result.get("details", {}).get("stop_price"),
                        "take_profits": result.get("take_profits") or result.get("details", {}).get("take_profits", []),
                        "rr_target": result.get("rr_target") or result.get("details", {}).get("rr_target"),
                    }
                    details = result.get("details", {}) or {}
                    details["plan"] = plan

                    Signal.objects.update_or_create(
                        strategy=strategy,
                        asset=asset,
                        date=today,
                        defaults={
                            "score": float(result.get("score", 0.0)),
                            "strength": result.get("strength", "WEAK"),
                            "details": details,
                        },
                    )
                    total_signals += 1
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"✓ {asset.symbol}: {result.get('score', 0):.1f} "
                            f"({result.get('strength')}) [{strategy.name}]"
                        )
                    )

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"✗ {asset.symbol}: {e}"))

        self.stdout.write(self.style.SUCCESS(f"\nTotal signals: {total_signals}"))
