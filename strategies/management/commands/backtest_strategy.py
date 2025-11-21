# app: strategies
# file: strategies/management/commands/backtest_strategy.py

from django.core.management.base import BaseCommand
import pandas as pd

from strategies.models import Strategy
from strategies.relative_strength import RelativeStrengthStrategy
from strategies.custom_strategy import CustomStrategy
from strategies.golden_cross import GoldenCrossStrategy
from strategies.volume_breakout import VolumeBreakoutStrategy

# Simulateur existant (fallback)
from strategies.services.backtest import simulate_one_symbol, BTParams
# Nouveau simulateur SL/TP avancé
from strategies.services.sltp import simulate_one_symbol_sltp

from indicators.models import Indicator
from market_data.services.store import read_parquet, bars_path


STRAT_MAP = {
    "CUSTOM": lambda s: CustomStrategy(code=s.code, parameters=s.parameters),
    "RS":     lambda s: RelativeStrengthStrategy(s.parameters),
    "GC":     lambda s: GoldenCrossStrategy(s.parameters),
    "VB":     lambda s: VolumeBreakoutStrategy(s.parameters),
}


class Command(BaseCommand):
    help = "Backtest simple d'une stratégie sur une liste de symboles (1D)."

    def add_arguments(self, parser):
        parser.add_argument("--strategy-id", type=int, required=True)
        parser.add_argument("--symbols", type=str, required=True)   # "AAPL,MSFT"
        parser.add_argument("--capital", type=float, default=100000)
        parser.add_argument("--risk", type=float, default=0.01)
        parser.add_argument("--warmup", type=int, default=252)

    def handle(self, *args, **opts):
        strategy_id = opts["strategy_id"]
        symbols_arg = opts["symbols"]
        capital = float(opts["capital"])
        risk = float(opts["risk"])
        warmup_bars = int(opts["warmup"])

        try:
            s = Strategy.objects.get(pk=strategy_id)
        except Strategy.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Strategy id={strategy_id} introuvable"))
            return

        ctor = STRAT_MAP.get(s.type)
        if ctor is None:
            self.stdout.write(self.style.ERROR(f"Type non supporté pour ce test: {s.type}"))
            return

        strat = ctor(s)
        params = BTParams(initial_capital=capital, risk_per_trade=risk)

        syms = [x.strip() for x in symbols_arg.split(",") if x.strip()]
        totals = {"n_trades": 0, "sum_R": 0.0, "symbols": []}

        # Active SLTP si présent dans parameters
        sltp_cfg = (s.parameters or {}).get("sltp")

        for sym in syms:
            dfp = read_parquet(bars_path(sym, "1D"))
            if dfp is None or dfp.empty:
                self.stdout.write(self.style.WARNING(f"{sym}: pas de données 1D"))
                continue

            dfi = pd.DataFrame(list(
                Indicator.objects.filter(asset__symbol=sym).order_by("date").values()
            ))
            if not dfi.empty:
                dfi.set_index("date", inplace=True)

            if sltp_cfg:
                res = simulate_one_symbol_sltp(
                    df_prices=dfp,
                    df_ind=dfi,
                    strat=strat,
                    initial_capital=capital,
                    risk_per_trade=risk,
                    sltp_cfg=sltp_cfg,
                    warmup_bars=warmup_bars,
                )
            else:
                res = simulate_one_symbol(dfp, dfi, strat, params, warmup_bars=warmup_bars)

            n = res["n_trades"]
            avg_R = res["avg_R"]
            wr = res["win_rate"]
            eq = res["equity_final"]

            totals["n_trades"] += n
            totals["sum_R"] += sum(t.get("r_multiple", 0.0) for t in res.get("trades", []))
            totals["symbols"].append({
                "symbol": sym,
                "n_trades": n,
                "win_rate": wr,
                "avg_R": avg_R,
                "equity_final": eq
            })

            self.stdout.write(
                self.style.SUCCESS(
                    f"{sym}: trades={n} win_rate={wr:.2%} avg_R={avg_R:.2f} equity={eq:.2f}"
                )
            )

        self.stdout.write(self.style.SUCCESS(
            f"TOTAL symbols={len(totals['symbols'])} trades={totals['n_trades']}"
        ))
