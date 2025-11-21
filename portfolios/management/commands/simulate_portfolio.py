# portfolios/management/commands/simulate_portfolio.py
from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db import transaction
import pandas as pd
import math
import json
import os

from portfolios.models import (
    Portfolio, PortfolioRun, PortfolioTrade, PortfolioEquity, PortfolioAllocation
)
from portfolios.services.simulator import simulate_portfolio_range, SimParams


class Command(BaseCommand):
    help = "Simule un portefeuille (allocations: stratégie+groupe) sur une période, avec récap Gains/Perte (R et monnaie)."

    def add_arguments(self, parser):
        parser.add_argument("--portfolio-id", type=int, required=True,
                            help="ID du portefeuille à simuler.")
        parser.add_argument("--start", type=str, required=True,
                            help="Date de début YYYY-MM-DD.")
        parser.add_argument("--end", type=str, required=True,
                            help="Date de fin YYYY-MM-DD.")
        parser.add_argument("--warmup", type=int, default=252,
                            help="Barres de chauffe (par défaut 252).")
        parser.add_argument("--commission-bps", type=float, default=1.0,
                            help="Commission en bps (ex: 1.0 = 0.01%).")
        parser.add_argument("--slippage-bps", type=float, default=2.0,
                            help="Slippage en bps (ex: 2.0 = 0.02%).")

        # Overrides optionnels (utiles pour tests)
        parser.add_argument("--override-portfolio-risk", type=float, default=None,
                            help="Override de Portfolio.risk_per_trade (ex: 0.01).")
        parser.add_argument("--override-max-total-positions", type=int, default=None,
                            help="Override de Portfolio.max_total_positions (0=illimité).")
        parser.add_argument("--override-allocation-risk", type=float, default=None,
                            help="Force per_trade_risk sur TOUTES les allocations actives (ex: 0.005).")
        parser.add_argument("--override-allocation-maxpos", type=int, default=None,
                            help="Force max_positions sur TOUTES les allocations actives (ex: 5).")

        # Formats / exports
        parser.add_argument("--dry-run", action="store_true",
                            help="N'enregistre rien en base (affiche uniquement le résumé).")
        parser.add_argument("--export-trades-csv", type=str, default="",
                            help="Chemin CSV pour exporter les trades.")
        parser.add_argument("--export-equity-csv", type=str, default="",
                            help="Chemin CSV pour exporter l'equity curve.")

    def handle(self, *args, **opts):
        pid = int(opts["portfolio_id"])
        start = pd.to_datetime(opts["start"]).normalize()
        end = pd.to_datetime(opts["end"]).normalize()
        warmup = int(opts["warmup"])
        commission_bps = float(opts["commission_bps"])
        slippage_bps = float(opts["slippage_bps"])

        override_portfolio_risk = opts.get("override_portfolio_risk")
        override_max_total_pos = opts.get("override_max_total_positions")
        override_alloc_risk = opts.get("override_allocation_risk")
        override_alloc_maxpos = opts.get("override_allocation_maxpos")

        dry_run = bool(opts.get("dry_run"))
        trades_csv = (opts.get("export_trades_csv") or "").strip()
        equity_csv = (opts.get("export_equity_csv") or "").strip()

        # ---- Charge le portefeuille + allocations actives
        try:
            ptf: Portfolio = Portfolio.objects.get(pk=pid)
        except Portfolio.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Portfolio id={pid} introuvable"))
            return

        allocs = list(ptf.allocations.select_related("strategy", "group").filter(is_active=True))
        if not allocs:
            self.stdout.write(self.style.ERROR("Aucune allocation active dans ce portefeuille"))
            return

        # ---- Overrides en mémoire (non persistés si dry-run)
        original_portfolio_risk = ptf.risk_per_trade
        original_max_total_pos = ptf.max_total_positions

        if override_portfolio_risk is not None:
            try:
                ptf.risk_per_trade = float(override_portfolio_risk)
            except Exception:
                self.stdout.write(self.style.WARNING("override_portfolio_risk invalide — ignoré"))
        if override_max_total_pos is not None:
            try:
                ptf.max_total_positions = int(override_max_total_pos)
            except Exception:
                self.stdout.write(self.style.WARNING("override_max_total_positions invalide — ignoré"))

        # Overrides par allocation
        original_alloc_fields = []
        if override_alloc_risk is not None or override_alloc_maxpos is not None:
            for a in allocs:
                original_alloc_fields.append((a.id, a.per_trade_risk, a.max_positions))
                if override_alloc_risk is not None:
                    try:
                        a.per_trade_risk = float(override_alloc_risk)
                    except Exception:
                        pass
                if override_alloc_maxpos is not None:
                    try:
                        a.max_positions = int(override_alloc_maxpos)
                    except Exception:
                        pass

        # ---- Map allocation -> symbols (depuis le groupe)
        symbols_by_alloc = {}
        for a in allocs:
            symbols_by_alloc[a.id] = list(a.group.assets.values_list("symbol", flat=True))

        params = SimParams(
            warmup_bars=warmup,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )

        # ---- Simulation
        self.stdout.write(self.style.MIGRATE_HEADING(
            f"Simulation portefeuille #{ptf.id} «{ptf.name}» — période {start.date()} → {end.date()}"
        ))
        self.stdout.write(
            f"Risque portefeuille: {ptf.risk_per_trade:.3%}  |  Max positions total: {ptf.max_total_positions or 0} (0=illimité)"
        )
        if override_alloc_risk is not None or override_alloc_maxpos is not None:
            self.stdout.write(
                f"Alloc overrides → per_trade_risk={override_alloc_risk if override_alloc_risk is not None else '—'} ; "
                f"max_positions={override_alloc_maxpos if override_alloc_maxpos is not None else '—'}"
            )
        self.stdout.flush()

        res = simulate_portfolio_range(ptf, allocs, symbols_by_alloc, start, end, params)

        summary = dict(res.get("summary", {}))
        trades = list(res.get("trades", []))
        equity_points = list(res.get("equity_points", []))

        # ---- Exposition (lecture depuis summary avec fallback via exposure_points)
        exposure_points = list(res.get("exposure_points", []))  # optionnel si renvoyé par le simulateur

        def _safe_mean(values):
            try:
                values = list(values)
                return (sum(values) / len(values)) if values else 0.0
            except Exception:
                return 0.0

        max_expo = float(summary.get(
            "max_exposure_pct",
            max((p.get("exposure_pct", 0.0) for p in exposure_points), default=0.0)
        ))
        avg_expo = float(summary.get(
            "avg_exposure_pct",
            _safe_mean(p.get("exposure_pct", 0.0) for p in exposure_points)
        ))

        # ---- Calculs gains / pertes (R et monnaie)
        sum_pos_R = 0.0
        sum_neg_R = 0.0
        sum_pos_pnl = 0.0
        sum_neg_pnl = 0.0

        for t in trades:
            r = float(t.get("r_multiple", 0.0))
            pnl = float(t.get("pnl", 0.0))
            if r >= 0:
                sum_pos_R += r
            else:
                sum_neg_R += r
            if pnl >= 0:
                sum_pos_pnl += pnl
            else:
                sum_neg_pnl += pnl

        net_R = sum_pos_R + sum_neg_R
        net_pnl = sum_pos_pnl + sum_neg_pnl

        # ---- Exports CSV éventuels
        if trades_csv:
            try:
                df_tr = pd.DataFrame(trades)
                os.makedirs(os.path.dirname(trades_csv), exist_ok=True) if os.path.dirname(trades_csv) else None
                df_tr.to_csv(trades_csv, index=False)
                self.stdout.write(self.style.SUCCESS(f"→ Trades exportés: {trades_csv}"))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Export trades CSV impossible: {e}"))

        if equity_csv:
            try:
                df_eq = pd.DataFrame(equity_points)
                os.makedirs(os.path.dirname(equity_csv), exist_ok=True) if os.path.dirname(equity_csv) else None
                df_eq.to_csv(equity_csv, index=False)
                self.stdout.write(self.style.SUCCESS(f"→ Equity exportée: {equity_csv}"))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Export equity CSV impossible: {e}"))

        # ---- Affichages synthèse
        n_trades = int(summary.get("n_trades", len(trades)))
        win_rate = float(summary.get("win_rate", 0.0))
        avg_R = float(summary.get("avg_R", 0.0))
        expectancy_R = float(summary.get("expectancy_R", 0.0))
        equity_final = float(summary.get("equity_final", ptf.initial_capital))

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("RÉSUMÉ (portefeuille)"))
        self.stdout.write(
            f"Trades: {n_trades}  |  Win rate: {win_rate:.2%}  |  avg_R: {avg_R:.3f}  |  expectancy_R: {expectancy_R:.3f}"
        )
        self.stdout.write(f"Equity finale: {equity_final:,.2f}")
        # >>> Nouveau log Exposition
        self.stdout.write(f"Exposition — moyenne: {avg_expo:.1%}  |  max: {max_expo:.1%}")
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Gains / Pertes"))
        self.stdout.write(
            f"Gains  : +{sum_pos_R:.2f} R   |  {sum_pos_pnl:,.2f} "
        )
        self.stdout.write(
            f"Pertes : {sum_neg_R:.2f} R   |  {sum_neg_pnl:,.2f} "
        )
        self.stdout.write(
            f"Net    : {net_R:.2f} R   |  {net_pnl:,.2f}"
        )
        self.stdout.flush()

        # ---- Persistance (sauf dry-run)
        if dry_run:
            self.stdout.write(self.style.WARNING("Dry-run activé → rien n'est enregistré en base."))
            # Restaure les champs modifiés en mémoire
            ptf.risk_per_trade = original_portfolio_risk
            ptf.max_total_positions = original_max_total_pos
            for a in allocs:
                # on ne persiste pas les alloc overrides
                pass
            return

        with transaction.atomic():
            run = PortfolioRun.objects.create(
                portfolio=ptf,
                start_date=start.date(),
                end_date=end.date(),
                capital_start=ptf.initial_capital,
                equity_final=equity_final,
                n_trades=n_trades,
                win_rate=win_rate,
                avg_R=avg_R,
                expectancy_R=expectancy_R,
                params={
                    "warmup": warmup,
                    "commission_bps": commission_bps,
                    "slippage_bps": slippage_bps,
                    "override": {
                        "portfolio_risk": ptf.risk_per_trade,
                        "max_total_positions": ptf.max_total_positions,
                        "allocation_risk": override_alloc_risk,
                        "allocation_max_positions": override_alloc_maxpos,
                    },
                },
                summary={
                    **summary,
                    "sum_pos_R": sum_pos_R,
                    "sum_neg_R": sum_neg_R,
                    "net_R": net_R,
                    "sum_pos_pnl": sum_pos_pnl,
                    "sum_neg_pnl": sum_neg_pnl,
                    "net_pnl": net_pnl,
                    # On peut persister l'expo si tu veux la revoir plus tard :
                    "avg_exposure_pct": avg_expo,
                    "max_exposure_pct": max_expo,
                },
            )

            # Trades
            rows = []
            for t in trades:
                rows.append(PortfolioTrade(
                    run=run,
                    allocation_id=t.get("allocation_id"),
                    strategy_id=t.get("strategy_id"),
                    symbol=t.get("symbol", ""),
                    entry_date=t.get("entry_date"),
                    entry_price=float(t.get("entry_price", 0.0)),
                    qty=int(t.get("qty", 0) or 0),
                    stop_price=t.get("stop_price"),
                    exit_date=t.get("exit_date"),
                    exit_price=float(t.get("exit_price", 0.0)),
                    outcome=str(t.get("outcome", "EOD")),
                    r_multiple=float(t.get("r_multiple", 0.0)),
                    pnl=float(t.get("pnl", 0.0)),
                    details={"score": t.get("score")},
                ))
            if rows:
                PortfolioTrade.objects.bulk_create(rows, batch_size=1000)

            # Equity curve
            eq_rows = []
            for p in equity_points:
                try:
                    eq_rows.append(PortfolioEquity(run=run, date=p["date"], equity=float(p["equity"])))
                except Exception:
                    # tolère un point corrompu
                    continue
            if eq_rows:
                PortfolioEquity.objects.bulk_create(eq_rows, batch_size=1000)

            self.stdout.write(self.style.SUCCESS(
                f"Run #{run.id} • {ptf.name}: trades={run.n_trades} win_rate={run.win_rate:.2%} equity={run.equity_final:.2f}"
            ))

        # Restaure les champs modifiés en mémoire (propreté)
        ptf.risk_per_trade = original_portfolio_risk
        ptf.max_total_positions = original_max_total_pos
        for (aid, old_risk, old_maxpos) in original_alloc_fields:
            try:
                a = next(x for x in allocs if x.id == aid)
                a.per_trade_risk = old_risk
                a.max_positions = old_maxpos
            except StopIteration:
                pass
