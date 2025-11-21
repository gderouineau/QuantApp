# portfolios/views.py
from __future__ import annotations
import pandas as pd

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy, reverse
from django.views.generic import ListView, CreateView, UpdateView, DeleteView, DetailView, FormView

from django.utils import timezone
from django.db import transaction
from django import forms
from strategies.models import Signal
from .models import (
    Portfolio, PortfolioAllocation, AssetGroup,
    PortfolioRun, PortfolioTrade, PortfolioEquity,
    Position, PositionEvent
)
from .forms import (
    PortfolioForm, PortfolioAllocationForm, AssetGroupForm, AcceptSignalForm,
    MoveSLForm, PartialExitForm, ClosePositionForm,
    PortfolioBacktestForm

)
from .services.portfolio_backtest import run_portfolio_backtest, PortfolioSimParams

# Service simu (MVP déjà fourni)
from .services.simulator import simulate_portfolio_range, SimParams

from market_data.services.store import read_parquet, bars_path
from indicators.models import Indicator


# --------- Mixins ---------
class OwnerRequiredMixin(UserPassesTestMixin):
    """Vérifie que l'objet appartient bien à l'utilisateur courant (ou staff)."""
    def test_func(self):
        obj = self.get_object()
        user = self.request.user
        if not user.is_authenticated:
            return False
        if user.is_staff or user.is_superuser:
            return True
        # Pour Portfolio / AssetGroup (objet a un attribut user)
        owner = getattr(obj, "user", None)
        return owner == user


class OwnerQuerysetMixin:
    """Limite les queryset aux objets de l'utilisateur (ou staff -> tout)."""
    def get_queryset(self):
        qs = super().get_queryset()
        u = self.request.user
        if not u.is_authenticated:
            return qs.none()
        if u.is_staff or u.is_superuser:
            return qs
        # Portfolio/AssetGroup
        if qs.model is AssetGroup:
            return qs.filter(is_global=True) | qs.filter(user=u)
        if qs.model is Portfolio:
            return qs.filter(user=u)
        if qs.model is PortfolioRun:
            return qs.filter(portfolio__user=u)
        if qs.model is PortfolioTrade:
            return qs.filter(run__portfolio__user=u)
        if qs.model is PortfolioEquity:
            return qs.filter(run__portfolio__user=u)
        return qs


# --------- Portfolios ---------
class PortfolioListView(LoginRequiredMixin, OwnerQuerysetMixin, ListView):
    model = Portfolio
    template_name = "portfolios/portfolio_list.html"
    context_object_name = "items"


class PortfolioCreateView(LoginRequiredMixin, CreateView):
    model = Portfolio
    form_class = PortfolioForm
    template_name = "portfolios/portfolio_form.html"

    def form_valid(self, form):
        obj = form.save(commit=False)
        obj.user = self.request.user
        obj.save()
        messages.success(self.request, "Portefeuille créé.")
        return redirect("portfolios:detail", pk=obj.pk)


class PortfolioUpdateView(LoginRequiredMixin, OwnerRequiredMixin, UpdateView):
    model = Portfolio
    form_class = PortfolioForm
    template_name = "portfolios/portfolio_form.html"

    def form_valid(self, form):
        resp = super().form_valid(form)
        messages.success(self.request, "Portefeuille mis à jour.")
        return resp

    def get_success_url(self):
        return reverse("portfolios:detail", kwargs={"pk": self.object.pk})


class PortfolioDeleteView(LoginRequiredMixin, OwnerRequiredMixin, DeleteView):
    model = Portfolio
    template_name = "portfolios/confirm_delete.html"
    success_url = reverse_lazy("portfolios:list")

    def delete(self, request, *args, **kwargs):
        messages.success(self.request, "Portefeuille supprimé.")
        return super().delete(request, *args, **kwargs)


class PortfolioDetailView(LoginRequiredMixin, OwnerRequiredMixin, DetailView):
    model = Portfolio
    template_name = "portfolios/portfolio_detail.html"
    context_object_name = "portfolio"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ptf: Portfolio = self.object
        allocs = ptf.allocations.select_related("strategy", "group").all().order_by("-weight")
        ctx["allocations"] = allocs

        # Derniers runs
        runs = ptf.runs.all().order_by("-started_at")[:10]
        ctx["runs"] = runs

        # Trades du dernier run (si existe)
        last_run = runs[0] if runs else None
        ctx["last_run"] = last_run
        ctx["last_trades"] = PortfolioTrade.objects.filter(
            run=last_run
        ).order_by(
            "-entry_date"
        )[:50] if last_run else []

        # Signaux récents (N jours) pour les actifs présents dans les groupes du portefeuille
        symbols = set()
        for a in allocs:
            symbols.update(a.group.assets.values_list("symbol", flat=True))
        days = int(self.request.GET.get("signals_days", 5) or 5)
        since = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
        ctx["recent_signals_days"] = days
        if symbols:
            ctx["recent_signals"] = (
                Signal.objects.filter(asset__symbol__in=symbols)
                .filter(date__gte=since.date())
                .select_related("strategy", "asset")
                .order_by("-date", "-score")[:200]
            )

        else:
            ctx["recent_signals"] = []

        ctx["open_positions"] = self.object.positions.filter(status="OPEN").select_related("asset", "strategy",
                                                                                           "allocation")

        return ctx


# --------- Allocations ---------
class AllocationCreateView(LoginRequiredMixin, FormView):
    template_name = "portfolios/allocation_form.html"
    form_class = PortfolioAllocationForm

    def dispatch(self, request, *args, **kwargs):

        self.portfolio = get_object_or_404(Portfolio, pk=kwargs["portfolio_id"])
        if not (request.user.is_staff or request.user.is_superuser or self.portfolio.user == request.user):
            raise Http404
        return super().dispatch(request, *args, **kwargs)

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["portfolio"] = self.portfolio
        return ctx

    def form_valid(self, form):
        obj = form.save(commit=False)
        obj.portfolio = self.portfolio
        obj.save()
        messages.success(self.request, "Allocation ajoutée.")
        return redirect("portfolios:detail", pk=self.portfolio.pk)


class AllocationUpdateView(LoginRequiredMixin, UpdateView):
    model = PortfolioAllocation
    form_class = PortfolioAllocationForm
    template_name = "portfolios/allocation_form.html"

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs

    def dispatch(self, request, *args, **kwargs):
        obj = self.get_object()
        if not (request.user.is_staff or request.user.is_superuser or obj.portfolio.user == request.user):
            raise Http404
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        resp = super().form_valid(form)
        messages.success(self.request, "Allocation mise à jour.")
        return resp

    def get_success_url(self):
        return reverse("portfolios:detail", kwargs={"pk": self.object.portfolio_id})


class AllocationDeleteView(LoginRequiredMixin, DeleteView):
    model = PortfolioAllocation
    template_name = "portfolios/confirm_delete.html"

    def dispatch(self, request, *args, **kwargs):
        obj = self.get_object()
        if not (request.user.is_staff or request.user.is_superuser or obj.portfolio.user == request.user):
            raise Http404
        self._portfolio_id = obj.portfolio_id
        return super().dispatch(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        messages.success(self.request, "Allocation supprimée.")
        return super().delete(request, *args, **kwargs)

    def get_success_url(self):
        return reverse("portfolios:detail", kwargs={"pk": self._portfolio_id})


# --------- Groupes d'actifs ---------
class GroupListView(LoginRequiredMixin, OwnerQuerysetMixin, ListView):
    model = AssetGroup
    template_name = "portfolios/group_list.html"
    context_object_name = "items"


class GroupCreateView(LoginRequiredMixin, CreateView):
    model = AssetGroup
    form_class = AssetGroupForm
    template_name = "portfolios/group_form.html"
    success_url = reverse_lazy("portfolios:groups")

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs

    def form_valid(self, form):
        obj = form.save(commit=False)
        # Propriétaire
        obj.user = self.request.user
        obj.save()
        form.save_m2m()
        messages.success(self.request, "Groupe créé.")
        return redirect(self.success_url)


class GroupUpdateView(LoginRequiredMixin, OwnerRequiredMixin, UpdateView):
    model = AssetGroup
    form_class = AssetGroupForm
    template_name = "portfolios/group_form.html"
    success_url = reverse_lazy("portfolios:groups")

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs

    def form_valid(self, form):
        resp = super().form_valid(form)
        messages.success(self.request, "Groupe mis à jour.")
        return resp


class GroupDeleteView(LoginRequiredMixin, OwnerRequiredMixin, DeleteView):
    model = AssetGroup
    template_name = "portfolios/confirm_delete.html"
    success_url = reverse_lazy("portfolios:groups")

    def delete(self, request, *args, **kwargs):
        messages.success(self.request, "Groupe supprimé.")
        return super().delete(request, *args, **kwargs)


# --------- Simulation depuis UI ---------
class PortfolioSimulateView(LoginRequiredMixin, OwnerRequiredMixin, FormView):
    """Formulaire simple pour lancer une simulation et rediriger vers le run."""
    template_name = "portfolios/simulate_form.html"

    def get_form_class(self):

        # Form ad-hoc sans classe dédiée
        class SimForm(forms.Form):
            start = forms.DateField(required=True, widget=forms.DateInput(attrs={"type": "date"}))
            end = forms.DateField(required=True, widget=forms.DateInput(attrs={"type": "date"}))
            warmup = forms.IntegerField(initial=252, min_value=0)
            commission_bps = forms.FloatField(initial=1.0, min_value=0)
            slippage_bps = forms.FloatField(initial=2.0, min_value=0)
        return SimForm

    def get_object(self):
        return get_object_or_404(Portfolio, pk=self.kwargs["pk"])

    def form_valid(self, form):
        ptf = self.get_object()
        start = pd.to_datetime(form.cleaned_data["start"]).normalize()
        end = pd.to_datetime(form.cleaned_data["end"]).normalize()
        params = SimParams(
            warmup_bars=form.cleaned_data["warmup"],
            commission_bps=form.cleaned_data["commission_bps"],
            slippage_bps=form.cleaned_data["slippage_bps"],
        )

        # Allocations actives
        allocs = list(ptf.allocations.select_related("strategy", "group").filter(is_active=True))
        if not allocs:
            messages.error(self.request, "Aucune allocation active dans ce portefeuille.")
            return redirect("portfolios:detail", pk=ptf.pk)

        symbols_by_alloc = {a.id: list(a.group.assets.values_list("symbol", flat=True)) for a in allocs}

        res = simulate_portfolio_range(ptf, allocs, symbols_by_alloc, start, end, params)

        run = PortfolioRun.objects.create(
            portfolio=ptf,
            start_date=start.date(),
            end_date=end.date(),
            capital_start=ptf.initial_capital,
            equity_final=res["summary"].get("equity_final", ptf.initial_capital),
            n_trades=res["summary"].get("n_trades", 0),
            win_rate=res["summary"].get("win_rate", 0.0),
            avg_R=res["summary"].get("avg_R", 0.0),
            expectancy_R=res["summary"].get("expectancy_R", 0.0),
            params={
                "warmup": params.warmup_bars,
                "commission_bps": params.commission_bps,
                "slippage_bps": params.slippage_bps,
            },
            summary=res["summary"],
        )

        # Persist trades/equity
        if res["trades"]:
            PortfolioTrade.objects.bulk_create([
                PortfolioTrade(
                    run=run,
                    allocation_id=t["allocation_id"],
                    strategy_id=t["strategy_id"],
                    symbol=t["symbol"],
                    entry_date=t["entry_date"],
                    entry_price=t["entry_price"],
                    qty=t["qty"],
                    stop_price=t["stop_price"],
                    exit_date=t["exit_date"],
                    exit_price=t["exit_price"],
                    outcome=t["outcome"],
                    r_multiple=t["r_multiple"],
                    pnl=t["pnl"],
                    details={"score": t.get("score")},
                ) for t in res["trades"]
            ], batch_size=1000)

        if res["equity_points"]:
            PortfolioEquity.objects.bulk_create([
                PortfolioEquity(run=run, date=p["date"], equity=p["equity"])
                for p in res["equity_points"]
            ], batch_size=1000)

        messages.success(self.request, f"Simulation lancée et enregistrée (Run #{run.id}).")
        return redirect("portfolios:run_detail", pk=run.pk)


class PortfolioRunDetailView(LoginRequiredMixin, OwnerQuerysetMixin, DetailView):
    model = PortfolioRun
    template_name = "portfolios/run_detail.html"
    context_object_name = "run"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        run: PortfolioRun = self.object
        ctx["trades"] = run.trades.all().order_by("entry_date")
        ctx["equity_points"] = run.equity_points.all().order_by("date")
        return ctx


class PortfolioSignalInboxView(LoginRequiredMixin, OwnerRequiredMixin, DetailView):
    """Liste des signaux récents pertinents pour le portefeuille, avec bouton 'Accepter'."""
    model = Portfolio
    template_name = "portfolios/portfolio_inbox.html"
    context_object_name = "portfolio"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ptf: Portfolio = self.object
        # Liste des symboles couverts par les allocations actives
        symbols = set()
        for a in ptf.allocations.select_related("group").filter(is_active=True):
            symbols.update(a.group.assets.values_list("symbol", flat=True))

        days = int(self.request.GET.get("days", 5))
        since = timezone.now().date() - pd.Timedelta(days=days)
        signals = []
        if symbols:
            qs = (Signal.objects
                  .filter(asset__symbol__in=list(symbols), date__gte=since)
                  .select_related("asset", "strategy")
                  .order_by("-date", "-score"))
            # Filtrer ceux déjà acceptés (liés à une position)
            taken_ids = set(Position.objects.filter(portfolio=ptf).values_list("source_signal_id", flat=True))
            signals = [s for s in qs if s.id not in taken_ids]

        ctx["signals_days"] = days
        ctx["signals"] = signals
        return ctx


class PortfolioAcceptSignalView(LoginRequiredMixin, FormView):
    """Accepte un signal → crée une Position avec sizing par risque."""
    form_class = AcceptSignalForm
    template_name = "portfolios/accept_signal.html"

    def dispatch(self, request, *args, **kwargs):
        self.portfolio = get_object_or_404(Portfolio, pk=kwargs["pk"])
        if not (request.user.is_staff or request.user.is_superuser or self.portfolio.user == request.user):
            raise Http404
        return super().dispatch(request, *args, **kwargs)

    def get_initial(self):
        initial = super().get_initial()
        sig_id = self.request.GET.get("signal_id")
        initial["signal_id"] = sig_id
        # entry_price par défaut = dernier close parquet
        if sig_id:
            sig = get_object_or_404(Signal, pk=sig_id)
            dfp = read_parquet(bars_path(sig.asset.symbol, "1D"))
            if not dfp.empty:
                initial["entry_price"] = float(dfp["close"].iloc[-1])
        return initial

    def form_valid(self, form):
        sig = get_object_or_404(Signal, pk=form.cleaned_data["signal_id"])
        ptf = self.portfolio

        # Trouver une allocation compatible (même stratégie + groupe contenant l'actif) ou None
        alloc = (ptf.allocations
                 .select_related("group", "strategy")
                 .filter(strategy=sig.strategy, group__assets__id=sig.asset_id, is_active=True)
                 .first())

        entry = float(form.cleaned_data["entry_price"])
        side = form.cleaned_data["side"]
        stop_method = form.cleaned_data["stop_method"]
        stop_val = float(form.cleaned_data["stop_value"])
        risk_pct = float(form.cleaned_data["risk_pct"]) / 100.0  # on convertit 1 → 0.01 si l’UI donne “1 = 1%”

        # Déterminer stop_price
        if stop_method == "PCT":
            stop_price = entry * (1 - stop_val / 100.0) if side == "LONG" else entry * (1 + stop_val / 100.0)
        elif stop_method == "ATR":
            # nécessite indicator atr_14
            ind = (Indicator.objects.filter(asset=sig.asset).order_by("-date").values("atr_14").first())
            atr = float(ind["atr_14"]) if ind and ind["atr_14"] is not None else None
            if not atr:
                messages.error(self.request, "ATR indisponible pour ce symbole.")
                return redirect("portfolios:inbox", pk=ptf.pk)
            mult = stop_val
            stop_price = entry - mult * atr if side == "LONG" else entry + mult * atr
        else:  # MANUAL
            stop_price = stop_val

        # Sizing
        risk_amount = ptf.initial_capital * (risk_pct or ptf.risk_per_trade)  # base simple
        stop_dist = abs(entry - stop_price)
        if stop_dist <= 0:
            messages.error(self.request, "Distance entrée/stop invalide.")
            return redirect("portfolios:inbox", pk=ptf.pk)

        qty = int(max(0, risk_amount // stop_dist))
        if qty == 0:
            messages.error(self.request, "Quantité calculée à 0 (risque trop faible / stop trop proche).")
            return redirect("portfolios:inbox", pk=ptf.pk)

        with transaction.atomic():
            pos = Position.objects.create(
                portfolio=ptf,
                allocation=alloc,
                strategy=sig.strategy,
                asset=sig.asset,
                source_signal=sig,
                side=side,
                qty=qty,
                entry_price=entry,
                entry_date=timezone.now().date(),
                stop_init=stop_price,
                stop_cur=stop_price,
                take_profits=[],
                details={
                    "signal_score": sig.score,
                    "signal_strength": sig.strength,
                    "init_qty": qty,
                },
            )
            PositionEvent.objects.create(position=pos, kind="CREATED", data={
                "from_signal": sig.id,
                "entry": entry,
                "stop": stop_price,
                "qty": qty,
                "risk_pct": risk_pct,
                "method": stop_method,
            })

        messages.success(self.request, f"Position ouverte sur {sig.asset.symbol} (qty={qty}).")
        return redirect("portfolios:detail", pk=ptf.pk)


class PositionOwnerRequiredMixin:
    def dispatch(self, request, *args, **kwargs):
        pos = self.get_object() if hasattr(self, "get_object") else get_object_or_404(Position, pk=kwargs["pk"])
        user = request.user
        if not user.is_authenticated:
            raise Http404
        if not (user.is_staff or user.is_superuser or pos.portfolio.user_id == user.id):
            raise Http404
        self.position = pos
        return super().dispatch(request, *args, **kwargs)

# --- Détail d'une position ---
class PositionDetailView(LoginRequiredMixin, PositionOwnerRequiredMixin, DetailView):
    model = Position
    template_name = "portfolios/position_detail.html"
    context_object_name = "position"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        pos: Position = self.object
        ctx["events"] = pos.events.all().order_by("-ts")
        return ctx

# --- Move SL ---
class PositionMoveSLView(LoginRequiredMixin, PositionOwnerRequiredMixin, FormView):
    form_class = MoveSLForm
    template_name = "portfolios/position_action_form.html"

    def get_object(self):
        return get_object_or_404(Position, pk=self.kwargs["pk"])

    def get_initial(self):
        return {"stop_price": self.position.stop_cur}

    def form_valid(self, form):
        pos: Position = self.position
        new_stop = float(form.cleaned_data["stop_price"])
        with transaction.atomic():
            pos.stop_cur = new_stop
            pos.save(update_fields=["stop_cur", "updated_at"])
            PositionEvent.objects.create(
                position=pos, kind="MOVE_SL", data={"new_stop": new_stop}
            )
        messages.success(self.request, f"Stop déplacé à {new_stop:.4f}.")
        return redirect("portfolios:position_detail", pk=pos.pk)

# --- Partial exit ---
class PositionPartialExitView(LoginRequiredMixin, PositionOwnerRequiredMixin, FormView):
    form_class = PartialExitForm
    template_name = "portfolios/position_action_form.html"

    def get_object(self):
        return get_object_or_404(Position, pk=self.kwargs["pk"])

    def form_valid(self, form):
        pos: Position = self.position
        qty_out = int(form.cleaned_data["qty"])
        px = float(form.cleaned_data["exit_price"])

        if qty_out > pos.qty:
            messages.error(self.request, "Quantité supérieure à la quantité restante.")
            return redirect("portfolios:position_detail", pk=pos.pk)

        # PnL partiel
        side = pos.side
        if side == "LONG":
            pnl = (px - pos.entry_price) * qty_out
            risk_per_share = max(1e-9, pos.entry_price - pos.stop_init)
        else:
            pnl = (pos.entry_price - px) * qty_out
            risk_per_share = max(1e-9, pos.stop_init - pos.entry_price)

        init_qty = int(pos.details.get("init_qty") or pos.qty)  # fallback
        # R multiple cumulé recalculé : (realized_pnl_total) / (risk_per_share * init_qty)
        with transaction.atomic():
            pos.realized_pnl += pnl
            pos.qty -= qty_out
            pos.r_multiple = pos.realized_pnl / (risk_per_share * max(1, init_qty))
            if pos.qty == 0:
                pos.status = "CLOSED"
                pos.exit_date = timezone.now().date()
                pos.exit_price = px
                pos.outcome = "MANUAL"
            pos.save()

            PositionEvent.objects.create(
                position=pos, kind="PARTIAL_EXIT",
                data={"qty": qty_out, "price": px, "pnl": pnl}
            )

        messages.success(self.request, f"Vente partielle {qty_out} @ {px:.4f} (PnL {pnl:.2f}).")
        return redirect("portfolios:position_detail", pk=pos.pk)

# --- Close position ---
class PositionCloseView(LoginRequiredMixin, PositionOwnerRequiredMixin, FormView):
    form_class = ClosePositionForm
    template_name = "portfolios/position_action_form.html"

    def get_object(self):
        return get_object_or_404(Position, pk=self.kwargs["pk"])

    def get_initial(self):
        pos: Position = self.position
        # Par défaut, prix = dernier entry_price (tu peux brancher le dernier close parquet ici)
        return {"exit_price": pos.entry_price, "outcome": "MANUAL"}

    def form_valid(self, form):
        pos: Position = self.position
        if pos.status != "OPEN":
            messages.error(self.request, "La position n'est pas ouverte.")
            return redirect("portfolios:position_detail", pk=pos.pk)

        px = float(form.cleaned_data["exit_price"])
        outcome = form.cleaned_data["outcome"]
        side = pos.side

        # PnL sur le reliquat
        if side == "LONG":
            pnl = (px - pos.entry_price) * pos.qty
            risk_per_share = max(1e-9, pos.entry_price - pos.stop_init)
        else:
            pnl = (pos.entry_price - px) * pos.qty
            risk_per_share = max(1e-9, pos.stop_init - pos.entry_price)

        init_qty = int(pos.details.get("init_qty") or pos.qty)
        with transaction.atomic():
            pos.realized_pnl += pnl
            pos.exit_price = px
            pos.exit_date = timezone.now().date()
            pos.status = "CLOSED"
            pos.outcome = outcome
            pos.r_multiple = pos.realized_pnl / (risk_per_share * max(1, init_qty))
            pos.save()

            PositionEvent.objects.create(
                position=pos, kind="CLOSE", data={"price": px, "pnl_added": pnl, "outcome": outcome}
            )

        messages.success(
            self.request,
            f"Position clôturée @ {px:.4f} (PnL total {pos.realized_pnl:.2f}, R={pos.r_multiple:.2f})."
        )
        return redirect("portfolios:position_detail", pk=pos.pk)




class PortfolioBacktestView(LoginRequiredMixin, OwnerRequiredMixin, FormView):
    form_class = PortfolioBacktestForm
    template_name = "portfolios/portfolio_backtest_form.html"
    context_object_name = "portfolio"

    def dispatch(self, request, *args, **kwargs):
        self.portfolio = get_object_or_404(Portfolio, pk=kwargs["pk"])
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["portfolio"] = self.portfolio
        return ctx

    def form_valid(self, form):
        p = self.portfolio
        params = PortfolioSimParams(
            start=form.cleaned_data.get("start"),
            end=form.cleaned_data.get("end"),
            warmup_bars=form.cleaned_data.get("warmup") or 252,
            capital_override=form.cleaned_data.get("capital_override"),
            risk_override=form.cleaned_data.get("risk_override"),
        )
        results = run_portfolio_backtest(p, params)

        # Enregistre un run
        run = PortfolioRun.objects.create(
            portfolio=p,
            started_at=timezone.now(),
            finished_at=timezone.now(),
            params={
                "start": results["params"]["start"],
                "end": results["params"]["end"],
                "warmup": results["params"]["warmup_bars"],
                "capital_override": results["params"]["capital_override"],
                "risk_override": results["params"]["risk_override"],
            },
            summary={
                "initial_capital": results["portfolio"]["initial_capital"],
                "n_trades": results["totals"]["n_trades"],
                "symbols": results["totals"]["symbols"],
                "equity_final": results["totals"]["equity_final"],
                "sum_R": results["totals"]["sum_R"],
            },
            details=results,
        )

        messages.success(self.request, "Simulation lancée et enregistrée.")
        return redirect("portfolios:run_detail", pk=run.pk)




