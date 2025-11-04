# strategies/views.py

from __future__ import annotations
import json
import pandas as pd
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import ListView, CreateView, UpdateView, DeleteView

from strategies.forms import StrategyCustomForm
from strategies.models import Strategy
from strategies.custom_strategy import CustomStrategy
from market_data.models import Asset
from indicators.models import Indicator
from market_data.services.store import read_parquet, bars_path


class StaffRequiredMixin(UserPassesTestMixin):
    def test_func(self):
        u = self.request.user
        return u.is_authenticated and (u.is_staff or u.is_superuser)


class CustomStrategyListView(LoginRequiredMixin, StaffRequiredMixin, ListView):
    template_name = "strategies/custom_list.html"
    context_object_name = "strategies"
    paginate_by = 50

    def get_queryset(self):
        qs = Strategy.objects.filter(type="CUSTOM").order_by("-created_at")
        q = self.request.GET.get("q", "").strip()
        if q:
            qs = qs.filter(name__icontains=q)
        return qs


class CustomStrategyCreateView(LoginRequiredMixin, StaffRequiredMixin, CreateView):
    model = Strategy
    form_class = StrategyCustomForm
    template_name = "strategies/custom_form.html"
    success_url = reverse_lazy("strategies:custom_list")

    def form_valid(self, form):
        # IMPORTANT : poser l'user sur l'instance avant l'appel parent
        form.instance.user = self.request.user
        form.instance.type = "CUSTOM"
        return super().form_valid(form)


class CustomStrategyUpdateView(LoginRequiredMixin, StaffRequiredMixin, UpdateView):
    model = Strategy
    form_class = StrategyCustomForm
    template_name = "strategies/custom_form.html"
    success_url = reverse_lazy("strategies:custom_list")

    def get_queryset(self):
        return Strategy.objects.filter(type="CUSTOM")

    def form_valid(self, form):
        # on renforce le type pour éviter tout changement indésirable
        form.instance.type = "CUSTOM"
        return super().form_valid(form)


class CustomStrategyDeleteView(LoginRequiredMixin, StaffRequiredMixin, DeleteView):
    model = Strategy
    template_name = "strategies/custom_confirm_delete.html"
    success_url = reverse_lazy("strategies:custom_list")

    def get_queryset(self):
        return Strategy.objects.filter(type="CUSTOM")


class CustomStrategyTestView(LoginRequiredMixin, StaffRequiredMixin, View):
    """
    Endpoint HTMX/GET pour tester une CustomStrategy sur un symbole.
    Retourne un snippet HTML (pre) avec le JSON pretty.
    """
    def get(self, request, pk: int):
        strategy = get_object_or_404(Strategy, pk=pk, type="CUSTOM")
        symbol = (request.GET.get("symbol") or "").strip()
        if not symbol:
            # symbole par défaut = premier equity actif
            symbol = Asset.objects.filter(is_active=True, type="equity").values_list("symbol", flat=True).first()
            if not symbol:
                return HttpResponseBadRequest("Aucun symbole disponible")

        # Charger data
        df_prices = read_parquet(bars_path(symbol, "1D"))
        if df_prices is None or df_prices.empty:
            return HttpResponseBadRequest(f"Aucune donnée pour {symbol}")

        asset = Asset.objects.filter(symbol=symbol).first()
        qs = Indicator.objects.filter(asset=asset).order_by("date").values()
        df_ind = pd.DataFrame(list(qs))
        if not df_ind.empty:
            df_ind.set_index("date", inplace=True)

        # Exécuter la stratégie custom
        executor = CustomStrategy(code=strategy.code, parameters=strategy.parameters)
        result = executor.evaluate(df_prices, df_ind)

        # Retour HTML simple (pour HTMX)
        html = render(request, "strategies/_test_result.html", {
            "symbol": symbol,
            "result_json": json.dumps(result, ensure_ascii=False, indent=2),
        }).content
        return HttpResponse(html)
