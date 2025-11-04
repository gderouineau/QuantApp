from django.contrib import admin
from django.urls import path
from django.http import JsonResponse, Http404
from django.utils.safestring import mark_safe
import pandas as pd

from strategies.models import Strategy, Signal
from strategies.relative_strength import RelativeStrengthStrategy
from strategies.custom_strategy import CustomStrategy
# (optionnel si tu as ajouté ces deux fichiers)
from strategies.golden_cross import GoldenCrossStrategy
from strategies.volume_breakout import VolumeBreakoutStrategy

from market_data.models import Asset
from indicators.models import Indicator
from market_data.services.store import read_parquet, bars_path


def build_strategy_instance(strategy: Strategy):
    t = strategy.type
    if t == "CUSTOM":
        return CustomStrategy(strategy.code, strategy.parameters)
    if t == "RS":
        return RelativeStrengthStrategy(strategy.parameters)
    if t == "GC":
        return GoldenCrossStrategy(strategy.parameters)
    if t == "VB":
        return VolumeBreakoutStrategy(strategy.parameters)
    raise ValueError(f"Type non géré: {t}")


@admin.register(Strategy)
class StrategyAdmin(admin.ModelAdmin):
    list_display = ("name", "type", "is_active", "weight", "created_at")
    list_filter = ("type", "is_active")
    search_fields = ("name",)
    change_form_template = "admin/strategies/strategy/change_form.html"

    def get_urls(self):
        urls = super().get_urls()
        custom = [
            path("<int:pk>/test/", self.admin_site.admin_view(self.test_view), name="strategies_strategy_test"),
        ]
        return custom + urls

    def test_view(self, request, pk: int):
        strategy = self.get_object(request, pk)
        if not strategy:
            raise Http404

        symbol = request.GET.get("symbol") or request.POST.get("symbol") or ""
        symbol = symbol.strip() or Asset.objects.filter(is_active=True, type="equity").values_list("symbol", flat=True).first()
        if not symbol:
            return JsonResponse({"ok": False, "error": "Aucun symbole disponible"}, status=400)

        # Charge data
        df_prices = read_parquet(bars_path(symbol, "1D"))
        if df_prices is None or df_prices.empty:
            return JsonResponse({"ok": False, "error": f"Aucune donnée prix pour {symbol}"}, status=400)

        asset = Asset.objects.filter(symbol=symbol).first()
        qs = Indicator.objects.filter(asset=asset).order_by("date").values()
        df_ind = pd.DataFrame(list(qs))
        if not df_ind.empty:
            df_ind.set_index("date", inplace=True)

        try:
            inst = build_strategy_instance(strategy)
            result = inst.evaluate(df_prices, df_ind)
            return JsonResponse({"ok": True, "symbol": symbol, "result": result})
        except Exception as e:
            return JsonResponse({"ok": False, "error": str(e)}, status=500)

    def render_change_form(self, request, context, *args, **kwargs):
        # Petit encart d’aide sous le titre
        context = super().render_change_form(request, context, *args, **kwargs)
        context["admin_test_help"] = mark_safe(
            "<p>Tester la stratégie sur un symbole (ex: AAPL, AIR.PA). "
            "Le test n'enregistre pas de Signal.</p>"
        )
        return context


@admin.register(Signal)
class SignalAdmin(admin.ModelAdmin):
    list_display = ("date", "asset", "strategy", "score", "strength")
    list_filter = ("date", "strategy", "strength")
    search_fields = ("asset__symbol",)
    ordering = ("-date", "-score")
