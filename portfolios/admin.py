from django.contrib import admin
from .models import AssetGroup, AssetGroupAsset, Portfolio, PortfolioAllocation, PortfolioRun, PortfolioTrade, PortfolioEquity


class AssetGroupAssetInline(admin.TabularInline):
    model = AssetGroupAsset
    extra = 0


@admin.register(AssetGroup)
class AssetGroupAdmin(admin.ModelAdmin):
    list_display = ("name", "user", "is_global")
    list_filter = ("is_global",)
    search_fields = ("name", "user__username")
    inlines = [AssetGroupAssetInline]


class PortfolioAllocationInline(admin.TabularInline):
    model = PortfolioAllocation
    extra = 0


@admin.register(Portfolio)
class PortfolioAdmin(admin.ModelAdmin):
    list_display = ("name", "user", "initial_capital", "risk_per_trade", "rebalance", "created_at")
    search_fields = ("name", "user__username")
    inlines = [PortfolioAllocationInline]


@admin.register(PortfolioRun)
class PortfolioRunAdmin(admin.ModelAdmin):
    list_display = ("id", "portfolio", "start_date", "end_date", "equity_final", "n_trades", "win_rate")
    list_filter = ("portfolio",)
    date_hierarchy = "start_date"


@admin.register(PortfolioTrade)
class PortfolioTradeAdmin(admin.ModelAdmin):
    list_display = ("run", "symbol", "entry_date", "exit_date", "outcome", "pnl", "r_multiple")
    list_filter = ("run", "outcome")
    search_fields = ("symbol",)


@admin.register(PortfolioEquity)
class PortfolioEquityAdmin(admin.ModelAdmin):
    list_display = ("run", "date", "equity")
    list_filter = ("run",)
