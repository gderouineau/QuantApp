# portfolios/models.py

from __future__ import annotations

from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone

from market_data.models import Asset
from strategies.models import Strategy
from strategies.models import Strategy, Signal

User = get_user_model()


class AssetGroup(models.Model):
    """Un panier d'actifs (ex: CAC40, US Mega, Tech)."""
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL, related_name="asset_groups")
    name = models.CharField(max_length=128)
    is_global = models.BooleanField(default=False)  # si True = utilisable par tous
    assets = models.ManyToManyField(Asset, through="AssetGroupAsset", related_name="groups")

    class Meta:
        unique_together = (("user", "name"),)
        ordering = ["name"]

    def __str__(self) -> str:
        owner = "global" if self.is_global else (self.user.username if self.user else "n/a")
        return f"{self.name} ({owner})"


class AssetGroupAsset(models.Model):
    group = models.ForeignKey(AssetGroup, on_delete=models.CASCADE)
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE)

    class Meta:
        unique_together = (("group", "asset"),)


class Portfolio(models.Model):
    """Portefeuille lié à un utilisateur."""
    REBALANCE_CHOICES = [
        ("NONE", "No rebalance"),
        ("MONTHLY", "Monthly"),
        ("QUARTERLY", "Quarterly"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="portfolios")
    name = models.CharField(max_length=128)
    initial_capital = models.FloatField(default=100000.0)
    risk_per_trade = models.FloatField(default=0.01)  # % du capital par trade
    rebalance = models.CharField(max_length=16, choices=REBALANCE_CHOICES, default="NONE")
    max_total_positions = models.PositiveIntegerField(default=0, help_text="0 = illimité (MVP)")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = (("user", "name"),)
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.user.username})"


class PortfolioAllocation(models.Model):
    """Lien portefeuille -> (stratégie + groupe d'actifs) avec poids et limites."""
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="allocations")
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE, related_name="portfolio_allocations")
    group = models.ForeignKey(AssetGroup, on_delete=models.CASCADE, related_name="allocations")
    weight = models.FloatField(default=0.5)  # 0..1, somme <= 1 (pas strictement imposé en MVP)
    max_positions = models.PositiveIntegerField(default=3)
    per_trade_risk = models.FloatField(null=True, blank=True, help_text="Override risk_per_trade du portefeuille")
    notes = models.CharField(max_length=255, blank=True, default="")
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["portfolio", "-weight"]
        unique_together = (("portfolio", "strategy", "group"),)

    def __str__(self) -> str:
        return f"{self.portfolio.name} • {self.strategy.name} @ {self.group.name}"


class PortfolioRun(models.Model):
    """Une exécution de simulation d'un portefeuille sur une période."""
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="runs")
    started_at = models.DateTimeField(default=timezone.now)
    start_date = models.DateField()
    end_date = models.DateField()
    capital_start = models.FloatField(default=0.0)
    equity_final = models.FloatField(default=0.0)
    n_trades = models.PositiveIntegerField(default=0)
    win_rate = models.FloatField(default=0.0)      # 0..1
    avg_R = models.FloatField(default=0.0)
    expectancy_R = models.FloatField(default=0.0)
    params = models.JSONField(default=dict, blank=True)
    summary = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return f"Run #{self.id} • {self.portfolio.name} [{self.start_date}→{self.end_date}]"


class PortfolioTrade(models.Model):
    """Trade simulé pendant un PortfolioRun."""
    run = models.ForeignKey(PortfolioRun, on_delete=models.CASCADE, related_name="trades")
    allocation = models.ForeignKey(PortfolioAllocation, on_delete=models.SET_NULL, null=True, blank=True)
    strategy = models.ForeignKey(Strategy, on_delete=models.SET_NULL, null=True, blank=True)
    symbol = models.CharField(max_length=64)

    entry_date = models.DateField()
    entry_price = models.FloatField()
    qty = models.PositiveIntegerField()
    stop_price = models.FloatField(null=True, blank=True)

    exit_date = models.DateField()
    exit_price = models.FloatField()
    outcome = models.CharField(max_length=32, default="EOD")  # SL / TPx / EOD
    r_multiple = models.FloatField(default=0.0)
    pnl = models.FloatField(default=0.0)

    details = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["entry_date"]


class PortfolioEquity(models.Model):
    """Courbe d'equity (snapshot par jour ou par événement) pour un run."""
    run = models.ForeignKey(PortfolioRun, on_delete=models.CASCADE, related_name="equity_points")
    date = models.DateField()
    equity = models.FloatField()

    class Meta:
        unique_together = (("run", "date"),)
        ordering = ["date"]


class Position(models.Model):
    SIDE_CHOICES = [("LONG", "Long"), ("SHORT", "Short")]
    STATUS_CHOICES = [("OPEN", "Open"), ("CLOSED", "Closed"), ("CANCELED", "Canceled")]
    OUTCOME_CHOICES = [("TP", "Take Profit"), ("SL", "Stop Loss"), ("MANUAL", "Manual"), ("EOD", "End of period"), ("NA", "N/A")]

    portfolio = models.ForeignKey("portfolios.Portfolio", on_delete=models.CASCADE, related_name="positions")
    allocation = models.ForeignKey("portfolios.PortfolioAllocation", on_delete=models.SET_NULL, null=True, blank=True, related_name="positions")
    strategy = models.ForeignKey(Strategy, on_delete=models.PROTECT, related_name="positions")
    asset = models.ForeignKey(Asset, on_delete=models.PROTECT, related_name="positions")
    source_signal = models.ForeignKey(Signal, on_delete=models.SET_NULL, null=True, blank=True, related_name="positions")

    side = models.CharField(max_length=5, choices=SIDE_CHOICES, default="LONG")
    status = models.CharField(max_length=8, choices=STATUS_CHOICES, default="OPEN")

    qty = models.PositiveIntegerField(default=0)

    entry_date = models.DateField(default=timezone.now)
    entry_price = models.FloatField()
    stop_init = models.FloatField()
    stop_cur = models.FloatField()
    take_profits = models.JSONField(default=list, blank=True)  # ex: [{"px": 1.25, "qty": 50}, ...]

    exit_date = models.DateField(null=True, blank=True)
    exit_price = models.FloatField(null=True, blank=True)
    outcome = models.CharField(max_length=10, choices=OUTCOME_CHOICES, default="NA")

    realized_pnl = models.FloatField(default=0.0)
    r_multiple = models.FloatField(default=0.0)
    details = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    sl_method = models.CharField(
        max_length=16,
        default="PERCENT",  # PERCENT | ATR | INDICATOR | SWING | CANDLE | INVALIDATION
        blank=True,
    )
    trailing = models.JSONField(default=dict,
                                blank=True)  # ex: {"method":"ATR","mult":3.0,"atr_col":"atr_14","activate_after_R":1.0}
    guardrails = models.JSONField(default=dict,
                                  blank=True)  # ex: {"min_stop_distance_pct":0.01,"max_stop_distance_pct":0.15,"never_widen":True}
    risk_per_share = models.FloatField(default=0.0)  # |entry - stop_init|
    risk_amount = models.FloatField(default=0.0)  # sizing * risk_per_share (en monnaie)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.portfolio.name} — {self.asset.symbol} ({self.status})"


class PositionEvent(models.Model):
    KIND_CHOICES = [
        ("CREATED", "Created"),
        ("MOVE_SL", "Move SL"),
        ("ADD_TP", "Add TP"),
        ("PARTIAL_EXIT", "Partial exit"),
        ("CLOSE", "Close"),
        ("NOTE", "Note"),
    ]
    position = models.ForeignKey(Position, on_delete=models.CASCADE, related_name="events")
    ts = models.DateTimeField(auto_now_add=True)
    kind = models.CharField(max_length=16, choices=KIND_CHOICES)
    data = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-ts"]

    def __str__(self):
        return f"{self.position.asset.symbol} {self.kind} @ {self.ts:%Y-%m-%d %H:%M}"

