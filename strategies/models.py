# strategies/models.py
from django.db import models
from django.contrib.auth.models import User
from market_data.models import Asset


class Strategy(models.Model):
    TYPE_CHOICES = [
        ('CUSTOM', 'Code personnalisé'),
        ('RS', 'Relative Strength'),
        ('VCP', 'Volatility Contraction'),
        ('PP', 'Pocket Pivot'),
        ('SMD', 'Smart Money Divergence'),
        ('GC', 'Golden Cross'),
        ('VB', 'Volume Breakout'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="strategies")
    name = models.CharField(max_length=100)
    type = models.CharField(max_length=10, choices=TYPE_CHOICES)

    # Code Python pour type CUSTOM
    code = models.TextField(blank=True, help_text="Fonction evaluate(df_prices, df_indicators, params) -> dict")

    parameters = models.JSONField(default=dict)
    optimize_parameters = models.JSONField(
        default=dict,
        blank=True,
        help_text=(
            "Plan d’optimisation (JSON). Exemple minimal : "
            "{\"grid\": {"
            "  \"trend_ma\": {\"ranges\": [[120,300]], \"step\": 10}, "
            "  \"rsi_max\": {\"values\": [35,40,45,50,55]}"
            "}, "
            "\"constraints\": [{\"lt\": [\"pullback_ma\", \"trend_ma\"]}], "
            "\"budget\": {\"n_trials\": 120, \"n_folds\": 3, \"step\": 126, \"min_train\": 630, \"sample_n_symbols\": 20}"
            "}"
        ),
    )
    weight = models.FloatField(default=1.0)
    is_active = models.BooleanField(default=True)

    # Qualité de vie
    description = models.TextField(blank=True, default="")
    version = models.CharField(max_length=20, blank=True, default="")
    last_run_at = models.DateTimeField(null=True, blank=True)
    last_error = models.TextField(blank=True, default="")

    # Garde-fous
    max_runtime_ms = models.PositiveIntegerField(default=1500)
    max_memory_mb = models.PositiveIntegerField(default=256)

    created_at = models.DateTimeField(auto_now_add=True)

    sltp_preset = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["is_active", "type"]),
        ]

    def __str__(self):
        return f"{self.name} ({self.get_type_display()})"


class Signal(models.Model):
    STRENGTH_CHOICES = [
        ('WEAK', 'Faible'),
        ('MEDIUM', 'Moyen'),
        ('STRONG', 'Fort'),
    ]
    DIRECTION = [('LONG', 'Long'), ('SHORT', 'Short')]

    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE, related_name="signals")
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name="signals")
    date = models.DateField()
    score = models.FloatField()  # 0-100
    strength = models.CharField(max_length=10, choices=STRENGTH_CHOICES)
    details = models.JSONField(default=dict, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    direction = models.CharField(max_length=5, choices=DIRECTION, default='LONG')
    entry_price = models.FloatField(null=True, blank=True)
    stop_price = models.FloatField(null=True, blank=True)
    take_profits = models.JSONField(default=list, blank=True)  # ex: [{"target":1.0,"price":123.4}, ...]
    rr_target = models.FloatField(null=True, blank=True)  # objectif en R si applicable
    atr_at_entry = models.FloatField(null=True, blank=True)
    valid_until = models.DateField(null=True, blank=True)

    class Meta:
        unique_together = (("strategy", "asset", "date"),)
        ordering = ["-date", "-score"]
        indexes = [
            models.Index(fields=["strategy", "asset", "date"]),
            models.Index(fields=["asset", "-date"]),
        ]

    def __str__(self):
        return f"{self.asset.symbol} - {self.score:.1f} ({self.date})"




class Trade(models.Model):
    STATUS = [('OPEN','Open'),('CLOSED','Closed'),('CANCELLED','Cancelled')]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="trades")
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE, related_name="trades")
    asset = models.ForeignKey('market_data.Asset', on_delete=models.CASCADE, related_name="trades")
    signal = models.ForeignKey(Signal, on_delete=models.SET_NULL, null=True, blank=True)

    direction = models.CharField(max_length=5, choices=Signal.DIRECTION, default='LONG')
    entry_date = models.DateField()
    entry_price = models.FloatField()
    stop_price = models.FloatField()
    take_profits = models.JSONField(default=list, blank=True)  # mêmes cibles que Signal
    qty = models.FloatField(default=0.0)                       # position sizing (unités)
    fees = models.FloatField(default=0.0)                      # frais totaux estimés

    exit_date = models.DateField(null=True, blank=True)
    exit_price = models.FloatField(null=True, blank=True)
    outcome = models.CharField(max_length=16, blank=True, default="")  # "SL" | "TP1" | "TP2" | "TSl" | etc.
    r_multiple = models.FloatField(null=True, blank=True)              # (PnL / risk_per_share)
    status = models.CharField(max_length=10, choices=STATUS, default='OPEN')

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-entry_date", "-id"]
