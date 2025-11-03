from __future__ import annotations
from django.db import models
from django.utils import timezone

class Asset(models.Model):
    TYPE_CHOICES = [("equity","Equity"),("index","Index"),("fx","Forex")]
    symbol = models.CharField(max_length=64, unique=True)          # canonique interne (ex: AAPL, HO.PA, EURUSD)
    type = models.CharField(max_length=12, choices=TYPE_CHOICES)
    exchange = models.CharField(max_length=32, blank=True, default="")
    currency = models.CharField(max_length=8, blank=True, default="")
    timezone = models.CharField(max_length=64, blank=True, default="UTC")

    # alias fournisseurs
    y_symbol = models.CharField(max_length=64, blank=True, default="")
    av_symbol = models.CharField(max_length=64, blank=True, default="")

    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["symbol"]

    def __str__(self) -> str:
        return self.symbol

    def for_provider(self, name: str) -> str:
        if name == "yahoo" and self.y_symbol:
            return self.y_symbol
        if name == "alphavantage" and self.av_symbol:
            return self.av_symbol
        return self.symbol

class DataFile(models.Model):
    KIND_CHOICES = [("bars_1D","Bars 1D"),("bars_1W","Bars 1W")]
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name="data_files")
    kind = models.CharField(max_length=16, choices=KIND_CHOICES)
    path = models.CharField(max_length=512)
    row_count = models.PositiveIntegerField(default=0)
    last_date = models.DateField(null=True, blank=True)
    file_size = models.BigIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (("asset","kind"),)

class IngestionRun(models.Model):
    SOURCE_CHOICES = [("yahoo","Yahoo"),("alphavantage","AlphaVantage")]
    TF_CHOICES = [("1D","Daily"),("1W","Weekly")]

    source = models.CharField(max_length=16, choices=SOURCE_CHOICES)
    timeframe = models.CharField(max_length=2, choices=TF_CHOICES)
    started_at = models.DateTimeField(default=timezone.now)
    finished_at = models.DateTimeField(null=True, blank=True)

    ok_count = models.PositiveIntegerField(default=0)
    fail_count = models.PositiveIntegerField(default=0)
    anomalies = models.JSONField(default=list, blank=True)
    errors = models.JSONField(default=list, blank=True)
    duration_ms = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-started_at"]
