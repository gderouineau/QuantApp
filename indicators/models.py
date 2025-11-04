# indicators/models.py
from django.db import models
from market_data.models import Asset

class Indicator(models.Model):
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name="indicators")
    date = models.DateField()

    sma_20 = models.FloatField(null=True, blank=True)
    sma_50 = models.FloatField(null=True, blank=True)
    sma_200 = models.FloatField(null=True, blank=True)
    ema_12 = models.FloatField(null=True, blank=True)
    ema_26 = models.FloatField(null=True, blank=True)

    rsi_14 = models.FloatField(null=True, blank=True)
    macd = models.FloatField(null=True, blank=True)
    macd_signal = models.FloatField(null=True, blank=True)
    macd_hist = models.FloatField(null=True, blank=True)

    bb_upper = models.FloatField(null=True, blank=True)
    bb_middle = models.FloatField(null=True, blank=True)
    bb_lower = models.FloatField(null=True, blank=True)
    atr_14 = models.FloatField(null=True, blank=True)

    volume_sma_20 = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = (("asset", "date"),)
        ordering = ["asset", "-date"]
        indexes = [
            models.Index(fields=["asset", "date"]),
            models.Index(fields=["-date"]),
        ]
