# portfolios/models.py (nouvelle app si tu veux)

from django.db import models
from django.contrib.auth.models import User

class AssetGroup(models.Model):
    name = models.CharField(max_length=100, unique=True)
    assets = models.ManyToManyField('market_data.Asset', related_name='groups')
    def __str__(self): return self.name

class Portfolio(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="portfolios")
    name = models.CharField(max_length=100)
    initial_capital = models.FloatField(default=100000.0)
    risk_per_trade = models.FloatField(default=0.01)
    rebalance = models.CharField(max_length=16, default="monthly")  # future
    def __str__(self): return f"{self.user.username} / {self.name}"

class PortfolioAllocation(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="allocations")
    strategy = models.ForeignKey('strategies.Strategy', on_delete=models.CASCADE)
    group = models.ForeignKey(AssetGroup, on_delete=models.CASCADE, related_name="allocations")
    weight = models.FloatField(default=0.25)          # part du capital
    max_positions = models.PositiveIntegerField(default=5)
    notes = models.TextField(blank=True, default="")
