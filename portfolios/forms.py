from __future__ import annotations
from django import forms
from django.contrib.auth import get_user_model

from market_data.models import Asset
from strategies.models import Strategy
from .models import Portfolio, PortfolioAllocation, AssetGroup, Position

User = get_user_model()


class PortfolioForm(forms.ModelForm):
    class Meta:
        model = Portfolio
        fields = [
            "name",
            "initial_capital",
            "risk_per_trade",
            "rebalance",
            "max_total_positions",
        ]


class PortfolioAllocationForm(forms.ModelForm):
    class Meta:
        model = PortfolioAllocation
        fields = [
            "strategy",
            "group",
            "weight",
            "max_positions",
            "per_trade_risk",
            "notes",
            "is_active",
        ]

    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)
        # Stratégies visibles: les siennes + actives (tu peux restreindre à user si tu préfères)
        qs_strat = Strategy.objects.all()
        if user and not user.is_superuser:
            qs_strat = qs_strat.filter(user=user)
        self.fields["strategy"].queryset = qs_strat.order_by("name")

        # Groupes visibles: globaux + appartenant à l'utilisateur
        qs_groups = AssetGroup.objects.filter(is_global=True)
        if user:
            qs_groups = (qs_groups | AssetGroup.objects.filter(user=user))
        self.fields["group"].queryset = qs_groups.order_by("name")


class AssetGroupForm(forms.ModelForm):
    # Edition des actifs via M2M direct (convenient)
    assets = forms.ModelMultipleChoiceField(
        queryset=Asset.objects.all().order_by("symbol"),
        required=False,
        widget=forms.SelectMultiple(attrs={"size": 15})
    )

    class Meta:
        model = AssetGroup
        fields = ["name", "is_global", "assets"]

    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)
        # Un non-staff ne peut pas cocher is_global
        if not (user and (user.is_staff or user.is_superuser)):
            self.fields["is_global"].disabled = True
        # Pré-cocher assets si instance existante
        if self.instance and self.instance.pk:
            self.fields["assets"].initial = self.instance.assets.all()

    def save(self, commit=True):
        inst = super().save(commit=False)
        if commit:
            inst.save()
            # Sync M2M via through model
            inst.assets.set(self.cleaned_data.get("assets", []))
        return inst


class AcceptSignalForm(forms.Form):
    signal_id = forms.IntegerField(widget=forms.HiddenInput)
    allocation_id = forms.IntegerField(required=False, widget=forms.HiddenInput)  # rempli côté vue
    side = forms.ChoiceField(choices=[("LONG","Long"),("SHORT","Short")], initial="LONG")
    entry_price = forms.FloatField(help_text="Par défaut: dernier close")
    stop_method = forms.ChoiceField(
        choices=[("PCT","% du prix"), ("ATR","ATR xN"), ("MANUAL","Manuel")],
        initial="PCT"
    )
    stop_value = forms.FloatField(help_text="Ex: 2 (pour 2%) ou 1.5 (pour 1.5x ATR) ou prix manuel", initial=2.0)
    risk_pct = forms.FloatField(help_text="Risque par trade (ex: 1 = 1%)", initial=1.0)

    def clean(self):
        c = super().clean()
        if c.get("entry_price") and c.get("stop_method") == "MANUAL" and c.get("stop_value") is None:
            raise forms.ValidationError("Stop MANUAL: fournissez un prix.")
        return c


class MoveSLForm(forms.Form):
    stop_price = forms.FloatField(label="Nouveau stop (prix)", min_value=0.0)


class PartialExitForm(forms.Form):
    qty = forms.IntegerField(label="Quantité à vendre", min_value=1)
    exit_price = forms.FloatField(label="Prix de sortie", min_value=0.0)


class ClosePositionForm(forms.Form):
    exit_price = forms.FloatField(label="Prix de sortie", min_value=0.0)
    outcome = forms.ChoiceField(choices=Position.OUTCOME_CHOICES, initial="MANUAL")

class PortfolioBacktestForm(forms.Form):
    start = forms.DateField(required=False, help_text="Optionnel")
    end = forms.DateField(required=False, help_text="Optionnel")
    warmup = forms.IntegerField(initial=252, min_value=0, label="Warmup bars")
    capital_override = forms.FloatField(required=False, label="Capital (override)")
    risk_override = forms.FloatField(required=False, label="Risque/trade (ex: 0.01)")

    def clean(self):
        c = super().clean()
        s, e = c.get("start"), c.get("end")
        if s and e and s > e:
            raise forms.ValidationError("La date de début doit précéder la date de fin.")
        return c