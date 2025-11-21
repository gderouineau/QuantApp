from django import forms
from .models import NSDataset

class BuildLabelsForm(forms.Form):
    dataset = forms.ModelChoiceField(queryset=NSDataset.objects.all())
    override_dates = forms.BooleanField(required=False, initial=False)

class BuildDatasetForm(forms.Form):
    dataset = forms.ModelChoiceField(queryset=NSDataset.objects.all())
    scaling_mode = forms.ChoiceField(choices=NSDataset.SCALING_MODES)

class BacktestForm(forms.Form):
    dataset = forms.ModelChoiceField(queryset=NSDataset.objects.all())
    tp_margin = forms.ChoiceField(choices=[("0.10","0.10"), ("0.15","0.15")])
    threshold = forms.FloatField(initial=0.85, min_value=0.5, max_value=0.99)
    fees_bps = forms.FloatField(initial=2.0)
    slippage_bps = forms.FloatField(initial=2.0)
