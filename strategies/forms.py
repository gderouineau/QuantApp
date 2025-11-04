# strategies/forms.py

from __future__ import annotations
import json
from django import forms
from strategies.models import Strategy


class StrategyCustomForm(forms.ModelForm):
    parameters_text = forms.CharField(
        label="Paramètres (JSON)",
        required=False,
        widget=forms.Textarea(attrs={"rows": 8, "spellcheck": "false", "class": "textarea"}),
        help_text='JSON valide (ex: {"lookback_days": 63}). Vide = {}.'
    )

    code = forms.CharField(
        label="Code Python",
        required=False,
        widget=forms.Textarea(attrs={
            "rows": 22,
            "spellcheck": "false",
            "class": "textarea mono",
            "placeholder": (
                "# Variables : asset_data, indicators, parameters\n"
                "signal = False\nscore = 0.0\nstrength = 'WEAK'\ndetails = {}\n"
            ),
        }),
        help_text="Doit définir: signal(bool), score(0..100), strength('WEAK'|'MEDIUM'|'STRONG'), details(dict)."
    )

    class Meta:
        model = Strategy
        fields = ["name", "code", "weight", "is_active"]  # parameters via parameters_text
        widgets = {
            "name": forms.TextInput(attrs={"class": "input"}),
            "weight": forms.NumberInput(attrs={"step": "0.1", "class": "input"}),
            "is_active": forms.CheckboxInput(attrs={"class": "checkbox"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        params = getattr(self.instance, "parameters", {}) or {}
        self.fields["parameters_text"].initial = json.dumps(params, ensure_ascii=False, indent=2)
        # snippet par défaut en création
        if not self.instance.pk and not (self.initial.get("code") or self.data.get("code")):
            self.fields["code"].initial = (
                "close = asset_data['close'].astype(float)\n"
                "volume = asset_data['volume'].astype(float)\n"
                "signal = False\nscore = 0.0\nstrength = 'WEAK'\ndetails = {}\n"
            )
        # tampon interne pour clean/save
        self._parsed_parameters = {}

    def clean(self):
        cleaned = super().clean()
        txt = cleaned.get("parameters_text") or ""
        if txt.strip():
            try:
                parsed = json.loads(txt)
                if not isinstance(parsed, dict):
                    raise forms.ValidationError("Les paramètres doivent être un objet JSON.")
            except json.JSONDecodeError as e:
                raise forms.ValidationError(f"JSON invalide : {e}")
        else:
            parsed = {}
        # mémorise pour save()
        self._parsed_parameters = parsed
        return cleaned

    def save(self, commit=True):
        obj: Strategy = super().save(commit=False)
        # force le type & paramètres ici (garanti avant save)
        obj.type = "CUSTOM"
        obj.parameters = self._parsed_parameters or {}
        if commit:
            obj.save()
        return obj
