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

    optimize_parameters_text = forms.CharField(
        label="Plan d’optimisation (JSON)",
        required=False,
        widget=forms.Textarea(attrs={
            "rows": 12,
            "spellcheck": "false",
            "class": "textarea mono",
            "placeholder": (
                '{\n'
                '  "grid": {\n'
                '    "trend_ma": {"ranges": [[120, 300]], "step": 10},\n'
                '    "rsi_max": {"values": [35, 40, 45, 50, 55]},\n'
                '    "sltp": {\n'
                '      "stop": {"percent": {"ranges": [[0.02, 0.06]], "step": 0.01}}\n'
                '    }\n'
                '  },\n'
                '  "constraints": [{"lt": ["pullback_ma", "trend_ma"]}, {"lt": ["vol_fast", "vol_slow"]}],\n'
                '  "budget": {"since": "2020-01-01", "sample_n_symbols": 20, "n_folds": 3, "min_train": 630, "step": 126, "warmup": 252}\n'
                '}'
            ),
        }),
        help_text=(
            "JSON valide. Clés supportées : grid (values|ranges+step sur chemins imbriqués), "
            "constraints (lt/le/gt/ge/eq/ne), budget (since/until/sample_n_symbols/n_folds/min_train/step/warmup). "
            "Laisse vide pour ne pas optimiser via grille."
        )
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
        fields = ["name", "code", "weight", "is_active"]  # paramètres via *_text
        widgets = {
            "name": forms.TextInput(attrs={"class": "input"}),
            "weight": forms.NumberInput(attrs={"step": "0.1", "class": "input"}),
            "is_active": forms.CheckboxInput(attrs={"class": "checkbox"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        params = getattr(self.instance, "parameters", {}) or {}
        self.fields["parameters_text"].initial = json.dumps(params, ensure_ascii=False, indent=2)

        optcfg = getattr(self.instance, "optimize_parameters", {}) or {}
        if optcfg:
            self.fields["optimize_parameters_text"].initial = json.dumps(optcfg, ensure_ascii=False, indent=2)

        # snippet par défaut en création
        if not self.instance.pk and not (self.initial.get("code") or self.data.get("code")):
            self.fields["code"].initial = (
                "close = asset_data['close'].astype(float)\n"
                "volume = asset_data['volume'].astype(float)\n"
                "signal = False\nscore = 0.0\nstrength = 'WEAK'\ndetails = {}\n"
            )

        # tampons internes pour clean/save
        self._parsed_parameters = {}
        self._parsed_optimize_parameters = {}

    def clean(self):
        cleaned = super().clean()

        # parameters_text
        txt = cleaned.get("parameters_text") or ""
        if txt.strip():
            try:
                parsed = json.loads(txt)
                if not isinstance(parsed, dict):
                    raise forms.ValidationError("Les paramètres doivent être un objet JSON.")
            except json.JSONDecodeError as e:
                raise forms.ValidationError(f"JSON invalide (Paramètres) : {e}")
        else:
            parsed = {}
        self._parsed_parameters = parsed

        # optimize_parameters_text
        otxt = cleaned.get("optimize_parameters_text") or ""
        if otxt.strip():
            try:
                oparsed = json.loads(otxt)
                if not isinstance(oparsed, dict):
                    raise forms.ValidationError("Le plan d’optimisation doit être un objet JSON.")
            except json.JSONDecodeError as e:
                raise forms.ValidationError(f"JSON invalide (Plan d’optimisation) : {e}")
        else:
            oparsed = {}
        self._parsed_optimize_parameters = oparsed

        return cleaned

    def save(self, commit=True):
        obj: Strategy = super().save(commit=False)
        # force type & champs JSON
        obj.type = "CUSTOM"
        obj.parameters = self._parsed_parameters or {}
        obj.optimize_parameters = self._parsed_optimize_parameters or {}
        if commit:
            obj.save()
        return obj