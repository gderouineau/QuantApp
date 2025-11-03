from django import forms
from market_data.models import Asset

class AssetForm(forms.ModelForm):
    class Meta:
        model = Asset
        fields = ["symbol","type","exchange","currency","timezone","y_symbol","av_symbol","is_active"]
