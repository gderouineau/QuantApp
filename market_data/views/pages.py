from django.shortcuts import render
from market_data.models import Asset, IngestionRun

def status_page(request):
    runs = IngestionRun.objects.all()[:50]
    return render(request, "market_data/status.html", {"runs": runs})

def chart_page(request):
    # symbol via ?symbol=AAPL (fallback sur premier Asset actif)
    symbol = request.GET.get("symbol")
    if not symbol:
        a = Asset.objects.filter(is_active=True).order_by("symbol").first()
        symbol = a.symbol if a else "AAPL"
    return render(request, "market_data/chart.html", {"symbol": symbol})
