from __future__ import annotations
from datetime import date
from pathlib import Path
import pandas as pd
from django.conf import settings
from django.http import JsonResponse, Http404
from django.db.models import Q

from market_data.models import Asset

DIR_MAP = {"1D": "1d", "1W": "1w", "1H": "1h"}


def bars_json(request):
    symbol = request.GET.get("symbol")
    tf = request.GET.get("tf", "1D")
    if not symbol or tf not in DIR_MAP:
        raise Http404("bad params")

    safe_symbol = symbol.replace("/", "_").replace(":", "_").replace("=", "_")
    fp = Path(settings.MARKET_DATA_DIR) / DIR_MAP[tf] / f"{safe_symbol}.parquet"
    if not fp.exists():
        # Pas encore téléchargé → vide
        return JsonResponse({"symbol": symbol, "tf": tf, "data": []})

    df = pd.read_parquet(fp)
    df = df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]].copy()

    # Filet de sécurité : coercition + drop des lignes OHLC NaN (jours fériés/trous)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    subset = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if subset:
        df = df.dropna(subset=subset, how="any")
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)

    # Filtrage (optionnel)
    start = request.GET.get("start")
    end = request.GET.get("end")
    if start:
        df = df[df.index.date >= date.fromisoformat(start)]
    if end:
        df = df[df.index.date <= date.fromisoformat(end)]

    # time en SECONDES (le front multiplie par 1000)
    out = []
    for ts, row in df.iterrows():
        ts = pd.Timestamp(ts)
        # si naïf -> localise UTC ; si tz-aware -> convertit UTC
        try:
            t_sec = int(ts.tz_localize("UTC").timestamp())
        except (TypeError, ValueError):
            t_sec = int(ts.tz_convert("UTC").timestamp())
        item = {
            "time": t_sec,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        if "volume" in df.columns:
            item["volume"] = float(row["volume"])
        out.append(item)

    return JsonResponse({"symbol": symbol, "tf": tf, "data": out})


def assets_search(request):
    q = (request.GET.get("q") or "").strip()
    qs = Asset.objects.filter(is_active=True)
    if q:
        qs = qs.filter(Q(symbol__icontains=q) | Q(y_symbol__icontains=q))
    qs = qs.order_by("symbol")[:20]
    data = [{"symbol": a.symbol, "y_symbol": a.y_symbol, "type": a.type} for a in qs]
    return JsonResponse({"results": data})
