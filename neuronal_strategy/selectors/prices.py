# neuronal_strategy/selectors/prices.py
from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
from django.conf import settings

from market_data.models import Asset, DataFile
from portfolios.models import AssetGroup, AssetGroupAsset

# bars_path est optionnel : on l'utilise en fallback si dispo
try:
    from market_data.services.store import bars_path
except Exception:
    bars_path = None  # type: ignore


# --- Timeframe canonicalisation -------------------------------------------------

# Canonique interne -> DataFile.kind
KIND_MAP: Dict[str, str] = {
    "1H": "bars_1H",
    "1D": "bars_1D",
    "1W": "bars_1W",
}

# Alias -> Canonique
TF_ALIAS_TO_CANON: Dict[str, str] = {
    "1H": "1H", "1h": "1H", "60": "1H", "60m": "1H", "H1": "1H", "h1": "1H",
    "1D": "1D", "1d": "1D", "D": "1D", "d": "1D",
    "1W": "1W", "1w": "1W", "W": "1W", "w": "1W",
}

REQUIRED_COLS = {"date", "open", "high", "low", "close", "volume"}


def _canon_tf(tf: str) -> Optional[str]:
    if not tf:
        return None
    return TF_ALIAS_TO_CANON.get(tf, tf if tf in KIND_MAP else None)


# --- Sélecteurs d'univers -------------------------------------------------------

def _assets_for_group(group_name: str) -> List[Asset]:
    try:
        grp = AssetGroup.objects.get(name=group_name)
    except AssetGroup.DoesNotExist:
        return []
    sym_qs = (
        AssetGroupAsset.objects.filter(group=grp)
        .select_related("asset")
        .values_list("asset__symbol", flat=True)
    )
    return list(Asset.objects.filter(symbol__in=sym_qs, is_active=True).order_by("symbol"))


def get_universe_instruments(universe) -> List[str]:
    """
    universe: NSUniverse (avec soit asset_group, soit asset_group_name, soit rien)
    Retourne la liste des symboles (Asset.symbol) actifs et ordonnés.
    """
    if getattr(universe, "asset_group", None):
        assets = _assets_for_group(universe.asset_group.name)
    elif getattr(universe, "asset_group_name", None):
        assets = _assets_for_group(universe.asset_group_name)
    else:
        assets = list(Asset.objects.filter(is_active=True).order_by("symbol"))
    return [a.symbol for a in assets]


# --- Lecture OHLCV --------------------------------------------------------------

def _fallback_paths(symbol: str, canon_tf: str) -> List[Path]:
    """
    Génère des chemins candidats pour lecture en fallback :
    - via bars_path(symbol, alias) si dispo
    - var/market_data/<tf lower>/<symbol>.parquet
    """
    candidates: List[Path] = []

    # Essais bars_path avec différents alias raisonnables
    if bars_path is not None:
        alias_try = {
            canon_tf,
            canon_tf.lower(),
            "1H" if canon_tf == "1H" else "",
            "1h" if canon_tf == "1H" else "",
            "60" if canon_tf == "1H" else "",
            "60m" if canon_tf == "1H" else "",
            "H1" if canon_tf == "1H" else "",
            "h1" if canon_tf == "1H" else "",
            "1D" if canon_tf == "1D" else "",
            "1d" if canon_tf == "1D" else "",
            "D" if canon_tf == "1D" else "",
            "1W" if canon_tf == "1W" else "",
            "1w" if canon_tf == "1W" else "",
            "W" if canon_tf == "1W" else "",
        }
        for tf in [a for a in alias_try if a]:
            try:
                p = Path(bars_path(symbol, tf))
                candidates.append(p)
            except Exception:
                pass

    # Fallback répertoire local
    base_dir = Path(getattr(settings, "BASE_DIR", Path.cwd()))
    candidates.append(base_dir / "var" / "market_data" / canon_tf.lower() / f"{symbol}.parquet")

    # Dédupe en conservant l'ordre
    seen: set[Path] = set()
    uniq: List[Path] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _read_ohlcv_parquet(parquet_path: Path) -> pd.DataFrame:
    # Lecture robuste (pandas choisira engine sinon)
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        df = pd.read_parquet(parquet_path, engine="pyarrow")

    # Si l'index est datetime et pas de colonne "date" → on réinitialise
    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex) or getattr(df.index, "name", None) in (None, "date"):
            df = df.reset_index()

    # Colonnes en minuscules pour standardiser
    df.columns = [str(c).lower() for c in df.columns]

    # Exige au minimum les colonnes nécessaires
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)} dans {parquet_path}")

    # Dates en UTC tz-aware
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    # Types numériques
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Nettoyage/tri
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    return df[["date", "open", "high", "low", "close", "volume"]]


def load_ohlcv(symbol: str, timeframe: str, date_from=None, date_to=None) -> pd.DataFrame:
    """
    Lecture prioritaire via DataFile (Asset + kind),
    puis fallback via bars_path()/var/market_data.
    Retourne un DataFrame avec colonnes: date, open, high, low, close, volume.
    """
    canon = _canon_tf(timeframe)
    if canon is None or canon not in KIND_MAP:
        return pd.DataFrame()

    # 1) Tentative via DataFile
    try:
        df_rec = DataFile.objects.get(asset__symbol=symbol, kind=KIND_MAP[canon])
        df = _read_ohlcv_parquet(Path(df_rec.path))
    except DataFile.DoesNotExist:
        df = pd.DataFrame()
    except Exception:
        # path cassé → on bascule en fallback
        df = pd.DataFrame()

    # 2) Fallback si vide
    if df.empty:
        for p in _fallback_paths(symbol, canon):
            if not p.exists():
                continue
            try:
                df = _read_ohlcv_parquet(p)
                break
            except Exception:
                continue

    if df.empty:
        return df

    # Fenêtrage par dates
    if date_from is not None:
        df = df[df["date"] >= pd.to_datetime(date_from, utc=True)]
    if date_to is not None:
        df = df[df["date"] <= pd.to_datetime(date_to, utc=True)]

    return df.sort_values("date")


def load_universe_ohlcv(universe, date_from=None, date_to=None, timeframe: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Charge un dict {symbol: df} pour l’univers.
    - timeframe: si None, on prend universe.timeframe
    """
    tf = timeframe or getattr(universe, "timeframe", "1D")
    syms = get_universe_instruments(universe)

    out: Dict[str, pd.DataFrame] = {}
    for sym in syms:
        df = load_ohlcv(sym, tf, date_from, date_to)
        if not df.empty:
            out[sym] = df
    return out
