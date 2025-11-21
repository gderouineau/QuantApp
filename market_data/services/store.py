# market_data/services/store.py

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from django.conf import settings
from market_data.models import Asset, DataFile

DIR_MAP = {"1D": "1d", "1W": "1w", "1H": "1h"}

def bars_path(symbol: str, timeframe: str) -> Path:
    base = Path(settings.MARKET_DATA_DIR) / DIR_MAP[timeframe]
    base.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("/", "_").replace(":", "_").replace("=", "_")
    return base / f"{safe}.parquet"

def write_parquet(df: pd.DataFrame, path: Path) -> tuple[int, pd.Timestamp | None]:
    if df is None or df.empty:
        if path.exists():
            ex = pd.read_parquet(path)
            return len(ex), getattr(ex.index, "max", lambda: None)()
        return 0, None
    df.to_parquet(path, compression="snappy")
    return len(df), df.index.max() if hasattr(df.index, "max") else None

def upsert_datafile(asset: Asset, kind: str, path: Path, row_count: int, last_date) -> DataFile:
    df, _ = DataFile.objects.get_or_create(asset=asset, kind=kind, defaults={"path": str(path)})
    df.path = str(path)
    df.row_count = int(row_count)
    df.last_date = getattr(last_date, "date", lambda: last_date)()
    df.file_size = os.path.getsize(path) if Path(path).exists() else 0
    df.save()
    return df

def read_parquet(pathlike) -> pd.DataFrame:
    """
    Accepte str ou Path. Retourne un DataFrame vide si le fichier n'existe pas.
    """
    p = Path(pathlike) if not isinstance(pathlike, Path) else pathlike
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)