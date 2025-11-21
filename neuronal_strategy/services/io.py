import os
import pandas as pd
from django.conf import settings

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def default_data_dir() -> str:
    base = getattr(settings, "NS_DATA_DIR", None)
    if not base:
        base = os.path.join(settings.BASE_DIR, "runs", "neuronal_strategy")
    os.makedirs(base, exist_ok=True)
    return base

def save_parquet(df: pd.DataFrame, rel_path: str) -> str:
    base = default_data_dir()
    path = os.path.join(base, rel_path)
    ensure_dir(path)
    df.to_parquet(path, index=True)
    return path
