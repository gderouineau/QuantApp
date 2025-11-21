# neuronal_strategy/management/commands/ns_dataset_indicators.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from neuronal_strategy.models import NSDataset, NSRun
from neuronal_strategy.services.io import default_data_dir
from neuronal_strategy.selectors.prices import load_universe_ohlcv

MIN_PRICE_COLS = ("open", "high", "low", "close")

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def _bbands(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = _sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return mid, upper, lower

def _volume_rel(vol: pd.Series, n: int = 20) -> pd.Series:
    med = vol.rolling(n, min_periods=1).median()
    rel = vol / med.replace(0, np.nan)
    return rel.fillna(0.0).clip(0, 1000)

def _map_to_grid(v: pd.Series, vmin: pd.Series, vmax: pd.Series, y_res: int) -> pd.Series:
    span = (vmax - vmin).replace(0, np.nan)
    z = (v - vmin) / span
    out = (z * y_res).clip(0, y_res)
    return out.round().astype("Int64")

class Command(BaseCommand):
    help = (
        "Construit le parquet de features (X=1) avec indicateurs: "
        "pos_open/pos_high/pos_low/pos_close, SMA20/50/100, Bollinger(20,2), vol_rel. "
        "Met à jour NSDataset.dataset_path. Respecte ds.timeframe (1D/1W/1H)."
    )

    def add_arguments(self, parser):
        parser.add_argument("--dataset_id", type=int, required=True)
        parser.add_argument("--y_resolution", type=int, default=None,
                            help="Override Y; sinon prend ds.y_resolution")
        parser.add_argument("--drop_na_head", action="store_true",
                            help="Drop débuts de série tant que SMA20 indisponible.")

    def handle(self, *args, **opts):
        ds = NSDataset.objects.get(pk=opts["dataset_id"])
        y_res = int(opts["y_resolution"]) if opts["y_resolution"] else int(ds.y_resolution or 500)
        if y_res <= 0:
            raise CommandError("y_resolution doit être > 0")

        tf = (ds.timeframe or "1D").upper().strip()
        run = NSRun.objects.create(dataset=ds, kind="build_dataset", status="pending")
        run.mark_running()

        try:
            data = load_universe_ohlcv(ds.universe, ds.date_from, ds.date_to, timeframe=tf)  # {instr: df}
            if not data:
                raise CommandError(f"Aucune donnée chargée (univers/TF={tf}). Vérifie chemins et colonnes.")

            parts = []
            skipped = 0

            for instr, df in data.items():
                try:
                    if df is None or df.empty:
                        self.stdout.write(self.style.WARNING(f"[skip] {instr}: dataframe vide"))
                        skipped += 1
                        continue

                    # DatetimeIndex propre
                    if not isinstance(df.index, pd.DatetimeIndex):
                        if "date" in df.columns:
                            df = df.set_index("date")
                        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
                    else:
                        # s'assure UTC
                        df.index = pd.to_datetime(df.index, utc=True)

                    df = df.sort_index()
                    df = df[~df.index.isna()]
                    if df.empty:
                        self.stdout.write(self.style.WARNING(f"[skip] {instr}: index datetime invalide"))
                        skipped += 1
                        continue

                    # Normalisation colonnes
                    ren = {
                        "Open": "open", "High": "high", "Low": "low", "Close": "close",
                        "Adj Close": "adj_close", "Volume": "volume",
                        "Adj_Close": "adj_close", "adj close": "adj_close",
                    }
                    df = df.rename(columns=ren)

                    if not set(MIN_PRICE_COLS).issubset(df.columns):
                        self.stdout.write(self.style.WARNING(
                            f"[skip] {instr}: colonnes manquantes {set(MIN_PRICE_COLS) - set(df.columns)}"
                        ))
                        skipped += 1
                        continue

                    # Colonnes numériques
                    open_ = pd.to_numeric(df["open"], errors="coerce")
                    high  = pd.to_numeric(df["high"], errors="coerce")
                    low   = pd.to_numeric(df["low"], errors="coerce")
                    close = pd.to_numeric(df["close"], errors="coerce")
                    vol   = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else pd.Series(0.0, index=df.index)

                    # Indicateurs
                    sma20  = _sma(close, 20)
                    sma50  = _sma(close, 50)
                    sma100 = _sma(close, 100)
                    bb_mid, bb_up, bb_lo = _bbands(close, 20, 2.0)
                    vol_rel = _volume_rel(vol, 20)

                    # Échelles min/max par barre
                    base_min = pd.concat([low, bb_lo, sma20, sma50, sma100], axis=1).min(axis=1)
                    base_max = pd.concat([high, bb_up, sma20, sma50, sma100], axis=1).max(axis=1)

                    # Mapping 0..Y
                    features = pd.DataFrame(
                        {
                            "instrument": instr,
                            "pos_open":   _map_to_grid(open_,  base_min, base_max, y_res),
                            "pos_high":   _map_to_grid(high,   base_min, base_max, y_res),
                            "pos_low":    _map_to_grid(low,    base_min, base_max, y_res),
                            "pos_close":  _map_to_grid(close,  base_min, base_max, y_res),
                            "pos_sma20":  _map_to_grid(sma20,  base_min, base_max, y_res),
                            "pos_sma50":  _map_to_grid(sma50,  base_min, base_max, y_res),
                            "pos_sma100": _map_to_grid(sma100, base_min, base_max, y_res),
                            "pos_bb_upper": _map_to_grid(bb_up, base_min, base_max, y_res),
                            "pos_bb_lower": _map_to_grid(bb_lo, base_min, base_max, y_res),
                            "vol_rel": vol_rel.astype("float32"),
                        },
                        index=df.index,  # datetime UTC
                    )

                    if opts["drop_na_head"]:
                        features = features[features["pos_sma20"].notna()]

                    if features.empty:
                        self.stdout.write(self.style.WARNING(f"[skip] {instr}: features vides après drop NA"))
                        skipped += 1
                        continue

                    parts.append(features)

                except Exception as e:
                    self.stdout.write(self.style.WARNING(f"[skip] {instr}: erreur {e!r}"))
                    skipped += 1
                    continue

            if not parts:
                raise CommandError("Aucun instrument avec données valides.")

            tmp = pd.concat(parts, axis=0)

            # Assure la colonne temporelle __ts__ de manière robuste
            # 1) si possible via reset_index(names="__ts__") (pandas >= 1.4)
            try:
                out = tmp.reset_index(names="__ts__")
            except TypeError:
                # 2) fallback universel
                out = tmp.reset_index()
                idx_name = tmp.index.name or "index"
                if "__ts__" not in out.columns:
                    if idx_name in out.columns:
                        out = out.rename(columns={idx_name: "__ts__"})
                    else:
                        # Ultime fallback: crée __ts__ depuis la première colonne si besoin
                        first_col = out.columns[0]
                        out["__ts__"] = out[first_col]

            # Force datetime UTC pour __ts__
            out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")

            # Tri + index sur __ts__
            out = out.sort_values(["instrument", "__ts__"]).set_index("__ts__")
            out.index.name = None

            # Écriture parquet
            out_dir = os.path.join(default_data_dir(), "datasets")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"features_ds{ds.id}.parquet")
            out.to_parquet(out_path, index=True)

            ds.dataset_path = out_path
            ds.y_resolution = y_res
            ds.save(update_fields=["dataset_path", "y_resolution"])

            run.mark_done(metrics={
                "rows": int(len(out)),
                "y_resolution": y_res,
                "features": list(out.columns),
                "skipped": skipped,
                "timeframe": tf,
            })
            self.stdout.write(self.style.SUCCESS(
                f"Dataset features saved → {out_path}  rows={len(out)}  skipped={skipped}  tf={tf}"
            ))

        except Exception as e:
            run.mark_failed(str(e))
            raise
