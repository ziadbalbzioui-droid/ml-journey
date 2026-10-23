"""
Loading. train_series.parquet is ~127M rows (one row per 5 s step), so it is read lazily with
polars and narrowed before collecting. Full table with these dtypes: ~2.5 GB in RAM.

Columns
    series_id  one accelerometer recording = one child (277 in train)
    step       integer index of the 5 s epoch within the series
    timestamp  local wall-clock time; the UTC offset is dropped (it changes with DST)
    anglez     arm angle relative to the horizontal plane, degrees
    enmo       Euclidean norm minus one (g), clipped at 0 — movement intensity
"""
from pathlib import Path

import pandas as pd
import polars as pl

DATA = Path(__file__).resolve().parents[3] / "data" / "cmi-sleep"
STEPS_PER_MIN = 12  # 5 s epochs
STEPS_PER_DAY = 12 * 60 * 24


def scan_series(split: str = "train") -> pl.LazyFrame:
    """Lazy frame over {train,test}_series.parquet with compact dtypes. Filter, then .collect()."""
    return pl.scan_parquet(DATA / f"{split}_series.parquet").with_columns(
        pl.col("series_id").cast(pl.Categorical),
        pl.col("step").cast(pl.UInt32),
        pl.col("timestamp").str.slice(0, 19).str.to_datetime("%Y-%m-%dT%H:%M:%S"),
        pl.col("anglez").cast(pl.Float32),
        pl.col("enmo").cast(pl.Float32),
    )


def load_series(series_ids: list[str] | None = None, split: str = "train") -> pd.DataFrame:
    """Eager pandas frame, optionally restricted to some series_ids. Sorted by (series_id, step)."""
    lf = scan_series(split)
    if series_ids is not None:
        lf = lf.filter(pl.col("series_id").cast(pl.String).is_in(series_ids))
    return lf.sort("series_id", "step").collect().to_pandas()


def series_ids(split: str = "train") -> list[str]:
    lf = pl.scan_parquet(DATA / f"{split}_series.parquet")
    return lf.select(pl.col("series_id").unique()).collect()["series_id"].sort().to_list()


def load_events() -> pd.DataFrame:
    """train_events.csv: one row per (series_id, night, event). step is NaN when the event was
    not labelled that night — keep that in mind before dropping or imputing anything."""
    ev = pd.read_csv(DATA / "train_events.csv")
    ev["step"] = ev["step"].astype("Int64")
    ev["timestamp"] = pd.to_datetime(ev["timestamp"].str.slice(0, 19))
    return ev
