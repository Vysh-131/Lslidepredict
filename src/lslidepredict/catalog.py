"""Loading and sampling the landslide inventory CSV."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import LANDSLIDE_CSV


def load_landslides(path: Path | str = LANDSLIDE_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["is_northeast"] = df["is_northeast"].astype(str).str.strip().str.lower().eq("true")
    df = df.dropna(subset=["Latitude", "Longitude", "event_date"])
    return df.reset_index(drop=True)


def select_northeast_sample(
    df: pd.DataFrame,
    n: int = 5,
    start_year: int = 2018,
    end_year: int = 2024,
    seed: int = 42,
    one_per_state: bool = True,
) -> pd.DataFrame:
    """Pick `n` North-East rows with event_date in [start_year, end_year].

    With `one_per_state` the sample is spread across different states (and therefore
    different terrain / MGRS tiles), which makes a better smoke test than 5 slides
    from the same hillside.
    """
    m = df["is_northeast"] & df["event_date"].dt.year.between(start_year, end_year)
    pool = df[m]
    if pool.empty:
        raise ValueError("No North-East rows in the requested year range")

    if not one_per_state:
        return pool.sample(n=min(n, len(pool)), random_state=seed).reset_index(drop=True)

    states = pool["State"].drop_duplicates().sample(frac=1, random_state=seed).tolist()
    picks = []
    for i, state in enumerate(states[:n]):
        picks.append(pool[pool["State"] == state].sample(n=1, random_state=seed + i))
    sample = pd.concat(picks)
    if len(sample) < n:  # fewer states than n: top up from the remaining rows
        rest = pool.drop(sample.index)
        sample = pd.concat([sample, rest.sample(n=n - len(sample), random_state=seed)])
    return sample.reset_index(drop=True)


def slide_key(slide_no: str) -> str:
    """Filesystem-safe id for a Slide_No like 'ASM/HLK/83D10/2020/003'."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(slide_no)).strip("_")
