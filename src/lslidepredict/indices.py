"""Spectral indices on mined patches (band order from config.PATCH_BANDS)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from .config import PATCH_BANDS, REFLECTANCE_SCALE, SCL_CLOUDY


def load_patch(path: Path | str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return ({band: float32 reflectance}, clear_mask)."""
    with rasterio.open(path) as src:
        data = src.read()
    bands = {b: data[i].astype(np.float32) / REFLECTANCE_SCALE for i, b in enumerate(PATCH_BANDS) if b != "SCL"}
    scl = data[PATCH_BANDS.index("SCL")]
    clear = ~np.isin(scl, SCL_CLOUDY) & (data[0] > 0)
    return bands, clear


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((a + b) != 0, (a - b) / (a + b), np.nan).astype(np.float32)


def ndvi(b: dict[str, np.ndarray]) -> np.ndarray:
    return _nd(b["B08"], b["B04"])


def bsi(b: dict[str, np.ndarray]) -> np.ndarray:
    """Bare Soil Index: ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue)).

    High for exposed / eroding soil; SWIR separates it from dry grass and rock shadow.
    """
    return _nd(b["B11"] + b["B04"], b["B08"] + b["B02"])


def ndwi(b: dict[str, np.ndarray]) -> np.ndarray:
    return _nd(b["B03"], b["B08"])


def patch_stats(path: Path | str) -> dict[str, float]:
    bands, clear = load_patch(path)
    if not clear.any():
        return {"mean_ndvi": float("nan"), "mean_bsi": float("nan")}
    return {
        "mean_ndvi": round(float(np.nanmean(ndvi(bands)[clear])), 3),
        "mean_bsi": round(float(np.nanmean(bsi(bands)[clear])), 3),
    }
