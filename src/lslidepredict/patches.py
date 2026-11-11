"""Cropping co-registered multi-band patches straight out of remote COG / JP2 files.

Every window (T0/T1/T2) of a slide is warped onto the *same* fixed 10 m UTM grid, so the
resulting patches are pixel-aligned and can be stacked directly for change detection,
even if the scenes come from different MGRS tiles or UTM zones.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from pystac import Item
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform, transform_bounds

from .config import BAND_DESCRIPTIONS, GDAL_ENV, PATCH_BANDS, REFLECTANCE_SCALE, SCL_CLOUDY
from .stac import Provider, scale_offset

log = logging.getLogger(__name__)


# --- network byte accounting -----------------------------------------------------------

class ByteMeter(logging.Handler):
    """Counts bytes GDAL fetches over HTTP by parsing its CPL_DEBUG range-request messages.

    GDAL logs e.g. ``VSICURL: Downloading 130105344-131465215 (https://...)``; rasterio
    forwards those to the ``rasterio`` logger when ``CPL_DEBUG`` is on.
    """

    _rx = re.compile(r"Downloading (\d+)-(\d+) \(")

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self._lock = threading.Lock()
        self.bytes = 0
        self.requests = 0

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Downloading" not in msg:
            return
        for a, b in self._rx.findall(msg):
            with self._lock:
                self.bytes += int(b) - int(a) + 1
                self.requests += 1

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self.bytes, self.requests


_METER: ByteMeter | None = None


def byte_meter() -> ByteMeter:
    """Install (once) and return the global GDAL byte meter."""
    global _METER
    if _METER is None:
        _METER = ByteMeter()
        rlog = logging.getLogger("rasterio")
        rlog.setLevel(logging.DEBUG)
        rlog.addHandler(_METER)
        rlog.propagate = False  # keep GDAL debug chatter out of the console
        warn = logging.StreamHandler()
        warn.setLevel(logging.WARNING)
        rlog.addHandler(warn)
    return _METER


# --- target grid ------------------------------------------------------------------------

@dataclass(frozen=True)
class PatchGrid:
    crs: CRS
    transform: Affine
    width: int
    height: int
    bbox_lonlat: tuple[float, float, float, float]

    @classmethod
    def around(cls, lat: float, lon: float, size_px: int = 256, res: float = 10.0) -> "PatchGrid":
        """Fixed size_px x size_px grid centred on (lat, lon), snapped to the Sentinel-2 10 m lattice.

        256 px at 10 m = 2.56 km, divisible by 32 for CNN/ViT/BiT backbones.
        """
        zone = int((lon + 180) // 6) + 1
        crs = CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)
        (x,), (y,) = transform("EPSG:4326", crs, [lon], [lat])
        half = size_px * res / 2
        minx = round((x - half) / res) * res   # S2 tile origins are multiples of 10 m, so
        maxy = round((y + half) / res) * res   # same-zone 10 m bands need no resampling
        maxx, miny = minx + size_px * res, maxy - size_px * res
        bbox = transform_bounds(crs, "EPSG:4326", minx, miny, maxx, maxy, densify_pts=21)
        return cls(crs=crs, transform=from_origin(minx, maxy, res, res),
                   width=size_px, height=size_px, bbox_lonlat=tuple(bbox))


# --- reading ----------------------------------------------------------------------------

def _read_band(path: str, grid: PatchGrid, resampling: Resampling, env: dict[str, str]) -> np.ndarray:
    with rasterio.Env(CPL_DEBUG=True, **GDAL_ENV, **env):
        with rasterio.open(path) as src:
            if src.crs == grid.crs and resampling is Resampling.bilinear and src.res[0] == grid.transform.a:
                resampling = Resampling.nearest  # aligned same-res band: plain copy
            with WarpedVRT(src, crs=grid.crs, transform=grid.transform, width=grid.width,
                           height=grid.height, resampling=resampling, nodata=0) as vrt:
                return vrt.read(1)


def _resampling_for(band: str) -> Resampling:
    if band == "SCL":
        return Resampling.nearest
    return Resampling.bilinear  # B11 20 m -> 10 m; 10 m bands become a copy when grids align


def read_scl(provider: Provider, item: Item, grid: PatchGrid) -> np.ndarray:
    href = item.assets[provider.band_assets["SCL"]].href
    return _read_band(provider.gdal_path(href), grid, Resampling.nearest, provider.gdal_env())


def patch_cloud_pct(scl: np.ndarray) -> float:
    return float(np.isin(scl, SCL_CLOUDY).mean() * 100.0)


def to_harmonised_dn(dn: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """uint16 DN -> uint16 reflectance*10000 with the baseline-04.00 offset removed. 0 = nodata."""
    refl = dn.astype(np.float32) * scale + offset
    out = np.clip(np.rint(refl * REFLECTANCE_SCALE), 1, 65535).astype(np.uint16)
    out[dn == 0] = 0
    return out


def read_patch(provider: Provider, item: Item, grid: PatchGrid, workers: int = 6,
               scl: np.ndarray | None = None) -> np.ndarray:
    """Return a (len(PATCH_BANDS), H, W) uint16 array for one scene."""
    env = provider.gdal_env()
    bands = [b for b in PATCH_BANDS if not (b == "SCL" and scl is not None)]

    def job(band: str) -> tuple[str, np.ndarray]:
        key = provider.band_assets[band]
        arr = _read_band(provider.gdal_path(item.assets[key].href), grid, _resampling_for(band), env)
        if band != "SCL":
            arr = to_harmonised_dn(arr, *scale_offset(item, key))
        return band, arr

    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = dict(pool.map(job, bands))
    if scl is not None:
        out["SCL"] = scl
    return np.stack([out[b].astype(np.uint16) for b in PATCH_BANDS])


def write_patch(path: Path, data: np.ndarray, grid: PatchGrid, tags: dict[str, object]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff", dtype="uint16", count=data.shape[0], width=grid.width, height=grid.height,
        crs=grid.crs, transform=grid.transform, nodata=0, compress="deflate", predictor=2,
        interleave="band",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for i, band in enumerate(PATCH_BANDS, start=1):
            dst.set_band_description(i, f"{band}: {BAND_DESCRIPTIONS[band]}")
        dst.update_tags(reflectance_scale=1 / REFLECTANCE_SCALE, band_order=",".join(PATCH_BANDS),
                        **{k: str(v) for k, v in tags.items()})
    return path.stat().st_size


@dataclass
class Timer:
    t0: float = 0.0
    b0: int = 0
    r0: int = 0

    def start(self) -> "Timer":
        self.t0 = time.perf_counter()
        self.b0, self.r0 = byte_meter().snapshot()
        return self

    def stop(self) -> tuple[float, int, int]:
        b, r = byte_meter().snapshot()
        return time.perf_counter() - self.t0, b - self.b0, r - self.r0
