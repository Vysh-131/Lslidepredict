"""STAC providers for Sentinel-2 L2A and scene selection.

Two catalogues are supported behind one interface:

* ``earth-search`` – Element84 Earth Search v1. Public Cloud-Optimised GeoTIFFs on AWS,
  so a 2 km patch is a handful of HTTP range requests. No credentials. Default download source.
* ``cdse`` – Copernicus Data Space Ecosystem STAC. Authoritative ESA catalogue
  (incl. the 2015-2021 N0500 reprocessing). Assets are JP2 on ``s3://eodata`` and need
  free CDSE S3 keys (env ``CDSE_S3_ACCESS_KEY`` / ``CDSE_S3_SECRET_KEY``).

Both are always searched so every run reports catalogue coverage side by side.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pystac import Item
from pystac_client import Client
from shapely.geometry import box, shape

from .windows import Window


@dataclass
class Provider:
    name: str
    url: str
    collection: str
    band_assets: dict[str, str]          # canonical band -> asset key
    _client: Client | None = field(default=None, repr=False)

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client.open(self.url)
        return self._client

    # --- access -------------------------------------------------------------------
    def can_download(self) -> bool:
        return True

    def gdal_path(self, href: str) -> str:
        return href

    def gdal_env(self) -> dict[str, str]:
        return {}

    # --- search -------------------------------------------------------------------
    def search(self, bbox: tuple[float, float, float, float], window: Window, max_cloud: float,
               limit: int = 200) -> tuple[list[Item], float]:
        t = time.perf_counter()
        search = self.client.search(
            collections=[self.collection],
            bbox=bbox,
            datetime=window.stac_datetime,
            query={"eo:cloud_cover": {"lt": max_cloud}},
            max_items=limit,
        )
        items = [it for it in search.items() if cloud_cover(it) is not None and cloud_cover(it) < max_cloud]
        return items, time.perf_counter() - t


class EarthSearch(Provider):
    def __init__(self) -> None:
        super().__init__(
            name="earth-search",
            url="https://earth-search.aws.element84.com/v1",
            collection="sentinel-2-l2a",
            band_assets={"B02": "blue", "B03": "green", "B04": "red", "B08": "nir",
                         "B11": "swir16", "SCL": "scl"},
        )

    def gdal_env(self) -> dict[str, str]:
        return {"AWS_NO_SIGN_REQUEST": "YES"}


class CDSE(Provider):
    def __init__(self) -> None:
        super().__init__(
            name="cdse",
            url="https://stac.dataspace.copernicus.eu/v1/",
            collection="sentinel-2-l2a",
            band_assets={"B02": "B02_10m", "B03": "B03_10m", "B04": "B04_10m", "B08": "B08_10m",
                         "B11": "B11_20m", "SCL": "SCL_20m"},
        )

    def can_download(self) -> bool:
        return bool(os.getenv("CDSE_S3_ACCESS_KEY") and os.getenv("CDSE_S3_SECRET_KEY"))

    def gdal_path(self, href: str) -> str:
        return href.replace("s3://", "/vsis3/", 1) if href.startswith("s3://") else href

    def gdal_env(self) -> dict[str, str]:
        return {
            "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu",
            "AWS_ACCESS_KEY_ID": os.getenv("CDSE_S3_ACCESS_KEY", ""),
            "AWS_SECRET_ACCESS_KEY": os.getenv("CDSE_S3_SECRET_KEY", ""),
            "AWS_HTTPS": "YES",
            "AWS_VIRTUAL_HOSTING": "FALSE",
        }


PROVIDERS: dict[str, type[Provider]] = {"earth-search": EarthSearch, "cdse": CDSE}


def get_providers(names: tuple[str, ...] | list[str]) -> list[Provider]:
    return [PROVIDERS[n]() for n in names]


# --- item helpers -------------------------------------------------------------------

def cloud_cover(item: Item) -> float | None:
    cc = item.properties.get("eo:cloud_cover")
    return float(cc) if cc is not None else None


def mgrs_tile(item: Item) -> str:
    code = item.properties.get("grid:code") or item.properties.get("s2:mgrs_tile")
    if code:
        return str(code).replace("MGRS-", "")
    m = re.search(r"_T?(\d{2}[A-Z]{3})_", item.id)
    return m.group(1) if m else "?"


def processing_baseline(item: Item) -> float:
    pb = item.properties.get("s2:processing_baseline")
    if pb is None:
        m = re.search(r"_N(\d{4})_", item.id)
        pb = f"{m.group(1)[:2]}.{m.group(1)[2:]}" if m else "0"
    try:
        return float(pb)
    except ValueError:
        return 0.0


def epsg_of(item: Item) -> int | None:
    code = item.properties.get("proj:epsg") or item.properties.get("proj:code")
    if code is None:
        return None
    return int(str(code).split(":")[-1])


def scale_offset(item: Item, asset_key: str) -> tuple[float, float]:
    """DN -> reflectance: refl = DN * scale + offset.

    Uses STAC ``raster:bands`` when present; otherwise falls back to the ESA rule that
    processing baseline >= 04.00 products carry a -1000 DN BOA offset.
    """
    # Earth Search has already subtracted the 1000 DN offset from the pixels when this flag
    # is True, but its raster:bands still advertises offset=-0.1. Applying it again would
    # zero out the visible bands, so the flag wins.
    if item.properties.get("earthsearch:boa_offset_applied") is True:
        return 1e-4, 0.0
    asset = item.assets[asset_key]
    rb = (asset.extra_fields.get("raster:bands") or [{}])[0]
    if "scale" in rb or "offset" in rb:
        return float(rb.get("scale", 1e-4)), float(rb.get("offset", 0.0))
    return 1e-4, (-0.1 if processing_baseline(item) >= 4.0 else 0.0)


def covers_bbox(item: Item, bbox: tuple[float, float, float, float]) -> bool:
    try:
        return shape(item.geometry).contains(box(*bbox))
    except Exception:
        return True


def rank_candidates(items: list[Item], bbox: tuple[float, float, float, float]) -> list[Item]:
    """Drop partial-footprint tiles and duplicate reprocessings, order by scene cloud cover."""
    full = [it for it in items if covers_bbox(it, bbox)] or items
    best: dict[tuple[str, str], Item] = {}
    for it in full:
        key = (it.datetime.date().isoformat() if it.datetime else it.id, mgrs_tile(it))
        cur = best.get(key)
        if cur is None or (processing_baseline(it), it.id) > (processing_baseline(cur), cur.id):
            best[key] = it
    return sorted(best.values(), key=lambda it: (cloud_cover(it) or 100.0, it.datetime))


def item_summary(item: Item | None) -> dict[str, Any]:
    if item is None:
        return {}
    return {
        "item_id": item.id,
        "date": item.datetime.date().isoformat() if item.datetime else None,
        "scene_cloud_pct": round(cloud_cover(item) or float("nan"), 2),
        "mgrs_tile": mgrs_tile(item),
        "baseline": processing_baseline(item),
    }
