"""Central configuration for the Sentinel-2 landslide data miner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LANDSLIDE_CSV = DATA_DIR / "tier1_landslide_data.csv"
PATCH_DIR = DATA_DIR / "patches"
RUNS_DIR = DATA_DIR / "runs"

# Canonical band order written into every patch GeoTIFF.
# B11 (SWIR1, 20 m) is resampled to the 10 m grid so the Bare Soil Index can be computed.
# SCL (scene classification, 20 m) is nearest-resampled and kept as the last band for masking.
PATCH_BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B08", "B11", "SCL")
REFLECTANCE_BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B08", "B11")
BAND_DESCRIPTIONS = {
    "B02": "Blue (490 nm, 10 m)",
    "B03": "Green (560 nm, 10 m)",
    "B04": "Red (665 nm, 10 m)",
    "B08": "NIR (842 nm, 10 m)",
    "B11": "SWIR1 (1610 nm, 20 m -> 10 m bilinear)",
    "SCL": "Scene classification (20 m -> 10 m nearest)",
}

# Reflectance values in patches are stored as uint16 = round(reflectance * 10000),
# with the processing-baseline >= 04.00 BOA offset already removed (harmonised across years).
REFLECTANCE_SCALE = 10_000

# SCL classes treated as "not clear" for patch-level cloud fraction.
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8 cloud medium, 9 cloud high, 10 cirrus
SCL_CLOUDY = (0, 1, 3, 8, 9, 10)

# Earth Search serves public Cloud-Optimised GeoTIFFs (no auth, HTTP range reads).
# CDSE serves JP2 on s3://eodata (needs free CDSE S3 keys; see README).
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff,.jp2",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "VSI_CACHE": "TRUE",
}


@dataclass
class MinerConfig:
    """Knobs for one mining run."""

    patch_px: int = 256               # patch is patch_px x patch_px at `resolution` (256 -> 2.56 km)
    max_cloud: float = 15.0           # scene-level eo:cloud_cover filter (%)
    max_patch_cloud: float = 10.0     # patch-level SCL cloud/shadow fraction to accept (%)
    max_candidates: int = 4           # how many scenes to try per window before giving up on patch cloud
    resolution: float = 10.0          # output grid (m)
    providers: tuple[str, ...] = ("earth-search", "cdse")  # all are searched; first downloadable wins
    out_dir: Path = field(default_factory=lambda: PATCH_DIR)
    runs_dir: Path = field(default_factory=lambda: RUNS_DIR)
    workers: int = 6                  # parallel band reads per patch
