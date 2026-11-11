# Land slide predictor
ML pipeline to map landslide prone risk area leveraging Bi Temporal Transformer method

## Sentinel-2 data miner (prototype)

Pulls co-registered, 10 m, 256 × 256 px (2.56 km) Sentinel-2 L2A patches centred on each landslide for three dry-season windows. Y is the calendar year of the event.

| Window | Role | Event Mar 1 – Dec 31 | Event Jan 1 – Feb 29 (split winter) |
|---|---|---|---|
| T0 | erosion baseline | Nov 1 (Y−2) → Feb 15 (Y−1) | Nov 1 (Y−2) → Feb 15 (Y−1) |
| T1 | pre-slide | Nov 1 (Y−1) → Feb 15 (Y) | Nov 1 (Y−1) → event − 1 day |
| T2 | post-slide | Nov 1 (Y) → Feb 15 (Y+1) | event + 1 day → Mar 15 (Y); falls back to Nov 1 (Y) → Feb 15 (Y+1) if that has no clear scene |

**Causal clamp.** A window can never contain the event: pre-event windows end the day before it, and T2 starts the day after it (this matters for Nov–Dec events). The window actually searched is recorded in `window_used`, `window_start` and `window_end`.

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python scripts/mine_sample.py            # 5 NE slides, 2018-2024
venv/bin/python scripts/mine_sample.py -n 20 --seed 7
```

**Sources.** Both STAC catalogues are searched for every window (`eo:cloud_cover < 15`):
- Element84 Earth Search (`earth-search.aws.element84.com/v1`): public COGs, only the needed byte ranges are fetched, no auth. This is the default download source.
- Copernicus Data Space (`stac.dataspace.copernicus.eu/v1`): the official ESA catalogue. Its assets are JP2 files on `s3://eodata`, so downloads need free CDSE S3 keys (`CDSE_S3_ACCESS_KEY`, `CDSE_S3_SECRET_KEY`); then pass `--providers cdse earth-search`.

**Patch format** (`data/patches/<slide>/<T>_<date>_<item>.tif`): 6-band uint16 GeoTIFF, `B02 B03 B04 B08 B11 SCL`.
Reflectance = value / 10000 (harmonised across processing baselines). B11 is bilinearly resampled from 20 m and SCL is nearest-resampled.
All windows for a slide share one UTM grid, so T0/T1/T2 stack pixel-for-pixel.

**Scene choice.** Partial-footprint tiles and duplicate reprocessings are dropped, then scenes are ranked by scene cloud cover.
The SCL band of the patch itself is then checked, and the first scene with ≤10 % cloud/shadow inside the patch is kept.

**Outputs per run** go to `data/runs/<timestamp>/`: `manifest.csv` (one row per slide × window, with dates, cloud, MB fetched, seconds, file size, mean BSI/NDVI) and `summary.md`.

Layout: `src/lslidepredict/` → `catalog` (CSV + sampling), `windows`, `stac` (providers + ranking), `patches` (grid, COG cropping, byte meter), `indices` (NDVI, BSI, NDWI), `pipeline` (orchestration + reports).
