"""End-to-end mining: slide row -> 3 seasonal windows -> STAC search -> cropped patches."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tabulate import tabulate

from .catalog import slide_key
from .config import MinerConfig
from .indices import patch_stats
from .patches import PatchGrid, Timer, byte_meter, patch_cloud_pct, read_patch, read_scl, write_patch
from .stac import Provider, cloud_cover, get_providers, item_summary, rank_candidates
from .windows import MIN_WINDOW_DAYS, Window, seasonal_windows

log = logging.getLogger(__name__)
MB = 1024 * 1024


def _add_note(rec: dict[str, Any], msg: str) -> None:
    rec["note"] = "; ".join(filter(None, [rec.get("note"), msg]))


Pick = tuple[float, Any, np.ndarray, tuple[float, int, int], Provider]


def _search_window(row: pd.Series, window: Window, grid: PatchGrid, providers: list[Provider],
                   cfg: MinerConfig, rec: dict[str, Any]) -> Pick | None:
    """Search all catalogues for one window and return the best (patch_cloud, item, scl, timing, provider)."""
    found: dict[str, list] = {}
    for p in providers:  # search every catalogue (for the coverage report)
        try:
            items, secs = p.search(grid.bbox_lonlat, window, cfg.max_cloud)
            ranked = rank_candidates(items, grid.bbox_lonlat)
            found[p.name] = ranked
            rec[f"n_{p.name}"] = len(ranked)
            rec[f"best_cloud_{p.name}"] = round(cloud_cover(ranked[0]), 2) if ranked else None
            rec[f"search_s_{p.name}"] = round(secs, 2)
        except Exception as exc:  # one catalogue being down must not kill the run
            log.warning("%s search failed for %s %s: %s", p.name, row["Slide_No"], window.name, exc)
            found[p.name] = []
            rec[f"n_{p.name}"] = None

    # download from the first provider we have access to
    downloader = next((p for p in providers if p.can_download() and found.get(p.name)), None)
    if downloader is None:
        _add_note(rec, f"{window.start}..{window.end}: no scene < {cfg.max_cloud:.0f}% cloud")
        return None

    tried: list[Pick] = []
    for item in found[downloader.name][: cfg.max_candidates]:
        t = Timer().start()
        try:
            scl = read_scl(downloader, item, grid)
        except Exception as exc:
            log.warning("SCL read failed for %s: %s", item.id, exc)
            continue
        tried.append((patch_cloud_pct(scl), item, scl, t.stop(), downloader))
        if tried[-1][0] <= cfg.max_patch_cloud:
            return tried[-1]
    if not tried:
        _add_note(rec, "all candidate reads failed")
        return None
    best = min(tried, key=lambda x: x[0])
    _add_note(rec, f"{window.start}..{window.end}: best patch still {best[0]:.1f}% cloudy/shadow")
    return best


def _mine_window(row: pd.Series, window: Window, grid: PatchGrid, providers: list[Provider],
                 cfg: MinerConfig) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "slide": row["Slide_No"], "state": row["State"], "district": row["District"],
        "lat": row["Latitude"], "lon": row["Longitude"],
        "event_date": row["event_date"].date().isoformat(),
        "window": window.name, "role": window.role,
        "found": False,
    }

    def describe(w: Window, used: str) -> None:
        rec.update({
            "window_used": used,
            "window_start": w.start.isoformat(), "window_end": w.end.isoformat(),
            "window_days": w.days, "window_clamped": w.clamped,
            "window_contains_event": w.contains(row["event_date"].date()),  # always False
        })
        if w.note:
            _add_note(rec, w.note)

    describe(window, "primary")
    if not window.valid:
        return rec

    pick = _search_window(row, window, grid, providers, cfg, rec)
    if window.fallback is not None and (pick is None or pick[0] > cfg.max_patch_cloud):
        log.info("   %s: no clear scene in %s..%s, trying %s..%s", window.name, window.start,
                 window.end, window.fallback.start, window.fallback.end)
        primary_rec = dict(rec)
        describe(window.fallback, "fallback")
        alt = _search_window(row, window.fallback, grid, providers, cfg, rec)
        if alt is not None and (pick is None or alt[0] < pick[0]):
            pick = alt
        else:  # primary was at least as good: restore its record (search counts, notes, dates)
            rec.clear()
            rec.update(primary_rec)
            _add_note(rec, "fallback window searched; primary kept")
    if pick is None:
        return rec

    pc, item, scl, (scl_s, scl_b, scl_r), downloader = pick
    rec["provider"] = downloader.name
    t = Timer().start()
    data = read_patch(downloader, item, grid, workers=cfg.workers, scl=scl)
    secs, nbytes, nreq = t.stop()
    secs, nbytes, nreq = secs + scl_s, nbytes + scl_b, nreq + scl_r

    # guard against double-applied BOA offsets: dark-clipped pixels mean bad harmonisation
    clipped = float((data[:5] == 1).mean() * 100)
    if clipped > 5:
        log.warning("%s: %.0f%% of reflectance pixels clipped to 0 - check scale/offset", item.id, clipped)
        _add_note(rec, f"{clipped:.0f}% reflectance clipped (offset?)")

    out = cfg.out_dir / slide_key(row["Slide_No"]) / f"{window.name}_{item.datetime:%Y%m%d}_{item.id}.tif"
    meta = item_summary(item)
    size = write_patch(out, data, grid, {
        "slide": row["Slide_No"], "window": window.name, "role": window.role,
        "event_date": rec["event_date"], "provider": downloader.name, **meta,
        "patch_cloud_pct": round(pc, 2),
    })

    rec.update(meta)
    rec.update({
        "found": True,
        "patch_cloud_pct": round(pc, 2),
        "fetched_mb": round(nbytes / MB, 2),
        "http_requests": nreq,
        "download_s": round(secs, 2),
        "mb_per_s": round(nbytes / MB / secs, 2) if secs > 0 else None,
        "file_kb": round(size / 1024, 1),
        "shape": f"{data.shape[0]}x{data.shape[1]}x{data.shape[2]}",
        "path": str(out),
        **patch_stats(out),
    })
    return rec


def mine_slides(sample: pd.DataFrame, cfg: MinerConfig | None = None) -> pd.DataFrame:
    cfg = cfg or MinerConfig()
    byte_meter()  # install before any GDAL I/O
    providers = get_providers(cfg.providers)
    for p in providers:
        if not p.can_download():
            log.info("%s: search only (no download credentials)", p.name)

    records = []
    for _, row in sample.iterrows():
        grid = PatchGrid.around(row["Latitude"], row["Longitude"], cfg.patch_px, cfg.resolution)
        for window in seasonal_windows(row["event_date"]):
            log.info("%s  %s %s..%s", row["Slide_No"], window.name, window.start, window.end)
            rec = _mine_window(row, window, grid, providers, cfg)
            records.append(rec)
            if rec["found"]:
                log.info("   -> %s cloud %.1f%% (patch %.1f%%), %.2f MB in %.1fs",
                         rec["date"], rec["scene_cloud_pct"], rec["patch_cloud_pct"],
                         rec["fetched_mb"], rec["download_s"])
            else:
                log.info("   -> not found: %s", rec.get("note"))
    return pd.DataFrame.from_records(records)


# --- reporting --------------------------------------------------------------------------

def summary_table(df: pd.DataFrame) -> str:
    cols = {
        "slide": "Slide", "state": "State", "event_date": "Event", "window": "Win",
        "found": "Found", "date": "Image date", "scene_cloud_pct": "Scene cld%",
        "patch_cloud_pct": "Patch cld%", "n_earth-search": "#ES", "n_cdse": "#CDSE",
        "fetched_mb": "Fetched MB", "download_s": "Secs", "mb_per_s": "MB/s", "file_kb": "File KB",
        "mean_bsi": "BSI", "mean_ndvi": "NDVI", "note": "Note",
    }
    view = df[[c for c in cols if c in df.columns]].rename(columns=cols).copy()
    view["Found"] = view["Found"].map({True: "yes", False: "NO"})
    for c in ("#ES", "#CDSE"):
        if c in view:
            view[c] = view[c].astype("Int64")
    view = view.astype(object).where(view.notna(), None)  # blank cells instead of "nan"
    return tabulate(view, headers="keys", tablefmt="github", showindex=False, floatfmt=".2f", missingval="-")


def coverage_matrix(df: pd.DataFrame) -> str:
    def cell(r: pd.Series) -> str:
        if r["found"]:
            return f"{r['date']} ({r['scene_cloud_pct']:.1f}%)"
        return "no valid window" if r["window_days"] < MIN_WINDOW_DAYS else "missing"

    m = df.assign(cell=df.apply(cell, axis=1)).pivot_table(
        index=["slide", "state", "event_date"], columns="window", values="cell", aggfunc="first")
    return tabulate(m.reset_index(), headers="keys", tablefmt="github", showindex=False)


def save_run(df: pd.DataFrame, cfg: MinerConfig) -> Path:
    run_dir = cfg.runs_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "manifest.csv", index=False)
    totals = df[df["found"]]
    tot_mb = totals["fetched_mb"].sum() if "fetched_mb" in totals else 0.0
    tot_s = totals["download_s"].sum() if "download_s" in totals else 0.0
    (run_dir / "summary.md").write_text(
        "# Mining run summary\n\n## Coverage\n\n" + coverage_matrix(df)
        + "\n\n## Detail\n\n" + summary_table(df)
        + f"\n\nPatches found: {int(df['found'].sum())}/{len(df)}. "
        + f"Total fetched {tot_mb:.1f} MB in {tot_s:.1f} s.\n"
    )
    return run_dir
