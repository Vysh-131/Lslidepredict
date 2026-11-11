#!/usr/bin/env python
"""Prototype run: pick N North-East slides (2018-2024) and mine T0/T1/T2 Sentinel-2 patches.

    python scripts/mine_sample.py                 # 5 slides, defaults
    python scripts/mine_sample.py -n 10 --seed 7
    python scripts/mine_sample.py --providers cdse earth-search   # prefer CDSE (needs S3 keys)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lslidepredict import (  # noqa: E402
    MinerConfig, coverage_matrix, load_landslides, mine_slides, save_run,
    select_northeast_sample, summary_table,
)
from lslidepredict.config import LANDSLIDE_CSV  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(LANDSLIDE_CSV))
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-year", type=int, default=2018)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--max-cloud", type=float, default=15.0)
    ap.add_argument("--max-patch-cloud", type=float, default=10.0)
    ap.add_argument("--patch-px", type=int, default=256, help="patch size in 10 m pixels (256 = 2.56 km)")
    ap.add_argument("--providers", nargs="+", default=["earth-search", "cdse"],
                    help="all are searched; the first one with download access is used")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    df = load_landslides(args.csv)
    sample = select_northeast_sample(df, n=args.n, start_year=args.start_year,
                                     end_year=args.end_year, seed=args.seed)
    print("\nSelected slides:")
    print(sample[["Slide_No", "State", "District", "Latitude", "Longitude", "event_date"]].to_string(index=False))

    cfg = MinerConfig(patch_px=args.patch_px, max_cloud=args.max_cloud,
                      max_patch_cloud=args.max_patch_cloud, providers=tuple(args.providers))
    result = mine_slides(sample, cfg)
    run_dir = save_run(result, cfg)

    print("\n== Coverage (image date, scene cloud %) ==\n")
    print(coverage_matrix(result))
    print("\n== Detail ==\n")
    print(summary_table(result))
    print(f"\nManifest + summary: {run_dir}")


if __name__ == "__main__":
    main()
