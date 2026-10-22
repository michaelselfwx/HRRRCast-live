#!/usr/bin/env python3
"""
Write (or repair) hrrrcast.<mem>.tHHz.pgrb2.fHH files from the per-hour NetCDF output, using
the same GRIB2 writer and grid definition as fcst.py uses for the members.

Meant for cases run before compute_pmm.py / get_hrrr_fcst.py wrote correct GRIB2 for the
ensemble products and the HRRR: earlier avg/spr GRIB2 files described the full 1799x1059
CONUS grid instead of the cropped one (and a wrong forecast hour), so they are rewritten when
their grid does not match the NetCDF. Correct files are left alone (--overwrite redoes them).

Usage:
    python src/nc_to_grib2.py 2025-07-03T18 48 --base_dir $DATAROOT              # avg spr hrrr lpmm
    python src/nc_to_grib2.py 2025-07-03T18 48 --base_dir $DATAROOT --members avg spr
Needs the NetCDF files (hrrrcast_<mem>_fHH.nc); the cleanup job deletes those, so for a
cleaned-up case rerun compute_pmm.py from the member NetCDF files (if still there) instead.
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import xarray as xr

import utils
from get_hrrr_fcst import nc_to_grib

logger = logging.getLogger("nc_to_grib2")


def grib_ok(grib_path: str, nc_path: str) -> bool:
    """True if grib_path exists and its grid size matches the NetCDF grid."""
    if not os.path.exists(grib_path) or os.path.getsize(grib_path) == 0:
        return False
    try:
        import grib2io
        with xr.open_dataset(nc_path, decode_timedelta=False) as ds:
            ny, nx = ds["latitude"].shape[-2:]
        g = grib2io.open(grib_path)
        try:
            if len(g) == 0:
                return False
            m = g[0]
            return int(m.nx) == nx and int(m.ny) == ny
        finally:
            g.close()
    except Exception as e:
        logger.debug(f"{os.path.basename(grib_path)}: cannot check ({e}); rewriting")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inittime", help="YYYY-MM-DDTHH")
    ap.add_argument("lead_hours", type=int, help="last forecast hour")
    ap.add_argument("--base_dir", default="./", help="data root (YYYYMMDD/HH/ under it)")
    ap.add_argument("--members", nargs="+", default=["avg", "spr", "hrrr", "lpmm"])
    ap.add_argument("--start_hour", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true", help="rewrite even correct GRIB2 files")
    ap.add_argument("--log_level", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")

    init_dt, y, mo, d, hh = utils.validate_datetime(a.inittime)
    case_dir = os.path.join(a.base_dir, f"{y}{mo}{d}", hh)
    jobs = []
    for mem in a.members:
        for h in range(a.start_hour, a.lead_hours + 1):
            nc = os.path.join(case_dir, f"hrrrcast_{mem}_f{h:02d}.nc")
            if not os.path.exists(nc):
                continue
            gb = os.path.join(case_dir, f"hrrrcast.{mem}.t{hh}z.pgrb2.f{h:02d}")
            if a.overwrite or not grib_ok(gb, nc):
                jobs.append((nc, gb, init_dt.isoformat(), h))
    if not jobs:
        logger.info(f"{case_dir}: nothing to do (no NetCDF without a matching GRIB2)")
        return
    logger.info(f"{case_dir}: writing {len(jobs)} GRIB2 files")
    ok = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(nc_to_grib, *j): j for j in jobs}
        for f in as_completed(futs):
            name = os.path.basename(futs[f][1])
            try:
                if f.result():
                    ok += 1
                    logger.info(f"Wrote {name}")
            except Exception as e:
                logger.error(f"{name}: {e}")
    logger.info(f"Done: {ok}/{len(jobs)} written")
    sys.exit(0 if ok == len(jobs) else 1)


if __name__ == "__main__":
    main()
