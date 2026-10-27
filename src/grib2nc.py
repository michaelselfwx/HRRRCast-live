#!/usr/bin/env python3
"""
Rebuild HRRRCast hourly NetCDF files from the GRIB2 files the cleanup job keeps.

The cleanup job deletes every hrrrcast_*.nc file but keeps hrrrcast.*.pgrb2.fHH. This script
reads those GRIB2 files back (the reverse of nc2grib.py, using the same parameter table in
cf_attributes.VARIABLE_METADATA) and writes

    YYYYMMDD/HH/hrrrcast.mNN.tHHz.pgrb2.fHH  ->  YYYYMMDD/HH/hrrrcast_mNN_fHH.nc
    YYYYMMDD/HH/hrrrcast.avg.tHHz.pgrb2.fHH  ->  YYYYMMDD/HH/hrrrcast_avg_fHH.nc
    YYYYMMDD/HH/hrrrcast.spr.tHHz.pgrb2.fHH  ->  YYYYMMDD/HH/hrrrcast_spr_fHH.nc

with the same layout fcst.py writes (dims lead_time, time, [level], latitude, longitude), so
derived_precip.py, compute_pmm.py and plot.py run on an old case as if it had just finished.
Values match the originals to within GRIB2 packing precision (e.g. 0.01 dBZ for REFC,
1e-4 mm for APCP). Existing NetCDF files are left alone unless --overwrite.

Usage:
  python src/grib2nc.py 2026-10-03T16 --forecast_dir $SCRATCH/hrrrcast-data
  python src/grib2nc.py 2026-10-03T16 --forecast_dir ... --runs avg spr --hours 1-18
"""

import argparse
import glob
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils  # noqa: E402
from cf_attributes import VARIABLE_METADATA, apply_cf_attributes, get_cf_encoding  # noqa: E402

logger = logging.getLogger("grib2nc")

MISSING = 255  # GRIB2 "no second fixed surface"


def _key(disc, cat, num, t1, v1, t2, v2):
    """Lookup key for one GRIB message; mirrors Netcdf2Grib._build_message."""
    return (int(disc), int(cat), int(num), int(t1), int(v1), None if int(t2) == MISSING else int(v2))


def section4_ids(raw: bytes):
    """(discipline, category, number, type1, value1, type2, value2) read straight from the
    message bytes. pygrib hands back some code-table keys as abbreviations ('sfc' for both
    surface (1) and height above ground (103)), so the raw octets are the reliable source.
    Octets 10-34 are laid out the same in product templates 4.0/4.1/4.8/4.11."""
    if raw[:4] != b"GRIB" or raw[7] != 2:
        raise ValueError("not a GRIB2 message")
    disc = raw[6]
    pos = 16
    while pos < len(raw) - 4:
        length = int.from_bytes(raw[pos:pos + 4], "big")
        if raw[pos + 4] == 4:
            s4 = raw[pos:pos + length]
            o = lambda a, b: int.from_bytes(s4[a - 1:b], "big")   # 1-based octets a..b
            return disc, s4[9], s4[10], s4[22], o(25, 28), s4[28], o(31, 34)
        if length <= 0:
            break
        pos += length
    raise ValueError("no section 4")


def build_lookup():
    """(single-level/layer key -> var) and ((disc, cat, num) -> pressure-level var)."""
    sfc, pl = {}, {}
    for var, meta in VARIABLE_METADATA.items():
        g = meta.get("grib2")
        if not g:
            continue
        disc, cat, num, stype, sval = g
        if stype == 100 and sval is None:            # pressure-level variable (level in Pa)
            pl[(disc, cat, num)] = var
        elif isinstance(sval, tuple):                # layer: (first, second), same surface type
            sfc[(disc, cat, num, stype, int(sval[0]), int(sval[1]))] = var
        else:                                        # single level (value 0 when None)
            sfc[(disc, cat, num, stype, int(sval or 0), None)] = var
    # nc2grib.py sets msg.discipline but grib2io writes discipline 0 into section 0, so LAND
    # (discipline 2 "land surface", 0/0) comes out of the files as 0/0/0 at the surface,
    # i.e. "surface temperature" (values are the 0-1 land fraction). No real variable uses that
    # code, so read it back as LAND.
    sfc.setdefault((0, 0, 0, 1, 0, None), "LAND")
    return sfc, pl


def _wrap(lon):
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


def read_grib(path):
    """Return (lats, lons, {sfc var: 2-D}, {pl var: {level hPa: 2-D}}, skipped names)."""
    import pygrib
    sfc_map, pl_map = build_lookup()
    sfc, pl, skipped = {}, {}, []
    g = pygrib.open(path)
    try:
        lats = lons = None
        for msg in g:
            if lats is None:
                lats, lons = msg.latlons()
            try:
                k = _key(*section4_ids(msg.tostring()))
            except Exception as e:
                skipped.append(f"msg{msg.messagenumber}({e})")
                continue
            vals = np.ma.filled(np.ma.asarray(msg.values, dtype=np.float32), np.nan)
            if k[3] == 100 and k[5] is None and k[:3] in pl_map:
                pl.setdefault(pl_map[k[:3]], {})[int(round(k[4] / 100.0))] = vals
            elif k in sfc_map:
                sfc[sfc_map[k]] = vals
            else:
                skipped.append(str(k))
    finally:
        g.close()
    return np.asarray(lats), _wrap(lons), sfc, pl, skipped


def to_dataset(init_dt, hour, lats, lons, sfc, pl):
    """Same structure as fcst.py create_xarray_dataset for one lead hour."""
    lead_times = [hour]
    valid = [init_dt + timedelta(hours=hour)]
    base = {
        "lead_time": ("lead_time", lead_times),
        "time": ("time", valid),
        "latitude": (("latitude", "longitude"), lats),
        "longitude": (("latitude", "longitude"), lons),
    }
    data_vars = {}
    for var, by_level in pl.items():
        levels = np.asarray(sorted(by_level), dtype=np.int32)
        arr = np.stack([by_level[int(lv)] for lv in levels])[None, None]
        data_vars[var] = xr.DataArray(arr, dims=("lead_time", "time", "level", "latitude", "longitude"),
                                      coords={**base, "level": ("level", levels)}, name=var)
    for var, vals in sfc.items():
        data_vars[var] = xr.DataArray(vals[None, None], dims=("lead_time", "time", "latitude", "longitude"),
                                      coords=base, name=var)
    ds = xr.Dataset(data_vars)
    return ds.assign_coords(forecast_reference_time=xr.DataArray(np.datetime64(init_dt, "ns")))


def convert_one(grib_path, nc_path, init_dt, hour):
    lats, lons, sfc, pl, skipped = read_grib(grib_path)
    if not sfc and not pl:
        raise ValueError(f"no recognised fields in {os.path.basename(grib_path)}")
    ds = to_dataset(init_dt, hour, lats, lons, sfc, pl)
    try:
        ds = apply_cf_attributes(ds, init_datetime=init_dt)
    except Exception as e:
        logger.debug(f"apply_cf_attributes skipped: {e}")
    tmp = nc_path + ".tmp"
    ds.to_netcdf(tmp, encoding=get_cf_encoding(ds, init_dt), engine=utils.netcdf_engine())
    os.replace(tmp, nc_path)
    return f"{os.path.basename(nc_path)}: {len(sfc)} sfc + {len(pl)} pl vars" + \
        (f" ({len(skipped)} unknown msgs skipped: {', '.join(skipped[:5])})" if skipped else "")


def parse_hours(spec, available):
    if not spec:
        return available
    want = set()
    for part in spec.replace(",", " ").split():
        a, _, b = part.partition("-")
        want.update(range(int(a), int(b or a) + 1))
    return [h for h in available if h in want]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inittime", help="YYYY-MM-DDTHH")
    ap.add_argument("--forecast_dir", default="./")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="which runs to rebuild, e.g. m00 m01 avg spr (default: all members + avg + spr found)")
    ap.add_argument("--hours", default=None, help="e.g. 1-18 (default: every hour found; f00 included)")
    ap.add_argument("--overwrite", action="store_true", help="rewrite NetCDF files that already exist")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    init_dt, y, m, d, hh = utils.validate_datetime(a.inittime)
    case_dir = os.path.join(a.forecast_dir, f"{y}{m}{d}", hh)
    pat = re.compile(rf"^hrrrcast\.(m\d+|avg|spr)\.t{hh}z\.pgrb2\.f(\d+)$")
    found = {}
    for f in glob.glob(os.path.join(case_dir, f"hrrrcast.*.t{hh}z.pgrb2.f*")):
        mt = pat.match(os.path.basename(f))
        if mt:
            found.setdefault(mt.group(1), {})[int(mt.group(2))] = f
    runs = a.runs or sorted(found, key=lambda r: (r[0] != "m", r))
    jobs = []
    for run in runs:
        if run not in found:
            logger.warning(f"No GRIB2 files for '{run}' in {case_dir}")
            continue
        for h in parse_hours(a.hours, sorted(found[run])):
            nc = os.path.join(case_dir, f"hrrrcast_{run}_f{h:02d}.nc")
            if os.path.exists(nc) and not a.overwrite:
                continue
            jobs.append((found[run][h], nc, init_dt, h))
    if not jobs:
        logger.info(f"Nothing to rebuild in {case_dir}")
        return 0
    logger.info(f"Rebuilding {len(jobs)} NetCDF files in {case_dir} from GRIB2")
    failed = 0
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(convert_one, *j): j for j in jobs}
        for fut in as_completed(futs):
            try:
                logger.info(fut.result())
            except Exception as e:
                failed += 1
                logger.error(f"{os.path.basename(futs[fut][0])}: {e}")
    logger.info(f"Done: {len(jobs) - failed} written, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
