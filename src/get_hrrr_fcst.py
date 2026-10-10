#!/usr/bin/env python3
"""
Operational HRRR forecast as a "member" for verification / side-by-side viewing.

Downloads the NOAA HRRR forecast from the same cycle HRRRCast was initialized from (same
AWS bucket get_ics.py uses), keeps only the fields HRRRCast predicts, and writes them in
the HRRRCast output layout:

    YYYYMMDD/HH/hrrrcast_hrrr_fHH.nc      (same variables, levels, grid and crop)

so plot.py (--members hrrr), derived_precip.py and the viewer treat it like any member.

Only the needed GRIB messages are fetched, using the .idx byte offsets (about 150 MB per
forecast hour instead of ~550 MB). Diagnostics (WIND_10M, HLCY_0_3km, ...) are computed
with the same code as the HRRRCast output, so the two are directly comparable.

The crop is copied from an existing hrrrcast_mNN_f01.nc in the case directory (so run
this after fcst.py), or given with --subset y0:y1,x0:x1, or --full for the whole CONUS grid.

APCP is the 1-h accumulation ending at fHH, as in HRRRCast.

Note: HRRR runs to f18 every hour and f48 at 00/06/12/18 UTC (f36 before mid-2018).

Usage:
  python src/get_hrrr_fcst.py 2026-10-03T16 18 --base_dir $SCRATCH/hrrrcast-data
"""

import argparse
import glob
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import xarray as xr

import utils
from utils import setup_logging

logger = logging.getLogger(__name__)

HOUR_THREADS = 3     # forecast hours downloaded at once
RANGE_THREADS = 8    # byte-range requests in flight per file
HRRR_BASE_URL = os.environ.get("HRRR_BASE_URL", "https://noaa-hrrr-bdp-pds.s3.amazonaws.com")

# Same variables / levels as make_ics.WeatherPreprocessConfig (kept here so this script does
# not need the normalization file).
PL_VARS = ["UGRD", "VGRD", "VVEL", "TMP", "HGT", "SPFH"]
PL_SHORT = {"UGRD": "u", "VGRD": "v", "VVEL": "w", "TMP": "t", "HGT": "gh", "SPFH": "q"}
LEVELS = [200, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 825, 850, 875, 900, 925,
          950, 975, 1000]
SFC_VARS = ["PRES", "MSLMA", "REFC", "T2M", "UGRD10M", "VGRD10M", "UGRD80M", "VGRD80M", "D2M",
            "TCDC", "LCDC", "MCDC", "HCDC", "VIS", "APCP", "HGTCC", "CAPE", "CIN"]
CONSTS = ["LAND", "OROG"]

# variable -> (idx "VAR:level" key, pygrib selection)
SFC_FIELDS = {
    "PRES":    ("PRES:surface",           {"shortName": "sp"}),
    "MSLMA":   ("MSLMA:mean sea level",   {"shortName": "mslma"}),
    "REFC":    ("REFC:entire atmosphere", {"shortName": "refc"}),
    "T2M":     ("TMP:2 m above ground",   {"shortName": "2t"}),
    "UGRD10M": ("UGRD:10 m above ground", {"shortName": "10u"}),
    "VGRD10M": ("VGRD:10 m above ground", {"shortName": "10v"}),
    "UGRD80M": ("UGRD:80 m above ground", {"shortName": "u", "typeOfLevel": "heightAboveGround", "level": 80}),
    "VGRD80M": ("VGRD:80 m above ground", {"shortName": "v", "typeOfLevel": "heightAboveGround", "level": 80}),
    "D2M":     ("DPT:2 m above ground",   {"shortName": "2d"}),
    "TCDC":    ("TCDC:entire atmosphere", {"shortName": "tcc", "typeOfLevel": "atmosphere"}),
    "LCDC":    ("LCDC:low cloud layer",   {"shortName": "lcc"}),
    "MCDC":    ("MCDC:middle cloud layer", {"shortName": "mcc"}),
    "HCDC":    ("HCDC:high cloud layer",  {"shortName": "hcc"}),
    "VIS":     ("VIS:surface",            {"shortName": "vis"}),
    "APCP":    ("APCP:surface",           {"shortName": "tp"}),
    "HGTCC":   ("HGT:cloud ceiling",      {"shortName": "gh", "typeOfLevel": "cloudCeiling"}),
    "CAPE":    ("CAPE:surface",           {"shortName": "cape", "typeOfLevel": "surface"}),
    "CIN":     ("CIN:surface",            {"shortName": "cin", "typeOfLevel": "surface"}),
    "LAND":    ("LAND:surface",           {"shortName": "lsm"}),
    "OROG":    ("HGT:surface",            {"shortName": "orog"}),
}


# --------------------------------------------------------------------------- download
def _session() -> requests.Session:
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=HOUR_THREADS * RANGE_THREADS + 4,
                                            max_retries=requests.adapters.Retry(
                                                total=4, backoff_factor=2,
                                                status_forcelist=(429, 500, 502, 503, 504)))
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def parse_idx(text: str) -> List[Tuple[int, str]]:
    """Return [(byte offset, 'VAR:level:ftime'), ...] from a wgrib2 .idx file."""
    out = []
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) >= 6:
            out.append((int(parts[1]), ":".join(parts[3:6])))
    return out


def wanted_ranges(entries: List[Tuple[int, str]], keys) -> List[Tuple[int, Optional[int]]]:
    """Byte ranges (start, end inclusive or None for EOF) for idx entries whose 'VAR:level'
    is in keys; adjacent messages are merged into one request."""
    ranges = []
    for i, (off, desc) in enumerate(entries):
        var_level = ":".join(desc.split(":")[:2])
        if var_level not in keys:
            continue
        end = entries[i + 1][0] - 1 if i + 1 < len(entries) else None
        if ranges and ranges[-1][1] is not None and ranges[-1][1] + 1 == off:
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((off, end))
    return ranges


def fetch_subset(url: str, keys, out_path: Path, session: requests.Session) -> bool:
    """Download only the GRIB messages matching keys from url into out_path."""
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info(f"Already have {out_path.name}")
        return True
    tmp = out_path.with_suffix(".part")
    r = session.get(url + ".idx", timeout=60)
    if r.status_code == 404:
        head = session.head(url, timeout=60)
        if head.status_code == 404:
            logger.warning(f"Not available: {url}")
            return False
        logger.warning(f"No .idx for {url}; downloading the whole file")
        ok = utils.download_file_with_retry(url, tmp)
        if ok:
            tmp.replace(out_path)
        return ok
    r.raise_for_status()
    ranges = wanted_ranges(parse_idx(r.text), keys)
    if not ranges:
        logger.error(f"None of the wanted fields are listed in {url}.idx")
        return False

    def get(rng):
        a, b = rng
        hdr = {"Range": f"bytes={a}-{'' if b is None else b}"}
        for attempt in range(4):
            try:
                resp = session.get(url, headers=hdr, timeout=300)
                resp.raise_for_status()
                return resp.content
            except requests.RequestException as e:
                if attempt == 3:
                    raise
                logger.warning(f"Range {a}-{b} of {url} failed ({e}); retrying")
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=RANGE_THREADS) as ex:
        chunks = list(ex.map(get, ranges))
    with open(tmp, "wb") as f:
        for c in chunks:
            f.write(c)
    tmp.replace(out_path)
    logger.info(f"Downloaded {out_path.name}: {len(ranges)} ranges, "
                f"{sum(len(c) for c in chunks) / 1e6:.0f} MB")
    return True


def download_hour(ymd: str, hh: str, h: int, grib_dir: Path, session) -> Optional[Tuple[Path, Path]]:
    base = f"{HRRR_BASE_URL}/hrrr.{ymd}/conus/hrrr.t{hh}z"
    sfc_keys = {v[0] for v in SFC_FIELDS.values()}
    prs_keys = {f"{v}:{lev} mb" for v in PL_VARS for lev in LEVELS}
    sfc = grib_dir / f"hrrr_t{hh}z_sfc_f{h:02d}.grib2"
    prs = grib_dir / f"hrrr_t{hh}z_prs_f{h:02d}.grib2"
    try:
        ok = (fetch_subset(f"{base}.wrfsfcf{h:02d}.grib2", sfc_keys, sfc, session)
              and fetch_subset(f"{base}.wrfprsf{h:02d}.grib2", prs_keys, prs, session))
    except Exception as e:
        logger.error(f"f{h:02d}: download failed: {e}")
        ok = False
    return (sfc, prs) if ok else None


# --------------------------------------------------------------------------- crop
def _wrap(lon):
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


REFERENCE_PATTERNS = (
    # HRRRCast NetCDF output ...
    "hrrrcast_m00_f01.nc", "hrrrcast_m*_f01.nc", "hrrrcast_m*_f[0-9]*.nc",
    # ... or its GRIB2 output, which the cleanup job keeps when it deletes the NetCDF files
    "hrrrcast.m00.t*z.pgrb2.f01", "hrrrcast.m*.t*z.pgrb2.f01", "hrrrcast.m*.t*z.pgrb2.f[0-9]*",
    "hrrrcast.avg.t*z.pgrb2.f01",
)


def find_reference(case_dir: Path) -> Optional[Path]:
    """An HRRRCast output file of this case whose grid the HRRR is cropped to."""
    for pat in REFERENCE_PATTERNS:
        hits = sorted(h for h in glob.glob(str(case_dir / pat))
                      if not h.endswith(".idx") and "_precip_" not in h)
        if hits:
            return Path(hits[0])
    return None


_REF_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def reference_latlons(ref: Path) -> Tuple[np.ndarray, np.ndarray]:
    """2-D (lat, lon) of a reference file: HRRRCast NetCDF or GRIB2 output."""
    key = str(ref)
    if key not in _REF_CACHE:
        if ref.suffix == ".nc":
            with xr.open_dataset(ref, decode_timedelta=True) as ds:
                lat, lon = ds["latitude"].values, ds["longitude"].values
        else:
            import pygrib
            g = pygrib.open(str(ref))
            try:
                lat, lon = g.message(1).latlons()
            finally:
                g.close()
        _REF_CACHE[key] = (np.asarray(lat), np.asarray(lon))
    return _REF_CACHE[key]


def crop_from_reference(ref: Path, lats: np.ndarray, lons: np.ndarray) -> Tuple[int, int, int, int]:
    """Index window of the full grid that matches the reference file's lat/lon grid."""
    rlat, rlon = reference_latlons(ref)
    rlon = _wrap(rlon)
    ny, nx = rlat.shape
    flon = _wrap(lons)
    d = (lats - rlat[0, 0]) ** 2 + ((flon - rlon[0, 0]) * np.cos(np.deg2rad(rlat[0, 0]))) ** 2
    y0, x0 = np.unravel_index(np.argmin(d), d.shape)
    y1, x1 = y0 + ny, x0 + nx
    if y1 > lats.shape[0] or x1 > lats.shape[1]:
        raise ValueError(f"Reference grid {ref.name} ({ny}x{nx}) does not fit inside the HRRR grid")
    err = max(np.abs(lats[y0:y1, x0:x1] - rlat).max(), np.abs(flon[y0:y1, x0:x1] - rlon).max())
    if err > 0.01:
        raise ValueError(f"Could not match the grid of {ref.name} (max lat/lon error {err:.3f} deg)")
    return int(y0), int(y1), int(x0), int(x1)


def crop_dataset_to_reference(ds: xr.Dataset, ref: Path) -> xr.Dataset:
    """Crop a (larger) HRRR dataset to the reference file's grid; unchanged if already equal."""
    rshape = reference_latlons(ref)[0].shape
    if ds["latitude"].shape == rshape:
        return ds
    y0, y1, x0, x1 = crop_from_reference(ref, ds["latitude"].values, ds["longitude"].values)
    ydim, xdim = ds["latitude"].dims
    return ds.isel({ydim: slice(y0, y1), xdim: slice(x0, x1)})


def recrop_existing(case_dir: Path, ref: Path, init_dt) -> int:
    """Crop hrrrcast_hrrr_*.nc files written on a bigger grid (e.g. the full CONUS grid because
    the HRRRCast output was not there yet) to the HRRRCast domain, in place. No download."""
    from cf_attributes import get_cf_encoding
    rshape = reference_latlons(ref)[0].shape
    n = 0
    for f in sorted(case_dir.glob("hrrrcast_hrrr_*f[0-9][0-9].nc")):
        with xr.open_dataset(f, decode_timedelta=False) as ds:   # keep lead_time as written
            if ds["latitude"].shape == rshape:
                continue
            try:
                out = crop_dataset_to_reference(ds, ref).load()
            except ValueError as e:
                logger.warning(f"Cannot crop {f.name} to {ref.name}: {e}")
                continue
        for v in out.variables:
            out[v].encoding = {}
        enc = get_cf_encoding(out, init_dt) if "_precip_" not in f.name else None
        tmp = str(f) + ".tmp"
        out.to_netcdf(tmp, encoding=enc, engine=utils.netcdf_engine())
        os.replace(tmp, f)
        n += 1
        logger.info(f"Cropped existing {f.name} to the HRRRCast domain {rshape[0]}x{rshape[1]}")
    return n


# --------------------------------------------------------------------------- convert
def _pick_apcp(msgs, h: int):
    """1-h accumulation ending at fH (HRRR sfc files also hold the 0-H total)."""
    for m in msgs:
        try:
            if int(m.endStep) - int(m.startStep) == 1:
                return m
        except Exception:
            pass
    if h == 1 and msgs:
        return msgs[0]
    return None


def convert_hour(init_iso: str, h: int, sfc_path: str, prs_path: str, out_path: str,
                 crop: Optional[Tuple[int, int, int, int]], ref_path: Optional[str]) -> str:
    """Read one forecast hour from the subset GRIBs and write hrrrcast_hrrr_fHH.nc."""
    import pygrib  # imported here so the download-only path works without it
    from datetime import datetime
    from diagnostics import compute_diagnostics
    from cf_attributes import apply_cf_attributes, get_cf_encoding

    init_dt = datetime.fromisoformat(init_iso)

    grbs = pygrib.open(sfc_path)
    lats, lons = grbs[1].latlons()
    if crop is None and ref_path:
        crop = crop_from_reference(Path(ref_path), lats, lons)
    y0, y1, x0, x1 = crop if crop else (0, lats.shape[0], 0, lats.shape[1])
    sl = (slice(y0, y1), slice(x0, x1))
    lats, lons = lats[sl], lons[sl]
    shape = lats.shape

    def vals(msg):
        v = msg.values[sl]
        if isinstance(v, np.ma.MaskedArray):
            v = v.filled(np.nan)
        return np.asarray(v, dtype=np.float32)

    sfc: Dict[str, np.ndarray] = {}
    for name in SFC_VARS + CONSTS:
        query = SFC_FIELDS[name][1]
        try:
            msgs = grbs.select(**query)
        except (ValueError, KeyError):
            msgs = []
        if name == "APCP":
            msg = _pick_apcp(msgs, h)
        else:
            msg = msgs[0] if msgs else None
        if msg is None:
            logger.warning(f"f{h:02d}: {name} not found in HRRR file; filling with NaN")
            sfc[name] = np.full(shape, np.nan, np.float32)
            continue
        sfc[name] = vals(msg)
    grbs.close()
    sfc["REFC"] = np.maximum(sfc["REFC"], 0)  # as make_ics / the model output
    sfc["APCP"] = np.maximum(sfc["APCP"], 0)

    grbs = pygrib.open(prs_path)
    pl: Dict[str, np.ndarray] = {}
    for var in PL_VARS:
        arr = np.full((len(LEVELS),) + shape, np.nan, np.float32)
        try:
            msgs = grbs.select(shortName=PL_SHORT[var], typeOfLevel="isobaricInhPa")
        except (ValueError, KeyError):
            msgs = []
        found = 0
        for m in msgs:
            if m.level in LEVELS:
                arr[LEVELS.index(m.level)] = vals(m)
                found += 1
        if found < len(LEVELS):
            logger.warning(f"f{h:02d}: {var} has {found}/{len(LEVELS)} levels")
        pl[var] = arr
    grbs.close()

    lead_times = [h]
    valid = [init_dt + timedelta(hours=h)]
    levels = np.asarray(LEVELS, dtype=np.int32)
    base_coords = {
        "lead_time": ("lead_time", lead_times),
        "time": ("time", valid),
        "latitude": (("latitude", "longitude"), lats),
        "longitude": (("latitude", "longitude"), lons),
    }
    data_vars = {}
    for var in PL_VARS:
        data_vars[var] = xr.DataArray(pl[var][None, None], name=var,
                                      dims=("lead_time", "time", "level", "latitude", "longitude"),
                                      coords={**base_coords, "level": ("level", levels)})
    for var in SFC_VARS + CONSTS:
        data_vars[var] = xr.DataArray(sfc[var][None, None], name=var,
                                      dims=("lead_time", "time", "latitude", "longitude"),
                                      coords=base_coords)
    ds = xr.Dataset(data_vars)
    ds = ds.assign_coords(forecast_reference_time=xr.DataArray(np.datetime64(init_dt, "ns")))

    ds = compute_diagnostics(ds)
    ds = apply_cf_attributes(ds, init_datetime=init_dt)
    ds.attrs["source"] = "NOAA operational HRRR forecast (noaa-hrrr-bdp-pds), via get_hrrr_fcst.py"
    ds.attrs["title"] = "HRRR forecast regridded to HRRRCast output layout"

    tmp = out_path + ".tmp"
    ds.to_netcdf(tmp, encoding=get_cf_encoding(ds, init_dt), engine=utils.netcdf_engine())
    os.replace(tmp, out_path)
    return out_path


# --------------------------------------------------------------------------- driver
def run(inittime: str, lead_hours: int, base_dir: str, start_hour: int = 1,
        subset: Optional[str] = None, full: bool = False, workers: int = 4,
        keep_grib: bool = False, overwrite: bool = False) -> int:
    init_dt, y, mo, d, hh = utils.validate_datetime(inittime)
    ymd = f"{y}{mo}{d}"
    case_dir = Path(base_dir) / ymd / hh
    case_dir.mkdir(parents=True, exist_ok=True)
    grib_dir = case_dir / "hrrr_fcst"
    grib_dir.mkdir(exist_ok=True)

    crop, ref = None, None
    if subset:
        a, b = subset.split(",")
        y0, y1 = (int(v) for v in a.split(":"))
        x0, x1 = (int(v) for v in b.split(":"))
        crop = (y0, y1, x0, x1)
    elif not full:
        ref = find_reference(case_dir)
        if ref is None:
            logger.warning(f"No HRRRCast output (hrrrcast_mNN_fHH.nc or hrrrcast.mNN.tHHz.pgrb2.fHH) in {case_dir} to copy the crop from; "
                           "writing the full HRRR grid. Rerun this script after fcst.py (it then "
                           "crops the existing files without downloading again) or pass --subset.")
        else:
            logger.info(f"Matching the grid of {ref.name}")
            recrop_existing(case_dir, ref, init_dt)

    hours = [h for h in range(start_hour, lead_hours + 1)
             if overwrite or not (case_dir / f"hrrrcast_hrrr_f{h:02d}.nc").exists()]
    if not hours:
        logger.info("All hrrrcast_hrrr_fHH.nc files already exist (use --overwrite to redo)")
        return 0
    logger.info(f"HRRR cycle {ymd} {hh}Z, hours {hours[0]}-{hours[-1]} -> {case_dir}")

    session = _session()
    done, failed = [], []
    with ThreadPoolExecutor(max_workers=HOUR_THREADS) as dl, ProcessPoolExecutor(max_workers=workers) as cv:
        dl_futs = {dl.submit(download_hour, ymd, hh, h, grib_dir, session): h for h in hours}
        cv_futs = {}
        for fut in as_completed(dl_futs):
            h = dl_futs[fut]
            paths = fut.result()
            if paths is None:
                failed.append(h)
                continue
            out = str(case_dir / f"hrrrcast_hrrr_f{h:02d}.nc")
            cv_futs[cv.submit(convert_hour, init_dt.isoformat(), h, str(paths[0]), str(paths[1]),
                              out, crop, str(ref) if ref else None)] = (h, paths)
        for fut in as_completed(cv_futs):
            h, paths = cv_futs[fut]
            try:
                fut.result()
                done.append(h)
                logger.info(f"Wrote hrrrcast_hrrr_f{h:02d}.nc")
                if not keep_grib:
                    for p in paths:
                        p.unlink(missing_ok=True)
            except Exception as e:
                logger.exception(f"f{h:02d}: conversion failed: {e}")
                failed.append(h)

    if not keep_grib and not any(grib_dir.iterdir()):
        grib_dir.rmdir()
    logger.info(f"HRRR forecast: {len(done)} hours written"
                + (f", failed/unavailable: {sorted(failed)}" if failed else ""))
    return 0 if done else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inittime", help="Cycle, YYYY-MM-DDTHH (same as the HRRRCast init time)")
    ap.add_argument("lead_hours", type=int, help="Last forecast hour")
    ap.add_argument("--base_dir", default="./", help="Data root (YYYYMMDD/HH/ under it); same as fcst.py --output_dir")
    ap.add_argument("--start_hour", type=int, default=1)
    ap.add_argument("--subset", default=None, help="Crop y0:y1,x0:x1 of the full grid (default: copy from hrrrcast_mNN_f01.nc)")
    ap.add_argument("--full", action="store_true", help="Write the full CONUS grid")
    ap.add_argument("--workers", type=int, default=4, help="Parallel conversion processes")
    ap.add_argument("--keep_grib", action="store_true", help="Keep the downloaded GRIB subsets (YYYYMMDD/HH/hrrr_fcst/)")
    ap.add_argument("--overwrite", action="store_true", help="Redo hours that already have hrrrcast_hrrr_fHH.nc")
    ap.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()
    global logger
    logger = setup_logging(args.log_level)
    sys.exit(run(args.inittime, args.lead_hours, args.base_dir, args.start_hour, args.subset,
                 args.full, args.workers, args.keep_grib, args.overwrite))


if __name__ == "__main__":
    main()
