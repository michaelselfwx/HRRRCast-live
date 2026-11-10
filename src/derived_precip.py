#!/usr/bin/env python3
"""
Derived precipitation products for HRRRCast forecasts.

HRRRCast writes APCP as the 1-hour accumulation ending at each forecast hour. From the
per-member files (YYYYMMDD/HH/hrrrcast_mNN_fHH.nc) this script writes, for every hour H:

  per member   hrrrcast_mNN_precip_fHH.nc
      APCP_TOT   total precipitation f00 -> fH
      APCP_6H    6-h total ending at fH   (H >= 6)
      APCP_12H   12-h total ending at fH  (H >= 12)
  HRRR         hrrrcast_hrrr_precip_fHH.nc  same, for the operational HRRR (get_hrrr_fcst.py), if present
  ensemble     hrrrcast_avg_precip_fHH.nc   domain-wide PMM of APCP_TOT / APCP_6H / APCP_12H
               hrrrcast_lpmm_fHH.nc         local PMM (LPMM) of APCP, APCP_TOT, APCP_6H, APCP_12H

PMM ("probability-matched mean", method 2 as in compute_pmm.py): take the ensemble-mean
pattern, then replace its values, rank for rank, with values drawn from the pooled member
distribution. LPMM does the same within local windows: the domain is cut into patches of
`--lpmm_patch` points, and each patch's PMM uses the pooled values from the patch plus a
`--lpmm_halo`-point border, so heavy rain in one region does not set the intensities of
another (Clark 2017; Snook et al. 2019). Defaults 16 / 24 points = 48 km patches with a
~190 km window on the 3-km grid.

Usage:
  python src/derived_precip.py 2026-10-03T16 18 --forecast_dir $SCRATCH/hrrrcast-data
"""

import argparse
import glob
import logging
import os
import re
import sys

import numpy as np
import xarray as xr

import utils
from utils import setup_logging

logger = logging.getLogger(__name__)

WINDOWS = {"APCP_6H": 6, "APCP_12H": 12}
ATTRS = {
    "APCP":     {"long_name": "1-h precipitation",              "units": "kg m-2"},
    "APCP_TOT": {"long_name": "Total precipitation since init", "units": "kg m-2"},
    "APCP_6H":  {"long_name": "6-h precipitation",              "units": "kg m-2"},
    "APCP_12H": {"long_name": "12-h precipitation",             "units": "kg m-2"},
}


# --------------------------------------------------------------------------- PMM / LPMM
def pmm_flat(stack: np.ndarray) -> np.ndarray:
    """PMM of (members, npoints) -> (npoints); method 2 (pooled sort, every M-th value)."""
    m = stack.shape[0]
    order = np.argsort(stack.mean(axis=0), kind="stable")
    pooled = np.sort(stack, axis=None)[::m]
    out = np.empty(stack.shape[1], dtype=stack.dtype)
    out[order] = pooled
    return out


def pmm(stack: np.ndarray) -> np.ndarray:
    """Domain-wide PMM of (members, ny, nx)."""
    m, ny, nx = stack.shape
    return pmm_flat(stack.reshape(m, -1)).reshape(ny, nx)


def lpmm(stack: np.ndarray, patch: int = 16, halo: int = 24, smooth: float = 0.0) -> np.ndarray:
    """Local PMM of (members, ny, nx): PMM per `patch` x `patch` tile using values pooled from
    the tile plus a `halo`-point border. Dry windows are skipped (output 0)."""
    m, ny, nx = stack.shape
    out = np.zeros((ny, nx), dtype=stack.dtype)
    for i0 in range(0, ny, patch):
        i1 = min(i0 + patch, ny)
        wi0, wi1 = max(0, i0 - halo), min(ny, i1 + halo)
        for j0 in range(0, nx, patch):
            j1 = min(j0 + patch, nx)
            wj0, wj1 = max(0, j0 - halo), min(nx, j1 + halo)
            win = stack[:, wi0:wi1, wj0:wj1]
            if not np.any(win > 0):
                continue
            local = pmm_flat(win.reshape(m, -1)).reshape(wi1 - wi0, wj1 - wj0)
            out[i0:i1, j0:j1] = local[i0 - wi0:i1 - wi0, j0 - wj0:j1 - wj0]
    if smooth and smooth > 0:
        from scipy.ndimage import gaussian_filter
        out = gaussian_filter(out, sigma=smooth)
    return np.maximum(out, 0)


# --------------------------------------------------------------------------- I/O helpers
def find_members(case_dir: str):
    mems = sorted({int(m.group(1)) for f in glob.glob(os.path.join(case_dir, "hrrrcast_m*_f01.nc"))
                   if (m := re.search(r"hrrrcast_m(\d+)_f01\.nc$", f))})
    return mems


def read_apcp(path: str):
    """Return (template DataArray, 2-D float32 values) for APCP in one hourly file."""
    with xr.open_dataset(path, decode_timedelta=True) as ds:
        da = ds["APCP"].load()
    return da, np.squeeze(da.values).astype(np.float32)


def write(path: str, template: xr.DataArray, fields: dict, extra_attrs: dict):
    """Write fields (name -> 2-D array) using the APCP template's dims/coords."""
    shape = template.shape
    data_vars = {}
    for name, vals in fields.items():
        da = template.copy(data=vals.reshape(shape).astype(np.float32))
        da.name = name
        da.attrs = {**template.attrs, **ATTRS.get(name, {}), **extra_attrs}
        data_vars[name] = da
    xr.Dataset(data_vars).to_netcdf(path, engine=utils.netcdf_engine())
    logger.info(f"Wrote {os.path.basename(path)}: {', '.join(fields)}")


# --------------------------------------------------------------------------- main
REFERENCE_RUNS = ("hrrr",)   # non-ensemble runs (get_hrrr_fcst.py): accumulations only, no PMM


def write_lpmm_grib(case_dir: str, init_dt, hh: str, h: int) -> None:
    """hrrrcast.lpmm.tHHz.pgrb2.fHH next to the members' GRIB2, with the same writer and grid
    (nc2grib.Netcdf2Grib via get_hrrr_fcst.nc_to_grib). It holds the 1-h APCP message, which
    lines up with the members' APCP; APCP_TOT/6H/12H stay in the NetCDF file because the
    writer has no GRIB2 code for multi-hour totals (they would look like 1-h APCP)."""
    nc = os.path.join(case_dir, f"hrrrcast_lpmm_f{h:02d}.nc")
    gb = os.path.join(case_dir, f"hrrrcast.lpmm.t{hh}z.pgrb2.f{h:02d}")
    try:
        from get_hrrr_fcst import nc_to_grib
        if nc_to_grib(nc, gb, init_dt.isoformat(), h):
            logger.info(f"Wrote {os.path.basename(gb)}")
    except Exception as e:
        logger.warning(f"{os.path.basename(gb)}: GRIB2 output failed ({e}); the NetCDF file is fine")


def _read_hourly(path: str) -> np.ndarray:
    """1-h APCP of one hourly file, >= 0, NaN -> 0."""
    return np.maximum(np.nan_to_num(read_apcp(path)[1]), 0)


def _hour_products(case_dir, h, init_dt, hh, run_fields, ens_members, hourly_ens, patch, halo, smooth,
                   do_lpmm, grib2):
    """Write every product of forecast hour h (runs in a worker process).

    run_fields: {run: {name: 2-D}} accumulations per member / reference run;
    ens_members: HRRRCast members in this hour's ensemble; hourly_ens: their 1-h APCP stack."""
    template = None
    for r, f in run_fields.items():
        template = read_apcp(os.path.join(case_dir, f"hrrrcast_{r}_f{h:02d}.nc"))[0]
        write(os.path.join(case_dir, f"hrrrcast_{r}_precip_f{h:02d}.nc"), template, f,
              {"source": "derived_precip.py"})
    if len(ens_members) >= 2:
        template = read_apcp(os.path.join(case_dir, f"hrrrcast_{ens_members[0]}_f{h:02d}.nc"))[0]
        names = list(run_fields[ens_members[0]])
        stacks = {n: np.stack([run_fields[m][n] for m in ens_members]) for n in names}
        write(os.path.join(case_dir, f"hrrrcast_avg_precip_f{h:02d}.nc"), template,
              {n: pmm(st) for n, st in stacks.items()},
              {"processing_method": "probability_matched_mean"})
        if do_lpmm:
            stacks = {"APCP": hourly_ens, **stacks}
            write(os.path.join(case_dir, f"hrrrcast_lpmm_f{h:02d}.nc"), template,
                  {n: lpmm(st, patch, halo, smooth) for n, st in stacks.items()},
                  {"processing_method": "local_probability_matched_mean",
                   "lpmm_patch_points": patch, "lpmm_halo_points": halo,
                   "lpmm_smooth_sigma": smooth})
            if grib2:
                write_lpmm_grib(case_dir, init_dt, hh, h)
    return h


def run(inittime: str, lead_hours: int, forecast_dir: str, patch: int, halo: int, smooth: float,
        do_lpmm: bool, grib2: bool = True, workers: int = 0):
    """Three steps: (1) read every run's 1-h APCP for every hour (parallel), (2) running totals
    (cheap, in order), (3) each hour's files, PMM and LPMM (parallel; hours are independent).
    workers: 0 = $HRRRCAST_PRECIP_WORKERS, else the available cores."""
    from concurrent.futures import ProcessPoolExecutor

    init_dt, y, mo, d, hh = utils.validate_datetime(inittime)
    case_dir = os.path.join(forecast_dir, f"{y}{mo}{d}", hh)
    members = [f"m{m:02d}" for m in find_members(case_dir)]
    refs = [r for r in REFERENCE_RUNS if os.path.exists(os.path.join(case_dir, f"hrrrcast_{r}_f01.nc"))]
    if not members and not refs:
        raise FileNotFoundError(f"No hrrrcast_mNN_f01.nc (or hrrrcast_hrrr_f01.nc) files in {case_dir}")
    logger.info(f"{case_dir}: members {members}, reference runs {refs}, hours 1-{lead_hours}")

    # a member (or the HRRR) whose files stop early is dropped from that hour on
    last = {}
    for r in members + refs:
        h = 0
        while h < lead_hours and os.path.exists(os.path.join(case_dir, f"hrrrcast_{r}_f{h + 1:02d}.nc")):
            h += 1
        last[r] = h
        if h < lead_hours:
            logger.warning(f"Missing hrrrcast_{r}_f{h + 1:02d}.nc; {r} products stop at f{h:02d}")
    n_hours = max(last.values())

    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    workers = workers or int(os.environ.get("HRRRCAST_PRECIP_WORKERS") or 0) or cores

    # (1) read all 1-h APCP fields (processes, not threads: netCDF4/HDF5 is not thread-safe)
    jobs = [(r, h) for r in members + refs for h in range(1, last[r] + 1)]
    paths = [os.path.join(case_dir, f"hrrrcast_{r}_f{h:02d}.nc") for r, h in jobs]
    n_read = max(1, min(workers, 16, len(jobs)))
    if n_read == 1:
        vals = [_read_hourly(pth) for pth in paths]
    else:
        with ProcessPoolExecutor(max_workers=n_read) as ex:
            vals = list(ex.map(_read_hourly, paths, chunksize=4))
    hourly = dict(zip(jobs, vals))
    logger.info(f"Read {len(jobs)} hourly APCP fields")

    # (2) running totals and window sums per run and hour
    fields = {}                                       # (run, h) -> {name: 2-D}
    for r in members + refs:
        tot = [None]
        for h in range(1, last[r] + 1):
            tot.append(hourly[(r, h)] if h == 1 else tot[h - 1] + hourly[(r, h)])
            f = {"APCP_TOT": tot[h]}
            for name, w in WINDOWS.items():
                if h >= w:
                    f[name] = tot[h] - tot[h - w] if h - w > 0 else tot[h]
            fields[(r, h)] = f

    # (3) per-hour products in parallel
    tasks = []
    for h in range(1, n_hours + 1):
        runs_h = [r for r in members + refs if last[r] >= h]
        ens_h = [m for m in members if last[m] >= h]
        hourly_ens = np.stack([hourly[(m, h)] for m in ens_h]) if len(ens_h) >= 2 else None
        tasks.append((case_dir, h, init_dt, hh, {r: fields[(r, h)] for r in runs_h}, ens_h, hourly_ens,
                      patch, halo, smooth, do_lpmm, grib2))
    n_workers = max(1, min(workers, len(tasks)))
    logger.info(f"Writing products for {len(tasks)} hours with {n_workers} worker processes")
    if n_workers == 1:
        for t in tasks:
            _hour_products(*t)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for h in ex.map(_hour_products, *zip(*tasks)):
                logger.info(f"Finished f{h:02d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inittime", help="YYYY-MM-DDTHH")
    ap.add_argument("lead_hours", type=int)
    ap.add_argument("--forecast_dir", default="./")
    ap.add_argument("--lpmm_patch", type=int, default=16, help="LPMM tile size in grid points (16 = 48 km)")
    ap.add_argument("--lpmm_halo", type=int, default=24, help="extra points pooled around each tile (24 = 72 km)")
    ap.add_argument("--lpmm_smooth", type=float, default=0.0,
                    help="Gaussian sigma (grid points) to soften tile seams; 0 = off")
    ap.add_argument("--no_lpmm", action="store_true", help="skip the LPMM products")
    ap.add_argument("--no_grib2", action="store_true",
                    help="do not write hrrrcast.lpmm.tHHz.pgrb2.fHH (GRIB2 of the LPMM 1-h APCP)")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel processes (default: $HRRRCAST_PRECIP_WORKERS, else all available cores)")
    ap.add_argument("--log_level", default="INFO")
    a = ap.parse_args()
    setup_logging(a.log_level)
    try:
        run(a.inittime, a.lead_hours, a.forecast_dir, a.lpmm_patch, a.lpmm_halo, a.lpmm_smooth,
            not a.no_lpmm, not a.no_grib2, a.workers)
    except Exception as e:
        logger.exception(f"derived_precip failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
