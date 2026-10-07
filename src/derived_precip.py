#!/usr/bin/env python3
"""
Derived precipitation products for HRRRCast forecasts.

HRRRCast writes APCP as the 1-hour accumulation ending at each forecast hour. From the
per-member files (YYYYMMDD/HH/hrrrcast_mNN_fHH.nc) this script writes, for every hour H:

  per member   hrrrcast_mNN_precip_fHH.nc
      APCP_TOT   total precipitation f00 -> fH
      APCP_6H    6-h total ending at fH   (H >= 6)
      APCP_12H   12-h total ending at fH  (H >= 12)
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
def run(inittime: str, lead_hours: int, forecast_dir: str, patch: int, halo: int, smooth: float,
        do_lpmm: bool):
    _, y, mo, d, hh = utils.validate_datetime(inittime)
    case_dir = os.path.join(forecast_dir, f"{y}{mo}{d}", hh)
    members = find_members(case_dir)
    if not members:
        raise FileNotFoundError(f"No hrrrcast_mNN_f01.nc files in {case_dir}")
    logger.info(f"{case_dir}: members {members}, hours 1-{lead_hours}")

    totals = {m: [np.zeros(0, np.float32)] for m in members}   # cumulative totals; index = hour
    for h in range(1, lead_hours + 1):
        hourly, template = {}, None
        for m in members:
            path = os.path.join(case_dir, f"hrrrcast_m{m:02d}_f{h:02d}.nc")
            if not os.path.exists(path):
                logger.warning(f"Missing {os.path.basename(path)}; stopping at f{h - 1:02d}")
                return
            template, vals = read_apcp(path)
            vals = np.maximum(np.nan_to_num(vals), 0)
            hourly[m] = vals
            prev = totals[m][-1] if totals[m][-1].size else np.zeros_like(vals)
            totals[m].append(prev + vals)

        # per-member accumulations
        member_fields = {}
        for m in members:
            f = {"APCP_TOT": totals[m][h]}
            for name, w in WINDOWS.items():
                if h >= w:
                    start = totals[m][h - w] if (h - w) > 0 else 0.0
                    f[name] = totals[m][h] - start
            member_fields[m] = f
            write(os.path.join(case_dir, f"hrrrcast_m{m:02d}_precip_f{h:02d}.nc"), template, f,
                  {"source": "derived_precip.py"})

        # ensemble products
        if len(members) >= 2:
            names = list(member_fields[members[0]])
            stacks = {n: np.stack([member_fields[m][n] for m in members]) for n in names}
            write(os.path.join(case_dir, f"hrrrcast_avg_precip_f{h:02d}.nc"), template,
                  {n: pmm(s) for n, s in stacks.items()},
                  {"processing_method": "probability_matched_mean"})
            if do_lpmm:
                stacks = {"APCP": np.stack([hourly[m] for m in members]), **stacks}
                write(os.path.join(case_dir, f"hrrrcast_lpmm_f{h:02d}.nc"), template,
                      {n: lpmm(s, patch, halo, smooth) for n, s in stacks.items()},
                      {"processing_method": "local_probability_matched_mean",
                       "lpmm_patch_points": patch, "lpmm_halo_points": halo,
                       "lpmm_smooth_sigma": smooth})

        # drop history no longer needed for the longest window
        keep = max(WINDOWS.values())
        for m in members:
            if h - keep - 1 >= 1:
                totals[m][h - keep - 1] = np.zeros(0, np.float32)


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
    ap.add_argument("--log_level", default="INFO")
    a = ap.parse_args()
    setup_logging(a.log_level)
    try:
        run(a.inittime, a.lead_hours, a.forecast_dir, a.lpmm_patch, a.lpmm_halo, a.lpmm_smooth,
            not a.no_lpmm)
    except Exception as e:
        logger.exception(f"derived_precip failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
