#!/usr/bin/env python3
"""
MRMS observations for an HRRRCast case, as member "mrms" (run on its own, any time after the case).

For every forecast hour H (valid time = init + H, top of the hour) this fetches these MRMS
products, cuts them to the lat/lon box of the case's HRRRCast grid and writes

    YYYYMMDD/HH/hrrrcast_mrms_fHH.nc          one variable per product (below)
    YYYYMMDD/HH/hrrrcast_mrms_precip_fHH.nc   APCP_TOT / APCP_6H / APCP_12H from the hourly QPE

so plot.py draws them like any member (`--members mrms`) and the viewer lists "MRMS (observed)".

    variable      MRMS product                        time used                 HRRRCast counterpart
    REFC          MergedReflectivityQCComposite       scan nearest HH:00        REFC
    APCP          MultiSensor_QPE_01H_Pass2           1 h ending at HH:00       APCP (1-h)
    ROTTRK_0_2km  RotationTrack60min (0-2 km AzShear) max over the past 60 min  (cf. MXUPHL_max_0_2km)
    MESH          MESH_Max_60min                      max over the past 60 min  (none: hail size)

Grid: the native ~1 km (0.01 deg) MRMS grid inside the domain's lat/lon box (default), or with
--grid hrrrcast the members' 3 km grid (each point takes the maximum of the nearby MRMS pixels,
or their mean for APCP). Missing/no-coverage values become NaN; "no echo" REFC becomes -10 dBZ.

Existing files are completed, not redone: products already in hrrrcast_mrms_fHH.nc are kept and
only the missing ones fetched (--overwrite redoes everything). Hours still in the future are skipped.

Sources (first that has the file): AWS Open Data s3://noaa-mrms-pds (Oct 2020 ->), then the
Iowa State MTArchive for older cases.

Usage:
  python src/get_mrms.py 2026-11-04T03 18 --base_dir $SCRATCH/hrrrcast-data
  python src/get_mrms.py 2026-11-04T03 18 --base_dir ... --products REFC APCP --grid hrrrcast
  (lead hours: 0 = take them from the case's HRRRCast output)
"""

import argparse
import glob
import gzip
import logging
import os
import re
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils  # noqa: E402

logger = logging.getLogger("get_mrms")

AWS = "https://noaa-mrms-pds.s3.amazonaws.com"
ISU = "https://mtarchive.geol.iastate.edu"

# variable -> MRMS product, how to pick the file, value cleanup, regrid aggregation, attributes
PRODUCTS = {
    "REFC": dict(product="MergedReflectivityQCComposite_00.50", isu="MergedReflectivityQCComposite",
                 floor=-10.0, agg="max", long_name="Composite reflectivity (MRMS)", units="dBZ"),
    "APCP": dict(product="MultiSensor_QPE_01H_Pass2_00.00", isu="MultiSensor_QPE_01H_Pass2",
                 exact=True, floor=None, agg="mean", long_name="1-h precipitation (MRMS multi-sensor QPE)",
                 units="kg m-2"),
    "ROTTRK_0_2km": dict(product="RotationTrack60min_00.50", isu="RotationTrack60min",
                         floor=0.0, agg="max", long_name="0-2 km rotation track, 60-min max (MRMS)",
                         units="s-1"),
    "MESH": dict(product="MESH_Max_60min_00.50", isu="MESH_Max_60min",
                 floor=0.0, agg="max", long_name="Max estimated hail size, 60-min max (MRMS)", units="mm"),
}
WINDOWS = {"APCP_6H": 6, "APCP_12H": 12}

# ecCodes (pygrib) and HDF5 (netCDF) are not thread-safe: downloads run in parallel threads,
# decoding and writing take this lock
_IO_LOCK = threading.Lock()
SCAN_RE = re.compile(r"_(\d{8})-(\d{6})\.grib2\.gz$")


# --------------------------------------------------------------------------- find a file
def _aws_keys(prefix: str):
    """Object keys under prefix (S3 ListObjectsV2, anonymous)."""
    keys, token = [], None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = requests.get(AWS + "/", params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        keys += [k.text for k in root.findall("s3:Contents/s3:Key", ns)]
        if root.findtext("s3:IsTruncated", default="false", namespaces=ns) != "true":
            return keys
        token = root.findtext("s3:NextContinuationToken", namespaces=ns)


def find_file(spec: dict, valid: datetime, max_offset_min: int):
    """(url, file time) of the product file closest to `valid`, or (None, None).
    exact=True (hourly QPE): only a file stamped exactly at `valid` counts."""
    prod = spec["product"]
    tol = 0 if spec.get("exact") else max_offset_min * 60
    cands = []
    for t in {valid - timedelta(hours=1), valid}:    # scans just before HH:00 are in the hour before
        prefix = f"CONUS/{prod}/{t:%Y%m%d}/MRMS_{prod}_{t:%Y%m%d-%H}"
        try:
            for k in _aws_keys(prefix):
                m = SCAN_RE.search(k)
                if m:
                    ft = datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S")
                    if abs((ft - valid).total_seconds()) <= tol:
                        cands.append((abs((ft - valid).total_seconds()), ft, f"{AWS}/{k}"))
        except Exception as e:
            logger.debug(f"AWS listing {prefix}: {e}")
    if cands:
        _, ft, url = min(cands)
        return url, ft
    # Iowa State archive (older cases): try HH:00 first, then nearby even minutes
    offs = [0] if spec.get("exact") else sorted(range(-max_offset_min, max_offset_min + 1), key=abs)
    for off in offs:
        t = valid + timedelta(minutes=off)
        url = f"{ISU}/{t:%Y/%m/%d}/mrms/ncep/{spec['isu']}/{prod}_{t:%Y%m%d-%H%M}00.grib2.gz"
        try:
            if requests.head(url, timeout=30).status_code == 200:
                return url, t
        except Exception:
            pass
    return None, None


# --------------------------------------------------------------------------- read / regrid
def read_mrms(url: str, spec: dict):
    """(lat 1-D ascending, lon 1-D in -180..180, values [lat, lon]) of one gzipped MRMS GRIB2."""
    import pygrib
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            raw = gzip.decompress(r.content)
            break
        except Exception as e:
            if attempt == 2:
                raise
            logger.warning(f"Download failed ({e}); retrying")
            time.sleep(5)
    with _IO_LOCK, tempfile.NamedTemporaryFile(suffix=".grib2") as tmp:
        tmp.write(raw)
        tmp.flush()
        g = pygrib.open(tmp.name)
        try:
            msg = g.message(1)
            vals = np.ma.filled(np.ma.asarray(msg.values, dtype=np.float32), np.nan)
            lat2, lon2 = msg.latlons()
        finally:
            g.close()
    lat, lon = lat2[:, 0].astype(np.float64), ((lon2[0, :] + 180.0) % 360.0) - 180.0
    if lat[0] > lat[-1]:                       # MRMS scans north -> south
        lat, vals = lat[::-1], vals[::-1]
    lat, lon, vals = coarsen_to_001(lat, lon, vals)
    # MRMS flags: -999 no coverage, -99 / -3 / -1 "no data" / below threshold
    if spec["floor"] is None:                  # QPE: any negative value is a flag
        vals = np.where(vals < 0, np.nan, vals)
    elif spec["floor"] < 0:                    # REFC: no coverage -> NaN, no echo -> floor
        vals = np.where(vals < -900, np.nan, np.maximum(vals, spec["floor"]))
    else:                                      # rotation / hail: flags -> NaN, values >= 0
        vals = np.where(vals < 0, np.nan, vals)
    return lat, lon, vals.astype(np.float32)


def coarsen_to_001(lat, lon, vals):
    """The rotation tracks come on a 0.005 deg grid (twice the 0.01 deg of REFC / QPE / MESH).
    Take the max of each 2x2 block so every product lands on the same 0.01 deg grid (the 0.005 deg
    cells pair up exactly into the 0.01 deg cells) and can share one file."""
    dlat = abs(float(lat[1] - lat[0]))
    f = int(round(0.01 / dlat)) if dlat < 0.0075 else 1
    if f <= 1:
        return lat, lon, vals
    ny, nx = (vals.shape[0] // f) * f, (vals.shape[1] // f) * f
    v = vals[:ny, :nx]
    out = v[0::f, 0::f]
    for di in range(f):
        for dj in range(f):
            if di or dj:
                out = np.fmax(out, v[di::f, dj::f])      # NaN only where the whole block is NaN
    lat_c = lat[:ny].reshape(-1, f).mean(axis=1)
    lon_c = lon[:nx].reshape(-1, f).mean(axis=1)
    return lat_c, lon_c, out


def box_indices(lat, lon, ref_lat, ref_lon):
    rlon = ((ref_lon + 180.0) % 360.0) - 180.0
    i = np.nonzero((lat >= ref_lat.min()) & (lat <= ref_lat.max()))[0]
    j = np.nonzero((lon >= rlon.min()) & (lon <= rlon.max()))[0]
    if not len(i) or not len(j):
        raise ValueError("HRRRCast domain is outside the MRMS grid")
    return slice(i[0], i[-1] + 1), slice(j[0], j[-1] + 1)


def to_hrrrcast_grid(lat, lon, vals, ref_lat, ref_lon, agg: str, half_width_km: float = 1.5):
    """Max (or mean) of the MRMS pixels within +/-half_width_km of each reference grid point."""
    rlon = ((ref_lon + 180.0) % 360.0) - 180.0
    dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
    ci = np.rint((ref_lat - lat[0]) / dlat).astype(int)
    cj = np.rint((rlon - lon[0]) / dlon).astype(int)
    ri = max(0, int(round(half_width_km / 111.0 / dlat)))
    rj = max(0, int(round(half_width_km / (111.0 * np.cos(np.deg2rad(np.nanmean(ref_lat)))) / dlon)))
    ny, nx = vals.shape
    stack = []
    for di in range(-ri, ri + 1):
        for dj in range(-rj, rj + 1):
            stack.append(vals[np.clip(ci + di, 0, ny - 1), np.clip(cj + dj, 0, nx - 1)])
    stack = np.stack(stack)
    with np.errstate(all="ignore"):
        out = np.nanmax(stack, axis=0) if agg == "max" else np.nanmean(stack, axis=0)
    out[~((ci >= 0) & (ci < ny) & (cj >= 0) & (cj < nx))] = np.nan
    return out.astype(np.float32)


# --------------------------------------------------------------------------- write
def _coords(init_dt, hour, lat2, lon2):
    return {"lead_time": ("lead_time", [hour]), "time": ("time", [init_dt + timedelta(hours=hour)]),
            "latitude": (("latitude", "longitude"), lat2), "longitude": (("latitude", "longitude"), lon2)}


def write_nc(path, init_dt, hour, lat2, lon2, fields: dict, attrs: dict, grid: str):
    base = _coords(init_dt, hour, lat2, lon2)
    data_vars = {}
    for name, (vals, vattrs) in fields.items():
        data_vars[name] = xr.DataArray(np.asarray(vals, np.float32)[None, None],
                                       dims=("lead_time", "time", "latitude", "longitude"),
                                       coords=base, attrs=vattrs)
    ds = xr.Dataset(data_vars).assign_coords(forecast_reference_time=xr.DataArray(np.datetime64(init_dt, "ns")))
    if grid == "hrrrcast":
        # on the members' Lambert grid: same CF metadata as their files (plain lat/lon otherwise)
        try:
            from cf_attributes import apply_cf_attributes
            ds = apply_cf_attributes(ds, init_datetime=init_dt)
            for name, (_, vattrs) in fields.items():
                ds[name].attrs.update(vattrs)
        except Exception as e:
            logger.debug(f"apply_cf_attributes skipped: {e}")
    ds.attrs.update(attrs)
    tmp = str(path) + ".tmp"
    with _IO_LOCK:
        ds.to_netcdf(tmp, engine=utils.netcdf_engine())
    os.replace(tmp, path)


def read_existing(path):
    """{var: (values 2-D, attrs)}, lat, lon, attrs of an existing hrrrcast_mrms_fHH.nc (or None)."""
    if not os.path.exists(path):
        return None
    with _IO_LOCK, xr.open_dataset(path, decode_timedelta=False) as ds:
        fields = {v: (np.squeeze(ds[v].values).astype(np.float32), dict(ds[v].attrs))
                  for v in ds.data_vars if v in PRODUCTS}
        return fields, np.asarray(ds["latitude"].values), np.asarray(ds["longitude"].values), dict(ds.attrs)


# --------------------------------------------------------------------------- main
def lead_from_case(case_dir: Path, hh: str) -> int:
    hours = [int(m.group(1)) for f in glob.glob(str(case_dir / "hrrrcast_m*_f*.nc"))
             if (m := re.search(r"hrrrcast_m\d+_f(\d+)\.nc$", f))]
    hours += [int(m.group(1)) for f in glob.glob(str(case_dir / f"hrrrcast.m*.t{hh}z.pgrb2.f*"))
              if (m := re.search(r"pgrb2\.f(\d+)$", f))]
    return max(hours) if hours else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inittime", help="YYYY-MM-DDTHH (HRRRCast init time)")
    ap.add_argument("lead_hours", type=int, nargs="?", default=0,
                    help="last forecast hour (default 0: read it from the case's HRRRCast output)")
    ap.add_argument("--base_dir", default="./", help="data root (YYYYMMDD/HH/ under it)")
    ap.add_argument("--products", nargs="+", default=list(PRODUCTS), choices=list(PRODUCTS),
                    help=f"default: all ({' '.join(PRODUCTS)})")
    ap.add_argument("--grid", choices=["native", "hrrrcast"], default="native",
                    help="native ~1 km MRMS grid in the domain box (default), or the members' 3 km grid")
    ap.add_argument("--max_offset_min", type=int, default=5, help="largest allowed scan time offset from HH:00")
    ap.add_argument("--start_hour", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads")
    ap.add_argument("--overwrite", action="store_true", help="refetch every product, not just missing ones")
    ap.add_argument("--log_level", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s")

    init_dt, y, mo, d, hh = utils.validate_datetime(a.inittime)
    case_dir = Path(a.base_dir) / f"{y}{mo}{d}" / hh
    lead = a.lead_hours or lead_from_case(case_dir, hh)
    if not lead:
        sys.exit(f"No HRRRCast output in {case_dir} to read the lead hours from; pass lead_hours")
    from get_hrrr_fcst import find_reference, reference_latlons
    ref = find_reference(case_dir)
    if ref is None:
        sys.exit(f"No HRRRCast output (hrrrcast_mNN_fHH.nc or GRIB2) in {case_dir} to take the domain from")
    ref_lat, ref_lon = reference_latlons(ref)
    logger.info(f"{case_dir}: f{a.start_hour:02d}-f{lead:02d}, products {' '.join(a.products)}, "
                f"domain {ref_lat.shape[0]}x{ref_lat.shape[1]} from {ref.name}, grid {a.grid}")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    status = {}

    def do_hour(h):
        out = case_dir / f"hrrrcast_mrms_f{h:02d}.nc"
        valid = init_dt + timedelta(hours=h)
        if valid > now - timedelta(minutes=a.max_offset_min):
            return "future"
        old = None if a.overwrite else read_existing(out)
        fields, attrs = (dict(old[0]), dict(old[3])) if old else ({}, {})
        todo = [p for p in a.products if p not in fields]
        if not todo:
            return "complete"
        lat2 = lon2 = None
        if old and fields:
            lat2, lon2 = old[1], old[2]
        got = []
        for p in todo:
            spec = PRODUCTS[p]
            url, ft = find_file(spec, valid, a.max_offset_min)
            if url is None:
                logger.warning(f"f{h:02d} {p}: no {spec['product']} file near {valid:%Y-%m-%d %H:%M}Z")
                continue
            lat, lon, vals = read_mrms(url, spec)
            if a.grid == "native":
                si, sj = box_indices(lat, lon, ref_lat, ref_lon)
                vals = vals[si, sj]
                lo2, la2 = np.meshgrid(lon[sj], lat[si])
            else:
                vals = to_hrrrcast_grid(lat, lon, vals, ref_lat, ref_lon, spec["agg"])
                la2, lo2 = ref_lat, ref_lon
            if lat2 is None:
                lat2, lon2 = la2, lo2
            elif la2.shape != lat2.shape:
                logger.warning(f"f{h:02d} {p}: grid {la2.shape} differs from {lat2.shape} in {out.name}; "
                               f"use --overwrite (e.g. after changing --grid)")
                continue
            fields[p] = (vals, {"long_name": spec["long_name"], "units": spec["units"],
                                "mrms_product": spec["product"], "mrms_file_time": f"{ft:%Y-%m-%dT%H:%M:%SZ}",
                                "mrms_url": url})
            got.append(p)
        if not got:
            return "missing" if not fields else "partial"
        attrs.update(source="MRMS (observed)", grid=a.grid,
                     title="MRMS observations on the HRRRCast domain (get_mrms.py)")
        write_nc(out, init_dt, h, lat2, lon2, fields, attrs, a.grid)
        logger.info(f"f{h:02d}: {out.name} + {' '.join(got)} ({lat2.shape[0]}x{lat2.shape[1]})")
        return "ok" if len(fields) == len(a.products) else "partial"

    hours = list(range(a.start_hour, lead + 1))
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for h, r in zip(hours, ex.map(lambda h: _safe(do_hour, h), hours)):
            status[h] = r

    if "APCP" in a.products:
        write_accumulations(case_dir, init_dt, lead)

    summary = {k: sum(v == k for v in status.values())
               for k in ("ok", "complete", "partial", "future", "missing", "error")}
    logger.info(f"MRMS {case_dir}: " + ", ".join(f"{k} {n}" for k, n in summary.items() if n))
    if summary["future"]:
        logger.info(f"{summary['future']} hour(s) not observed yet; rerun later to fill them in")
    return 1 if summary["error"] else 0


def write_accumulations(case_dir: Path, init_dt, lead: int):
    """hrrrcast_mrms_precip_fHH.nc: run total / 6 h / 12 h sums of the hourly QPE (as derived_precip.py
    does for the members). An hour without QPE makes every sum that includes it missing."""
    from derived_precip import write as write_precip
    tot = {0: None}
    for h in range(1, lead + 1):
        path = case_dir / f"hrrrcast_mrms_f{h:02d}.nc"
        if not path.exists():
            break
        with xr.open_dataset(path, decode_timedelta=True) as ds:
            if "APCP" in ds:
                template = ds["APCP"].load()
                vals = np.squeeze(template.values).astype(np.float32)
            else:                                # QPE missing this hour: sums through it are NaN
                template = ds[next(iter(ds.data_vars))].load()
                vals = np.full(np.squeeze(template.values).shape, np.nan, np.float32)
        tot[h] = vals if h == 1 else tot[h - 1] + vals          # NaN carries forward
        f = {"APCP_TOT": tot[h]}
        for name, w in WINDOWS.items():
            if h >= w:
                f[name] = tot[h] - tot[h - w] if h - w > 0 else tot[h]
        write_precip(str(case_dir / f"hrrrcast_mrms_precip_f{h:02d}.nc"), template, f,
                     {"source": "get_mrms.py (sums of MRMS 1-h multi-sensor QPE)"})


def _safe(fn, h):
    try:
        return fn(h)
    except Exception as e:
        logger.error(f"f{h:02d}: {e}")
        return "error"


if __name__ == "__main__":
    sys.exit(main())
