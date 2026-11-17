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
    MESH          MESH_Max_60min                      max over the past 60 min  (none: hail size)

Grid: the native ~1 km (0.01 deg) MRMS grid inside the domain's lat/lon box (default), or with
--grid hrrrcast the members' 3 km grid (each point takes the maximum of the nearby MRMS pixels,
or their mean for APCP). Missing/no-coverage values become NaN; "no echo" REFC becomes -10 dBZ.

GRIB2 for verification (MET etc.): every hour also gets

    YYYYMMDD/HH/hrrrcast.mrms.tHHz.pgrb2.fHH  REFC + APCP on the members' 3 km grid

written with the same GRIB2 writer and grid definition as the members (Netcdf2Grib), so it lines
up point for point with hrrrcast.mNN.tHHz.pgrb2.fHH and survives the cleanup job. A native-grid
file is put on the 3 km grid first, the same way --grid hrrrcast does (max for REFC, mean for
APCP); no-coverage points are written as missing. Existing GRIB2 files are rewritten when their
grid does not match the members' grid or the hour was refetched; --no_grib2 skips this step.

Speed: one S3 listing per product and day finds every file; downloads run in threads, and the
slow part -- decoding the PNG-packed 3500 x 7000 CONUS GRIB2 -- runs in parallel processes
(--workers, default: all cores), each returning only the cut-out domain.

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
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
    "MESH": dict(product="MESH_Max_60min_00.50", isu="MESH_Max_60min",
                 floor=0.0, agg="max", long_name="Max estimated hail size, 60-min max (MRMS)", units="mm"),
}
WINDOWS = {"APCP_6H": 6, "APCP_12H": 12}
SCAN_RE = re.compile(r"_(\d{8})-(\d{6})\.grib2\.gz$")


# --------------------------------------------------------------------------- find files
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


def list_day(product: str, day: datetime):
    """[(file time, url)] of one product on one day (one S3 listing, ~720 keys for 2-min scans)."""
    prefix = f"CONUS/{product}/{day:%Y%m%d}/"
    try:
        out = []
        for k in _aws_keys(prefix):
            m = SCAN_RE.search(k)
            if m:
                out.append((datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S"), f"{AWS}/{k}"))
        return out
    except Exception as e:
        logger.debug(f"AWS listing {prefix}: {e}")
        return []


def pick(files, spec: dict, valid: datetime, max_offset_min: int):
    """(url, file time) of the listed file closest to `valid` within tolerance, or (None, None)."""
    tol = 0 if spec.get("exact") else max_offset_min * 60
    best = min(((abs((t - valid).total_seconds()), t, u) for t, u in files), default=None)
    if best and best[0] <= tol:
        return best[2], best[1]
    return None, None


def isu_url(spec: dict, valid: datetime, max_offset_min: int):
    """Iowa State archive fallback (older cases): HH:00 first, then nearby even minutes."""
    offs = [0] if spec.get("exact") else sorted(range(-max_offset_min, max_offset_min + 1), key=abs)
    for off in offs:
        t = valid + timedelta(minutes=off)
        url = f"{ISU}/{t:%Y/%m/%d}/mrms/ncep/{spec['isu']}/{spec['product']}_{t:%Y%m%d-%H%M}00.grib2.gz"
        try:
            if requests.head(url, timeout=30).status_code == 200:
                return url, t
        except Exception:
            pass
    return None, None


def download(url: str) -> bytes:
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            return r.content
        except Exception as e:
            if attempt == 2:
                raise
            logger.warning(f"Download failed ({e}); retrying")
            time.sleep(5)


# --------------------------------------------------------------------------- decode + cut (worker process)
def decode_cut(raw_gz: bytes, var: str, grid: str, ref_lat, ref_lon):
    """Decode one gzipped MRMS GRIB2 and return (values, lat2, lon2) on the output grid.
    Runs in a worker process: ecCodes decoding is CPU-bound and not thread-safe."""
    import pygrib
    spec = PRODUCTS[var]
    msg = pygrib.fromstring(gzip.decompress(raw_gz))
    vals = np.ma.filled(np.ma.asarray(msg.values, dtype=np.float32), np.nan)
    # regular lat/lon grid: build the axes from the header instead of 2 x 24.5M-point latlons()
    nj, ni = vals.shape
    la1, lo1 = msg["latitudeOfFirstGridPointInDegrees"], msg["longitudeOfFirstGridPointInDegrees"]
    la2, lo2 = msg["latitudeOfLastGridPointInDegrees"], msg["longitudeOfLastGridPointInDegrees"]
    lat = np.linspace(la1, la2, nj)
    lon = ((np.linspace(lo1, lo2, ni) + 180.0) % 360.0) - 180.0
    if lat[0] > lat[-1]:                       # MRMS scans north -> south
        lat, vals = lat[::-1], vals[::-1]
    rlon = ((ref_lon + 180.0) % 360.0) - 180.0
    if grid == "native":
        i = np.nonzero((lat >= ref_lat.min()) & (lat <= ref_lat.max()))[0]
        j = np.nonzero((lon >= rlon.min()) & (lon <= rlon.max()))[0]
        if not len(i) or not len(j):
            raise ValueError("HRRRCast domain is outside the MRMS grid")
        vals = vals[i[0]:i[-1] + 1, j[0]:j[-1] + 1]
        lon2, lat2 = np.meshgrid(lon[j[0]:j[-1] + 1], lat[i[0]:i[-1] + 1])
    else:
        lat2, lon2 = ref_lat, ref_lon
    vals = clean(vals, spec)
    if grid != "native":
        vals = to_hrrrcast_grid(lat, lon, vals, ref_lat, rlon, spec["agg"])
    return vals.astype(np.float32), lat2, lon2


def clean(vals, spec):
    """MRMS flags: -999 no coverage, -99 / -3 / -1 "no data" / below threshold."""
    if spec["floor"] is None:                  # QPE: any negative value is a flag
        return np.where(vals < 0, np.nan, vals)
    if spec["floor"] < 0:                      # REFC: no coverage -> NaN, no echo -> floor
        return np.where(vals < -900, np.nan, np.maximum(vals, spec["floor"]))
    return np.where(vals < 0, np.nan, vals)    # hail: flags -> NaN, values >= 0


def to_hrrrcast_grid(lat, lon, vals, ref_lat, rlon, agg: str, half_width_km: float = 1.5):
    """Max (or mean) of the MRMS pixels within +/-half_width_km of each reference grid point."""
    dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
    ci = np.rint((ref_lat - lat[0]) / dlat).astype(int)
    cj = np.rint((rlon - lon[0]) / dlon).astype(int)
    ri = max(0, int(round(half_width_km / 111.0 / dlat)))
    rj = max(0, int(round(half_width_km / (111.0 * np.cos(np.deg2rad(np.nanmean(ref_lat)))) / dlon)))
    ny, nx = vals.shape
    stack = np.stack([vals[np.clip(ci + di, 0, ny - 1), np.clip(cj + dj, 0, nx - 1)]
                      for di in range(-ri, ri + 1) for dj in range(-rj, rj + 1)])
    with np.errstate(all="ignore"):
        out = np.nanmax(stack, axis=0) if agg == "max" else np.nanmean(stack, axis=0)
    out[~((ci >= 0) & (ci < ny) & (cj >= 0) & (cj < nx))] = np.nan
    return out


# --------------------------------------------------------------------------- write
def write_nc(path, init_dt, hour, lat2, lon2, fields: dict, attrs: dict, grid: str):
    base = {"lead_time": ("lead_time", [hour]), "time": ("time", [init_dt + timedelta(hours=hour)]),
            "latitude": (("latitude", "longitude"), lat2), "longitude": (("latitude", "longitude"), lon2)}
    data_vars = {name: xr.DataArray(np.asarray(vals, np.float32)[None, None],
                                    dims=("lead_time", "time", "latitude", "longitude"), coords=base, attrs=vattrs)
                 for name, (vals, vattrs) in fields.items()}
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
    ds.to_netcdf(tmp, engine=utils.netcdf_engine())
    os.replace(tmp, path)


def grib2_grid_ok(path, shape) -> bool:
    """True if the GRIB2 file exists and its first message has the members' grid size."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        import pygrib
        g = pygrib.open(str(path))
        try:
            m = g.message(1)
            return (int(m["Nj"]), int(m["Ni"])) == tuple(shape)
        finally:
            g.close()
    except Exception:
        return False


def write_grib2_3km(grib_path, init_dt, hour, fields: dict, lat2, lon2, ref_lat, ref_lon) -> bool:
    """hrrrcast.mrms.tHHz.pgrb2.fHH: REFC/APCP on the members' 3 km grid, written like the members.
    fields come from hrrrcast_mrms_fHH.nc on either grid (lat2/lon2 = that file's grid)."""
    try:
        from nc2grib import Netcdf2Grib
    except Exception as e:                     # grib2io / ecCodes writer missing in this environment
        logger.warning(f"f{hour:02d}: GRIB2 skipped, nc2grib unavailable ({e})")
        return False
    keep = [v for v in ("REFC", "APCP") if v in fields]
    if not keep:
        return False
    ny, nx = ref_lat.shape
    rlon = ((np.asarray(ref_lon) + 180.0) % 360.0) - 180.0
    data = {}
    for v in keep:
        vals = np.asarray(fields[v][0], np.float32)
        if vals.shape != (ny, nx):             # native lat/lon box -> members' grid
            lat1 = np.asarray(lat2)[:, 0]
            lon1 = ((np.asarray(lon2)[0, :] + 180.0) % 360.0) - 180.0
            vals = to_hrrrcast_grid(lat1, lon1, vals, ref_lat, rlon, PRODUCTS[v]["agg"]).astype(np.float32)
        data[v] = xr.DataArray(vals[None, None], dims=("lead_time", "time", "y", "x"))
    ds = xr.Dataset(data, coords={"lead_time": ("lead_time", [hour]),
                                  "time": ("time", [init_dt + timedelta(hours=hour)]),
                                  "latitude": (("y", "x"), np.asarray(ref_lat)),
                                  "longitude": (("y", "x"), np.asarray(ref_lon))})
    section3 = Netcdf2Grib.construct_section3_hrrr(nx=nx, ny=ny, lat1=float(ref_lat[0, 0]),
                                                   lon1=float(ref_lon[0, 0]) % 360.0)
    tmp = str(grib_path) + ".tmp"
    for f in (tmp, tmp + ".idx"):
        if os.path.exists(f):
            os.remove(f)
    Netcdf2Grib(section3=section3).save_grib2(init_dt, ds, tmp)
    if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        logger.warning(f"f{hour:02d}: GRIB2 writer produced no output")
        return False
    os.replace(tmp, grib_path)
    if os.path.exists(tmp + ".idx"):
        os.replace(tmp + ".idx", str(grib_path) + ".idx")
    return True


def read_existing(path):
    """{var: (values 2-D, attrs)}, lat, lon, attrs of an existing hrrrcast_mrms_fHH.nc (or None)."""
    if not os.path.exists(path):
        return None
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        fields = {v: (np.squeeze(ds[v].values).astype(np.float32), dict(ds[v].attrs))
                  for v in ds.data_vars if v in PRODUCTS}
        return fields, np.asarray(ds["latitude"].values), np.asarray(ds["longitude"].values), dict(ds.attrs)


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


# --------------------------------------------------------------------------- main
def lead_from_case(case_dir: Path, hh: str) -> int:
    hours = [int(m.group(1)) for f in glob.glob(str(case_dir / "hrrrcast_m*_f*.nc"))
             if (m := re.search(r"hrrrcast_m\d+_f(\d+)\.nc$", f))]
    hours += [int(m.group(1)) for f in glob.glob(str(case_dir / f"hrrrcast.m[0-9][0-9].t{hh}z.pgrb2.f*"))
              if (m := re.search(r"pgrb2\.f(\d+)$", f))]
    return max(hours) if hours else 0


def _cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


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
    ap.add_argument("--workers", type=int, default=0,
                    help="decode processes (default 0: all cores; each needs ~1 GB RAM)")
    ap.add_argument("--download_threads", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true", help="refetch every product, not just missing ones")
    ap.add_argument("--no_grib2", action="store_true",
                    help="don't write hrrrcast.mrms.tHHz.pgrb2.fHH (REFC/APCP on the members' 3 km grid)")
    ap.add_argument("--log_level", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s")
    t0 = time.time()

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

    # 1. what is needed: per hour, the products not already in its file
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    status, have, need = {}, {}, []
    rewritten = set()                            # hours whose NetCDF was (re)written in this run
    for h in range(a.start_hour, lead + 1):
        valid = init_dt + timedelta(hours=h)
        if valid > now - timedelta(minutes=a.max_offset_min):
            status[h] = "future"
            continue
        old = None if a.overwrite else read_existing(case_dir / f"hrrrcast_mrms_f{h:02d}.nc")
        have[h] = old
        todo = [p for p in a.products if not old or p not in old[0]]
        if not todo:
            status[h] = "complete"
        need += [(h, p) for p in todo]
    if not need:
        logger.info("Nothing to fetch")
    else:
        # 2. find the files: one S3 listing per product and day (in parallel)
        days = {(PRODUCTS[p]["product"], (init_dt + timedelta(hours=h + dh)).date())
                for h, p in need for dh in (-1, 0)}
        with ThreadPoolExecutor(max_workers=max(1, a.download_threads)) as ex:
            listed = dict(zip(days, ex.map(lambda pd: list_day(pd[0], datetime.combine(pd[1], datetime.min.time())),
                                           days)))
        files = {}
        for (prod, _), lst in listed.items():
            files.setdefault(prod, []).extend(lst)
        urls = {}
        missing_isu = []
        for h, p in need:
            valid = init_dt + timedelta(hours=h)
            url, ft = pick(files.get(PRODUCTS[p]["product"], []), PRODUCTS[p], valid, a.max_offset_min)
            if url:
                urls[(h, p)] = (url, ft)
            else:
                missing_isu.append((h, p))
        if missing_isu:                          # older cases: Iowa State archive
            with ThreadPoolExecutor(max_workers=max(1, a.download_threads)) as ex:
                res = ex.map(lambda hp: isu_url(PRODUCTS[hp[1]], init_dt + timedelta(hours=hp[0]), a.max_offset_min),
                             missing_isu)
                for (h, p), (url, ft) in zip(missing_isu, res):
                    if url:
                        urls[(h, p)] = (url, ft)
                    else:
                        logger.warning(f"f{h:02d} {p}: no {PRODUCTS[p]['product']} file near "
                                       f"{init_dt + timedelta(hours=h):%Y-%m-%d %H:%M}Z")
        logger.info(f"{len(urls)} of {len(need)} files found ({time.time() - t0:.0f}s); "
                    f"downloading and decoding")

        # 3. download (threads) -> decode + cut (processes), as each download finishes
        n_proc = max(1, min(a.workers or _cores(), len(urls) or 1))
        got = {}
        with ThreadPoolExecutor(max_workers=max(1, a.download_threads)) as dl, \
                ProcessPoolExecutor(max_workers=n_proc) as px:
            dls = {dl.submit(download, u): hp for hp, (u, _) in urls.items()}
            decs = {}
            for f in as_completed(dls):
                hp = dls[f]
                try:
                    decs[px.submit(decode_cut, f.result(), hp[1], a.grid, ref_lat, ref_lon)] = hp
                except Exception as e:
                    logger.error(f"f{hp[0]:02d} {hp[1]}: download failed ({e})")
            for f in as_completed(decs):
                hp = decs[f]
                try:
                    got[hp] = f.result()
                except Exception as e:
                    logger.error(f"f{hp[0]:02d} {hp[1]}: decode failed ({e})")
        logger.info(f"Decoded {len(got)} files with {n_proc} processes ({time.time() - t0:.0f}s)")

        # 4. write one file per hour (existing products kept)
        for h in sorted({h for h, _ in need}):
            new = {p: got[(h, p)] for p in a.products if (h, p) in got}
            old = have.get(h)
            fields, attrs = (dict(old[0]), dict(old[3])) if old else ({}, {})
            lat2 = lon2 = None
            if old and fields:
                lat2, lon2 = old[1], old[2]
            added = []
            for p, (vals, la2, lo2) in new.items():
                if lat2 is None:
                    lat2, lon2 = la2, lo2
                elif la2.shape != lat2.shape:
                    logger.warning(f"f{h:02d} {p}: grid {la2.shape} differs from {lat2.shape} in "
                                   f"hrrrcast_mrms_f{h:02d}.nc; use --overwrite (e.g. after changing --grid)")
                    continue
                url, ft = urls[(h, p)]
                spec = PRODUCTS[p]
                fields[p] = (vals, {"long_name": spec["long_name"], "units": spec["units"],
                                    "mrms_product": spec["product"],
                                    "mrms_file_time": f"{ft:%Y-%m-%dT%H:%M:%SZ}", "mrms_url": url})
                added.append(p)
            if not added:
                status[h] = "missing" if not fields else "partial"
                continue
            attrs.update(source="MRMS (observed)", grid=a.grid,
                         title="MRMS observations on the HRRRCast domain (get_mrms.py)")
            write_nc(case_dir / f"hrrrcast_mrms_f{h:02d}.nc", init_dt, h, lat2, lon2, fields, attrs, a.grid)
            logger.info(f"f{h:02d}: hrrrcast_mrms_f{h:02d}.nc + {' '.join(added)} ({lat2.shape[0]}x{lat2.shape[1]})")
            rewritten.add(h)
            status[h] = "ok" if len(fields) == len(a.products) else "partial"

    if "APCP" in a.products:
        write_accumulations(case_dir, init_dt, lead)

    # 5. GRIB2 on the members' 3 km grid (REFC, APCP) for verification; kept by the cleanup job
    if not a.no_grib2:
        n_grib = 0
        for h in range(a.start_hour, lead + 1):
            nc = case_dir / f"hrrrcast_mrms_f{h:02d}.nc"
            gb = case_dir / f"hrrrcast.mrms.t{hh}z.pgrb2.f{h:02d}"
            if not nc.exists():
                continue
            if h not in rewritten and not a.overwrite and grib2_grid_ok(gb, ref_lat.shape):
                continue                         # already there, on the right grid
            old = read_existing(nc)
            try:
                if old and write_grib2_3km(gb, init_dt, h, old[0], old[1], old[2], ref_lat, ref_lon):
                    n_grib += 1
            except Exception as e:
                logger.error(f"f{h:02d}: GRIB2 failed ({e})")
        if n_grib:
            logger.info(f"Wrote {n_grib} hrrrcast.mrms.t{hh}z.pgrb2.fHH files (REFC/APCP on the 3 km grid)")

    summary = {k: sum(v == k for v in status.values()) for k in ("ok", "complete", "partial", "future", "missing")}
    logger.info(f"MRMS {case_dir}: " + ", ".join(f"{k} {n}" for k, n in summary.items() if n)
                + f" ({time.time() - t0:.0f}s)")
    if summary["future"]:
        logger.info(f"{summary['future']} hour(s) not observed yet; rerun later to fill them in")
    return 0


if __name__ == "__main__":
    sys.exit(main())
