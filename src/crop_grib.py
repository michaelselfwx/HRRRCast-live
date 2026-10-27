#!/usr/bin/env python3
"""
Cut the HRRRCast GRIB2 output down to smaller verification domains.

For every hrrrcast.<mem>.tHHz.pgrb2.fHH in a case folder this writes a copy that holds only
a sub-box of the grid (same Lambert projection, same messages, same packing):

    YYYYMMDD/HH/<domain>/hrrrcast.<mem>.tHHz.pgrb2.fHH

Domains: any name from plot.py DOMAINS (e.g. hcfcd_hrrr / hcfcd_href: Harris County boxes, texas), or your own box with
--bbox lat_min,lat_max,lon_min,lon_max --name myname. The box is cut along the model grid:
the smallest grid rectangle that contains the domain corners (plus --pad points).

Uses wgrib2 -ijsmall_grib when wgrib2 is available ($WGRIB2 or on PATH), otherwise the
eccodes Python library (pip install eccodes). Every output file is checked: its grid
size and corner lat/lon must match the requested box.

Usage:
  python src/crop_grib.py 2026-10-03T16 --forecast_dir $SCRATCH/hrrrcast-data --domains hcfcd_hrrr hcfcd_href
  python src/crop_grib.py 2026-10-03T16 --forecast_dir ... --bbox 29.5,30.2,-95.9,-94.9 --name houston
  python src/crop_grib.py 2026-10-03T16 --forecast_dir ... --domains hcfcd_hrrr --members m00 m01 avg
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

FNAME_RE = re.compile(r"^hrrrcast\.([a-z0-9]+)\.t(\d\d)z\.pgrb2\.f(\d+)$")


def wrap(lon):
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def grid_of(path):
    import pygrib
    g = pygrib.open(path)
    try:
        m = g.message(1)
        if m["iScansNegatively"] != 0 or m["jScansPositively"] != 1:
            raise ValueError("only grids scanning west->east, south->north (like HRRR) are supported")
        lat, lon = m.latlons()
    finally:
        g.close()
    return np.asarray(lat), wrap(lon)


def box_from_corners(lat, lon, corners, pad):
    """Grid rectangle (r0, r1, c0, c1), inclusive 0-based, containing the nearest points to corners."""
    rows, cols = [], []
    for la, lo in corners:
        d = (lat - la) ** 2 + ((lon - lo) * np.cos(np.deg2rad(la))) ** 2
        r, c = np.unravel_index(np.argmin(d), d.shape)
        if np.sqrt(d[r, c]) > 0.05:
            raise ValueError(f"corner ({la}, {lo}) is outside this grid")
        rows.append(r); cols.append(c)
    ny, nx = lat.shape
    return (max(0, min(rows) - pad), min(ny - 1, max(rows) + pad),
            max(0, min(cols) - pad), min(nx - 1, max(cols) + pad))


def box_from_bbox(lat, lon, bbox, pad):
    la0, la1, lo0, lo1 = bbox
    inside = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    if not inside.any():
        raise ValueError(f"bbox {bbox} does not overlap this grid")
    rr, cc = np.nonzero(inside)
    ny, nx = lat.shape
    return (max(0, rr.min() - pad), min(ny - 1, rr.max() + pad),
            max(0, cc.min() - pad), min(nx - 1, cc.max() + pad))


def find_wgrib2():
    """$WGRIB2 or wgrib2 on PATH, but only if it actually runs (a conda wgrib2 built against a
    different libg2c dies with "undefined symbol: g2c_enc_aec", exit status 127)."""
    w = shutil.which(os.environ.get("WGRIB2", "wgrib2"))
    if not w:
        return None
    # "wgrib2 -version" prints the version but exits non-zero, so only treat it as broken
    # when it cannot start (missing library/symbol: exit 126/127 or a loader error message)
    try:
        r = subprocess.run([w, "-version"], capture_output=True, text=True, timeout=30)
        out = (r.stdout + r.stderr).strip()
        broken = (r.returncode in (126, 127) or "symbol lookup error" in out
                  or "error while loading shared libraries" in out or not out)
    except Exception as e:
        broken, out = True, str(e)
    if broken:
        print(f"wgrib2 at {w} does not run ({out.splitlines()[-1] if out else 'no output'}); "
              "set WGRIB2 to a working wgrib2", file=sys.stderr)
        return None
    return w


def crop_wgrib2(wgrib2, src, dst, box):
    r0, r1, c0, c1 = box                                   # wgrib2 i/j are 1-based, j=1 south
    tmp = dst + ".tmp"
    subprocess.run([wgrib2, src, "-set_grib_type", "same",
                    "-ijsmall_grib", f"{c0 + 1}:{c1 + 1}", f"{r0 + 1}:{r1 + 1}", tmp],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    os.replace(tmp, dst)
    with open(dst + ".idx", "w") as f:                     # inventory, like the originals
        subprocess.run([wgrib2, dst, "-s"], check=True, stdout=f, stderr=subprocess.DEVNULL)


def crop_eccodes(src, dst, box, lat, lon):
    import eccodes as ec
    r0, r1, c0, c1 = box
    ny_new, nx_new = r1 - r0 + 1, c1 - c0 + 1
    la1, lo1 = float(lat[r0, c0]), float(lon[r0, c0]) % 360.0
    tmp = dst + ".tmp"
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        while True:
            h = ec.codes_grib_new_from_file(fin)
            if h is None:
                break
            try:
                nx, ny = ec.codes_get(h, "Nx"), ec.codes_get(h, "Ny")
                vals = ec.codes_get_values(h).reshape(ny, nx)[r0:r1 + 1, c0:c1 + 1]
                missing = ~np.isfinite(vals)
                if ec.codes_get(h, "bitmapPresent"):
                    mv = ec.codes_get(h, "missingValue")
                    missing |= vals == mv
                ec.codes_set(h, "Nx", nx_new)
                ec.codes_set(h, "Ny", ny_new)
                ec.codes_set(h, "latitudeOfFirstGridPointInDegrees", la1)
                ec.codes_set(h, "longitudeOfFirstGridPointInDegrees", lo1)
                if missing.any():
                    ec.codes_set(h, "bitmapPresent", 1)
                    vals = np.where(missing, ec.codes_get(h, "missingValue"), vals)
                ec.codes_set_values(h, np.ascontiguousarray(vals, dtype=float).ravel())
                ec.codes_write(h, fout)
            finally:
                ec.codes_release(h)
    os.replace(tmp, dst)


def check(dst, box, lat, lon):
    """Output grid must be exactly the requested sub-box of the input grid."""
    import pygrib
    r0, r1, c0, c1 = box
    g = pygrib.open(dst)
    try:
        n = g.messages
        la, lo = g.message(1).latlons()
    finally:
        g.close()
    exp_la, exp_lo = lat[r0:r1 + 1, c0:c1 + 1], lon[r0:r1 + 1, c0:c1 + 1]
    if la.shape != exp_la.shape:
        raise RuntimeError(f"{os.path.basename(dst)}: grid {la.shape} != expected {exp_la.shape}")
    err = max(np.abs(la - exp_la).max(), np.abs(wrap(lo) - exp_lo).max())
    if err > 1e-3:
        raise RuntimeError(f"{os.path.basename(dst)}: lat/lon off by {err:.4f} deg")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inittime", help="YYYY-MM-DDTHH")
    ap.add_argument("--forecast_dir", default="./", help="data root (YYYYMMDD/HH/ under it)")
    ap.add_argument("--domains", nargs="+", default=[], help="named domains from plot.py DOMAINS, e.g. hcfcd_hrrr hcfcd_href texas")
    ap.add_argument("--bbox", default=None, help="lat_min,lat_max,lon_min,lon_max for a custom box")
    ap.add_argument("--name", default="bbox", help="folder name for --bbox (default: bbox)")
    ap.add_argument("--pad", type=int, default=0, help="extra grid points around the box (default 0)")
    ap.add_argument("--members", nargs="+", default=None, help="only these, e.g. m00 m01 avg spr (default: all)")
    ap.add_argument("--engine", choices=["auto", "wgrib2", "eccodes"], default="auto")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2})$", a.inittime)
    if not m:
        sys.exit("inittime must be YYYY-MM-DDTHH")
    case_dir = os.path.join(a.forecast_dir, f"{m[1]}{m[2]}{m[3]}", m[4])
    files = sorted(f for f in glob.glob(os.path.join(case_dir, "hrrrcast.*.pgrb2.f*"))
                   if FNAME_RE.match(os.path.basename(f))
                   and (not a.members or FNAME_RE.match(os.path.basename(f))[1] in a.members))
    if not files:
        sys.exit(f"No hrrrcast.<mem>.tHHz.pgrb2.fHH files in {case_dir}")

    wgrib2 = find_wgrib2() if a.engine in ("auto", "wgrib2") else None
    if a.engine == "wgrib2" and not wgrib2:
        sys.exit("wgrib2 not found (set $WGRIB2 or put it on PATH)")
    engine = "wgrib2" if wgrib2 else "eccodes"
    if engine == "eccodes":
        try:
            import eccodes  # noqa: F401
        except ImportError:
            sys.exit("No working wgrib2 and no eccodes Python package: set WGRIB2 to a working "
                     "wgrib2 (e.g. export WGRIB2=$HOME/.conda/envs/wgrib2/bin/wgrib2) or pip install eccodes")

    lat, lon = grid_of(files[0])
    targets = []
    if a.domains:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from plot import DOMAINS, DOMAIN_ALIASES
        a.domains = [DOMAIN_ALIASES.get(d.lower(), d.lower()) for d in a.domains]
        for d in a.domains:
            spec = DOMAINS.get(d.lower())
            if not spec:
                sys.exit(f"Unknown domain '{d}' (plot.py DOMAINS: {', '.join(k for k, v in DOMAINS.items() if v)})")
            targets.append((d.lower(), box_from_corners(lat, lon, spec["corners"], a.pad)))
    if a.bbox:
        bb = [float(x) for x in a.bbox.split(",")]
        targets.append((a.name, box_from_bbox(lat, lon, bb, a.pad)))
    if not targets:
        sys.exit("Give --domains and/or --bbox")

    print(f"{case_dir}: {len(files)} GRIB2 files, engine {engine}")
    for name, (r0, r1, c0, c1) in targets:
        print(f"  {name}: rows {r0}-{r1}, cols {c0}-{c1} -> {r1 - r0 + 1} x {c1 - c0 + 1} points, "
              f"lat {lat[r0:r1+1, c0:c1+1].min():.3f}..{lat[r0:r1+1, c0:c1+1].max():.3f}, "
              f"lon {lon[r0:r1+1, c0:c1+1].min():.3f}..{lon[r0:r1+1, c0:c1+1].max():.3f}")

    def job(src, name, box):
        out_dir = os.path.join(case_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, os.path.basename(src))
        if os.path.exists(dst) and not a.overwrite:
            return "skip"
        if engine == "wgrib2":
            crop_wgrib2(wgrib2, src, dst, box)
        else:
            crop_eccodes(src, dst, box, lat, lon)
        check(dst, box, lat, lon)
        return "ok"

    done = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [(src, name, ex.submit(job, src, name, box)) for name, box in targets for src in files]
        for src, name, f in futs:
            try:
                r = f.result()
                done += r == "ok"; skipped += r == "skip"
            except Exception as e:
                failed += 1
                print(f"  FAILED {name}/{os.path.basename(src)}: {e}", file=sys.stderr)
    print(f"Done: {done} written, {skipped} already there, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
