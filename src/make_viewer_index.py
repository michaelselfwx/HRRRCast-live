#!/usr/bin/env python3
"""
Build the HRRRCast forecast viewer.

Scans plot.py output (<base_dir>/YYYYMMDD/HH/<member>_leadNNh/<PRODUCT>_leadNNh.png, and
<base_dir>/YYYYMMDD/HH/<domain>/<member>_leadNNh/... for extra plot domains such as hcfcd),
writes <base_dir>/viewer_manifest.js describing what exists, and copies the viewer page
to <base_dir>/viewer.html.

Usage:
    python src/make_viewer_index.py                 # base_dir = ./
    python src/make_viewer_index.py --base_dir /data/hrrrcast
Then open viewer.html (double-click works) or run `python -m http.server` in base_dir.

Images hosted somewhere else (viewer page in one web folder, plots at another public URL):
    python src/make_viewer_index.py --base_dir /data/hrrrcast \
        --image_base_url https://example.edu/hrrrcast-plots/ --output_dir /var/www/html/viewer
--base_dir is a local copy of the plot folders (only scanned, to list what exists); the
viewer loads every image from <image_base_url>/YYYYMMDD/HH/... instead of next to viewer.html.
"""

import argparse
import fnmatch
import os
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

CASE_RE = re.compile(r"^\d{8}$")
HOUR_RE = re.compile(r"^\d{2}$")
# member folder: m00 / memm00 (older plot.py) / mem0 / avg / memavg / spr ...  + _leadNNh
MEMDIR_RE = re.compile(r"^(?:mem)?(m?\d+|avg|spr|pmm|lpmm|hrrr)_lead(\d+)h$")  # m00, older memm00 / mem0, avg, spr
PNG_RE = re.compile(r"^(.+)_lead(\d+)h\.png$")


def product_wanted(product: str, patterns) -> bool:
    """Same matching as plot.py --products: full name or variable name, shell wildcards."""
    if not patterns:
        return True
    var = product[:-len("_surface")] if product.endswith("_surface") else product
    if var.endswith("hPa") and "_" in var:
        var = var.rsplit("_", 1)[0]
    base = re.sub(r"_(TOT|\d+H)$", "", var)  # APCP_TOT / APCP_6H also match "APCP"
    return any(fnmatch.fnmatchcase(product, p) or fnmatch.fnmatchcase(var, p) or fnmatch.fnmatchcase(base, p)
               for p in patterns)


def member_id(raw: str) -> str:
    if raw in ("avg", "pmm"):
        return "avg"
    if raw in ("spr", "lpmm", "hrrr"):
        return raw
    return f"m{int(raw.lstrip('m')):02d}"


def member_sort_key(m: str):
    # viewer order: operational HRRR (reference), members m00.., PMM/mean, LPMM, spread last
    special = {"hrrr": (-1, 0), "avg": (1, 0), "lpmm": (1, 1), "spr": (1, 2)}
    if m in special:
        return special[m]
    if m.startswith("m"):
        return (0, int(m[1:]))
    return (2, 0)


DOMAIN_RE = re.compile(r"^[a-z][a-z0-9_]*$")   # plot.py --domains subfolders, e.g. hcfcd/
DEFAULT_DOMAIN = "tx"                          # plots directly in YYYYMMDD/HH/


def scan_dir(folder: Path, products=None) -> dict:
    """Members / leads / products of the <member>_leadNNh folders directly inside folder."""
    # (member, folder prefix) -> {product: {leads}}; one member can appear under several
    # prefixes (e.g. old "mem0_" and newer "memm00_" folders), so keep the fullest one.
    found = {}
    for d in folder.iterdir():
        m = MEMDIR_RE.match(d.name) if d.is_dir() else None
        if not m:
            continue
        mem, lead = member_id(m.group(1)), int(m.group(2))
        prefix = d.name[: d.name.rindex("_lead")]
        prods = found.setdefault((mem, prefix), {})
        for f in d.iterdir():
            pm = PNG_RE.match(f.name)
            if pm and product_wanted(pm.group(1), products):
                prods.setdefault(pm.group(1), set()).add(lead)
    members, case_prods, leads = {}, {}, set()
    best = {}
    for (mem, prefix), prods in found.items():
        n = sum(len(v) for v in prods.values())
        if n and (mem not in best or n > best[mem][0]):
            best[mem] = (n, prefix)
    for mem, (_, prefix) in best.items():
        members[mem] = prefix
        for prod, ls in found[(mem, prefix)].items():
            case_prods.setdefault(prod, {})[mem] = ls
            leads |= ls
    if not case_prods:
        return {}
    return {
        "members": sorted(members, key=member_sort_key),
        "memdirs": members,
        "leads": sorted(leads),
        "products": {
            p: {mem: sorted(ls) for mem, ls in sorted(v.items(), key=lambda kv: member_sort_key(kv[0]))}
            for p, v in sorted(case_prods.items())
        },
    }


def scan(base: Path, products=None) -> dict:
    """{case: {"domains": {domain: {..., "prefix": subfolder}}}}"""
    cases = {}
    for day in sorted(p for p in base.iterdir() if p.is_dir() and CASE_RE.match(p.name)):
        for hh in sorted(p for p in day.iterdir() if p.is_dir() and HOUR_RE.match(p.name)):
            domains = {}
            top = scan_dir(hh, products)
            if top:
                domains[DEFAULT_DOMAIN] = {**top, "prefix": ""}
            for sub in sorted(p for p in hh.iterdir() if p.is_dir() and DOMAIN_RE.match(p.name)):
                if MEMDIR_RE.match(sub.name):
                    continue
                info = scan_dir(sub, products)
                if info:
                    domains[sub.name] = {**info, "prefix": sub.name + "/"}
            if domains:
                cases[f"{day.name}/{hh.name}"] = {"domains": domains}
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", default="./", help="Directory holding YYYYMMDD/HH plot folders (plot.py --output_dir)")
    ap.add_argument("--products", nargs="+", default=None,
                    help="Only list these products in the viewer (names or wildcards, same as plot.py). "
                         "Default: $HRRRCAST_PLOT_PRODUCTS if set, else everything found")
    ap.add_argument("--image_base_url", default="https://hdwx.tamu.edu/products/wxgen3/HRRRCast/",
                    help="Public URL the YYYYMMDD/HH plot folders are served from, e.g. "
                         "https://example.edu/hrrrcast/ (default: images next to viewer.html)")
    ap.add_argument("--output_dir", default=None,
                    help="Where to write viewer.html + viewer_manifest.js (default: --base_dir)")
    args = ap.parse_args()
    base = Path(args.base_dir).resolve()
    out = Path(args.output_dir).resolve() if args.output_dir else base
    out.mkdir(parents=True, exist_ok=True)
    image_base = args.image_base_url.strip()
    if image_base and not image_base.endswith("/"):
        image_base += "/"
    raw = args.products or os.environ.get("HRRRCAST_PLOT_PRODUCTS", "").split()
    products = [p for v in raw for p in v.replace(",", " ").split()] or None

    cases = scan(base, products)
    if not cases:
        print(f"No plot folders found under {base} (expected YYYYMMDD/HH/<member>_leadNNh/*.png)")
        sys.exit(1)

    manifest = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "image_base": image_base, "cases": cases}
    (out / "viewer_manifest.js").write_text(
        "window.HRRRCAST_MANIFEST = " + json.dumps(manifest, separators=(",", ":")) + ";\n"
    )
    template = Path(__file__).with_name("viewer_template.html")
    shutil.copyfile(template, out / "viewer.html")

    n_img = sum(len(ls) for c in cases.values() for dm in c["domains"].values()
                for p in dm["products"].values() for ls in p.values())
    print(f"Indexed {len(cases)} case(s), {n_img} images -> {out / 'viewer_manifest.js'}")
    if image_base:
        print(f"Images will load from {image_base}YYYYMMDD/HH/...")
    for k, c in cases.items():
        for name, dm in c["domains"].items():
            print(f"  {k} [{name}]: members {', '.join(dm['members'])}; "
                  f"leads f{dm['leads'][0]:02d}-f{dm['leads'][-1]:02d}; {len(dm['products'])} products")
    print(f"Open {out / 'viewer.html'}  (or: cd {out} && python -m http.server, then http://localhost:8000/viewer.html)")


if __name__ == "__main__":
    main()
