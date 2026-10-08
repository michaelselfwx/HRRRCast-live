#!/usr/bin/env python3
"""
Build the HRRRCast forecast viewer.

Scans plot.py output (<base_dir>/YYYYMMDD/HH/<member>_leadNNh/<PRODUCT>_leadNNh.png),
writes <base_dir>/viewer_manifest.js describing what exists, and copies the viewer page
to <base_dir>/viewer.html.

Usage:
    python src/make_viewer_index.py                 # base_dir = ./
    python src/make_viewer_index.py --base_dir /data/hrrrcast
Then open viewer.html (double-click works) or run `python -m http.server` in base_dir.
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
    if m == "hrrr":
        return (-1, 0)          # operational HRRR first, as the reference
    if m.startswith("m"):
        return (0, int(m[1:]))
    return (1, {"avg": 0, "lpmm": 1, "spr": 2}.get(m, 9))


def scan(base: Path, products=None) -> dict:
    cases = {}
    for day in sorted(p for p in base.iterdir() if p.is_dir() and CASE_RE.match(p.name)):
        for hh in sorted(p for p in day.iterdir() if p.is_dir() and HOUR_RE.match(p.name)):
            case_key = f"{day.name}/{hh.name}"
            # (member, folder prefix) -> {product: {leads}}; one member can appear under several
            # prefixes (e.g. old "mem0_" and newer "memm00_" folders), so keep the fullest one.
            found = {}
            for d in hh.iterdir():
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
                continue
            cases[case_key] = {
                "members": sorted(members, key=member_sort_key),
                "memdirs": members,
                "leads": sorted(leads),
                "products": {
                    p: {mem: sorted(ls) for mem, ls in sorted(v.items(), key=lambda kv: member_sort_key(kv[0]))}
                    for p, v in sorted(case_prods.items())
                },
            }
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", default="./", help="Directory holding YYYYMMDD/HH plot folders (plot.py --output_dir)")
    ap.add_argument("--products", nargs="+", default=None,
                    help="Only list these products in the viewer (names or wildcards, same as plot.py). "
                         "Default: $HRRRCAST_PLOT_PRODUCTS if set, else everything found")
    args = ap.parse_args()
    base = Path(args.base_dir).resolve()
    raw = args.products or os.environ.get("HRRRCAST_PLOT_PRODUCTS", "").split()
    products = [p for v in raw for p in v.replace(",", " ").split()] or None

    cases = scan(base, products)
    if not cases:
        print(f"No plot folders found under {base} (expected YYYYMMDD/HH/<member>_leadNNh/*.png)")
        sys.exit(1)

    manifest = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "cases": cases}
    (base / "viewer_manifest.js").write_text(
        "window.HRRRCAST_MANIFEST = " + json.dumps(manifest, separators=(",", ":")) + ";\n"
    )
    template = Path(__file__).with_name("viewer_template.html")
    shutil.copyfile(template, base / "viewer.html")

    n_img = sum(len(ls) for c in cases.values() for p in c["products"].values() for ls in p.values())
    print(f"Indexed {len(cases)} case(s), {n_img} images -> {base / 'viewer_manifest.js'}")
    for k, c in cases.items():
        print(f"  {k}: members {', '.join(c['members'])}; leads f{c['leads'][0]:02d}-f{c['leads'][-1]:02d}; "
              f"{len(c['products'])} products")
    print(f"Open {base / 'viewer.html'}  (or: cd {base} && python -m http.server, then http://localhost:8000/viewer.html)")


if __name__ == "__main__":
    main()
