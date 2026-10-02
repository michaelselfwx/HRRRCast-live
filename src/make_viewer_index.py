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
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

CASE_RE = re.compile(r"^\d{8}$")
HOUR_RE = re.compile(r"^\d{2}$")
# member folder: m00 / memm00 (older plot.py) / mem0 / avg / memavg / spr ...  + _leadNNh
MEMDIR_RE = re.compile(r"^(?:mem)?(m?\d+|avg|spr|pmm)_lead(\d+)h$")
PNG_RE = re.compile(r"^(.+)_lead(\d+)h\.png$")


def member_id(raw: str) -> str:
    if raw in ("avg", "pmm"):
        return "avg"
    if raw == "spr":
        return "spr"
    return f"m{int(raw.lstrip('m')):02d}"


def member_sort_key(m: str):
    if m.startswith("m"):
        return (0, int(m[1:]))
    return (1, {"avg": 0, "spr": 1}.get(m, 9))


def scan(base: Path) -> dict:
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
                    if pm:
                        prods.setdefault(pm.group(1), set()).add(lead)
            members, products, leads = {}, {}, set()
            best = {}
            for (mem, prefix), prods in found.items():
                n = sum(len(v) for v in prods.values())
                if n and (mem not in best or n > best[mem][0]):
                    best[mem] = (n, prefix)
            for mem, (_, prefix) in best.items():
                members[mem] = prefix
                for prod, ls in found[(mem, prefix)].items():
                    products.setdefault(prod, {})[mem] = ls
                    leads |= ls
            if not products:
                continue
            cases[case_key] = {
                "members": sorted(members, key=member_sort_key),
                "memdirs": members,
                "leads": sorted(leads),
                "products": {
                    p: {mem: sorted(ls) for mem, ls in sorted(v.items(), key=lambda kv: member_sort_key(kv[0]))}
                    for p, v in sorted(products.items())
                },
            }
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", default="./", help="Directory holding YYYYMMDD/HH plot folders (plot.py --output_dir)")
    args = ap.parse_args()
    base = Path(args.base_dir).resolve()

    cases = scan(base)
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
