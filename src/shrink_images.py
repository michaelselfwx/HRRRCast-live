#!/usr/bin/env python3
"""
Shrink existing plot PNGs in place (same names and folders, so the viewer keeps working).

Each PNG wider than --max_width is resized to that width, then saved as an 8-bit palette PNG
(the plots are flat contour colours, so 256 colours look the same) with compression on.
300-dpi plots (~3260 px wide, ~1 MB) come out around 1100 px and ~100-200 kB.

Usage:
  python src/shrink_images.py /path/to/plots --dry_run      # report only
  python src/shrink_images.py /path/to/plots                # shrink everything below it
  python src/shrink_images.py /path/to/plots --max_width 1400 --workers 16
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image


def shrink(path: str, max_width: int, colors: int, dry_run: bool):
    """Returns (bytes before, bytes after, changed?)."""
    before = os.path.getsize(path)
    with Image.open(path) as im:
        w, h = im.size
        if w <= max_width and im.mode == "P":
            return before, before, False              # already done
        im = im.convert("RGB")                         # plots have no real transparency
        if w > max_width:
            im = im.resize((max_width, round(h * max_width / w)), Image.LANCZOS)
        im = im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    if dry_run:
        import io
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return before, buf.tell(), True
    tmp = path + ".tmp.png"
    im.save(tmp, format="PNG", optimize=True)
    after = os.path.getsize(tmp)
    if after >= before:                                # never make a file bigger
        os.remove(tmp)
        return before, before, False
    os.replace(tmp, path)
    return before, after, True


def _job(args):
    try:
        return shrink(*args)
    except Exception as e:                             # one bad file shouldn't stop the run
        print(f"  skipped {args[0]}: {e}", file=sys.stderr)
        return 0, 0, False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir", help="folder holding the YYYYMMDD/HH plot folders (searched recursively)")
    ap.add_argument("--max_width", type=int, default=1100, help="max image width in px (default 1100)")
    ap.add_argument("--colors", type=int, default=256, help="palette size (default 256)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--dry_run", action="store_true", help="report the savings without changing files")
    a = ap.parse_args()

    files = [str(p) for p in Path(a.base_dir).rglob("*.png") if not p.name.endswith(".tmp.png")]
    if not files:
        print(f"No PNG files under {a.base_dir}")
        return
    print(f"{len(files)} PNG files under {a.base_dir}" + (" (dry run)" if a.dry_run else ""))
    total_b = total_a = changed = 0
    jobs = [(f, a.max_width, a.colors, a.dry_run) for f in files]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (b, aft, ch) in enumerate(ex.map(_job, jobs, chunksize=16), 1):
            total_b += b; total_a += aft; changed += ch
            if i % 500 == 0 or i == len(files):
                print(f"  {i}/{len(files)}: {total_b / 1e9:.2f} GB -> {total_a / 1e9:.2f} GB")
    verb = "would shrink" if a.dry_run else "shrank"
    print(f"{verb} {changed} files: {total_b / 1e9:.2f} GB -> {total_a / 1e9:.2f} GB "
          f"({100 * (1 - total_a / max(total_b, 1)):.0f}% smaller)")


if __name__ == "__main__":
    main()
