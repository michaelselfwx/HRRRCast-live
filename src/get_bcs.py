#!/usr/bin/env python3
"""
GFS Lateral Boundary Conditions Downloader
Downloads GFS GRIB2 files for lateral boundary conditions.
"""

import argparse
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import utils
from utils import setup_logging, create_output_directory, download_file_with_retry

# -------------------------------
# Configuration
# -------------------------------
class Config:
    """Configuration class for GFS data downloader."""
    
    # Base URLs
    # AWS Open Data: 0.25 deg, hourly forecast output. Archive starts in 2021.
    GFS_BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
    # NCEI THREDDS archive (https://www.ncei.noaa.gov/thredds/catalog/model/gfs.html), output every 3 h.
    # Coverage differs a lot by month, so files are located via each day's catalog.xml:
    #   *-004-files[-old]      gfs_4_*    0.5 deg forecasts (full range only ~2019-08..2020-05)
    #   *-g4-anl-files[-old]   gfsanl_4_* 0.5 deg analysis set, f000-f006 only (2004-03..)
    #   *-003-files[-old]      gfs_3_*    1.0 deg forecasts
    #   *-g3-anl-files[-old]   gfsanl_3_* 1.0 deg analysis set, f000-f006 only
    # Earlier collections win when the same (grid, cycle, fh) appears in several.
    THREDDS_BASE_URL = "https://www.ncei.noaa.gov/thredds"
    THREDDS_COLLECTIONS = [
        "model-gfs-004-files-old", "model-gfs-004-files",
        "model-gfs-g4-anl-files-old", "model-gfs-g4-anl-files",
        "model-gfs-003-files-old", "model-gfs-003-files",
        "model-gfs-g3-anl-files-old", "model-gfs-g3-anl-files",
    ]
    THREDDS_GRID_PREFERENCE = ["4", "3"]   # grid 4 = 0.5 deg first, grid 3 = 1.0 deg as last resort
    THREDDS_STEP_HOURS = 3
    # Cycles before this date default to THREDDS when --source auto
    AWS_FIRST_DATE = datetime(2021, 1, 1)
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    TIMEOUT = 300    # seconds


# -------------------------------
# GFS Download Functions
# -------------------------------
def get_gfs_cycle_and_hours(year: str, month: str, day: str, hour: str, lead_hours: int) -> Tuple[datetime, List[int]]:
    """Pick the GFS cycle at or before the init time and list the forecast hours needed.

    Returns (cycle_datetime, forecast_hours). Forecast hours cover init+1 .. init+lead_hours
    (hour 0 skipped), plus the next synoptic hour after the last valid time if that is not
    synoptic (make_bcs uses it for APCP).
    """
    hour_int = int(hour)
    cycle_hours = [0, 6, 12, 18]

    # Find the appropriate GFS cycle (must be synoptic hour for initialization)
    if hour_int in cycle_hours:
        init_cycle = hour_int
        init_date_str = f"{year}{month}{day}"
    else:
        # Use the most recent synoptic hour
        previous_cycle = max([c for c in cycle_hours if c < hour_int], default=18)
        if previous_cycle >= hour_int:
            # Need to go to previous day
            dt = datetime(int(year), int(month), int(day)) - timedelta(days=1)
            init_date_str = dt.strftime("%Y%m%d")
            init_cycle = 18
        else:
            init_date_str = f"{year}{month}{day}"
            init_cycle = previous_cycle

    # Calculate forecast hours needed
    if hour_int in cycle_hours:
        start_forecast_hour = 0
    else:
        # Calculate offset from the initialization cycle
        if init_date_str != f"{year}{month}{day}":
            # Previous day's 18Z cycle
            start_forecast_hour = (24 - 18) + hour_int
        else:
            start_forecast_hour = hour_int - init_cycle

    init_dt = datetime(int(init_date_str[:4]), int(init_date_str[4:6]), int(init_date_str[6:8]), init_cycle)
    fhs = [fh_ + start_forecast_hour for fh_ in range(1, lead_hours + 1)]

    if lead_hours > 0:
        # if the last valid time is not a synoptic hour, also get the next synoptic hour file
        valid_dt = init_dt + timedelta(hours=fhs[-1])
        if valid_dt.hour not in cycle_hours:
            next_syn_hour = min([c for c in cycle_hours if c > valid_dt.hour], default=0)
            if next_syn_hour == 0:
                next_syn_dt = valid_dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_syn_dt = valid_dt.replace(hour=next_syn_hour, minute=0, second=0, microsecond=0)
            fhs.append(int((next_syn_dt - init_dt).total_seconds() // 3600))

    return init_dt, fhs


def _gfs_filename(init_dt: datetime, fh: int) -> str:
    """Local file name keyed by valid time (what make_bcs expects)."""
    valid_dt = init_dt + timedelta(hours=fh)
    return f"gfs_{valid_dt.strftime('%Y%m%d_%H')}.grib2"


def aws_url(init_dt: datetime, fh: int) -> str:
    d, c = init_dt.strftime("%Y%m%d"), init_dt.strftime("%H")
    return f"{Config.GFS_BASE_URL}/gfs.{d}/{c}/atmos/gfs.t{c}z.pgrb2.0p25.f{fh:03d}"


def thredds_forecast_hours(fhs: List[int]) -> List[int]:
    """THREDDS files are every 3 h: replace each hour with the 3-hourly files bracketing it.

    make_bcs linearly interpolates in time for the hours in between.
    """
    step = Config.THREDDS_STEP_HOURS
    out = set()
    for fh in fhs:
        lo = (fh // step) * step
        out.add(lo)
        if fh != lo:
            out.add(lo + step)
    return sorted(out)


_THREDDS_NS = "{http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0}"
_THREDDS_FILE_RE = re.compile(r"^(?:gfs|gfsanl)_(\d)_(\d{8})_(\d{2})00_(\d{3})\.grb2$")
_thredds_day_cache: Dict[str, Dict[Tuple[str, int, int], str]] = {}


def thredds_day_index(day: datetime) -> Dict[Tuple[str, int, int], str]:
    """Index of GFS files NCEI THREDDS holds for one day: (grid, cycle_hour, fh) -> download URL.

    Collections are searched in Config.THREDDS_COLLECTIONS order; the first hit wins.
    """
    logger = logging.getLogger(__name__)
    key = day.strftime("%Y%m%d")
    if key in _thredds_day_cache:
        return _thredds_day_cache[key]
    index: Dict[Tuple[str, int, int], str] = {}
    for coll in Config.THREDDS_COLLECTIONS:
        cat_url = f"{Config.THREDDS_BASE_URL}/catalog/{coll}/{day:%Y%m}/{key}/catalog.xml"
        try:
            resp = requests.get(cat_url, timeout=Config.TIMEOUT)
            if resp.status_code == 404:
                logger.debug(f"No THREDDS catalog: {cat_url}")
                continue
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as e:
            logger.warning(f"Could not read THREDDS catalog {cat_url}: {e}")
            continue
        n = 0
        for ds in root.iter(f"{_THREDDS_NS}dataset"):
            url_path = ds.get("urlPath")
            if not url_path:
                continue
            m = _THREDDS_FILE_RE.match(os.path.basename(url_path))
            if not m:
                continue
            grid, _, cyc, fh = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            index.setdefault((grid, cyc, fh), f"{Config.THREDDS_BASE_URL}/fileServer/{url_path}")
            n += 1
        if n:
            logger.info(f"THREDDS {coll}/{key}: {n} GFS files")
    _thredds_day_cache[key] = index
    return index


def _thredds_lookup(prefix: str, cycle_dt: datetime, fh: int) -> Optional[str]:
    return thredds_day_index(cycle_dt).get((prefix, cycle_dt.hour, fh))


def plan_thredds(init_dt: datetime, fhs: List[int], stitch: bool
                 ) -> Tuple[Optional[str], List[Tuple[str, str]], List[str], List[str]]:
    """Choose one grid (prefix) and the THREDDS files to download.

    Returns (prefix, [(url, local filename)], notes, missing). A single grid is used for the
    whole case because make_bcs builds one set of regrid weights per run. With stitch=True a
    valid time missing from the GFS cycle is taken from the earliest later cycle that has it
    (shorter lead; this uses information from after the init time).
    """
    best = None
    for prefix in Config.THREDDS_GRID_PREFERENCE:
        urls, notes, missing = [], [], []
        for fh in thredds_forecast_hours(fhs):
            valid = init_dt + timedelta(hours=fh)
            url = _thredds_lookup(prefix, init_dt, fh)
            if url is None and stitch:
                cyc = init_dt + timedelta(hours=6)
                while url is None and cyc <= valid:
                    fh2 = int((valid - cyc).total_seconds() // 3600)
                    url = _thredds_lookup(prefix, cyc, fh2)
                    if url is not None:
                        notes.append(f"{valid:%Y-%m-%d %H}Z from {cyc:%Y-%m-%d %H}Z cycle f{fh2:03d} (stitched)")
                    cyc += timedelta(hours=6)
            if url is None:
                missing.append(f"{valid:%Y-%m-%d %H}Z (grid {prefix} {init_dt:%Y%m%d %H}Z f{fh:03d})")
            else:
                urls.append((url, _gfs_filename(init_dt, fh)))
        if not missing:
            return prefix, urls, notes, []
        if best is None or len(missing) < len(best[3]):
            best = (prefix, urls, notes, missing)
    return best


def get_gfs_urls(year: str, month: str, day: str, hour: str, lead_hours: int,
                 source: str = "aws", stitch: bool = False) -> List[Tuple[str, str]]:
    """Generate (url, local filename) pairs for one source ('aws' or 'thredds'), skipping hour 0."""
    logger = logging.getLogger(__name__)
    init_dt, fhs = get_gfs_cycle_and_hours(year, month, day, hour, lead_hours)
    if source == "thredds":
        prefix, urls, notes, missing = plan_thredds(init_dt, fhs, stitch)
        grid = {"4": "0.5 deg", "3": "1.0 deg"}.get(prefix, prefix)
        logger.info(f"THREDDS grid {prefix} ({grid}), cycle {init_dt:%Y-%m-%d %H}Z")
        if prefix == "3":
            logger.warning("Only 1.0 deg GFS available for this case; forcing will be coarser than training data")
        for n in notes:
            logger.warning(n)
        if missing:
            logger.error("Not on THREDDS: " + "; ".join(missing))
            if not stitch:
                logger.error("NCEI only keeps f000-f006 for many months (the analysis set). Re-run with "
                             "--stitch_cycles to fill these from later GFS cycles")
        return urls
    return [(aws_url(init_dt, fh), _gfs_filename(init_dt, fh)) for fh in fhs]


def resolve_source(source: str, year: str, month: str, day: str, hour: str, lead_hours: int) -> str:
    """'auto' -> 'thredds' for cycles before Config.AWS_FIRST_DATE, else 'aws'."""
    if source != "auto":
        return source
    init_dt, _ = get_gfs_cycle_and_hours(year, month, day, hour, lead_hours)
    return "thredds" if init_dt < Config.AWS_FIRST_DATE else "aws"


def _download_set(urls: List[Tuple[str, str]], output_dir: Path) -> List[bool]:
    """Download a list of (url, filename) pairs in parallel; returns per-file success."""
    logger = logging.getLogger(__name__)
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_url = {
            executor.submit(download_file_with_retry, url, str(output_dir / filename)): (url, filename)
            for url, filename in urls
        }
        for future in as_completed(future_to_url):
            url, filename = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
                if result:
                    logger.info(f"Downloaded: {filename}")
                else:
                    # don't leave a truncated/empty file behind for make_bcs to trip over
                    try:
                        (output_dir / filename).unlink()
                    except FileNotFoundError:
                        pass
            except Exception as e:
                logger.error(f"Error downloading {filename}: {e}")
                results.append(False)
    return results


def download_gfs_files(year: str, month: str, day: str, hour: str, lead_hours: int, output_dir: Path,
                       source: str = "auto", stitch: bool = False) -> List[bool]:
    """Download GFS GRIB2 files for boundary conditions.

    source: 'aws' (0.25 deg hourly, 2021+), 'thredds' (NCEI 0.5/1.0 deg 3-hourly archive), or
    'auto' (pick by date; if the chosen source yields nothing, try the other one).
    """
    logger = logging.getLogger(__name__)
    chosen = resolve_source(source, year, month, day, hour, lead_hours)
    logger.info(f"GFS source: {chosen}" + (" (auto)" if source == "auto" else ""))
    
    if lead_hours == 0:
        logger.info(f"Downloading GFS data for {year}-{month}-{day} {hour}:00 UTC")
    else:
        logger.info(f"Downloading GFS boundary conditions: {year}-{month}-{day} {hour}:00 UTC + {lead_hours} hours")
    
    urls = get_gfs_urls(year, month, day, hour, lead_hours, source=chosen, stitch=stitch)
    logger.info(f"Total files to download: {len(urls)}")
    for url in urls:
        logger.info(f"{url[0]} -> {url[1]}")

    results = _download_set(urls, output_dir) if urls else []

    init_dt, _ = get_gfs_cycle_and_hours(year, month, day, hour, lead_hours)
    if source == "auto" and not any(results) and (chosen == "aws" or init_dt >= Config.AWS_FIRST_DATE):
        other = "aws" if chosen == "thredds" else "thredds"
        logger.warning(f"No files from {chosen}; retrying with {other}")
        urls = get_gfs_urls(year, month, day, hour, lead_hours, source=other, stitch=stitch)
        for url in urls:
            logger.info(f"{url[0]} -> {url[1]}")
        results = _download_set(urls, output_dir)

    logger.info(f"GFS downloads completed: {sum(results)}/{len(results)} successful")
    return results

# -------------------------------
# Main Functions
# -------------------------------
def download_gfs_data(datetime_str: str, lead_hours: int, base_dir: str = "./", source: str = "auto",
                      stitch: bool = False) -> dict:
    """Download GFS boundary condition data for specified date and time."""
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    init_datetime, year, month, day, hour = utils.validate_datetime(datetime_str)
    date_str = f"{year}{month}{day}/{hour}"
    
    # Create output directory
    output_dir = create_output_directory(base_dir, date_str)
    logger.info(f"Output directory: {output_dir}")
    
    results = {'gfs': []}
    
    # Download GFS data
    try:
        gfs_results = download_gfs_files(year, month, day, hour, lead_hours, output_dir, source=source, stitch=stitch)
        results['gfs'] = gfs_results
    except Exception as e:
        logger.error(f"Error downloading GFS data: {e}")
        results['gfs'] = [False]
    
    return results

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download GFS lateral boundary conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('inittime',
                       help='Forecast initialization time in format YYYY-MM-DDTHH (e.g., "2024-05-06T23")')
    parser.add_argument('lead_hours', type=int, help='Lead time in hours for boundary conditions')
    parser.add_argument('--base_dir', default='./', help='Base directory for downloads (default: ./)')
    parser.add_argument('--source', default='auto', choices=['auto', 'aws', 'thredds'],
                       help="GFS source: aws (0.25 deg hourly, 2021+), thredds (NCEI 0.5 deg 3-hourly archive), "
                            "auto = thredds for cycles before 2021-01-01, else aws (default: auto)")
    parser.add_argument('--stitch_cycles', action='store_true',
                       help="THREDDS only: fill valid times missing from the GFS cycle with short leads from "
                            "later cycles (some months only keep f000-f006). Uses post-init information.")
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Validate lead_hours
    if args.lead_hours < 0:
        logger.error("Lead hours must be >= 0")
        sys.exit(1)
    
    try:
        # Download GFS data
        results = download_gfs_data(
            args.inittime, args.lead_hours, args.base_dir, source=args.source,
            stitch=args.stitch_cycles
        )
        
        # Summary
        total_successful = sum(results['gfs'])
        total_attempted = len(results['gfs'])
        
        logger.info(f"Download summary: {total_successful}/{total_attempted} files successful")
        
        if total_successful == 0:
            logger.error("No files were downloaded successfully")
            sys.exit(1)
        elif total_successful < total_attempted:
            logger.warning("Some downloads failed")
            sys.exit(2)
        else:
            logger.info("All downloads completed successfully")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    main()
