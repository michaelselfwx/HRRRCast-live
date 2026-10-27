"""
GRIB2 writer using grib2io for HRRRCast outputs.

This module converts NetCDF forecast output to GRIB2 format using grib2io,
inspired by NOAA-EMC MLGlobal's grib2writer.py.

The Netcdf2Grib class handles per-member forecast writes, supporting both
single-hour and multi-hour datasets. Per-hour writes enable overlapped I/O
during autoregressive forecasting.

Notes/assumptions:
- Grid Definition: We require a valid GRIB2 Section 3 for the HRRRCast Lambert
    Conformal grid. Provide via Netcdf2Grib(section3=...) constructor or set the
    environment variable NETCDF2GRIB_SECTION3 to a .npy file. If neither is provided,
    we auto-construct a canonical HRRR Lambert Conformal Section 3 for the full
    3 km grid (Nx=1799, Ny=1059).
- Template Numbers: Product Definition Template Numbers (pdtn) and Data Representation
  Template Numbers (drtn) default to 0 (instantaneous forecast, simple packing).
  For accumulated fields (e.g., APCP), adjust pdtn and duration semantics to match
  downstream consumers.
- Member IDs: Each member forecast can be written independently, allowing per-member
  outputs with consistent naming conventions.
- Encoder: messages are encoded with ecCodes through pygrib by default
  (NETCDF2GRIB_ENGINE=eccodes). NETCDF2GRIB_ENGINE=grib2io uses the original grib2io path,
  which writes discipline 0 for every field (so LAND, discipline 2, comes out as surface TMP).
- Index: the .idx (wgrib2 -s style inventory) is written in Python from the encoded messages,
  so wgrib2 is not needed. NETCDF2GRIB_INDEX=wgrib2 runs $WGRIB2 -s instead; =none skips it.
"""

import os
import subprocess
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import xarray as xr

try:  # only needed for NETCDF2GRIB_ENGINE=grib2io
    import grib2io
except Exception:  # pragma: no cover
    grib2io = None

from cf_attributes import VARIABLE_METADATA
from utils import setup_logging


def _set_data(msg, vals2d) -> None:
    """Attach a 2-D field to a message. NaNs (e.g. HRRR cloud ceiling where there is no
    ceiling) are written as missing through a bitmap; without one grib2io turns them into
    Python objects and the packing fails ("Buffer dtype mismatch ... Python object")."""
    arr = np.asarray(vals2d, dtype=np.float32)
    if np.isnan(arr).any():
        try:
            from grib2io.templates import Grib2Metadata
            msg.bitMapFlag = Grib2Metadata(0, table="6.0")
        except Exception:
            msg.bitMapFlag = 0
    msg.data = arr

logger = setup_logging("INFO")


# Derive GRIB parameter map from consolidated metadata
# Format: var -> (discipline, category, number, surface_type, surface_value)
GRIB_PARAM_MAP = {
    var: meta["grib2"]
    for var, meta in VARIABLE_METADATA.items()
    if "grib2" in meta
}


ENGINE = os.environ.get("NETCDF2GRIB_ENGINE", "eccodes").lower()
INDEX = os.environ.get("NETCDF2GRIB_INDEX", "python").lower()

# 200-byte GRIB2 message (Lambert conformal 3.30, PDT 4.0, simple packing) used as the starting
# point for every message; all keys that matter are overwritten in _encode_eccodes.
_TEMPLATE_B64 = (
    "R1JJQv//AAIAAAAAAAAAyAAAABUBAAcAAAIBAQfXAxcMAAAAAQAAAFEDAAAAAAQAAAAeAQAAYTed////////////"
    "/wAAAAIAAAACAUKKyw4knNgIAkt2oA+lbqAALcbAAC3GwABAAkt2oAJLdqAAAAAAAAAAAAAAACIEAAAAAAAAAv+A"
    "AAAAAQAAAAAB//////////////8AAAAVBQAAAAQAAAAAAACAFgAAGAAAAAAGBv8AAAARBwAAAEAAAIAAAMAAADc3Nzc="
)

# NCEP abbreviations and wgrib2-style level names, for the .idx inventory
_ABBR = {
    (0, 0, 0): "TMP", (0, 0, 2): "POT", (0, 0, 6): "DPT",
    (0, 1, 0): "SPFH", (0, 1, 1): "RH", (0, 1, 3): "PWAT", (0, 1, 8): "APCP",
    (0, 1, 33): "CRAIN", (0, 1, 34): "CFRZR",
    (0, 2, 1): "WIND", (0, 2, 2): "UGRD", (0, 2, 3): "VGRD", (0, 2, 8): "VVEL", (0, 2, 12): "RELV",
    (0, 2, 15): "VUCSH", (0, 2, 16): "VVCSH", (0, 2, 22): "GUST", (0, 2, 194): "USTM",
    (0, 2, 195): "VSTM", (0, 2, 220): "MAXUVV", (0, 2, 221): "MAXDVV",
    (0, 3, 0): "PRES", (0, 3, 5): "HGT", (0, 3, 198): "MSLMA",
    (0, 6, 1): "TCDC", (0, 6, 3): "LCDC", (0, 6, 4): "MCDC", (0, 6, 5): "HCDC",
    (0, 7, 6): "CAPE", (0, 7, 7): "CIN", (0, 7, 8): "HLCY", (0, 7, 199): "MXUPHL", (0, 7, 200): "MNUPHL",
    (0, 16, 196): "REFC", (0, 19, 0): "VIS", (2, 0, 0): "LAND",
}
_LEVEL_NAMES = {1: "surface", 4: "0C isotherm", 10: "entire atmosphere", 101: "mean sea level",
                214: "low cloud layer", 215: "cloud ceiling", 224: "middle cloud layer",
                234: "high cloud layer"}


def _level_text(stype, value) -> str:
    """wgrib2-style level description."""
    if isinstance(value, tuple):
        a, b = value
        if stype == 100:
            return f"{a / 100:g}-{b / 100:g} mb"
        return f"{a:g}-{b:g} m above ground" if stype == 103 else f"{a:g}-{b:g} level {stype}"
    if stype == 100:
        return f"{(value or 0) / 100:g} mb"
    if stype == 103:
        return f"{value or 0:g} m above ground"
    return _LEVEL_NAMES.get(stype, f"level type {stype} {value or 0:g}")


def _precision(var_name: str, surface_value) -> Tuple[int, int]:
    """(decimal scale factor, spatial differencing order) per variable."""
    sv = surface_value if not isinstance(surface_value, tuple) else None
    if var_name in ("SPFH", "SPFH_0C", "SPFH2M"):
        if sv and 5000 <= sv <= 10000:
            dec = 12
        elif sv and 15000 <= sv <= 40000:
            dec = 10
        else:
            dec = 8
    elif var_name == "PWAT":
        dec = 3      # precipitable water: 0-80 mm
    elif var_name in ("CRAIN", "CFRZR", "APCP"):
        dec = 4      # small hourly precipitation amounts
    elif var_name in ("VUCSH_0_1km", "VVCSH_0_1km", "VUCSH_0_6km", "VVCSH_0_6km",
                      "RELV_max_0_1km", "RELV_max_0_2km"):
        dec = 5      # shear / vorticity in 1/s
    else:
        dec = 2
    order = 0 if var_name in ("VIS", "HGTCC") else 2   # no differencing for discontinuous fields
    return dec, order


def _put(msg, key, value) -> None:
    """Set a key on a pygrib message; code-table keys only take strings in pygrib."""
    try:
        msg[key] = value
    except TypeError:
        msg[key] = str(value)


def _encode_eccodes(section3, ref_time: datetime, lead_hour: int, var_name: str,
                    surface_type: int, surface_value, vals2d) -> bytes:
    """One GRIB2 message with ecCodes (through pygrib, which bundles ecCodes)."""
    import base64
    import pygrib

    s3 = [int(v) for v in section3]
    disc, cat, num, _, _ = GRIB_PARAM_MAP[var_name]
    dec, order = _precision(var_name, surface_value)
    arr = np.asarray(vals2d, dtype=np.float64)
    ny, nx = arr.shape
    if (s3[12], s3[13]) != (nx, ny):
        raise ValueError(f"{var_name}: data {ny}x{nx} does not match the grid {s3[13]}x{s3[12]}")

    m = pygrib.fromstring(base64.b64decode(_TEMPLATE_B64))
    for key, val in (
        # section 0/1
        ("discipline", disc), ("centre", 7), ("subCentre", 0), ("tablesVersion", 2),
        ("localTablesVersion", 1), ("significanceOfReferenceTime", 1),
        ("dataDate", ref_time.year * 10000 + ref_time.month * 100 + ref_time.day),
        ("dataTime", ref_time.hour * 100 + ref_time.minute),
        ("productionStatusOfProcessedData", 0), ("typeOfProcessedData", 1),
        # section 3 (template 3.30, same integers as the grib2io Section 3 array)
        ("shapeOfTheEarth", s3[5]), ("scaleFactorOfRadiusOfSphericalEarth", s3[6]),
        ("scaledValueOfRadiusOfSphericalEarth", s3[7]), ("Nx", nx), ("Ny", ny),
        ("latitudeOfFirstGridPoint", s3[14]), ("longitudeOfFirstGridPoint", s3[15]),
        ("resolutionAndComponentFlags", s3[16]), ("LaD", s3[17]), ("LoV", s3[18]),
        ("Dx", s3[19]), ("Dy", s3[20]), ("projectionCentreFlag", s3[21]), ("scanningMode", s3[22]),
        ("Latin1", s3[23]), ("Latin2", s3[24]),
        # section 4 (template 4.0)
        ("parameterCategory", cat), ("parameterNumber", num), ("typeOfGeneratingProcess", 2),
        ("indicatorOfUnitForForecastTime", 1), ("forecastTime", int(lead_hour)),
    ):
        _put(m, key, val)

    _put(m, "typeOfFirstFixedSurface", surface_type)
    _put(m, "scaleFactorOfFirstFixedSurface", 0)
    if isinstance(surface_value, tuple):
        _put(m, "scaledValueOfFirstFixedSurface", int(surface_value[0]))
        _put(m, "typeOfSecondFixedSurface", surface_type)
        _put(m, "scaleFactorOfSecondFixedSurface", 0)
        _put(m, "scaledValueOfSecondFixedSurface", int(surface_value[1]))
    else:
        _put(m, "scaledValueOfFirstFixedSurface", int(surface_value or 0))
        _put(m, "typeOfSecondFixedSurface", 255)

    # packing: complex (+ 2nd-order spatial differencing) with enough bits for 10^-dec precision.
    # pygrib only sees keys that exist when it parses a message, so re-parse after changing
    # the packing type.
    _put(m, "packingType", "grid_complex_spatial_differencing" if order else "grid_complex")
    m = pygrib.fromstring(m.tostring())
    if order:
        _put(m, "orderOfSpatialDifferencing", order)
    finite = arr[np.isfinite(arr)]
    span = float(finite.max() - finite.min()) * 10.0 ** dec if finite.size else 0.0
    nbits = int(np.ceil(np.log2(span + 1.0))) + 1 if span > 0 else 0
    _put(m, "decimalScaleFactor", dec)
    _put(m, "bitsPerValue", min(nbits, 32))
    if finite.size < arr.size:            # NaN -> missing via bitmap
        m["values"] = np.ma.masked_invalid(arr).ravel()
    else:
        m["values"] = arr.ravel()
    return m.tostring()


def _write_python_index(outfile: str, entries) -> None:
    """wgrib2 -s style inventory: n:offset:d=YYYYMMDDHH:ABBR:level:time:"""
    with open(f"{outfile}.idx", "w") as f:
        for i, (offset, ref_time, lead, var_name, stype, sval) in enumerate(entries, start=1):
            disc, cat, num, _, _ = GRIB_PARAM_MAP[var_name]
            abbr = _ABBR.get((disc, cat, num), f"var discipline={disc} cat={cat} num={num}")
            when = "anl" if int(lead) == 0 else f"{int(lead)} hour fcst"
            f.write(f"{i}:{offset}:d={ref_time:%Y%m%d%H}:{abbr}:{_level_text(stype, sval)}:{when}:\n")


class Netcdf2Grib:
    # Class-level lock for grib2io operations (g2c library may not be thread-safe)
    _grib2io_lock = threading.Lock()

    def __init__(self, section3: Optional[np.ndarray] = None, pdtn_default: int = 0, drtn_default: int = 3):
        self.section3 = self._resolve_section3(section3)
        self.pdtn_default = pdtn_default
        self.drtn_default = drtn_default

    @staticmethod
    def construct_section3_hrrr(nx: int = 1799, ny: int = 1059,
                                lat1: float = 21.138123, lon1: float = 237.280472) -> np.ndarray:
        """Construct GRIB2 Section 3 for HRRR-like CONUS Lambert Conformal grid at 3 km.

        This uses canonical HRRR projection parameters and the full-resolution dimensions
        defined in preprocessing (grid_width=1799, grid_height=1059).

        Parameters used:
        - First grid point (La1/Lo1): 21.138123N, 237.280472E
        - Orientation longitude (LoV): 262.5E
        - Standard parallels (Latin1, Latin2): 38.5N, 38.5N
        - Grid spacing (Dx/Dy): 3000 m
        - Earth radius: 6371229 m

        Returns a numpy array suitable for the `section3` argument of grib2io.Grib2Message.

        Note: If grib2io provides a helper for LCC Section 3 creation in your environment,
        this function will attempt to use it. Otherwise, it constructs a fixed array using
        canonical HRRR parameters. You can override via NETCDF2GRIB_SECTION3.
        """
        # Canonical HRRR LCC parameters (matching HRRR docs)
        # lat1/lon1: first (south-west) grid point; defaults are the full HRRR CONUS grid.
        # Pass the corner of a sub-domain to describe a cropped grid.
        lov = 262.5         # degrees East
        latin1 = 38.5       # degrees North
        latin2 = 38.5       # degrees North
        dx = 3000           # meters
        dy = 3000           # meters
        earth_radius = 6371229  # meters (spherical)

        # Build a best-effort fixed array for GRIB2 Template 3.30 (Lambert Conformal)
        # Values are encoded as scaled integers:
        # - Lat/Lon in microdegrees (deg * 1e6)
        # - Dx/Dy in millimeters (m * 1e3)
        # Note: Field positions follow common GRIB2 3.30 usage; some decoders may require
        # exact scan mode or earth-shape codes. Adjust if downstream tools complain.

        micro = 1_000_000
        milli = 1_000

        la1 = int(round(lat1 * micro))
        lo1 = int(round(lon1 * micro))
        lov_i = int(round(lov * micro))
        latin1_i = int(round(latin1 * micro))
        latin2_i = int(round(latin2 * micro))
        dx_mm = int(round(dx * milli))
        dy_mm = int(round(dy * milli))

        # Common defaults
        shape_of_earth = 1  # spherical with given radius
        # Resolution and component flags: 8 -> winds(grid) per wgrib2 'res 8'
        res_flags = 8
        # Projection centre flag: 0 = north, 1 = south
        proj_center_flag = 0

        # Section 3 structure (template 3.30 Lambert Conformal) matching grib_dump order:
        # Fields reflect wgrib2/grib_dump output: res=8, scanningMode=64 (WE:SN), LaD=38500000
        section3 = np.array([
            0,                   # Source of grid definition
            nx * ny,             # Number of data points = Ni * Nj
            0,                   # Number of octets for number of points
            0,                   # Interpretation of number of points
            30,                  # Grid definition template number (3.30)
            shape_of_earth,      # Shape of Earth (1 = spherical, producer-specified radius)
            0,                   # Scale factor of radius of spherical Earth
            earth_radius,        # Scaled value of spherical Earth radius (meters)
            0,                   # Scale factor of Earth major axis
            0,                   # Scaled value of Earth major axis
            0,                   # Scale factor of Earth minor axis
            0,                   # Scaled value of Earth minor axis
            nx,                  # Nx
            ny,                  # Ny
            la1,                 # Latitude of first grid point (microdegrees)
            lo1,                 # Longitude of first grid point (microdegrees)
            res_flags,           # Resolution and component flags (8 -> winds(grid))
            38_500_000,          # LaD (Latitude of grid orientation, microdegrees)
            lov_i,               # LoV (orientation longitude, microdegrees)
            dx_mm,               # Dx (grid length in x, millimeters)
            dy_mm,               # Dy (grid length in y, millimeters)
            proj_center_flag,    # Projection centre flag (0 = north)
            64,                  # Scanning mode (WE:SN)
            latin1_i,            # Latin1 (first standard parallel, microdegrees)
            latin2_i,            # Latin2 (second standard parallel, microdegrees)
            0,                   # Latitude of southern pole
            0,                   # Longitude of southern pole
        ], dtype=np.int64)

        return section3

    def _resolve_section3(self, section3: Optional[np.ndarray]) -> np.ndarray:
        if section3 is not None:
            return np.asarray(section3, dtype=np.int64)
        env_path = os.environ.get("NETCDF2GRIB_SECTION3", "")
        if env_path and os.path.isfile(env_path):
            try:
                arr = np.load(env_path)
                return np.asarray(arr, dtype=np.int64)
            except Exception as e:
                raise RuntimeError(f"Failed to load section3 from {env_path}: {e}")
        # Fallback: attempt to construct HRRR-like 3 km LCC Section 3 using known dims (Nx=1799, Ny=1059)
        try:
            return self.construct_section3_hrrr(nx=1799, ny=1059)
        except Exception as e:
            raise RuntimeError(
                "GRIB2 Section 3 (grid definition) is required and could not be auto-constructed. "
                "Provide 'section3' to Netcdf2Grib, set NETCDF2GRIB_SECTION3 to a .npy file, or ensure grib2io LCC helper is available. "
                f"Error: {e}"
            )

    def _build_message(
        self,
        var_name: str,
        ref_time: datetime,
        lead_hour: int,
        surface_type: Optional[int] = None,
        surface_value: Optional[float] = None,
        pdtn: Optional[int] = None,
        drtn: Optional[int] = None,
    ) -> "grib2io.Grib2Message":

        # 1. Define Section 1 (Identification Section)
        section1 = np.array([
            7,               # Center: 7 (NCEP)
            0,               # Subcenter: 0
            2,               # Master Tables Version: 2
            1,               # Local Tables Version: 1
            1,               # Significance of Ref Time: 1 (Start of Forecast)
            ref_time.year,
            ref_time.month,
            ref_time.day,
            ref_time.hour,
            ref_time.minute,
            ref_time.second,
            0,               # Production Status: 0 (Operational)
            1                # Type of Data: 1 (Forecast)
        ], dtype=np.int64)

        # 2. Construct message
        msg = grib2io.Grib2Message(
            section1=section1,
            section3=self.section3,
            pdtn=self.pdtn_default if pdtn is None else pdtn,
            drtn=self.drtn_default if drtn is None else drtn,
        )

        # 3. Set parameter keys
        if var_name not in GRIB_PARAM_MAP:
            raise ValueError(f"Unknown variable {var_name} not in GRIB_PARAM_MAP")
        disc, cat, num, default_surface, _ = GRIB_PARAM_MAP[var_name]
        msg.discipline = disc
        msg.parameterCategory = cat
        msg.parameterNumber = num
        msg.typeOfFirstFixedSurface = surface_type if surface_type is not None else default_surface

        if surface_value is not None:
            # Check if surface_value is a tuple (layer) or a single value
            if isinstance(surface_value, tuple):
                # Layer specification: (top, bottom)
                top_value, bottom_value = surface_value
                msg.scaledValueOfFirstFixedSurface = int(top_value)
                msg.scaleFactorOfFirstFixedSurface = 0
                msg.typeOfSecondFixedSurface = surface_type if surface_type is not None else default_surface
                msg.scaledValueOfSecondFixedSurface = int(bottom_value)
                msg.scaleFactorOfSecondFixedSurface = 0
            else:
                # Single level specification
                msg.scaledValueOfFirstFixedSurface = int(surface_value)
                msg.scaleFactorOfFirstFixedSurface = 0
                msg.typeOfSecondFixedSurface = 255
                msg.scaleFactorOfSecondFixedSurface = 0
                msg.scaledValueOfSecondFixedSurface = 0
        else:
            msg.scaledValueOfFirstFixedSurface = 0
            msg.scaleFactorOfFirstFixedSurface = 0

            msg.typeOfSecondFixedSurface = 255
            msg.scaleFactorOfSecondFixedSurface = 0
            msg.scaledValueOfSecondFixedSurface = 0

        # 4. Time metadata
        msg.unitOfForecastTime = 1  # hours
        msg.leadTime = timedelta(hours=int(lead_hour))

        # 5. Statistical processing
        msg.typeOfStatisticalProcessing = 0
        msg.numberOfTimeRanges = 0

        # 6./7. Decimal scale factor and spatial differencing order (shared with the ecCodes path)
        msg.binaryScaleFactor = 0
        msg.decScaleFactor, msg.spatialDifferenceOrder = _precision(var_name, surface_value)

        return msg

    def _get_surface_type_and_value(self, var_name: str, ds: xr.Dataset, da: xr.DataArray) -> Tuple[int, Optional[float]]:
        if var_name not in GRIB_PARAM_MAP:
            raise ValueError(f"Unknown variable {var_name} not in GRIB_PARAM_MAP")
        _, _, _, surface_type, surface_value = GRIB_PARAM_MAP[var_name]
        return surface_type, surface_value

    def save_grib2(self, forecast_starttime: datetime, ds_hour: xr.Dataset, output_path: str) -> None:
        """Write a single-hour GRIB2 file from an xarray.Dataset.

        ds_hour is expected to have dims (lead_time=1, time=1, [level], y, x) and contain
        both pressure-level and surface variables. Encoded with ecCodes (default) or grib2io
        (NETCDF2GRIB_ENGINE=grib2io); a .idx inventory is written next to it.
        """
        try:
            lead = int(np.asarray(ds_hour["lead_time"]).item())
        except Exception:
            lead = 0
        outfile = output_path

        # (label, var_name, surface_type, surface_value, 2-D values) for every message
        specs = []
        try:
            ds_loc = ds_hour
            if "y" not in ds_loc.dims or "x" not in ds_loc.dims:
                if "latitude" in ds_loc.dims and "longitude" in ds_loc.dims:
                    ds_loc = ds_loc.rename_dims({"latitude": "y", "longitude": "x"})
                else:
                    logger.warning("Dataset missing y/x dims; attempting to infer from data variable shapes.")

            for var_name in sorted(ds_loc.data_vars):
                da = ds_loc[var_name]
                if var_name not in GRIB_PARAM_MAP:
                    logger.debug(f"Skipping unknown variable {var_name}")
                    continue
                surface_type, surface_value = self._get_surface_type_and_value(var_name, ds_loc, da)
                if "level" in da.coords:
                    for level in np.atleast_1d(da["level"].values):
                        plevel = float(level)
                        if plevel < 2000:  # hPa -> Pa
                            plevel *= 100.0
                        vals = np.squeeze(da.sel(level=level).values)
                        while vals.ndim > 2:
                            vals = vals[0]
                        specs.append((f"{var_name}:{plevel / 100:g}hPa", var_name, 100, plevel, vals))
                else:
                    vals = np.squeeze(da.values)
                    while vals.ndim > 2:
                        vals = vals[0]
                    specs.append((var_name, var_name, surface_type, surface_value, vals))
        except Exception as e:
            logger.warning("Error preparing GRIB messages: %s", e)
            return

        index_entries = []
        # Serialize the encoding/writing (ecCodes / g2c may not be thread-safe)
        with self._grib2io_lock:
            if os.path.isfile(outfile):
                os.remove(outfile)
            if ENGINE == "grib2io":
                if grib2io is None:
                    raise RuntimeError("NETCDF2GRIB_ENGINE=grib2io but grib2io is not installed")
                g2 = grib2io.open(outfile, mode="w")
                try:
                    for name, var_name, stype, sval, vals in specs:
                        try:
                            msg = self._build_message(var_name, forecast_starttime, lead,
                                                      surface_type=stype, surface_value=sval)
                            _set_data(msg, vals)
                            msg.pack()
                            g2.write(msg)
                        except Exception as e:
                            logger.warning("Skipping GRIB message %s: %s", name, e)
                finally:
                    g2.close()
            else:
                offset = 0
                with open(outfile, "wb") as fout:
                    for name, var_name, stype, sval, vals in specs:
                        try:
                            raw = _encode_eccodes(self.section3, forecast_starttime, lead, var_name,
                                                  stype, sval, vals)
                        except Exception as e:  # one bad field is skipped, not the whole file
                            logger.warning("Skipping GRIB message %s: %s", name, e)
                            continue
                        fout.write(raw)
                        index_entries.append((offset, forecast_starttime, lead, var_name, stype, sval))
                        offset += len(raw)

        # Index: Python inventory by default; wgrib2 -s on request (or for the grib2io path)
        if INDEX == "none":
            return
        try:
            t0 = time.time()
            if INDEX == "wgrib2" or ENGINE == "grib2io":
                wgrib2 = os.environ.get("WGRIB2", "wgrib2")
                with open(f"{outfile}.idx", "w") as f_out:
                    subprocess.run([wgrib2, "-s", outfile], stdout=f_out, check=True)
            else:
                _write_python_index(outfile, index_entries)
            logger.info(f"Index created in {time.time() - t0:.2f}s: {outfile}.idx")
        except Exception as e:
            logger.warning(f"Skipping index creation: {e}")
