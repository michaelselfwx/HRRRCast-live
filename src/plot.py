#!/usr/bin/env python3
"""
Forecast Visualization Script

This script plots each variable from the forecast output and saves them as separate PNG files.
It handles both pressure level and surface variables from the HRRR forecast data.

Usage:
        python plot_forecast.py <init_time> <lead_hour> <member> [--forecast_dir DIR] [--output_dir DIR]
    
        Expects per-hour NetCDF files:
            - Member average (PMM/mean): hrrrcast_avg_fXX.nc
            - Individual members:        hrrrcast_mN_fXX.nc
"""

import argparse
import fnmatch
import re
import logging
import os
import sys
from datetime import timedelta
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

# Local imports
import utils
from utils import setup_logging
from cf_attributes import VARIABLE_METADATA


def parse_product_patterns(values) -> Optional[List[str]]:
    """Turn --products / HRRRCAST_PLOT_PRODUCTS into a list of patterns (None = everything).

    Accepts space- or comma-separated entries, e.g. "REFC APCP T2M HGT_500hPa summary".
    """
    if not values:
        return None
    if isinstance(values, str):
        values = [values]
    pats = [p.strip() for v in values for p in v.replace(",", " ").split() if p.strip()]
    return pats or None


# Derived precipitation products written by derived_precip.py
# (hrrrcast_<mem>_precip_fHH.nc for members/avg, hrrrcast_lpmm_fHH.nc for the local PMM)
DERIVED_PRECIP_VARS = ["APCP", "APCP_TOT", "APCP_6H", "APCP_12H"]
DERIVED_META = {
    "APCP_TOT": {"long_name": "Total precipitation since init", "units": "mm"},
    "APCP_6H":  {"long_name": "6-h precipitation",              "units": "mm"},
    "APCP_12H": {"long_name": "12-h precipitation",             "units": "mm"},
}


def ensure_county_shapes() -> bool:
    """Download/locate the Natural Earth 10m US county shapefile (cached by cartopy after the
    first run). Returns False, with a warning, if it can't be fetched (e.g. no internet)."""
    if not CARTOPY_AVAILABLE:
        return False
    try:
        from cartopy.io import shapereader
        shapereader.natural_earth(resolution="10m", category="cultural", name="admin_2_counties")
        return True
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"County borders unavailable ({e}); plotting without them. "
            "On HPRC compute nodes load the WebProxy module, or run once on a login node to cache them."
        )
        return False


def product_wanted(product: str, patterns: Optional[List[str]]) -> bool:
    """True if a product (e.g. "REFC_surface", "HGT_500hPa", "summary") should be plotted.

    A pattern matches the full product name or just the variable name, and may use shell
    wildcards: "REFC" -> REFC_surface; "HGT" -> HGT at every level; "HGT_500hPa" -> one
    level; "*_850hPa" -> every variable at 850 hPa; "HLCY_*" -> both SRH layers.
    """
    if not patterns:
        return True
    var = product
    for suffix in ("_surface",):
        if var.endswith(suffix):
            var = var[: -len(suffix)]
    if var.endswith("hPa") and "_" in var:
        var = var.rsplit("_", 1)[0]
    base = re.sub(r"_(TOT|\d+H)$", "", var)  # APCP_TOT / APCP_6H also match "APCP"
    return any(fnmatch.fnmatchcase(product, p) or fnmatch.fnmatchcase(var, p) or fnmatch.fnmatchcase(base, p)
               for p in patterns)

logger = None


def _normalize_range(range_values: Optional[List[float]], range_name: str) -> Optional[tuple]:
    """Validate and normalize a two-value [min, max] numeric range."""
    if range_values is None:
        return None
    if len(range_values) != 2:
        raise ValueError(f"{range_name} must contain exactly 2 values: min max")
    low, high = float(range_values[0]), float(range_values[1])
    if low == high:
        raise ValueError(f"{range_name} min and max cannot be equal")
    return (min(low, high), max(low, high))


def save_png(fig, path: str, dpi: int, quantize: bool = True) -> None:
    """Save a figure as a small PNG: rendered at `dpi`, then stored as an 8-bit palette image
    (contour plots have few colours, so it looks the same at ~1/4 of the size)."""
    if not quantize:
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        return
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    with Image.open(buf) as im:
        im = im.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                        dither=Image.Dither.NONE)
        im.save(path, format="PNG", optimize=True)


class ForecastPlotterConfig:
    """Configuration class for forecast plotting parameters."""
    
    def __init__(self):
        # prefix for plot titles, e.g. "HRRR " for the operational HRRR run
        self.run_label = ""
        # Variable definitions matching the preprocessor
        self.pl_vars = ["UGRD", "VGRD", "VVEL", "TMP", "HGT", "SPFH"]
        # Updated surface variable list (matches preprocessing)
        self.sfc_vars = [
            "PRES", "MSLMA", "REFC", "T2M", "UGRD10M", "VGRD10M", "UGRD80M", "VGRD80M",
            "D2M", "R2M", "SPFH2M", "POT2M", "TCDC", "LCDC", "MCDC", "HCDC", "VIS", "APCP", "HGTCC", "CAPE", "CIN",
            "PWAT", "CRAIN", "RAIN_MASK", "CFRZR", "FRZR_MASK", "WARM_LAYER_DEPTH", "COLD_LAYER_DEPTH",
            "GUST", "GUST_FACTOR", "GUST_CONV", "WIND_10M", "WIND_MAX",
            "VUCSH_0_1km", "VVCSH_0_1km", "VUCSH_0_6km", "VVCSH_0_6km",
            "RELV_max_0_1km", "RELV_max_0_2km", "USTM_0_6km", "VSTM_0_6km",
            "HLCY_0_1km", "HLCY_0_3km", "MXUPHL_max_0_2km", "MNUPHL_min_0_2km",
            "MXUPHL_max_0_3km", "MNUPHL_min_0_3km", "MXUPHL_max_2_5km", "MNUPHL_min_2_5km",
            "MAXUVV_max_100_1000mb", "MAXDVV_max_100_1000mb",
            "HGT_0C", "UGRD_0C", "VGRD_0C", "WIND_0C", "SPFH_0C", "RH_0C",
            "DU_SFC_0C", "DV_SFC_0C", "SHEAR_SFC_0C"
        ]
        
        # Pressure levels (hPa)
        self.levels = [200, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 825, 850, 875, 900, 925, 950, 975, 1000]

        # Plot settings
        self.figure_size = (12, 8)
        # 100 dpi -> ~1100 px wide, plenty for a browser (300 dpi made ~3300 px / ~1 MB files).
        # Override with HRRRCAST_PLOT_DPI; HRRRCAST_PLOT_QUANTIZE=0 keeps full-colour PNGs.
        self.dpi = int(os.environ.get("HRRRCAST_PLOT_DPI", "100"))
        self.quantize = os.environ.get("HRRRCAST_PLOT_QUANTIZE", "1") not in ("0", "false", "no")
        self.cmap_default = 'viridis'
        self.zoom_extent = None
        self.domains = ["tx"]   # plot domains (see DOMAINS); --domains / HRRRCAST_PLOT_DOMAINS
        self.outline = None     # (lons, lats) polygon drawn on the map (set per domain)
        self.county_lw = 0.12
        self.products = None   # list of product patterns; None = plot everything
        # US county borders (Natural Earth 10m admin_2_counties; downloaded once by cartopy).
        # Turn off with --no-counties or HRRRCAST_PLOT_COUNTIES=0.
        self.counties = os.environ.get("HRRRCAST_PLOT_COUNTIES", "1") not in ("0", "false", "no")


class ForecastPlotter:
    """Handles forecast data visualization."""
    
    def __init__(self, config: ForecastPlotterConfig):
        self.config = config
        self.use_cartopy = CARTOPY_AVAILABLE
        if not self.use_cartopy:
            logger.warning("Cartopy not available, using simple plotting")

    _counties_feature = None   # built once per process, shared by every plot

    def _add_map_features(self, ax) -> None:
        """Coastlines, country and state borders, plus (optionally) county borders."""
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=3)
        ax.add_feature(cfeature.STATES, linewidth=0.3, zorder=3)
        outline = getattr(self.config, "outline", None)
        if outline:
            ax.plot(outline[0], outline[1], color="black", linewidth=1.0, linestyle="--",
                    transform=ccrs.PlateCarree(), zorder=4)
        if not getattr(self.config, "counties", False):
            return
        if ForecastPlotter._counties_feature is None:
            # cartopy only fetches the shapefile when the figure is drawn, so a failed download
            # would break savefig; fetch it here first and skip counties if that fails.
            if not ensure_county_shapes():
                self.config.counties = False
                return
            ForecastPlotter._counties_feature = cfeature.NaturalEarthFeature(
                category="cultural", name="admin_2_counties", scale="10m",
                facecolor="none", edgecolor="0.45",
            )
        # thin grey lines under the state borders but above the filled field
        ax.add_feature(ForecastPlotter._counties_feature,
                       linewidth=getattr(self.config, "county_lw", 0.12), zorder=2.5)
    
    def load_forecast_data(self, forecast_file: str) -> xr.Dataset:
        """Load forecast data from NetCDF file."""
        if not os.path.exists(forecast_file):
            raise FileNotFoundError(f"Forecast file not found: {forecast_file}")
        
        try:
            logger.info(f"Loading forecast data from {forecast_file}")
            ds = xr.open_dataset(forecast_file, decode_timedelta=True)
            return ds
        except Exception as e:
            logger.error(f"Error loading forecast data: {e}")
            raise
    
    @staticmethod
    def _sample_cmap(name, n):
        base = plt.get_cmap(name)
        return [mcolors.to_hex(base(i/(n-1))) for i in range(n)]


    @staticmethod
    def get_refc_cmap() -> tuple:
        """Return a colormap and normalization for reflectivity (REFC)."""
        reflectivity_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
        reflectivity_colors = [
            "#FFFFFF", "#00F9F9", "#0080FF", "#0004FF", "#00FF00", "#00C100", "#008000", "#F5FA00",
            "#FFBF00", "#FF8200", "#FF0400", "#BF0000", "#820000", "#FF00FF", "#9062CD",
        ]
        vmin, vmax = min(reflectivity_levels), max(reflectivity_levels)
        cmap = mcolors.ListedColormap(reflectivity_colors)
        norm = mcolors.BoundaryNorm(reflectivity_levels, cmap.N)
        return cmap, norm, vmin, vmax

    @staticmethod
    def get_apcp_accum_cmap() -> tuple:
        """Colormap + norm for multi-hour precipitation totals (mm), extends to 250 mm."""
        levels = [0.25, 1, 2.5, 5, 10, 15, 20, 25, 35, 50, 75, 100, 125, 150, 200, 250]
        colors = [
            "#E1F5FE", "#B0E2FF", "#7EC0EE", "#00FA9A", "#32CD32", "#228B22", "#FFFF00", "#FFD700",
            "#FFA500", "#FF4500", "#FF0000", "#B22222", "#8B0000", "#9400D3", "#4B0082",
        ]
        cmap = mcolors.ListedColormap(colors)
        cmap.set_under("white", alpha=0)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm, min(levels), max(levels)

    def plot_derived_precip(self, ds: xr.Dataset, lead_hour: int, output_dir: str,
                            timestamp_str: str, label: str = "") -> None:
        """Plot whichever derived precipitation fields are in ds (see DERIVED_PRECIP_VARS)."""
        lats = ds['latitude'].values
        lons = ds['longitude'].values
        title_suffix = f"{label}\n{self.config.run_label}Forecast: {timestamp_str} + {lead_hour}h"
        for var_name in DERIVED_PRECIP_VARS:
            if var_name not in ds.variables or not product_wanted(f"{var_name}_surface", self.config.products):
                continue
            try:
                data = np.squeeze(ds[var_name].values)
                fig = self.create_plot(data, lats, lons, var_name, None, title_suffix)
                filename = f"{var_name}_surface_lead{lead_hour:02d}h.png"
                save_png(fig, os.path.join(output_dir, filename), self.config.dpi, self.config.quantize)
                plt.close(fig)
                logger.info(f"Saved: {filename}")
            except Exception as e:
                logger.error(f"Error plotting derived {var_name}: {e}")

    @staticmethod
    def get_apcp_cmap() -> tuple:
        """Return colormap + norm for accumulated precipitation (APCP)."""
        apcp_levels = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 25, 35, 45, 60, 80, 100]
        apcp_colors = [
            "#FFFFFF", "#B0E2FF", "#7EC0EE", "#00FA9A", "#32CD32", "#FFFF00", "#FFD700",
            "#FFA500", "#FF4500", "#FF0000", "#8B0000", "#9400D3", "#8B008B", "#4B0082",
        ]
        vmin, vmax = min(apcp_levels), max(apcp_levels)
        cmap = mcolors.ListedColormap(apcp_colors)
        norm = mcolors.BoundaryNorm(apcp_levels, cmap.N)
        return cmap, norm, vmin, vmax

    @staticmethod
    def get_cape_cmap() -> tuple:
        """Colormap for CAPE (0-7000 J/kg) using 'inferno'."""
        levels = [0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000]
        colors = ForecastPlotter._sample_cmap("inferno", len(levels)-1)
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm, min(levels), max(levels)
    
    @staticmethod
    def get_cin_cmap() -> tuple:
        """Colormap for CIN (-2000 to 0 J/kg) using 'PuBuGn_r'."""
        levels = [-2000, -1500, -1000, -750, -500, -300, -200, -150, -100, -75, -50, -25, -10, -1, 0]
        colors = ForecastPlotter._sample_cmap("PuBuGn_r", len(levels)-1)
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm, min(levels), max(levels)
    
    @staticmethod
    def get_vis_cmap() -> tuple:
        """Colormap for VIS (0-100000 m) using 'YlOrBr_r' and log-ish spaced levels."""
        levels = [10, 50, 100, 200, 400, 800, 1500, 3000, 6000, 12000, 24000, 48000, 100000]
        colors = ForecastPlotter._sample_cmap("YlOrBr_r", len(levels)-1)
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm, min(levels), max(levels)
    
    @staticmethod
    def get_hgtcc_cmap() -> tuple:
        """Colormap for HGTCC (0-20000 m) using 'viridis'."""
        levels = [0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 20000]
        colors = ForecastPlotter._sample_cmap("viridis", len(levels)-1)
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm, min(levels), max(levels)

    
    def create_plot(self, data: np.ndarray, lats: np.ndarray, lons: np.ndarray, 
                   var_name: str, level: Optional[int] = None, 
                   title_suffix: str = "") -> plt.Figure:
        """Create a plot for a given variable."""
        
        # Get variable configuration from VARIABLE_METADATA
        var_meta = VARIABLE_METADATA.get(var_name) or DERIVED_META.get(var_name, {})
        units = var_meta.get('units', '')
        long_name = var_meta.get('long_name', var_name)
        
        # Special handling for categorical / thresholded fields
        norm = None
        if var_name == 'REFC':
            cmap, norm, vmin, vmax = self.get_refc_cmap()
        elif var_name == 'APCP':
            cmap, norm, vmin, vmax = self.get_apcp_cmap()
        elif var_name in DERIVED_META:
            cmap, norm, vmin, vmax = self.get_apcp_accum_cmap()
        elif var_name == 'CAPE':
            cmap, norm, vmin, vmax = self.get_cape_cmap()
        elif var_name == 'CIN':
            cmap, norm, vmin, vmax = self.get_cin_cmap()
        elif var_name == 'VIS':
            cmap, norm, vmin, vmax = self.get_vis_cmap()
        elif var_name == 'HGTCC':
            cmap, norm, vmin, vmax = self.get_hgtcc_cmap()
        else:
            cmap = var_meta.get('cmap', self.config.cmap_default)
            norm = None
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
        
        # Create figure
        if self.use_cartopy:
            fig = plt.figure(figsize=self.config.figure_size)
            ax = plt.axes(projection=ccrs.PlateCarree())
            self._add_map_features(ax)
            if self.config.zoom_extent is not None:
                ax.set_extent(self.config.zoom_extent, crs=ccrs.PlateCarree())

            ax.gridlines(draw_labels=False)
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create the plot
        if norm is not None:
            im = ax.contourf(lons, lats, data, levels=norm.boundaries, 
                           cmap=cmap, norm=norm, extend='both')
        else:
            im = ax.contourf(lons, lats, data, levels=20, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label(f'{long_name} ({units})', fontsize=10)
        
        # Set title
        level_str = f" at {level} hPa" if level is not None else ""
        title = f"{long_name}{level_str}{title_suffix}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Set labels
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_pressure_level_variables(self, ds: xr.Dataset, lead_hour: int, 
                                    output_dir: str, timestamp_str: str) -> None:
        """Plot all pressure level variables."""
        logger.info("Plotting pressure level variables...")
        
        # Get coordinate data
        lats = ds['latitude'].values
        lons = ds['longitude'].values
        
        # Create title suffix with forecast information
        title_suffix = f"\n{self.config.run_label}Forecast: {timestamp_str} + {lead_hour}h"
        
        # Plot each variable at each level
        levels_in_ds = ds['level'].values if 'level' in ds.dims or 'level' in ds.coords else self.config.levels
        for var_idx, var_name in enumerate(self.config.pl_vars):
            if var_name not in ds.variables:
                logger.warning(f"Variable {var_name} not found in dataset")
                continue

            for level_idx, level in enumerate(levels_in_ds):
                if not product_wanted(f"{var_name}_{level}hPa", self.config.products):
                    continue
                try:
                    # Extract data for this variable and level
                    # Use lead_time dimension and select first time step (time=0)
                    # Per-hour file has a single lead_time slot; index 0
                    data = ds[var_name].isel(time=0, lead_time=0, level=level_idx).values
                    logger.info(f"{var_name} stats - mean: {np.nanmean(data):.2f}, std: {np.nanstd(data):.2f}, min: {np.nanmin(data):.2f}, max: {np.nanmax(data):.2f}")
                    
                    # Create plot
                    fig = self.create_plot(data, lats, lons, var_name, level, title_suffix)

                    # Save plot
                    filename = f"{var_name}_{level}hPa_lead{lead_hour:02d}h.png"
                    filepath = os.path.join(output_dir, filename)
                    save_png(fig, filepath, self.config.dpi, self.config.quantize)
                    plt.close(fig)
                    
                    logger.info(f"Saved: {filename}")
                    
                except Exception as e:
                    logger.error(f"Error plotting {var_name} at {level} hPa: {e}")
                    continue
    
    def plot_surface_variables(self, ds: xr.Dataset, lead_hour: int, 
                              output_dir: str, timestamp_str: str) -> None:
        """Plot surface variables."""
        logger.info("Plotting surface variables...")
        
        # Get coordinate data
        lats = ds['latitude'].values
        lons = ds['longitude'].values
        
        # Create title suffix with forecast information
        title_suffix = f"\n{self.config.run_label}Forecast: {timestamp_str} + {lead_hour}h"
        
        # Plot each surface variable
        for var_name in self.config.sfc_vars:
            if not product_wanted(f"{var_name}_surface", self.config.products):
                continue

            if var_name not in ds.variables:
                logger.warning(f"Variable {var_name} not found in dataset")
                continue
            
            try:
                # Extract data for this variable
                # Use lead_time dimension and select first time step (time=0)
                # Per-hour file has a single lead_time slot; index 0
                data = ds[var_name].isel(time=0, lead_time=0).values
                # log mean, std, min, max of data
                logger.info(f"{var_name} stats - mean: {np.nanmean(data):.2f}, std: {np.nanstd(data):.2f}, min: {np.nanmin(data):.2f}, max: {np.nanmax(data):.2f}")
                
                # Create plot
                fig = self.create_plot(data, lats, lons, var_name, None, title_suffix)
                
                # Save plot
                filename = f"{var_name}_surface_lead{lead_hour:02d}h.png"
                filepath = os.path.join(output_dir, filename)
                save_png(fig, filepath, self.config.dpi, self.config.quantize)
                plt.close(fig)
                
                logger.info(f"Saved: {filename}")
                
            except Exception as e:
                logger.error(f"Error plotting surface variable {var_name}: {e}")
                continue
    
    def create_summary_plot(self, ds: xr.Dataset, lead_hour: int, 
                           output_dir: str, timestamp_str: str) -> None:
        """Create a summary plot with key variables."""
        logger.info("Creating summary plot...")
        
        try:
            # Get coordinate data
            lats = ds['latitude'].values
            lons = ds['longitude'].values
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            if self.use_cartopy:
                # Recreate with cartopy if available
                fig = plt.figure(figsize=(16, 12))
                axes = []
                for i in range(4):
                    ax = plt.subplot(2, 2, i+1, projection=ccrs.PlateCarree())
                    self._add_map_features(ax)
                    if self.config.zoom_extent is not None:
                        ax.set_extent(self.config.zoom_extent, crs=ccrs.PlateCarree())
                    axes.append(ax)
            else:
                axes = axes.flatten()
            
            # Plot key variables
            plots = [
                ('T2M', 'T2M', None, 'Temperature at 2m'),
                ('REFC', 'REFC', None, 'Composite Reflectivity'),
                ('TMP', 'TMP', 850, 'Temperature at 850 hPa'),
                ('UGRD', 'UGRD', 850, 'U-Wind at 850 hPa'),
            ]
            
            for i, (var_name, var_display, level, title) in enumerate(plots):
                if var_name not in ds.variables:
                    continue
                
                # Get data
                if level is not None:
                    # Find level index
                    level_idx = self.config.levels.index(level) if level in self.config.levels else 0
                    data = ds[var_name].isel(time=0, lead_time=0, level=level_idx).values
                else:
                    data = ds[var_name].isel(time=0, lead_time=0).values
                
                # Get colormap from VARIABLE_METADATA
                var_meta = VARIABLE_METADATA.get(var_display, {})
                cmap = var_meta.get('cmap', self.config.cmap_default)
                
                # Special handling for REFC/APCP
                if var_display == 'REFC':
                    cmap_refc, norm_refc, *_ = self.get_refc_cmap()
                    im = axes[i].contourf(
                        lons, lats, data, levels=norm_refc.boundaries,
                        cmap=cmap_refc, norm=norm_refc, extend='both'
                    )
                elif var_display == 'APCP':
                    cmap_apcp, norm_apcp, *_ = self.get_apcp_cmap()
                    im = axes[i].contourf(
                        lons, lats, data, levels=norm_apcp.boundaries,
                        cmap=cmap_apcp, norm=norm_apcp, extend='both'
                    )
                else:
                    im = axes[i].contourf(lons, lats, data, levels=20, cmap=cmap, extend='both')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[i], shrink=0.4)
                
                # Set title
                axes[i].set_title(f"{title}\n{self.config.run_label}Forecast: {timestamp_str} + {lead_hour}h", 
                                fontsize=10, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save summary plot
            filename = f"summary_lead{lead_hour:02d}h.png"
            filepath = os.path.join(output_dir, filename)
            save_png(fig, filepath, self.config.dpi, self.config.quantize)
            plt.close(fig)
            
            logger.info(f"Saved: {filename}")
            
        except Exception as e:
            logger.error(f"Error creating summary plot: {e}")


# Plot domains. "tx" = the whole forecast grid (plots in YYYYMMDD/HH/<member>_leadNNh/);
# any other domain is a sub-box of it, plotted into YYYYMMDD/HH/<domain>/<member>_leadNNh/.
# Corners are (lat, lon) of HRRR grid points (SW, SE, NE, NW), so the box follows the grid.
DOMAINS = {
    "tx": None,
    # Harris County Flood Control District box (HRRR grid corners; TSC ft in the comments)
    "hcfcd": {
        "label": "HCFCD",
        "corners": [
            (29.340298, -96.144918),   # SW  X 2,877,994.82  Y 13,682,372.01
            (29.312892, -94.739308),   # SE  X 3,325,953.48  Y 13,686,035.04
            (30.833933, -94.686062),   # NE  X 3,322,509.65  Y 14,239,458.32
            (30.861971, -96.118770),   # NW  X 2,872,690.00  Y 14,235,821.99
        ],
        "pad": 3,             # extra grid points kept around the box so contours fill the frame
        "county_lw": 0.5,     # county lines matter at this scale
    },
}


def parse_domains(values) -> List[str]:
    if not values:
        return ["tx"]
    if isinstance(values, str):
        values = [values]
    out = []
    for v in values:
        for d in v.replace(",", " ").lower().split():
            if d not in DOMAINS:
                raise ValueError(f"Unknown domain '{d}' (known: {', '.join(DOMAINS)})")
            if d not in out:
                out.append(d)
    return out


def domain_view(ds: xr.Dataset, name: str):
    """(cropped dataset, map extent, outline) for a plot domain; (None, ...) if it's off-grid."""
    spec = DOMAINS[name]
    if spec is None:
        return ds, None, None
    lats = ds["latitude"].values
    lons = ((ds["longitude"].values + 180.0) % 360.0) - 180.0
    ys, xs = [], []
    for la, lo in spec["corners"]:
        d = (lats - la) ** 2 + ((lons - lo) * np.cos(np.deg2rad(la))) ** 2
        y, x = np.unravel_index(np.argmin(d), d.shape)
        if np.sqrt(d[y, x]) > 0.05:      # corner not inside this grid (~5 km tolerance)
            return None, None, None
        ys.append(y); xs.append(x)
    pad = spec.get("pad", 3)
    ny, nx = lats.shape
    y0, y1 = max(0, min(ys) - pad), min(ny, max(ys) + pad + 1)
    x0, x1 = max(0, min(xs) - pad), min(nx, max(xs) + pad + 1)
    ydim, xdim = ds["latitude"].dims
    sub = ds.isel({ydim: slice(y0, y1), xdim: slice(x0, x1)})
    clat = [c[0] for c in spec["corners"]]
    clon = [c[1] for c in spec["corners"]]
    m = 0.02
    extent = (min(clon) - m, max(clon) + m, min(clat) - m, max(clat) + m)
    outline = (clon + clon[:1], clat + clat[:1])
    return sub, extent, outline


def crop_like_members(ds, ds_path):
    """Show the operational HRRR on the same domain as the HRRRCast members, even if its file
    holds a bigger grid (get_hrrr_fcst.py ran before the HRRRCast output existed)."""
    try:
        from pathlib import Path
        from get_hrrr_fcst import find_reference, crop_dataset_to_reference
        ref = find_reference(Path(os.path.dirname(ds_path)))
        return crop_dataset_to_reference(ds, ref) if ref else ds
    except Exception as e:
        logging.warning(f"Could not crop HRRR to the HRRRCast domain: {e}")
        return ds


def plot_lead_hour(h, ds_path, init_datetime, init_year, init_month, init_day, init_hh, output_dir, date_str, member, config_dict):
    # Reconstruct config and plotter
    config = ForecastPlotterConfig()
    for k, v in config_dict.items():
        setattr(config, k, v)
    if member == "hrrr":
        config.run_label = "HRRR "
    plotter = ForecastPlotter(config)
    base_zoom, base_county_lw = config.zoom_extent, config.county_lw
    ds = xr.open_dataset(ds_path, decode_timedelta=True)
    if member == "hrrr":
        ds = crop_like_members(ds, ds_path)
    # accumulated precipitation from derived_precip.py, if it has been run
    precip_path = ds_path.replace(f"_f{h:02d}.nc", f"_precip_f{h:02d}.nc")
    dsp = None
    if member != "lpmm" and os.path.exists(precip_path):
        dsp = xr.open_dataset(precip_path, decode_timedelta=True)
        if member == "hrrr":
            dsp = crop_like_members(dsp, ds_path)
    try:
        timestamp_str = f"{init_year}-{init_month}-{init_day} {init_hh}:00 UTC"
        for dom in (config.domains or ["tx"]):
            dsd, extent, outline = domain_view(ds, dom)
            if dsd is None:
                logging.warning(f"Domain {dom} is not inside {os.path.basename(ds_path)}; skipped")
                continue
            spec = DOMAINS[dom] or {}
            config.zoom_extent = extent if extent else base_zoom
            config.outline = outline
            config.county_lw = spec.get("county_lw", base_county_lw)
            # member is m00 / avg / spr / lpmm / hrrr; non-default domains get their own folder
            sub = "" if dom == "tx" else f"{dom}/"
            output_subdir = f"{output_dir}/{date_str}/{sub}{member}_lead{h:02d}h"
            utils.make_directory(output_subdir)
            if member == "lpmm":
                # LPMM file only holds the precipitation products
                plotter.plot_derived_precip(dsd, h, output_subdir, timestamp_str, label=" (local PMM)")
            else:
                plotter.plot_pressure_level_variables(dsd, h, output_subdir, timestamp_str)
                plotter.plot_surface_variables(dsd, h, output_subdir, timestamp_str)
                if product_wanted("summary", config.products):
                    plotter.create_summary_plot(dsd, h, output_subdir, timestamp_str)
                if dsp is not None:
                    dspd = domain_view(dsp, dom)[0]
                    if dspd is not None:
                        label = " (PMM)" if member == "avg" else ""
                        plotter.plot_derived_precip(dspd, h, output_subdir, timestamp_str, label=label)
            logging.info(f"Plots for lead hour {h} ({dom}) saved to: {output_subdir}")
    finally:
        ds.close()
        if dsp is not None:
            dsp.close()

def plot_forecast_data(datetime_str: str,
                      lead_hour: str, member: str,
                      zoom_extent: Optional[tuple] = None,
                      forecast_dir: str = "./", output_dir: str = "./",
                      products: Optional[List[str]] = None,
                      counties: Optional[bool] = None,
                      domains: Optional[List[str]] = None):
    """Main plotting function. Plots all hours from 1 to lead_hour (inclusive) in parallel."""
    try:
        # Validate inputs
        init_datetime, init_year, init_month, init_day, init_hh = utils.validate_datetime(datetime_str)
        date_str = f"{init_year}{init_month}{init_day}/{init_hh}"
        lead_hour_int = int(lead_hour)
        
        # Normalize 'pmm' alias to 'avg'
        if member == "pmm":
            member = "avg"
        mem_str = str(member)
        if mem_str not in {"avg", "spr", "lpmm", "hrrr"}:
            mem_str = f"m{int(member):02d}"

        # Initialize plotter config (for passing to subprocesses)
        config = ForecastPlotterConfig()
        config.zoom_extent = zoom_extent
        config.products = products
        config.domains = domains or ["tx"]
        if counties is not None:
            config.counties = counties
        if config.counties:
            # fetch the county shapefile once here, before the worker processes start,
            # so parallel workers don't all try to download it at the same time
            config.counties = ensure_county_shapes()
        config_dict = config.__dict__
        
        n_workers = lead_hour_int
        logger.info(f"Parallel plotting using {n_workers} workers (one per lead hour)")
        # Parallel plotting over lead hours
        args_list = []
        for h in range(1, lead_hour_int + 1):
            # Build per-hour file path
            ds_path = f"{forecast_dir}/{date_str}/hrrrcast_{mem_str}_f{h:02d}.nc"
            if not os.path.exists(ds_path):
                logger.warning(f"Skipping hour f{h:02d}: file not found {ds_path}")
                continue
            args_list.append((h, ds_path, init_datetime, init_year, init_month, init_day, init_hh, output_dir, date_str, mem_str, config_dict))
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(plot_lead_hour, *args) for args in args_list]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in parallel plotting: {e}")
        logger.info(f"Plotting completed successfully for all hours 1 to {lead_hour_int}.")
        
    except Exception as e:
        logger.error(f"Plotting failed: {e}")
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot Forecast Variables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('inittime',
                       help='Forecast initialization time in format YYYY-MM-DDTHH (e.g., "2024-05-06T23")')
    parser.add_argument("lead_hour", help="Lead hour for forecast (0, 1, 2, ...)")
    parser.add_argument("--members", nargs='+', required=True, help="List/range of member IDs (e.g., 0-2 4 6-7 pmm spr lpmm hrrr)")
    parser.add_argument("--forecast_dir", default="./", help="Directory containing forecast files")
    parser.add_argument("--output_dir", default="./", help="Output directory for plots")
    parser.add_argument(
        "--lat-range",
        dest="lat_range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LAT_MIN", "LAT_MAX"),
        help="Latitude zoom bounds for map extent (e.g., --lat-range 36 50)",
    )
    parser.add_argument(
        "--lon-range",
        dest="lon_range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LON_MIN", "LON_MAX"),
        help="Longitude zoom bounds for map extent (e.g., --lon-range 259 272)",
    )
    parser.add_argument("--domains", nargs="+", default=None,
                        help=f"Plot domains: {', '.join(DOMAINS)} (default: $HRRRCAST_PLOT_DOMAINS, else tx). "
                             "tx = the full forecast grid; others go in YYYYMMDD/HH/<domain>/")
    parser.add_argument("--no-counties", dest="counties", action="store_false", default=None,
                        help="Do not draw county borders (default: drawn; or set HRRRCAST_PLOT_COUNTIES=0)")
    parser.add_argument("--products", nargs="+", default=None,
                        help="Only plot these products (names or wildcards), e.g. REFC APCP T2M HGT_500hPa summary. "
                             "Default: $HRRRCAST_PLOT_PRODUCTS if set, else everything")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    global logger
    args = parse_arguments()
    logger = setup_logging(args.log_level)

    try:
        try:
            lat_bounds = _normalize_range(args.lat_range, "lat_range")
            lon_bounds = _normalize_range(args.lon_range, "lon_range")
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

        zoom_extent = None
        if lat_bounds is not None and lon_bounds is not None:
            zoom_extent = (lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1])
        elif lat_bounds is not None or lon_bounds is not None:
            logger.error("Both --lat-range and --lon-range must be provided together for zooming")
            sys.exit(1)

        def expand_member_arg(m):
            result = []
            for part in m.split(","):
                part = part.strip()
                if "-" in part and part.replace("-", "").isdigit():
                    start, end = part.split("-")
                    result.extend([str(i) for i in range(int(start), int(end) + 1)])
                elif part != "":
                    result.append(part)
            return result

        members = []
        for m in args.members:
            members.extend(expand_member_arg(m))
        members = sorted(set(members), key=lambda x: (not x.isdigit(), x))

        products = parse_product_patterns(args.products or os.environ.get("HRRRCAST_PLOT_PRODUCTS"))
        logger.info(f"Products: {' '.join(products) if products else 'all'}")
        try:
            domains = parse_domains(args.domains or os.environ.get("HRRRCAST_PLOT_DOMAINS"))
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)
        logger.info(f"Domains: {' '.join(domains)}")

        for member in members:
            plot_forecast_data(
                datetime_str=args.inittime,
                lead_hour=args.lead_hour,
                member=member,
                zoom_extent=zoom_extent,
                forecast_dir=args.forecast_dir,
                output_dir=args.output_dir,
                products=products,
                counties=args.counties,
                domains=domains,
            )
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
