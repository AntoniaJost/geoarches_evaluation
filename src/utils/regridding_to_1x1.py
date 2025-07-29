# Parts of this script are taken from Nikolay Koldunov and all credits belong to him!
# Check out his repository: https://github.com/koldunovn/aimip (native_to_1degree.py)

import xarray as xr
import numpy as np
import os
import argparse
from datetime import datetime
import pytz
import ast
import sys

# Setting up logger - making sure it logs into same pipeline.log as rest of repo
_here = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path.insert(0, project_root)
from src.config_loader import ConfigLoader
from src.logger import setup_logger, WARN
# Attaching logs from here to overall logfile
cfg = ConfigLoader("config.yaml")
general = cfg.get("general") if hasattr(cfg, "get") else cfg.general
logger = setup_logger(
    general["log_dir"],
    general["log_level"]
)

def create_bounds(coord_vals):
    """Creates a bounds array for a given 1D coordinate array."""
    bounds = np.zeros((len(coord_vals), 2))
    diffs = np.diff(coord_vals) / 2
    bounds[:-1, 1] = coord_vals[:-1] + diffs
    bounds[1:, 0] = coord_vals[1:] - diffs
    bounds[0, 0] = coord_vals[0] - diffs[0]
    bounds[-1, 1] = coord_vals[-1] + diffs[-1]
    return bounds


def regrid_to_template_grid(native_ds: xr.Dataset, template_ds: xr.Dataset) -> xr.Dataset:
    """
    Regrid the main variable of a native-resolution ERA5 dataset to the grid of a CMIP6 template,
    while preserving key coordinates and bounds like time_bnds, lat_bnds, lon_bnds.
    """
    # identify main variable name
    var_name = list(native_ds.data_vars)[0]

    # regrid the data to the template lat/lon grid
    regridded_da = native_ds[var_name].interp(
        lat=template_ds.lat,
        lon=template_ds.lon,
        method='linear'
    )

    # build a new dataset with regridded variable
    regridded_ds = xr.Dataset({var_name: regridded_da})

    # copy over time + time_bnds from native data (if present)
    if "time" in native_ds:
        regridded_ds["time"] = native_ds["time"]
        regridded_ds["time"].attrs = native_ds["time"].attrs.copy()
    if "time_bnds" in native_ds:
        regridded_ds["time_bnds"] = native_ds["time_bnds"]
        regridded_ds["time_bnds"].attrs = native_ds["time_bnds"].attrs.copy()
        if "bounds" not in regridded_ds["time"].attrs:
            regridded_ds["time"].attrs["bounds"] = "time_bnds"

    # copy height if exists
    if "height" in native_ds:
        regridded_ds["height"] = native_ds["height"]
        regridded_ds["height"].attrs = native_ds["height"].attrs.copy()

    # coordinate cleanup (lat/lon already interpolated)
    regridded_ds["lat"].attrs = template_ds["lat"].attrs.copy()
    regridded_ds["lon"].attrs = template_ds["lon"].attrs.copy()

    return regridded_ds

def create_cmor_file(template_ds, regridded_ds, output_file, metadata_overrides):
    """Creates a CMOR-compliant NetCDF file from a template and new data."""
    logger.info(f"[Regridding] Creating CMOR File: {output_file}")
    
    # Determine the main variable name robustly
    main_var_name = template_ds.attrs.get('variable_id')
    if not main_var_name or main_var_name not in regridded_ds.data_vars:
        logger.warning(f"{WARN} [Regridding] 'variable_id' attribute not found or not in regridded data. Inferring main variable.")
        candidate_vars = {
            name: da.ndim 
            for name, da in regridded_ds.data_vars.items() 
            if '_bnds' not in name and 'bnds' not in da.dims
        }
        if not candidate_vars:
            raise ValueError("Could not determine the main data variable in the provided regridded dataset.")
        main_var_name = max(candidate_vars, key=candidate_vars.get)

    cmor_ds = regridded_ds.copy(deep=True)
    logger.debug(f"[Regridding] Main data variable: {main_var_name}")

    cmor_ds.attrs = template_ds.attrs.copy()
    logger.debug("[Regridding] Copied global attributes.")

    logger.info("[Regridding] Copying Variable Attributes and Restoring Missing Value")
    encoding_keys = ['calendar', '_FillValue']
    for var_name in cmor_ds.variables:
        if var_name in template_ds:
            attrs = template_ds[var_name].attrs.copy()
            # remove 'units' manually from 'time' to avoid xarray conflict
            if var_name == "time" and "units" in attrs:
                del attrs["units"]
            for k in encoding_keys:
                if k in attrs:
                    del attrs[k]
            cmor_ds[var_name].attrs = attrs
            logger.debug(f"[Regridding] Copied attributes for: {var_name}")

    # Restore missing_value attribute on main variable, which might not be in attrs
    if main_var_name in cmor_ds.data_vars and main_var_name in template_ds:
        template_encoding = template_ds[main_var_name].encoding
        fill_value = template_encoding.get('_FillValue') or template_encoding.get('missing_value')
        if fill_value is not None:
            cmor_ds[main_var_name].attrs['missing_value'] = fill_value
            logger.debug(f"[Regridding] Ensured 'missing_value' attribute exists for '{main_var_name}' with value {fill_value}")

    logger.info("[Regridding] Copying Non-Spatial Variables from Template")
    template_vars_to_copy = [v for v in template_ds.variables if v not in cmor_ds.variables]
    for var_name in template_vars_to_copy:
        if var_name == 'height': continue # Handle height separately
        cmor_ds[var_name] = template_ds[var_name]
        logger.debug(f"[Regridding] Copied variable: {var_name} with dims {template_ds[var_name].dims}")

    if 'height' in template_ds:
        logger.debug("[Regridding] Processing 'height' coordinate")
        original_height = template_ds['height']
        logger.debug(f"[Regridding] Original height dims: {original_height.dims}")
        scalar_height = original_height.isel({dim: 0 for dim in original_height.dims}, drop=True)
        scalar_height.attrs = original_height.attrs.copy()
        cmor_ds['height'] = scalar_height
        if 'coordinates' not in cmor_ds[main_var_name].attrs:
            cmor_ds[main_var_name].attrs['coordinates'] = 'height'
        logger.debug(f"[Regridding] Processed height as scalar with dims: {cmor_ds['height'].dims}")

    logger.info("[Regridding] Creating new lat/lon bounds")
    cmor_ds['lat_bnds'] = xr.DataArray(create_bounds(cmor_ds['lat'].values), dims=['lat', 'bnds'], name='lat_bnds')
    cmor_ds['lon_bnds'] = xr.DataArray(create_bounds(cmor_ds['lon'].values), dims=['lon', 'bnds'], name='lon_bnds')
    cmor_ds['lat'].attrs['bounds'] = 'lat_bnds'
    cmor_ds['lon'].attrs['bounds'] = 'lon_bnds'
    # Add fallback attributes to lat/lon
    cmor_ds['lat'].attrs.setdefault('units', 'degrees_north')
    cmor_ds['lat'].attrs.setdefault('standard_name', 'latitude')
    cmor_ds['lat'].attrs.setdefault('long_name', 'Latitude')
    cmor_ds['lat'].attrs.setdefault('axis', 'Y')

    cmor_ds['lon'].attrs.setdefault('units', 'degrees_east')
    cmor_ds['lon'].attrs.setdefault('standard_name', 'longitude')
    cmor_ds['lon'].attrs.setdefault('long_name', 'Longitude')
    cmor_ds['lon'].attrs.setdefault('axis', 'X')

    logger.debug("[Regridding] Created 'lat_bnds' and 'lon_bnds'")

    logger.info("[Regridding] Applying Metadata Overrides")
    for key, value in metadata_overrides.items():
        old = cmor_ds.attrs.get(key, "<not set>")
        logger.debug(f"[Regridding] Setting global attribute '{key}': '{old}' -> '{value}'")
        cmor_ds.attrs[key] = value

    history_update = f"{datetime.now(pytz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}: Data regridded and CMORized."
    cmor_ds.attrs['history'] = f"{history_update} ; {template_ds.attrs.get('history', '')}"
    logger.debug("[Regridding] Updated history attribute.")

    # Demote 'height' from a coordinate to a data variable to prevent xarray
    # from automatically adding 'coordinates' attributes to other variables.
    if 'height' in cmor_ds.coords:
        cmor_ds = cmor_ds.reset_coords(['height'])
        logger.debug("[Regridding] Demoted 'height' from coordinate to data variable to fix attribute propagation.")

    logger.info("[Regridding] Preparing Encodings and Final Cleanup")
    encoding = {}
    valid_keys = {'_FillValue', 'dtype', 'scale_factor', 'add_offset', 'units', 'calendar', 'zlib', 'complevel', 'shuffle', 'fletcher32', 'contiguous', 'chunksizes', 'least_significant_digit'}

    for var_name in cmor_ds.variables:
        da = cmor_ds[var_name]
        var_encoding = template_ds[var_name].encoding.copy() if var_name in template_ds else {}

        # General cleanup
        if "_FillValue" in var_encoding and var_encoding["_FillValue"] is None:
            del var_encoding["_FillValue"]
            logger.debug(f"[Regridding] Removed _FillValue from encoding for: {var_name}")

        if "chunksizes" in var_encoding:
            shape = cmor_ds[var_name].shape
            chunks = var_encoding["chunksizes"]

            if chunks is None or any(c > s for c, s in zip(chunks, shape)):
                del var_encoding["chunksizes"]
                logger.debug(f"[Regridding] Removed invalid chunksizes from {var_name}: {chunks} for shape {shape}")

        if "contiguous" in var_encoding:
            del var_encoding["contiguous"]
            logger.debug(f"[Regridding] Removed 'contiguous' from encoding for: {var_name}")

        # Drop bad compression keys if chunking isn't defined
        if "zlib" in var_encoding and "chunksizes" not in var_encoding:
            for key in ["zlib", "shuffle", "complevel", "fletcher32"]:
                var_encoding.pop(key, None)
            logger.debug(f"[Regridding] Removed compression-related keys from: {var_name}")

        # Remove 'coordinates' attr for non-main variables
        if "coordinates" in da.attrs and var_name != main_var_name:
            del da.attrs["coordinates"]
            logger.debug(f"[Regridding] Removed 'coordinates' attribute from: {var_name}")

        # Exclude time from encoding altogether
        if var_name == "time":
            logger.debug("[Regridding] Skipping encoding for 'time'")
            continue

        encoding[var_name] = {k: v for k, v in var_encoding.items() if k in valid_keys}

    # Ensure bnds variables have minimal encoding so they are preserved
    for bnd_var in ['lat_bnds', 'lon_bnds', 'time_bnds']:
        if bnd_var in cmor_ds and bnd_var not in encoding:
            encoding[bnd_var] = {'dtype': 'double'}
            logger.debug(f"[Regridding] Added fallback encoding for {bnd_var}")

    logger.info("[Regridding] Saving to NetCDF")
    cmor_ds.to_netcdf(output_file, encoding=encoding, unlimited_dims=['time'])
    logger.info(f"[Regridding] Successfully created {output_file}")
