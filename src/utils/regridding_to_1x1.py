# Besides some minor changes, this script is taken from Nikolay Koldunov and all credits belong to him!
# Check out his repository: https://github.com/koldunovn/aimip (native_to_1degree.py)

import xarray as xr
import numpy as np
import os
import argparse
from datetime import datetime
import pytz
import ast
import sys
from logger import WARN

# Setting up logger - making sure it logs into same pipeline.log as rest of repo
_here = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path.insert(0, project_root)
from src.config_loader import ConfigLoader
from src.logger import setup_logger
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

def generate_example_regridded_data(template_ds):
    """Generates example regridded data based on a template dataset."""
    logger.info("[Regridding] Generating Example Regridded Data")
    
    main_var_name = template_ds.attrs.get('variable_id')
    if not main_var_name or main_var_name not in template_ds.data_vars:
        raise ValueError("Cannot determine main variable from 'variable_id' attribute in template.")
    logger.debug(f"[Regridding] Identified main variable for interpolation: {main_var_name}")

    # Define the target regular grid
    new_lon = np.linspace(0.5, 359.5, 360)
    new_lat = np.linspace(-89.5, 89.5, 180)

    # Add cyclic point for longitude to handle wrap-around
    wrap_ds = xr.concat([template_ds, template_ds.isel(lon=0)], dim='lon', compat='override', coords='all')
    wrap_ds['lon'] = np.append(template_ds['lon'], 360.0)

    # Interpolate only the main data variable
    regridded_da = wrap_ds[main_var_name].interp(lat=new_lat, lon=new_lon, method='linear')
    
    regridded_ds = regridded_da.to_dataset(name=main_var_name)

    logger.debug("[Regridding] Example data generated.")
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
    for var_name in cmor_ds.variables:
        if var_name in template_ds:
            cmor_ds[var_name].attrs = template_ds[var_name].attrs.copy()
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
        scalar_height.attrs = original_height.attrs
        cmor_ds['height'] = scalar_height
        cmor_ds[main_var_name].attrs['coordinates'] = 'height'
        logger.debug(f"[Regridding] Processed height as scalar with dims: {cmor_ds['height'].dims}")

    logger.info("[Regridding] Creating new lat/lon bounds")
    cmor_ds['lat_bnds'] = xr.DataArray(create_bounds(cmor_ds['lat'].values), dims=['lat', 'bnds'], name='lat_bnds')
    cmor_ds['lon_bnds'] = xr.DataArray(create_bounds(cmor_ds['lon'].values), dims=['lon', 'bnds'], name='lon_bnds')
    cmor_ds['lat'].attrs['bounds'] = 'lat_bnds'
    cmor_ds['lon'].attrs['bounds'] = 'lon_bnds'
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

        if "chunksizes" in var_encoding and var_encoding["chunksizes"] is None:
            del var_encoding["chunksizes"]
            logger.debug(f"[Regridding] Removed invalid chunksizes from: {var_name}")

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

    logger.info("[Regridding] Saving to NetCDF")
    cmor_ds.to_netcdf(output_file, encoding=encoding, unlimited_dims=['time'])
    logger.info(f"[Regridding] Successfully created {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a CMOR-compliant NetCDF file from a template.')
    parser.add_argument('template_file', type=str, help='Path to the template NetCDF file.')
    parser.add_argument(
        '--override', '-o',
        action='append',
        dest='overrides',
        default=[],
        help='Metadata override in the form KEY=VALUE (changes tbd in config.yaml)'
    )
    args = parser.parse_args()

    METADATA_OVERRIDES = {}
    for item in args.overrides:
        key, val = item.split("=", 1)
        try:
            val = ast.literal_eval(val)
        except Exception:
            pass
        METADATA_OVERRIDES[key] = val

    with xr.open_dataset(args.template_file, decode_times=False) as template_ds:
        # 1. Generate some example data (in a real scenario, this would be provided)
        regridded_data = generate_example_regridded_data(template_ds)

        # 2. Define the output file path
        output_filename = os.path.basename(args.template_file).replace('_gn_', '_gr_')
        output_path = os.path.join(os.path.dirname(args.template_file), output_filename)

        # 3. Create the CMOR file
        create_cmor_file(template_ds, regridded_data, output_path, METADATA_OVERRIDES)
