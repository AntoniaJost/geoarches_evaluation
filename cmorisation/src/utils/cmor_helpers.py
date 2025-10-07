import numpy as np
import xarray as xr
import sys, os
import logging
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))
from logger import WARN

logger = logging.getLogger("cmor")

def add_lat_lon_bounds(ds):
    """
    adds CMOR-compliant lat/lon bounds to a dataset if missing
    """
    if 'lat_bnds' not in ds and 'lat' in ds:
        lat = ds['lat'].values
        lat_bnds = _infer_bounds(lat, is_lon=False)
        ds['lat_bnds'] = (('lat', 'bnds'), lat_bnds)
        ds['lat'].attrs.update({
            "bounds": "lat_bnds",
            "units": "degrees_north",
            "long_name": "Latitude",
            "standard_name": "latitude"
        })

    if 'lon_bnds' not in ds and 'lon' in ds:
        lon = ds['lon'].values
        lon_bnds = _infer_bounds(lon, is_lon=True)
        ds['lon_bnds'] = (('lon', 'bnds'), lon_bnds)
        ds['lon'].attrs.update({
            "bounds": "lon_bnds",
            "units": "degrees_east",
            "long_name": "Longitude",
            "standard_name": "longitude"
        })

    return ds

def add_time_bounds(ds):
    """
    adds time bounds if missing
    """
    time = ds['time'].values
    ds['time'].attrs.update({"long_name": "time"})
    if 'time_bnds' not in ds and len(time) > 1:
        dt = (time[1] - time[0]) / 2
        bounds = np.stack([time - dt, time + dt], axis=1)
        ds['time_bnds'] = (('time', 'bnds'), bounds)
        ds['time'].attrs.update({
            "bounds": "time_bnds"
        })
    return ds

def fix_plev(ds):
    """
    convert from hPa/mbar to Pa if needed
    updates pressure level metadata to cmor standard
    """
    if 'plev' in ds.coords:
        pressure = ds.coords['plev']
        # units = pressure.attrs.get("units").lower()
        units = (pressure.attrs.get("units", "") or "").lower()
        # decide if conversion is needed:
        # either units explicitly say hPa/mbar, 
        # or magnitudes look like hPa (typical max < 2000)
        needs_convert = (
            units in {"hpa", "hectopascal", "millibar", "mbar"} or 
            (pressure.size > 0 and
            (float(pressure.max()) if pressure.dtype.kind in "fi" else float(pressure.astype("float64").max())) < 2000.0)
        )
        if needs_convert:
            ds['plev'] = pressure.astype("float64") * 100
        # update attributes    
        ds['plev'].attrs.update({
            "units": "Pa", 
            "positive": "down",
            "axis": "Z",
            "long_name": "pressure",
            "standard_name": "air_pressure"
        })
    return ds

def fix_variable_metadata(ds, var_name, metadata_dict):
    """
    injects metadata attributes into variable
    uses values from provided metadata dictionary
    """
    if var_name not in ds.data_vars:
        return ds  

    coords = ds[var_name]
    if var_name not in list(metadata_dict.keys()):
        logger.info(f"{WARN} No metadata available for {var_name}, using default values which might not be CMOR conform.")
        attrs = coords.attrs
        attrs.setdefault("standard_name", var_name)
        attrs.setdefault("long_name", var_name)
        attrs.setdefault("comment", "unknown")
        attrs.setdefault("units", "unknown")
        attrs.setdefault("cell_methods", "(area:) time: mean")
        attrs.setdefault("cell_measures", "area: areacella")
        ds[var_name].attrs.update(attrs)

    elif var_name in metadata_dict:
        for key in ["standard_name", "long_name", "comment", "units", "cell_methods", "cell_measures"]:
            if key in metadata_dict[var_name]:
                coords.attrs[key] = metadata_dict[var_name][key]
    coords.attrs.setdefault("history", "Processed for AIMIP")
    return ds


def inject_height_if_needed(ds, var):
    """
    injects a "height" coordinate (2m or 10m) if required
    only applies to tas, uas and vas
    """
    if var in ['tas', 'uas', 'vas'] and 'height' not in ds.coords:
        height_value = 2.0 if var == 'tas' else 10.0
        ds.coords['height'] = height_value
        ds['height'].attrs.update({
            "units": "m",
            "positive": "up",
            "axis": "Z",
            "long_name": "height",
            "standard_name": "height"
        })
    return ds

def _infer_bounds(values, is_lon=False):
    """
    infers bounds (for lat/lon) by computing midpoints
    assumes regularly spaced coordinates
    handles wrap-around for longitude if is_lon=True
    """
    logger.debug(f"\n[_infer_bounds] called for {'lon' if is_lon else 'lat'}")
    logger.debug(f"values[:10] = {values[:10]} ... total={len(values)}")

    # step = np.diff(values)
    # d = step[0]
    values = np.asarray(values)
    d = np.median(np.diff(values))
    logger.debug(f"d = {d}")
    
    lower = values - d/2
    upper = values + d/2
    logger.debug(f"lower[:5] (before wrap) = {lower[:5]}")
    logger.debug(f"upper[:5] (before wrap) = {upper[:5]}")

    if is_lon:
        # Wrap into [0, 360)
        lower = (lower + 360) % 360
        upper = (upper + 360) % 360
        logger.debug(f"lower[:5] (after wrap) = {lower[:5]}")
        logger.debug(f"upper[:5] (after wrap) = {upper[:5]}")

        # Special case: first cell should connect to last
        lower[0] = values[0] - d/2   # may be negative, e.g. -0.5
        upper[0] = values[0] + d/2   # small positive, e.g. 0.5
        logger.debug(f"lower[0] fixed = {lower[0]}")
        logger.debug(f"upper[-1] fixed = {upper[-1]}")
            # --- Fix last cell to end before 360° ---
        lower[-1] = values[-1] - d/2
        upper[-1] = values[-1] + d/2  # should be just below 360

    bounds = np.stack([lower, upper], axis=1)
    logger.debug(f"bounds[:5] = \n{bounds[:5]}")
    logger.debug(f"bounds[-5:] = \n{bounds[-5:]}")

    return bounds
