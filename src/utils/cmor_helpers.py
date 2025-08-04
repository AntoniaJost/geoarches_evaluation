import numpy as np
import xarray as xr
import sys, os
import logging
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))
from logger import WARN

logger = logging.getLogger("cmor")

def add_lat_lon_bounds(ds):
    if 'lat_bnds' not in ds and 'lat' in ds:
        lat = ds['lat'].values
        lat_bnds = _infer_bounds(lat)
        ds['lat_bnds'] = (('lat', 'bnds'), lat_bnds)
        ds['lat'].attrs.update({
            "bounds": "lat_bnds",
            "units": "degrees_north",
            "long_name": "Latitude",
            "standard_name": "latitude"
        })

    if 'lon_bnds' not in ds and 'lon' in ds:
        lon = ds['lon'].values
        lon_bnds = _infer_bounds(lon)
        ds['lon_bnds'] = (('lon', 'bnds'), lon_bnds)
        ds['lon'].attrs.update({
            "bounds": "lon_bnds",
            "units": "degrees_east",
            "long_name": "Longitude",
            "standard_name": "longitude"
        })

    return ds

def add_time_bounds(ds):
    time = ds['time'].values
    if 'time_bnds' not in ds and len(time) > 1:
        dt = (time[1] - time[0]) / 2
        bounds = np.stack([time - dt, time + dt], axis=1)
        ds['time_bnds'] = (('time', 'bnds'), bounds)
        ds['time'].attrs.update({
            "bounds": "time_bnds",
            "long_name": "time"
        })
    return ds

def fix_plev(ds):
    if 'plev' in ds.coords:
        ds['plev'].attrs.update({
            "units": "Pa", # ?? or hPa?
            "positive": "down",
            "long_name": "pressure",
            "standard_name": "air_pressure"
        })
    return ds

def fix_variable_metadata(ds, var_name, metadata_dict):
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

def _infer_bounds(values):
    step = np.diff(values)
    lower = values - step[0]/2
    upper = values + step[0]/2
    return np.stack([lower, upper], axis=1)
