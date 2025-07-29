# This script is inspired by Nikolay Koldunov's native_to_1degree.py.
# Check out his repository: https://github.com/koldunovn/aimip

import xarray as xr

def regrid_to_template_grid(native_ds: xr.Dataset, template_ds: xr.Dataset, output_path: str) -> xr.Dataset:
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

    # copy attributes with variable_id
    if "variable_id" in native_ds.attrs:
        regridded_ds.attrs["variable_id"] = native_ds.attrs["variable_id"]

    # coordinate cleanup (lat/lon already interpolated)
    regridded_ds["lat"].attrs = template_ds["lat"].attrs.copy()
    regridded_ds["lon"].attrs = template_ds["lon"].attrs.copy()

    regridded_ds.to_netcdf(output_path)

    return regridded_ds
