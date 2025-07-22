# Besides some minor changes, the cmorize_data_with_template function is taken from Nikolay Koldunov and all credits belong to him!
# Check out his repository: https://github.com/koldunovn/aimip (cmor_utils.py)

import xarray as xr
import os, sys
from datetime import datetime
import pytz
import numpy as np
import glob
import logging
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))
from logger import WARN

logger = logging.getLogger("cmor")

def cmorize_data_with_template(user_data_array, template_path, output_path, metadata_overrides):
    """
    Replaces the data of the main variable in a template NetCDF file with data from a user's
    in-memory NumPy array, updates metadata, and saves a new CMOR-compliant file.

    Args:
        user_data_array (np.ndarray): The user's n-dimensional data array.
        template_path (str): Path to the CMOR-compliant template NetCDF file.
        output_path (str): Path for the new output NetCDF file.
        metadata_overrides (dict): Dictionary of global attributes to update (e.g., {'source_id': 'NewModel'}).
    """
    logger.info(f"[cmorutils] --- Starting CMORization Process ---")
    logger.debug(f"[cmorutils] User data: In-memory NumPy array of shape {user_data_array.shape}")
    logger.debug(f"[cmorutils] Template: {template_path}")
    logger.debug(f"[cmorutils] Output: {output_path}")

    with xr.open_dataset(template_path, decode_times=False) as template_ds:

        # 1. Determine the main variable name from the template
        main_var_name = template_ds.attrs.get('variable_id')
        if not main_var_name or main_var_name not in template_ds.data_vars:
            raise ValueError("[cmorutils] Cannot determine main variable from 'variable_id' in template.")
        logger.debug(f"[cmorutils] Identified main data variable: '{main_var_name}'")

        # 2. Start with a deep copy of the template to preserve all metadata
        cmor_ds = template_ds.copy(deep=True)

        # 3. Replace the data array
        logger.info("[cmorutils] Replacing data with user-provided array.")
        # Ensure dimensions match before replacing data
        template_shape = cmor_ds[main_var_name].shape
        if user_data_array.shape != template_shape:
            raise ValueError(f"[cmorutils] Shape of user data array {user_data_array.shape} does not match shape of template data {template_shape}.")
        cmor_ds[main_var_name].data = user_data_array

        # 4. Apply metadata overrides
        logger.info("[cmorutils] Applying metadata overrides...")
        for key, value in metadata_overrides.items():
            if key in cmor_ds.attrs:
                logger.info(f"[cmorutils] Overriding global attribute '{key}': '{cmor_ds.attrs.get(key)}' -> '{value}'")
                cmor_ds.attrs[key] = value
            else:
                logger.info(f"[cmorutils] Adding new global attribute '{key}': '{value}'")
                cmor_ds.attrs[key] = value

        # 5. Update history
        history_update = f"{datetime.now(pytz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}: Data replaced with user-provided in-memory data."
        cmor_ds.attrs['history'] = f"{history_update} ; {template_ds.attrs.get('history', '')}"
        logger.info("[cmorutils] Updated history attribute.")

        # 6. Prepare encodings and perform final cleanup (lessons learned)
        logger.info("[cmorutils] Preparing encodings and performing final cleanup...")
        encoding = {}
        valid_keys = {'_FillValue', 'dtype', 'scale_factor', 'add_offset', 'units', 'calendar', 'zlib', 'complevel', 'shuffle', 'fletcher32', 'contiguous', 'chunksizes', 'least_significant_digit'}

        # Demote 'height' to prevent attribute propagation issues
        if 'height' in cmor_ds.coords:
            cmor_ds = cmor_ds.reset_coords(['height'])
            logger.debug("[cmorutils] Demoted 'height' from coordinate to data variable.")

        for var_name in cmor_ds.variables:
            da = cmor_ds[var_name]
            var_encoding = template_ds[var_name].encoding.copy() if var_name in template_ds else {}

            # Disable _FillValue for coordinates, bounds, and our special 'height' case
            if var_name in cmor_ds.coords or '_bnds' in var_name or var_name == 'height':
                var_encoding['_FillValue'] = None
            
            # Clean up rogue 'coordinates' attributes
            if 'coordinates' in da.attrs and var_name != main_var_name:
                 del da.attrs['coordinates']

            encoding[var_name] = {k: v for k, v in var_encoding.items() if k in valid_keys}

        # 7. Save the final file
        logger.debug("[cmorutils] Saving to NetCDF...")
        cmor_ds.to_netcdf(output_path, encoding=encoding, unlimited_dims=['time'])
        logger.info(f"[cmorutils] Successfully created {output_path}")

    return


def find_template_for_variable(variable: str, frequency: str, cfg: dict) -> str:
    """
    Find the correct CMOR template file based on variable and frequency.
    """
    # aimip specification
    cmor_freq = {
        "monthly": "Amon",
        "daily": "day"  
    }

    if frequency not in cmor_freq:
        raise ValueError(f"[CmorUtils] Unsupported frequency: {frequency}")

    cmor_subfolder = cmor_freq[frequency]

    base_template_dir = cfg.get('cmor_template_base_dir')
    #"/home/a/a270220/projects/cmorisation/data_read/MPI-M-local/MPI-ESM1-2-LR/aimip/r1i1p1f1"
    target_dir = os.path.join(base_template_dir, cmor_subfolder, variable, "gr", "v20190815")
    pattern = os.path.join(target_dir, f"{variable}_{cmor_subfolder}_MPI-ESM1-2-LR_amip_r1i1p1f1_gr_*.nc")

    # find and return first match (sorted)
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[0]


def adjust_template_time(
    template_ds: xr.Dataset,
    desired_len: int,
    desired_time_values: np.ndarray,
    logger: logging.Logger | None = None,
) -> xr.Dataset:
    """
    Ensure the time dimension of template_ds equals *desired_len*.
    Params:
        template_ds: The original CMOR template
        desired_len: # of timesteps in the user data array that will be written into the template later
        desired_time_values: 1D numpy array or pandas datetime index with the actual time stamps from the user dataset (will replace the template's)
        logger
    Returns:
        xr.Dataset: A copy of template_ds whose time coordinate now has desired_len

    • If the template already has the correct length it is returned unchanged.  
    • If the template is longer, the routine truncates the excess tail.  
    • If it is shorter, the data variables are repeated along time and then trimmed to exact length.
    The function drops time_bnds before repeating (bc bounds cannot be repeated safely) and re‑attaches the coordinate arrays afterwards.
    """
    current_len = template_ds.sizes.get("time", 0)
    if current_len == desired_len:
        return template_ds

    if logger:
        logger.debug(f"{WARN} [template_adjust] time mismatch: %s → %s timesteps", current_len, desired_len)

    # strip bounds we cannot broadcast safely
    template_core = template_ds.drop_vars(
        ["time_bnds", "lat_bnds", "lon_bnds"], errors="ignore"
    )
    # template too long
    if current_len > desired_len:  # simply slice off the tail
        adjusted = template_core.isel(time=slice(0, desired_len))
    # template too short
    else:  # repeat the shorter template in chunks
        reps = int(np.ceil(desired_len / current_len))
        repeated = xr.concat([template_core] * reps, dim="time")
        adjusted = repeated.isel(time=slice(0, desired_len))

    # replace the time coordinate with the real one
    adjusted = adjusted.assign_coords(time=("time", desired_time_values))

    # attach spatial bounds again
    for bnd in ("lat_bnds", "lon_bnds"):
        if bnd in template_ds:
            adjusted = adjusted.assign_coords({bnd: template_ds[bnd]})

    return adjusted


def adjust_template_grid(
    template_ds: xr.Dataset,
    new_lat: np.ndarray,
    new_lon: np.ndarray,
    logger: logging.Logger | None = None,
) -> xr.Dataset:
    """
    Replace latitude / longitude axes (and bounds) in template_ds when they differ from the user data grid.
    Params:
        template_ds: Dataset obtained from adjust_template_time (or the untouched template if time already matched).
        new_lat, new_lon: 1D numpy arrays containing the latitude / longitude values of the user dataset that will be written later.
        logger 
    Returns: 
        xr.Dataset: A copy of template_ds with lat / lon replaced if shapes differ. 
        If the shapes are identical the input dataset is returned unmodified.

    ❗️ This function does NOT perform spatial interpolation! It merely tells the template to pretend its data are on the new grid. 
    """
    ds = template_ds

    # get original grid size from template
    old_nlat, old_nlon = ds.sizes["lat"], ds.sizes["lon"]
    # get target grid size from user data
    new_nlat, new_nlon = new_lat.size, new_lon.size

    # skip if template and user grid already match
    if (old_nlat, old_nlon) == (new_nlat, new_nlon):
        return ds

    if logger:
        logger.debug(f"{WARN}[template_adjust] grid mismatch – {old_nlat}×{old_nlon} → {new_nlat}×{new_nlon}")

    # Bring the grid to the shape of the native grid
    # if template is larger → slice
    if old_nlat >= new_nlat and old_nlon >= new_nlon:
        ds = ds.isel(lat=slice(0, new_nlat), lon=slice(0, new_nlon))
    else:
        # if template smaller → pad with NaNs to requested size
        pad_lat = new_nlat - old_nlat
        pad_lon = new_nlon - old_nlon

        if pad_lat < 0 or pad_lon < 0:
            raise ValueError("Unexpected negative padding – logic error")

        # pad a single dataArray along one dimension with NaNs
        def _pad_dim(da, dim_name, new_len):
            if da.sizes.get(dim_name, None) is None:
                return da  # variable does not have this dimension
            pad_shape = list(da.shape)
            axis = da.sizes.index(dim_name)
            pad_shape[axis] = new_len - da.shape[axis]
            # construct a full NaN array of the correct shape
            pad_array = xr.full_like(
                da.isel({dim_name: 0}), # grab representative slice
                np.nan
            ).broadcast(**{dim_name: new_len - da.shape[axis]})

            # concatenate the original data and the NaN padding
            return xr.concat([da, pad_array], dim=dim_name)

        # pad all data variables as needed
        for var in ds.data_vars:
            ds[var] = _pad_dim(ds[var], "lat", new_nlat)
            ds[var] = _pad_dim(ds[var], "lon", new_nlon)

    # replace coordinates and drop the old bounds
    ds = ds.drop_vars(
        ["lat_bnds", "lon_bnds", "areacella"],
        errors="ignore"
    ).assign_coords(
        lat=("lat", new_lat),
        lon=("lon", new_lon),
    )

    return ds
