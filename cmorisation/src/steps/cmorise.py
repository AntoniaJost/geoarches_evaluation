import os
import xarray as xr
import uuid
import pandas as pd
import glob
import yaml
import subprocess
from steps.step_base import Step
from utils.cmor_helpers import (
    add_lat_lon_bounds, add_time_bounds,
    fix_plev, fix_variable_metadata, inject_height_if_needed
)
from logger import GREEN, RED, WARN

def _post_ncatted(path, var, treat_as_plev=False):
    """
    Post-process NetCDF file using ncatted to remove or fix
    metadata attributes, depending on the variable.
    """
    base_cmd = ["ncatted", "-O"]
    # choose flags depending on the variable
    if var in ("tas", "uas", "vas"):
        # remove coordinates="height" for these three and then same as for ts and psl
        args = [
            "-a", "coordinates,time_bnds,d,,",
            "-a", "coordinates,lat_bnds,d,,",
            "-a", "coordinates,lon_bnds,d,,",
            "-a", "_FillValue,height,d,,",
            "-a", "_FillValue,time_bnds,d,,",
            "-a", "missing_value,time_bnds,d,,",
            "-a", "_FillValue,time,d,,",
            "-a", "_FillValue,lat,d,,",
            "-a", "_FillValue,lon,d,,",
            "-a", "_FillValue,lat_bnds,d,,",
            "-a", "_FillValue,lon_bnds,d,,",
        ]
    elif not treat_as_plev: #var in ("ts", "psl", "zg"): # ts, psl, (zg)
        # no plev
        args = [
            "-a", "_FillValue,time_bnds,d,,",
            "-a", "missing_value,time_bnds,d,,",
            "-a", "_FillValue,time,d,,",
            "-a", "_FillValue,lat,d,,",
            "-a", "_FillValue,lon,d,,",
            "-a", "_FillValue,lat_bnds,d,,",
            "-a", "_FillValue,lon_bnds,d,,",
        ]
    else:
        # (zg), ua, va, ta, hus  → include plev
        args = [
            "-a", "_FillValue,time_bnds,d,,",
            "-a", "missing_value,time_bnds,d,,",
            "-a", "_FillValue,time,d,,",
            "-a", "_FillValue,lat,d,,",
            "-a", "_FillValue,lon,d,,",
            "-a", "_FillValue,plev,d,,",
            "-a", "_FillValue,lat_bnds,d,,",
            "-a", "_FillValue,lon_bnds,d,,",
        ]

    cmd = base_cmd + args + [path, path]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # log but don’t crash the entire pipeline
        print(f"{WARN} ncatted failed on {path}: {e}")

class CmoriseStep(Step):
    def run(self):

        cfg = self.cfg
        work_dir       = cfg['base_dir']
        metadata_path  = cfg['metadata_path']
        scales         = cfg.get('scales', [])
        grid           = cfg.get('gn_subdir', 'gn')
        model          = cfg.get('model', 'AMM')
        ensemble       = cfg.get('ensemble', 'r1i1p1f1')
        out_base       = cfg['output_root']
        freq_map   = {'monthly': 'Amon', 'daily': 'day'}
        frequency = {'monthly': 'mon', 'daily': 'day'} 
        zg_to_500      = self.full_cfg["rename_vars"].get('zg_to_500', False)

        # generate tracking id (cmip convention)
        tracking_id = str(uuid.uuid4())

        # load variable metadata from yaml file
        def load_variable_metadata(path):
            with open(path, "r") as f:
                return yaml.safe_load(f)
        metadata = load_variable_metadata(metadata_path)

        # process each temporal scale (daily/monthly)
        for scale in scales:
            cfreq     = freq_map.get(scale, scale)
            input_dir = os.path.join(work_dir, scale, '*', grid, '*.nc')
            files     = sorted(glob.glob(input_dir))

            for fname in files:
                if not fname.endswith(".nc"):
                    continue
                # get variable name from parent folder structure
                var = os.path.basename(os.path.dirname(os.path.dirname(fname)))

                self.logger.info(f"[CmoriseStep] Processing {fname}")
                try:
                    ds = xr.open_dataset(fname)

                    # extract time range from data
                    tmin = pd.to_datetime(str(ds.time.min().values))
                    tmax = pd.to_datetime(str(ds.time.max().values))
                    # format time stamps for filenames
                    if scale == 'monthly':
                        start = tmin.strftime('%Y%m')
                        end   = tmax.strftime('%Y%m')
                    else:
                        start = tmin.strftime('%Y%m%d')
                        end   = tmax.strftime('%Y%m%d')
                    # build aimip conform filename and output path
                    subdir   = grid
                    filename = f"{var}_{cfreq}_{model}_aimip_{ensemble}_{subdir}_{start}-{end}.nc"
                    out_dir  = os.path.join(out_base, cfreq, var, grid)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, filename)

                    # skip file if already processed
                    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                        print(f"[CmoriseStep] Skipping; already exists: {out_path}")
                        self.logger.info(f"{GREEN} [CmoriseStep] Skipping; already exists: {out_path}")
                        ds.close()
                        continue

                    # continue with cmor edits
                    ds = add_lat_lon_bounds(ds)
                    ds = add_time_bounds(ds)
                    ds = fix_plev(ds)
                    ds = inject_height_if_needed(ds, var)
                    ds = fix_variable_metadata(ds, var, metadata)

                    # global attributes
                    global_attributes = cfg.get("global_attributes", {})
                    # normalize it to a flat dict
                    if isinstance(global_attributes, dict):
                        global_attrs = global_attributes
                    elif isinstance(global_attributes, list):
                        global_attrs = {}
                        for entry in global_attributes:
                            if isinstance(entry, dict):
                                global_attrs.update(entry)
                            else:
                                self.logger.warning(f"Skipping invalid global_attributes entry: {entry!r}")
                    else:
                        self.logger.warning(f"global_attributes has unexpected type {type(global_attributes)}, skipping")
                        global_attrs = {}
                    # add attributes to dataset
                    ds.attrs.update(global_attrs)
                    freq = frequency.get(scale)
                    ds.attrs.update({
                        "tracking_id": tracking_id,
                        "frequency": freq
                    })

                    has_plev = ('plev' in ds[var].dims) if var in ds else False
                    treat_as_plev = has_plev or (var == "zg" and zg_to_500)

                    # save dataset to netcdf
                    ds.to_netcdf(out_path)
                    self.logger.info(f"{GREEN} [CmoriseStep] Saved {out_path}")
                    ds.close()

                    # post‐process with ncatted to clean metadata
                    _post_ncatted(out_path, var, treat_as_plev)
                    self.logger.info(f"{GREEN} [CmoriseStep] Saved & post-processed {out_path}")

                except Exception as e:
                    self.logger.error(f"{RED} [CmoriseStep] Failed to process {fname}: {e}")
                    ds.close()
