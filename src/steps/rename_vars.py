import os
import xarray as xr
import pandas as pd
import numpy as np
import json
from steps.step_base import Step
from utils.change_tracker import get_force_rerun_flag
from logger import GREEN, RED, WARN

class RenameVarsStep(Step):
    """
    Renames variables and units in daily and monthly mean files, applies pressure level filtering and time slicing, 
    and outputs one combined NetCDF file per frequency.
    """
    def run(self):
        cfg = self.cfg
        input_dirs = cfg.get('input_dirs', [])
        out_base   = cfg['output_dir']
        self.logger.info(f"[RenameVarsStep] Starting. input_dirs={input_dirs}, out_base={out_base}")
        os.makedirs(out_base, exist_ok=True)

        # config mappings
        var_map    = cfg.get('var_mapping', {})
        unit_map   = cfg.get('unit_mapping', {})
        levels     = cfg.get('pressure_levels', [])
        time_slice = cfg.get('time_slice_daily', None)

        for in_dir in input_dirs:
            # determine processing type from folder name
            base_name = 'daily' if 'daily' in os.path.basename(in_dir).lower() else 'monthly'
            force_rerun = get_force_rerun_flag(base_name, self.logger)
            date_format = '%Y%m%d' if base_name == 'daily' else '%Y%m'

            # gather all files to process
            files = sorted([os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith('.nc')])
            if not files:
                self.logger.info(f"{GREEN}[RenameVarsStep] No files to process in {in_dir}; skipping")
                continue

            # infer time range from first and last file for naming output
            try:
                t0 = xr.open_dataset(files[0], decode_times=True).time.min().values
                tN = xr.open_dataset(files[-1], decode_times=True).time.max().values
                start = pd.to_datetime(str(t0)).strftime(date_format)
                end   = pd.to_datetime(str(tN)).strftime(date_format)
            except Exception as e:
                self.logger.error(f"{RED} [RenameVarsStep] Could not read time range from {files[0]}–{files[-1]}: {e}")
                continue

            # If rerun is forced, delete all daily_*.nc or monthly_*.nc before continuing
            if force_rerun:
                removed = 0
                for fname in os.listdir(out_base):
                    full = os.path.join(out_base, fname)
                    if fname.startswith(f"{base_name}_") and fname.endswith(".nc") and os.path.isfile(full):
                        try:
                            os.remove(full)
                            self.logger.warning(f"{WARN} [RenameVarsStep] Deleted outdated file: {full}")
                            removed += 1
                        except Exception as e:
                            self.logger.warning(f"{RED} [RenameVarsStep] Could not delete {full}: {e}")
                if removed == 0:
                    self.logger.info(f"[RenameVarsStep] No old {base_name} files found to delete.")

            # define output file
            outfile = os.path.join(out_base, f"{base_name}_{start}_{end}.nc")

            # skip processing if valid output already exists and rerun is not forced
            if not force_rerun and os.path.exists(outfile) and os.path.getsize(outfile) > 0:
                print(f"[RenameVarsStep] Skipping {base_name} processing: output exists {outfile}")
                self.logger.info(f"{GREEN} [RenameVarsStep] Skipping {base_name} processing: output exists {outfile}")
                continue

            self.logger.info(f"[RenameVarsStep] Processing directory: {in_dir}")
            processed = []

            for fp in files:
                self.logger.debug(f"[RenameVarsStep] Reading {fp}")
                try:
                    ds = xr.open_dataset(fp)
                    # rename variables based on mapping
                    ds = ds.rename({k: v for k, v in var_map.items() if k in ds.variables})
                    # retain only renamed variables
                    keep = [v for v in var_map.values() if v in ds.data_vars]
                    aux_vars = [v for v in ds.data_vars if v.endswith('_bnds')]
                    ds = ds[keep + aux_vars]

                    # standardise dimensions
                    dim_renames = {}
                    if 'longitude' in ds.dims: dim_renames['longitude'] = 'lon'
                    if 'latitude'  in ds.dims: dim_renames['latitude']  = 'lat'
                    if 'level'     in ds.dims: dim_renames['level']     = 'plev'
                    if dim_renames: ds = ds.rename(dim_renames)

                    # Convert units for precipitation (from mm/day → kg/m²/s)
                    if 'pr' in ds:
                        if ds.sizes.get('time', 0) == 0:
                            self.logger.warning(f"{RED} [RenameVarsStep] 'pr' has no time in {fp}; skipping var")
                        else:
                            conv = float(unit_map.get('pr'))
                            ds['pr'] = ds['pr'] / conv

                    # if pressure levels are specified, retain only those levels
                    if 'plev' in ds.dims and levels:
                        vars_with_level = [v for v in ds.data_vars if 'plev' in ds[v].dims]
                        if vars_with_level:
                            ds = ds.sel(plev=[lev for lev in levels if lev in ds.plev.values])
                            self.logger.debug(f"[RenameVarsStep] Filtered {vars_with_level} to levels {ds.plev.values}")

                    # zg special case, for geopotential height only keep 500 hPa
                    if 'zg' in ds and 'level' in ds['zg'].dims:
                        ds['zg'] = ds['zg'].sel(plev=500, drop=False)

                    # for daily data only keep restriced time range 
                    if base_name == 'daily' and time_slice and 'time' in ds:
                        ds = ds.sel(time=slice(time_slice[0], time_slice[1]))
                        if ds.sizes.get('time', 0) == 0:
                            self.logger.warning(f"{WARN} [RenameVarsStep] No time steps remain in {fp}; Moving on.")
                            continue

                    # save variable_id in global attrs (used in CMOR step)
                    for var in ds.data_vars:
                        ds.attrs['variable_id'] = var
                        # assign missing_value attribute to each variable
                        missing_val = np.float32(1e20)
                        ds[var].attrs['missing_value'] = missing_val
                        ds[var].encoding['_FillValue'] = missing_val

                    processed.append(ds)

                except Exception as e:
                    self.logger.error(f"{RED} [RenameVarsStep] Error processing {fp}: {e}")
                    raise

            # merge datasets across time and save results
            if processed:
                combined = xr.concat(processed, dim='time')
                self.logger.debug(f"[RenameVarsStep] Writing output to {outfile}")
                combined.to_netcdf(outfile)
                self.logger.info(f"{GREEN} [RenameVarsStep] Saved {outfile}")
