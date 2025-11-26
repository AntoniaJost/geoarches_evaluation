import os
import xarray as xr
import pandas as pd
from steps.step_base import Step
from logger import GREEN, RED, WARN
from utils.change_tracker import get_force_rerun_flag

class SplitVarsStep(Step):
    """
    Splits concatenated NetCDF files (daily/monthly) into separate files per variable.
    Each variable is saved in its own directory: <output_base>/<frequency>/<var>/gn/.
    Skips variables if output file already exists and is non-empty.
    Supports forced re-splitting by checking change-tracking flags.
    """
    def run(self):
        # load config
        cfg = self.cfg
        in_dir = cfg['input_dir']
        out_base = cfg['output_dir']
        daily_pref = cfg.get('daily_prefix', 'daily_')
        monthly_pref = cfg.get('monthly_prefix', 'monthly_')
        time_slice = cfg.get('time_slice_daily', None)

        # process both daily and monthly files
        for is_daily, prefix in [(True, daily_pref), (False, monthly_pref)]:
            freq = 'daily' if is_daily else 'monthly'
            self.logger.debug(f"[SplitVarsStep] Searching {freq} file in {in_dir} with prefix '{prefix}'")

            # check if the frequency requires a forced rerun
            force_rerun = get_force_rerun_flag(freq, self.logger)

            # find the matching concatenated file in input directory
            if is_daily and time_slice:
                # daily files use daily time slice
                t0 = pd.to_datetime(time_slice[0]).strftime('%Y%m%d')
                t1 = pd.to_datetime(time_slice[1]).strftime('%Y%m%d')
                expected_name = f"{daily_pref}{t0}_{t1}.nc"
            else:
                # there typically only exists one file for monthly covering the entire time span 
                mon_files = sorted([
                    f for f in os.listdir(in_dir)
                    if f.startswith(monthly_pref) and f.endswith(".nc")
                ])
                if not mon_files:
                    self.logger.error(f"{RED} [SplitVarsStep] No monthly files found in {in_dir}")
                    continue
                expected_name = mon_files[-1]

            concat_path = os.path.join(in_dir, expected_name)

            if not os.path.exists(concat_path):
                self.logger.error(
                    f"{RED} [SplitVarsStep] Expected file '{expected_name}' not found in {in_dir}"
                )
                continue

            # open dataset once and list all contained variables
            ds = xr.open_dataset(concat_path)
            self.logger.info(f"{GREEN} [SplitVarsStep] Opened {concat_path} with vars: {list(ds.data_vars)}")

            # extract start and end dates from the time dimension (monthly) or time slice (daily)
            fmt = '%Y%m%d' if is_daily else '%Y%m'
            if is_daily and time_slice:
                try:
                    t0 = pd.to_datetime(time_slice[0]).strftime(fmt)
                    t1 = pd.to_datetime(time_slice[1]).strftime(fmt)
                    self.logger.info(
                        f"[SplitVarsStep] Using time_slice_daily for naming: {t0}–{t1}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"{RED} [SplitVarsStep] Invalid time_slice_daily format: {time_slice} ({e})"
                    )
                    raise
            else:
                tmin = pd.to_datetime(str(ds.time.min().values))
                tmax = pd.to_datetime(str(ds.time.max().values))
                t0 = tmin.strftime(fmt)
                t1 = tmax.strftime(fmt)
                self.logger.info(f"[SplitVarsStep] Using dataset time range for naming: {t0}–{t1}")

            # loop over each variable and save it into its own file
            for var in ds.data_vars:
                if var == "time_bnds":
                    continue
                # Sanitize variable name to prevent issues like \2m_temperature
                sanitized_var = var.replace("\\", "").replace(" ", "_")
                out_dir = os.path.join(out_base, freq, sanitized_var, 'gn')
                os.makedirs(out_dir, exist_ok=True)

                # delete existing files for this var if force rerun is enabled
                if force_rerun:
                    deleted = 0
                    for fname in os.listdir(out_dir):
                        full_path = os.path.join(out_dir, fname)
                        if fname.endswith(".nc") and os.path.isfile(full_path):
                            try:
                                os.remove(full_path)
                                self.logger.warning(f"{WARN} [SplitVarsStep] Deleted outdated file: {fname}")
                                deleted += 1
                            except Exception as e:
                                self.logger.warning(f"{RED} [SplitVarsStep] Could not delete {fname}: {e}")
                    if deleted == 0:
                        self.logger.info(f"[SplitVarsStep] No old {freq} files to delete in {out_dir}")

                # build output file path for the variable
                out_file = os.path.join(out_dir, f"{sanitized_var}_gn_{t0}_{t1}.nc")

                # skip if exists and non-empty
                if not force_rerun and os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                    self.logger.info(f"{GREEN} [SplitVarsStep] Skipping existing file: {out_file}")
                    continue
                elif os.path.exists(out_file):
                    self.logger.warning(f"{RED} [SplitVarsStep] Overwriting empty file: {out_file}")

                # Write single-variable dataset
                vars_to_keep = [var]
                # making sure time_bnds don't get lost
                if "time_bnds" in ds.data_vars:
                    vars_to_keep.append("time_bnds")

                # Create dataset with both
                var_ds = ds[vars_to_keep].copy()
                var_ds.attrs['variable_id'] = sanitized_var
                var_ds.to_netcdf(out_file)
                self.logger.info(f"{GREEN} [SplitVarsStep] Saved {out_file}")
                