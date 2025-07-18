import os
import numpy as np
from datetime import datetime
import xarray as xr
from glob import glob
from collections import defaultdict
import json
from steps.step_base import Step
from logger import GREEN, RED, WARN

class CalcMeansStep(Step):
    """
    Calculates daily and monthly means from ERA5 NetCDF input files.

    This step:
    - Loads NetCDF files from the specified source directory
    - Groups files by year using filename parsing
    - Applies optional filtering by user-specified years
    - Verifies expected number of files per year
    - Computes daily and monthly means using xarray
    - Saves output into dedicated daily and monthly folders
    - Tracks which years were newly added or deleted
    - Writes a JSON change flag to signal downstream steps
    """
    def run(self):
        cfg = self.cfg
        src_dir = cfg['source_dir']
        out_base = cfg['intermediate_dir']
        daily_dir = os.path.join(out_base, cfg['daily_subdir'])
        monthly_dir = os.path.join(out_base, cfg['monthly_subdir'])
        os.makedirs(daily_dir, exist_ok=True)
        os.makedirs(monthly_dir, exist_ok=True)

        # define input file pattern
        pattern = cfg.get('file_pattern', '*.nc')

        # read requested time span or years from config
        requested_years = cfg.get('years', [])
        if isinstance(requested_years, str) and ":" in requested_years:
            start, end = map(int, requested_years.split(":"))
            requested_years = [str(y) for y in range(start, end + 1)]
        elif isinstance(requested_years, list):
            requested_years = [str(y) for y in requested_years]
        else:
            raise ValueError("Invalid 'years' format in config")
        requested_years = set(requested_years)  

        expected = cfg.get('expected_files_per_year', None)
        # time range for daily mean filtering (for 1978-1979 only)
        start_daily = np.datetime64(cfg.get('daily_start'))
        end_daily   = np.datetime64(cfg.get('daily_end'))

        # detect which years already exist in daily/monthly output
        existing_daily_years = {f.split('_')[-1].replace('.nc', '') for f in os.listdir(daily_dir) if f.endswith('.nc')}
        existing_monthly_years = {f.split('_')[-1].replace('.nc', '') for f in os.listdir(monthly_dir) if f.endswith('.nc')}
        combined_existing = existing_daily_years.union(existing_monthly_years)

        #  Group all input files by year from filename (e.g., era5_240_1979_6h.nc → 1979)
        file_groups = defaultdict(list)
        for fpath in glob(os.path.join(src_dir, pattern)):
            parts = os.path.basename(fpath).split("_")
            if len(parts) < 3:
                continue
            year = parts[2]
            if requested_years and year not in requested_years:
                continue
            file_groups[year].append(fpath)

        # filter again 
        if requested_years:
            file_groups = {yr: fls for yr, fls in file_groups.items() if yr in requested_years}

        # track changes
        changed_years = set()
        all_done = True

        # Process each year; compute daily/monthly means per year
        for year in sorted(file_groups):
            files = sorted(file_groups[year])
            if expected and len(files) != expected:
                self.logger.warning(f"{RED} [CalcMeansStep] Skipping {year}: found {len(files)} files, expected {expected}")
                continue
            try:
                daily_out   = os.path.join(daily_dir, f"era5_daily_mean_{year}.nc")
                monthly_out = os.path.join(monthly_dir, f"era5_monthly_mean_{year}.nc")
                skip_daily   = os.path.exists(daily_out) and os.path.getsize(daily_out) > 0
                skip_monthly = os.path.exists(monthly_out) and os.path.getsize(monthly_out) > 0

                if skip_daily and skip_monthly:
                    print(f"[CalcMeansStep] Skipping {year} (daily & monthly outputs already exist)")
                    self.logger.info(f"{GREEN} [CalcMeansStep] Skipping {year}: daily & monthly outputs exist")
                    continue
                
                # load and concatenate all quarterly files for the year
                # ‼️ BOTTLENECK ‼️ But using combine='by_coords' does not work
                ds = xr.open_mfdataset(files, engine='h5netcdf', combine='nested', concat_dim='time', parallel=True, chunks={})
                ds = ds.sortby('time')

                # Daily mean only for 1978 and 1979
                if not skip_daily:
                    if year in ("1978", "1979"):
                        ds_daily = ds
                        if "time" in ds.coords:
                            ds_daily = ds.sel(time=slice(start_daily, end_daily))
                        ds_daily_mean = ds_daily.resample(time='1D').mean()
                        ds_daily_mean.to_netcdf(daily_out)
                        self.logger.info(f"{GREEN} [CalcMeansStep] Saved daily mean: {daily_out}")
                    else:
                        self.logger.info(f"[CalcMeansStep] Skipped daily mean for {year} (not in 1978–1979)")

                # monthly mean, always computed
                if not skip_monthly:
                    ds.resample(time='1MS').mean().to_netcdf(monthly_out)
                    print(f"[CalcMeansStep] Saved monthly mean for {year}")
                    self.logger.info(f"{GREEN} [CalcMeansStep] Saved monthly mean: {monthly_out}")
                else:
                    print(f"[CalcMeansStep] Skipped monthly mean for {year} (already exists)")
                    self.logger.info(f"[CalcMeansStep] Skipped monthly mean for {year}; already exists")

            except Exception as e:
                self.logger.error(f"{RED} Failed processing {year}: {e}")
                raise

        
        # remove obsolete files if requested_years is fewer than previously
        deleted_years = combined_existing - requested_years
        for y in deleted_years:
            dfile = os.path.join(daily_dir, f"era5_daily_mean_{y}.nc")
            mfile = os.path.join(monthly_dir, f"era5_monthly_mean_{y}.nc")
            for f in [dfile, mfile]:
                if os.path.exists(f):
                    os.remove(f)
                    self.logger.warning(f"{WARN} [CalcMeansStep] Deleted obsolete mean file: {f}")
                    changed_years.add(y)
                    all_done = False

        # detect also newly added years
        added_years = requested_years - combined_existing
        if added_years:
            self.logger.info(f"[CalcMeansStep] Newly added years detected: {sorted(added_years)}")
            changed_years.update(added_years)
            all_done = False

        # Write change signal for downstream steps into tracking file
        flag_path = os.path.join(out_base, "means_changed.json")
        if not all_done:
            with open(flag_path, "w") as f:
                json.dump({
                    "changed": True,
                    "years_changed": sorted(changed_years)
                }, f)
            self.logger.info(f"[CalcMeansStep] Wrote change flag to: {flag_path}")
        else:
            if os.path.exists(flag_path):
                os.remove(flag_path)
            self.logger.info(f"[CalcMeansStep] No changes detected, no flag written.")
