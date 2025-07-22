import os
import glob
import uuid
import pandas as pd
import xarray as xr
import numpy as np
import tempfile
import shutil
from steps.step_base import Step
from logger import GREEN, RED, WARN
from utils.cmor_utils import cmorize_data_with_template, find_template_for_variable, adjust_template_time, adjust_template_grid
from utils.change_tracker import get_force_rerun_flag


def generate_tracking_id():
    """
    Generate a unique tracking identifier for CMOR files.
    """
    return f"hdl:21.14100/{uuid.uuid4()}"


class CmoriseStep(Step):
    """
    CMORises regridded NetCDF data using AIMIP-compliant template files.

    This step:
    - Iterates over input files for each frequency (e.g. daily, monthly)
    - Selects the appropriate template for each variable
    - Modifies the template if the time dimension does not match
    - Applies a (dirty) workaround for 'zg' if enabled (aka modifying the template)
    - Writes output files in AIMIP-compliant format under `output_root/<frequency>/`
    - Skips processing if outputs already exist (unless forced rerun is triggered)
    """
    def run(self):
        # create a temp dir for modified templates
        temp_dir = tempfile.mkdtemp(prefix='cmorise_')
        self.logger.debug(f"[CmoriseStep] Created temp_dir: {temp_dir}")
        try: 
            # load config entries
            cfg = self.cfg
            base_dir       = cfg['base_dir']
            scales         = cfg.get('scales', [])
            gn_subdir      = cfg.get('gn_subdir', 'gn')
            gr_subdir      = cfg.get('gr_subdir', 'gr')
            output_root    = cfg['output_root']
            source_id      = cfg['source_id']
            institution    = cfg['institution']
            contact        = cfg['contact']
            title          = cfg['title']
            experiment     = cfg['experiment']
            experiment_id  = cfg['experiment_id']
            institution_id = cfg['institution_id']
            nominal_res    = cfg['nominal_resolution']
            license        = cfg['license']
            model          = cfg.get('model', 'MMM')
            ensemble       = cfg.get('ensemble', 'r1i1p1f1')
            use_native     = cfg.get('use_native', False)
            use_regridded = cfg.get('use_regridded', False)

            if not (use_native or use_regridded):
                self.logger.warning(f"{WARN} [CmoriseStep] Neither native nor regridded requested; skipping CMORisation")
                return

            # map frequency names to cmor frequency labels
            freq_map = {'monthly': 'Amon', 'daily': 'day'}

            # loop over each present frequency (mainly daily and monthly)
            for frequency in scales:
                cfreq = freq_map.get(frequency, frequency)
                self.logger.info(f"[CmoriseStep] Processing frequency: {frequency}")

                # Check if we need to force a re-run (needed if previous files have been changed)
                force_rerun = get_force_rerun_flag(frequency, self.logger)
                if force_rerun:
                    removed = 0
                    for grid in (['gn'] if use_native else []) + (['gr'] if use_regridded else []):
                        # out_dir_grid = os.path.join(output_root, frequency, '*', grid)
                        for old in glob.glob(os.path.join(output_root, frequency, '*', grid, '*.nc')):
                            try:
                                os.remove(old)
                                self.logger.warning(f"{WARN} [CmoriseStep] Deleted outdated CMOR file: {old}")
                                removed += 1
                            except Exception as e:
                                self.logger.warning(f"{RED} [CmoriseStep] Could not delete {old}: {e}")

                    if removed == 0:
                        self.logger.info(f"[CmoriseStep] No existing CMORised files to delete in {frequency}")

                # for each requested grid, pick the right input files
                for grid, subdir in (('gn', gn_subdir), ('gr', gr_subdir)):
                    if grid == 'gn' and not use_native:    continue
                    if grid == 'gr' and not use_regridded: continue

                    # gather all netcdf input files in subdir
                    in_pattern = os.path.join(base_dir, frequency, '*', subdir, '*.nc')
                    files = sorted(glob.glob(in_pattern))
                    self.logger.info(f"[CmoriseStep] Found {len(files)} files to CMORise in {frequency}/{grid}")

                    for fpath in files:
                        self.logger.info(f"[CmoriseStep] Processing {grid} file {fpath}")
                        # load variable data and reshape to match convention
                        with xr.open_dataset(fpath) as ds:
                            main_var = ds.attrs.get('variable_id') or list(ds.data_vars.keys())[0]
                            raw_data = ds[main_var].to_numpy() + 5  # dummy variation
                            # transpose to (time, level, lat, lon) or (time, lat, lon); needed due to earlier processing step
                            if raw_data.ndim == 4:
                                user_data_array = raw_data.transpose(0, 1, 3, 2)
                            elif raw_data.ndim == 3:
                                user_data_array = raw_data.transpose(0, 2, 1)
                            else:
                                raise ValueError(f"Unsupported number of dimensions: {raw_data.ndim}")
                        self.logger.info(f"{GREEN} [CmoriseStep] Loaded '{main_var}' with shape {user_data_array.shape}")

                        # extract variable name from path and build output filename
                        parts = fpath.split(os.sep)
                        try:
                            variable = parts[-3]
                        except IndexError:
                            self.logger.error(f"{RED} [CmoriseStep] Unexpected path structure: {fpath}")
                            continue

                        # determine time range from input for naming
                        ds = xr.open_dataset(fpath)
                        tmin = pd.to_datetime(str(ds.time.min().values))
                        tmax = pd.to_datetime(str(ds.time.max().values))
                        if frequency == 'monthly':
                            start = tmin.strftime('%Y%m')
                            end   = tmax.strftime('%Y%m')
                        else:
                            start = tmin.strftime('%Y%m%d')
                            end   = tmax.strftime('%Y%m%d')

                        # build output filename following AIMIP convention
                        filename = (
                            f"{variable}_{cfreq}_{model}_aimip_{ensemble}_{subdir}_{start}-{end}.nc"
                        )
                        out_dir = os.path.join(output_root, frequency, variable, grid)
                        os.makedirs(out_dir, exist_ok=True)
                        out_file = os.path.join(out_dir, filename)

                        # Skip if exists and non-empty, unless force_rerun is set
                        if not force_rerun and os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                            self.logger.info(f"{GREEN} [CmoriseStep] Skipping CMORisation for {fpath}; exists {out_file}")
                            continue
                        elif os.path.exists(out_file):
                            self.logger.warning(f"{WARN} [CmoriseStep] Overwriting empty file: {out_file}")

                        # find appropiate CMOR template for this variable & frequency
                        template_path = find_template_for_variable(variable, frequency, cfg)  
                        if template_path is None:
                            self.logger.warning(f"{WARN} [CmoriseStep] No template for variable '{variable}', skipping {fpath}")
                            continue

                        # "Dirty workaround" for zg monthly: drop plev dimension in template
                        if cfg.get('dirty_workaround','').upper()=='YES' and variable=='zg' and frequency=='monthly':
                            dirty_tpl = os.path.join(temp_dir, os.path.basename(template_path).replace('.nc','_dirty.nc'))
                            if not os.path.exists(dirty_tpl):
                                self.logger.warning(f"‼️ [CmoriseStep] Applying dirty_workaround on {template_path}")
                                with xr.open_dataset(template_path) as dt:
                                    ds_d = dt.isel(plev=0, drop=True)
                                    ds_d.to_netcdf(dirty_tpl)
                            template_path = dirty_tpl

                        # load the original template and user input
                        tpl_orig = xr.open_dataset(template_path, decode_times=False)
                        ds_in    = xr.open_dataset(fpath)

                        # fix time axis (of template) if needed
                        desired_len = user_data_array.shape[0]
                        orig_len    = tpl_orig.sizes.get("time", 0)
                        if orig_len != desired_len:
                            if self.logger:
                                self.logger.debug(f"[CmoriseStep] Time mismatch: {orig_len} → {desired_len} steps, adjusting template")
                            tpl_time_ok = adjust_template_time(
                                tpl_orig,
                                desired_len,
                                ds_in.time.values,
                                logger=self.logger,
                            )
                        else:
                            tpl_time_ok = tpl_orig  # no change

                        # fix lon/lat grid if needed (needed if native grid is chosen and that is not 1x1 deg)
                        need_grid_fix = (
                            tpl_time_ok.sizes.get("lat")  != ds_in.sizes["lat"]
                            or tpl_time_ok.sizes.get("lon") != ds_in.sizes["lon"]
                        )
                        if grid == "gn" and need_grid_fix:
                            if self.logger:
                                self.logger.debug(
                                    f"[CmoriseStep] Grid mismatch: "
                                    f"{tpl_time_ok.sizes['lat']}×{tpl_time_ok.sizes['lon']} → "
                                    f"{ds_in.sizes['lat']}×{ds_in.sizes['lon']}, adjusting template"
                                )
                            tpl_full_ok = adjust_template_grid(
                                tpl_time_ok,
                                ds_in.lat.values,
                                ds_in.lon.values,
                                logger=self.logger,
                            )
                        else:
                            tpl_full_ok = tpl_time_ok

                        # save adjusted template (temporarily)
                        slice_path = os.path.join(
                            temp_dir,
                            os.path.basename(template_path).replace(".nc","_adj.nc")
                        )
                        tpl_full_ok.to_netcdf(slice_path)

                        # clean up
                        if tpl_time_ok is not tpl_orig:
                            tpl_time_ok.close()
                        if tpl_full_ok is not tpl_time_ok:
                            tpl_full_ok.close()
                        tpl_orig.close()
                        ds_in.close()

                        template_to_use = slice_path

                        # generate tracking ID and run CMORisation
                        tracking_id = generate_tracking_id()
                        self.logger.info(f"[CmoriseStep] CMORising: {variable}/{frequency} -> {out_file}")

                        try:
                            cmorize_data_with_template(
                                user_data_array,
                                template_to_use,
                                out_file,
                                {
                                    "source_id": source_id,
                                    "institution": institution,
                                    "contact": contact,
                                    "title": title,
                                    "experiment": experiment,
                                    "experiment_id": experiment_id,
                                    "institution_id": institution_id,
                                    "nominal_resolution": nominal_res,
                                    "license": license,
                                    "tracking_id": tracking_id,
                                }
                            )
                            self.logger.info(f"{GREEN} [CmoriseStep] CMORisation successful: {out_file}")
                        except Exception as e:
                            self.logger.error(f"{RED} [CmoriseStep] Failed CMORisation for {fpath}: {e}")
                            raise
                
        finally:
            # cleanup temporary directory
            try:
                shutil.rmtree(temp_dir)
                self.logger.debug(f"[CmoriseStep] Removed temp_dir: {temp_dir}")
            except Exception as e:
                self.logger.warning(f"{RED} [CmoriseStep] Could not remove temp_dir {temp_dir}: {e}")
