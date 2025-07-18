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
from utils.cmor_utils import cmorize_data_with_template, find_template_for_variable
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

            # map frequency names to cmor frequency labels
            freq_map = {'monthly': 'Amon', 'daily': 'day'}

            for frequency in scales:
                cfreq = freq_map.get(frequency, frequency)
                in_freq_dir = os.path.join(base_dir, frequency)
                if not os.path.isdir(in_freq_dir):
                    self.logger.warning(f"{RED} [CmoriseStep] Frequency directory not found: {in_freq_dir}, skipping")
                    continue
                self.logger.info(f"[CmoriseStep] Processing frequency: {frequency}")

                # Check if we need to force a re-run (needed if previous files have been changed)
                force_rerun = get_force_rerun_flag(frequency, self.logger)
                if force_rerun:
                    removed = 0
                    if os.path.isdir(output_root):
                        for f in os.listdir(os.path.join(output_root, frequency)):
                            full_path = os.path.join(output_root, frequency, f)
                            if f.endswith(".nc") and os.path.isfile(full_path):
                                try:
                                    os.remove(full_path)
                                    self.logger.warning(f"{WARN} [CmoriseStep] Deleted outdated CMOR file: {full_path}")
                                    removed += 1
                                except Exception as e:
                                    self.logger.warning(f"{RED} [CmoriseStep] Could not delete {full_path}: {e}")
                    if removed == 0:
                        self.logger.info(f"[CmoriseStep] No existing CMORised files to delete in {frequency}")

                # gather all netcdf input files in gr_subdir
                pattern = os.path.join(in_freq_dir, '*', gr_subdir, '*.nc')
                files = sorted(glob.glob(pattern))
                self.logger.info(f"[CmoriseStep] Found {len(files)} files to CMORise in {frequency}")

                for fpath in files:
                    self.logger.info(f"[CmoriseStep] Processing {fpath}")
                    # load variable data
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
                        f"{variable}_{cfreq}_{model}_aimip_{ensemble}_{gr_subdir}_{start}-{end}.nc"
                    )
                    out_dir = os.path.join(output_root, frequency)
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

                    # adjust template to match time dimension of data
                    # OLD VERSION, WORKS GREAT! BUT ONLY IF TEMPLATE IS TOO LONG
                    tpl_ds = xr.open_dataset(template_path)
                    tpl_time_len = tpl_ds.sizes.get('time', None)
                    tpl_ds.close()
                    # if tpl_time_len != user_data_array.shape[0]:
                    #     sliced_tpl = os.path.join(temp_dir, os.path.basename(template_path).replace('.nc','_sliced.nc'))
                    #     if not os.path.exists(sliced_tpl):
                    #         self.logger.info(f"[CmoriseStep] Slicing template to {user_data_array.shape[0]} timesteps to match time dimension of input")
                    #         with xr.open_dataset(template_path) as ft:
                    #             sliced = ft.isel(time=slice(0, user_data_array.shape[0]))
                    #             sliced.to_netcdf(sliced_tpl)
                    #     template_to_use = sliced_tpl
                    # else:
                    #     template_to_use = template_path

                    # NEW VERSION, ANTICIPATING THAT TEMPLATE MIGHT BE TOO SHORT - NOT TESTED for that case yet!!!!
                    sliced_tpl = os.path.join(temp_dir, os.path.basename(template_path).replace('.nc','_sliced.nc'))
                    if tpl_time_len != user_data_array.shape[0]:
                        if not os.path.exists(sliced_tpl):
                            self.logger.info(f"[CmoriseStep] Adjusting template: {tpl_time_len} → {user_data_array.shape[0]} timesteps")
                            with xr.open_dataset(template_path) as ft:
                                if tpl_time_len > user_data_array.shape[0]:
                                    # Template too long: slice
                                    sliced = ft.isel(time=slice(0, user_data_array.shape[0]))
                                else:
                                    # Template too short: repeat
                                    reps = int(np.ceil(user_data_array.shape[0] / tpl_time_len))
                                    ft_repeated = xr.concat([ft] * reps, dim="time")
                                    sliced = ft_repeated.isel(time=slice(0, user_data_array.shape[0]))
                                sliced.to_netcdf(sliced_tpl)
                        template_to_use = sliced_tpl
                    else:
                        template_to_use = template_path

                    # "Dirty workaround" for zg monthly: drop plev dimension in template
                    if cfg.get('dirty_workaround','').upper()=='YES' and variable=='zg' and frequency=='monthly':
                        dirty_tpl = os.path.join(temp_dir, os.path.basename(template_to_use).replace('.nc','_dirty.nc'))
                        if not os.path.exists(dirty_tpl):
                            self.logger.warning(f"‼️ [CmoriseStep] Applying dirty_workaround on {template_to_use}")
                            with xr.open_dataset(template_to_use) as dt:
                                ds_d = dt.isel(plev=0, drop=True)
                                ds_d.to_netcdf(dirty_tpl)
                        template_to_use = dirty_tpl

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
