import os
from glob import glob
import xarray as xr
from steps.step_base import Step
from logger import GREEN, RED, WARN
from utils.change_tracker import get_force_rerun_flag
from utils.regridding_to_1x1 import regrid_to_template_grid

class RunRegriddingStep(Step):
    """
    - Regrids all *_gn_*.nc files to *_gr_*.nc for each variable and scale
    - Moves the regridded *_gr_*.nc result into a separate 'gr' subdirectory
    - Supports forced regridding via change-tracking flag
    """
    def run(self):
        # load config
        cfg          = self.cfg
        regrid       = cfg.get('regrid', False)
        base_dir     = cfg['base_dir']
        scales       = cfg.get('scales', [])
        gn_sub       = cfg.get('gn_subdir', 'gn')
        gr_sub       = cfg.get('gr_subdir', 'gr')
        template_dir = cfg['template_dir']
        version      = cfg.get('template_version', 'v20190815')
        suffix       = cfg.get('template_suffix', 'MPI-ESM1-2-LR_amip_r1i1p1f1_gr')

        # check if regridding to 1x1° (resolution of template) is wanted, otherwise skip
        if not regrid:
            self.logger.warning(f"{WARN} [RunRegriddingStep] No regridding wanted. Skipping entire step.")
            return
        
        freq_map = {
            'monthly': 'Amon',
            'daily': 'day'
        }

        for scale in scales:
            # get frequency: daily or monthly
            freq_code = freq_map.get(scale)
            if not freq_code:
                self.logger.warning(f"{WARN} [RunRegriddingStep] Unknown frequency: {scale}")
                continue
            scale_dir = os.path.join(base_dir, scale)
            if not os.path.isdir(scale_dir):
                self.logger.warning(f"{RED} [RunRegriddingStep] Scale directory not found, skipping: {scale_dir}")
                continue

            # check if force rerun flag is set for this frequency
            force_rerun = get_force_rerun_flag(scale, self.logger)

            # iterate over each variable in daily/monthly
            for var in os.listdir(scale_dir):
                gn_dir = os.path.join(scale_dir, var, gn_sub)
                gr_dir = os.path.join(scale_dir, var, gr_sub)
                # skip variables with no native grid resolution
                if not os.path.isdir(gn_dir):
                    continue
                os.makedirs(gr_dir, exist_ok=True)

                # if force_rerun, remove all old gr files
                if force_rerun:
                    removed = 0
                    for f in os.listdir(gr_dir):
                        if f.endswith(".nc") and "_gr_" in f:
                            try:
                                os.remove(os.path.join(gr_dir, f))
                                self.logger.warning(f"{WARN} [RunRegriddingStep] Deleted outdated: {f}")
                                removed += 1
                            except Exception as e:
                                self.logger.warning(f"{RED} [RunRegriddingStep] Could not delete {f}: {e}")
                    if removed == 0:
                        self.logger.info(f"[RunRegriddingStep] No existing {scale} gr files to delete.")

                # determine template path
                template_pattern = os.path.join(
                    template_dir, freq_code, var, 'gr', version, f"{var}_{freq_code}_{suffix}_*.nc"
                )
                matching_templates = glob(template_pattern)
                if not matching_templates:
                    self.logger.error(f"{RED} [RunRegriddingStep] No template found for {var} ({freq_code})")
                    continue
                template_file = matching_templates[0]                       

                # process each input gn‑file
                for gn_path in glob(os.path.join(gn_dir, "*_gn_*.nc")):
                    fname = os.path.basename(gn_path)
                    gr_fname = fname.replace("_gn_", "_gr_")
                    gr_path = os.path.join(gr_dir, gr_fname)

                    # skip if target file already exists and non‑empty
                    if not force_rerun and os.path.exists(gr_path) and os.path.getsize(gr_path) > 0:
                        print(f"[RunRegriddingStep] Skipping regrid; output exists {gr_path}")
                        self.logger.info(f"{GREEN} [RunRegriddingStep] Skipping regrid for {gn_path}; output exists {gr_path}")
                        continue
                    elif os.path.exists(gr_path):
                        self.logger.warning(f"{RED} [RunRegriddingStep] Overwriting empty file: {gr_path}")

                    self.logger.info(f"[RunRegriddingStep] Regridding {gn_path} → {gr_path}, using template {template_file}")
                    try:
                        with xr.open_dataset(template_file, decode_times=False) as template_ds, \
                             xr.open_dataset(gn_path) as native_ds:

                            # regrid
                            regrid_to_template_grid(native_ds, template_ds, gr_path)

                    except Exception as e:
                        self.logger.error(f"{RED} [RunRegriddingStep] Failed on {gn_path}: {e}")
                        raise
