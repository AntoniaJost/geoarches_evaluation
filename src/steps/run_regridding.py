import os
import subprocess
from glob import glob
from steps.step_base import Step
from logger import GREEN, RED, WARN
from utils.change_tracker import get_force_rerun_flag

class RunRegriddingStep(Step):
    """
    Regrids all *_gn_*.nc files to *_gr_*.nc for each variable and scale:
    - Traverses base_dir/<scale>/<var>/gn/*.nc
    - Runs a user-specified external regridding script on each input file
    - Moves the regridded *_gr_*.nc result into a separate 'gr' subdirectory
    - Supports forced regridding via change-tracking flag
    """
    def run(self):
        # load config
        cfg      = self.cfg
        base_dir = cfg['base_dir']
        scales   = cfg.get('scales', [])
        gn_sub   = cfg.get('gn_subdir', 'gn')
        gr_sub   = cfg.get('gr_subdir', 'gr')
        script   = cfg['script']
        meta_over = cfg.get('metadata_overrides', {})

        for scale in scales:
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

                    self.logger.info(f"[RunRegriddingStep] Regridding {gn_path} → {gr_path}")
                    
                    # construct command to call external regridding script
                    cmd = ["python", script, gn_path]

                    # Append metadata_overrides from config
                    for key, val in self.cfg.get("metadata_overrides", {}).items():
                        cmd += ["--override", f"{key}={val}"]

                    try:
                        # run that regridding subprocess
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        self.logger.error(f"{RED} [RunRegriddingStep] Regridding failed for {gn_path}: {e}")
                        raise

                    # move output file from gn folder to final gr folder to align with convention
                    generated = os.path.join(gn_dir, gr_fname)
                    if not os.path.exists(generated):
                        self.logger.error(f"{RED} [RunRegriddingStep] Expected output not found: {generated}")
                        raise FileNotFoundError(generated)

                    os.replace(generated, gr_path)
                    self.logger.info(f"{GREEN} [RunRegriddingStep] Moved to: {gr_path}")
