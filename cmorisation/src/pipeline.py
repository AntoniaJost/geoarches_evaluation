# pipeline.py — Main entry point for running all processing steps in sequence.
# 
# Steps:
# 1. Load config and set up logger
# 2. Run: RenameVars → SplitVars → Cmorise
# 3. Optionally delete intermediate outputs

import sys
import argparse
from config_loader import ConfigLoader
import shutil
import os
from logger import setup_logger, success, error, WARN, GREEN

# import processing steps
from steps.rename_vars import RenameVarsStep
from steps.split_vars import SplitVarsStep
from steps.cmorise import CmoriseStep

def main():
    # parse command line arguments
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    # load config
    cfg = ConfigLoader(args.config)
    general = cfg.get('general') if hasattr(cfg, 'get') else cfg.general
    log_level = general.get("log_level", "INFO")
    log_dir = general.get("log_dir", "logs")

    # initialize logger
    logger = setup_logger(log_dir, log_level)

    # define processing steps (order is important!)
    steps = [
        RenameVarsStep( cfg.get('rename_vars'), logger),
        SplitVarsStep(cfg.get('split_vars'), logger),
        CmoriseStep(cfg.get('cmorise'), logger),
    ]

    # run each step and report success or failure
    for step in steps:
        try:
            step.run()
            success(logger, f"➡️ {step.name} done")
            print(f"➡️ {step.name} done")
        except Exception as e:
            error(logger, f"{step.name} failed: {e}")
            sys.exit(1)

    # optionally delete all intermediate output folders besides the cmorised output, if requested
    if general.get('delete_intermediate_outputs', False):
        folders_to_delete = [
            general['work_dir'] + "/1_means",
            general['work_dir'] + "/2_renamed",
            general['work_dir'] + "/3_split",
        ]
        for folder in folders_to_delete:
            if os.path.isdir(folder):
                try:
                    shutil.rmtree(folder)
                    logger.info(f"{WARN} [pipeline] Deleted intermediate folder: {folder}")
                except Exception as e:
                    logger.warning(f"[pipeline] Could not delete folder {folder}: {e}")
        print(f"{GREEN} [pipeline] Cleanup complete: intermediate outputs deleted.")
        logger.info(f"{GREEN} [pipeline] Cleanup complete: intermediate outputs deleted.")
    else:
        logger.info("➡️ [pipeline] Keeping intermediate outputs (set delete_intermediate_outputs to False if you want to keep more than the final cmorised output.)")


if __name__ == '__main__': main()