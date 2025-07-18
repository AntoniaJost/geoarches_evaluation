import os
import json
from logger import RED
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def get_force_rerun_flag(base_name: str, logger=None) -> bool:
    """
    Determine whether to force a rerun of a pipeline step based on detected changes.

    Logic depends on the `means_changed.json` file, which tracks whether upstream means
    were recalculated. The decision rule is:
    
    - For 'daily' steps: rerun only if any changed year is ≤ 1979
    - For 'monthly' steps: rerun if any year has changed

    Parameters:
    base_name : str
        Either "daily" or "monthly", depending on which step is being evaluated.
    logger : logging.Logger, optional
        Logger for debug or warning messages.

    Returns
    bool: True if rerun is required, False otherwise.
    """
    # retrieve means_chaned_path from config
    path = config['general']['means_changed_path']
    if not path:
        if logger:
            logger.warning("[change_tracker] No means_changed_path found in config.")
        return False
    
    force_rerun = False

    if os.path.exists(path):
        try:
            # load list of changed years
            with open(path) as f:
                changed_years = json.load(f).get("years_changed", [])
            if logger:
                logger.debug(f"[change_tracker] {base_name}: checking changes {changed_years}")

            # if files changed that are about years < 1979, daily steps don't have to be rerun as they are anyway not affected; saves compute power
            if base_name == "daily":
                force_rerun = any(int(y) <= 1979 for y in changed_years)
            elif base_name == "monthly":
                force_rerun = len(changed_years) > 0
        except Exception as e:
            if logger:
                logger.warning(f"{RED} [change_tracker] Failed to parse means_changed.json: {e}")

    return force_rerun
