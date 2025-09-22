

import os
import numpy as np

import hydra
from omegaconf import OmegaConf

import torch

from geoarches.lightning_modules.base_module import load_module

# set mixed precision for torch matmul
torch.set_float32_matmul_precision('medium')


@hydra.main(version_base=None, config_path="configs", config_name="rollout_config")
def main(cfg):
    """
    Main function to run the rollout for detection.
    """
    OmegaConf.resolve(cfg)

    # Load the module based on the configuration
    print("Loading module for rollout ...", end=' ')
    module, loaded_config = load_module(cfg.model_name)
    print("Done.")

    # Get dataset
    print("Loading dataset for rollout ...", end=' ')
    loaded_config.dataloader.dataset.domain = "all"
    loaded_config.dataloader.dataset._target_ = "geoarches.dataloaders.era5.Era5Forecast"

    if hasattr(loaded_config.dataloader.dataset, "pred_path"):
        del loaded_config.dataloader.dataset.pred_path
        del loaded_config.dataloader.dataset.load_hard_neg
    
    dataset = hydra.utils.instantiate(loaded_config.dataloader.dataset, loaded_config.stats)
    print("Done.")


    # Create the rollout module
    print("Creating rollout module ...", end=' ')
    rollout_module = hydra.utils.instantiate(cfg.aimip, module, dataset)
    print("Done.")

    print(rollout_module)
    
    # Convert start and end timestamps to numpy datetime64
    start_timestamp = np.datetime64(cfg.start_timestamp).astype('datetime64[ns]')
    end_timestamp = np.datetime64(cfg.end_timestamp).astype('datetime64[ns]')

    rollout_module.rollout(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )


if __name__ == "__main__":
    main()