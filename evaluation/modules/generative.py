

import numpy as np
import torch

from geoarches.dataloaders.era5_constants import arches_default_pressure_levels
from modules.base import AAIMIPRollout

from tensordict import TensorDict

class GenerativeAAIMIPRollout(AAIMIPRollout):
    def __init__(
            self, module, dataset, 
            aimip_name, storage_path, 
            storage_type="monthly", member=1, 
            grid_type='gn', init_method=1, 
            physics=1, forcings=1, sst_scenario='', 
            continue_rollout=True, replace_nans_from_state=True, 
            replacement_method='daily', store_initial_condition=True):
        
        super().__init__(
            module, dataset, aimip_name, 
            storage_path, storage_type, member, 
            grid_type, init_method, physics, 
            forcings, sst_scenario, continue_rollout, 
            replace_nans_from_state, replacement_method,
            store_initial_condition)
        
    
    @torch.no_grad()
    def _rollout(self, start_timestamp, rollout_steps, batch):

        loop_batch = batch.copy()
        loop_batch = TensorDict({k: v.unsqueeze(0) for k, v in loop_batch.items()}, batch_size=[1])
        loop_batch = loop_batch.to('cuda')

        if self.store_initial_condition:
            states.append(
                self.dataset.convert_to_xarray(
                    self.dataset.denormalize(loop_batch['state']), 
                    loop_batch['timestamp'], 
                    levels=arches_default_pressure_levels
                )
            )

        for i, steps in enumerate(rollout_steps):


            output = self.module.sample_rollout(
                loop_batch, batch_nb=i, 
                member=self.member, iterations=steps,
                return_format="list"
            )
                
            outputs = [self.dataset.convert_to_xarray(
                self.dataset.denormalize(o), 
                loop_batch['timestamp'].detach().cpu() + self.lead_time_hours * 3600 * (i + 1), 
                levels=arches_default_pressure_levels)
                for i, o in enumerate(output)
            ]

            states.extend(outputs)
            current_timestamp = start_timestamp + np.timedelta64(sum(rollout_steps[:i+1]), 'D')
            self.dump_to_netcdf(states, current_timestamp)

            # Update the batch with the model's output
            loop_batch = self.update_batch(loop_batch, output[-1], mult=steps)
            states = []

    def rollout(self, start_timestamp, end_timestamp, store_initial_condition):

        # Convert timestamps to datetime64[ns]
        start_timestamp = np.datetime64(start_timestamp).astype('datetime64[ns]')
        end_timestamp = np.datetime64(end_timestamp).astype('datetime64[ns]')

        # Calculate the number of rollout steps in years 
        # (gen_module.sample_trajectory takes in number of steps, thus days per year are needed)
        rollout_steps = (end_timestamp - start_timestamp).astype('timedelta64[Y]').astype(int)

        # Get the initial batch
        initial_batch = self.get_initial_batch(start_timestamp)

        # Perform the rollout
        self._rollout(start_timestamp, rollout_steps, initial_batch)
