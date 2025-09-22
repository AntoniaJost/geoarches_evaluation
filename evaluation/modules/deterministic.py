import numpy as np
import pandas as pd
import xarray as xr

import torch
from tensordict import TensorDict
from modules.base import AAIMIPRollout


class DeterministicAAIMIPRollout(AAIMIPRollout):
    def __init__(
            self, 
            module, 
            dataset, 
            aimip_name, 
            storage_path, 
            storage_type="monthly", 
            member=1, 
            grid_type='gn', 
            init_method=1, 
            physics=1, 
            forcings=1,
            sst_scenario='',
            continue_rollout=True,
            replace_nans_from_state=True,
            store_initial_condition=True,
        ):

        """
        Initialize the AAIMIPRollout class.
        Args:
            module: The model module to be used for the rollout.
            dataset: The dataset from which to draw data for the rollout.
            aimip_name: Name of the model.
            storage_path: Path where the output files will be stored.
            storage: Type of storage, either 'monthly' or 'daily'.
            member: Member number for the model.
            grid_type: Type of grid used in the model.
            init_method: Initialization method used in the model.
            physics: Physics configuration used in the model.
            forcings: Forcings configuration used in the model.
        """
        super().__init__(
            module=module,
            dataset=dataset,
            aimip_name=aimip_name,
            storage_path=storage_path,
            storage_type=storage_type,
            member=member,
            grid_type=grid_type,
            init_method=init_method,
            physics=physics,
            forcings=forcings,
            sst_scenario=sst_scenario,
            continue_rollout=continue_rollout,
            replace_nans_from_state=replace_nans_from_state,
            store_initial_condition=store_initial_condition,
        )

    
    def update_batch(self, pred, loop_batch):
        # Update the batch with the model's output
        new_loop_batch = {k: v for k, v in loop_batch.items()}
        new_loop_batch['prev_state'] = loop_batch['state']
        new_loop_batch['state'] = pred
        new_loop_batch['timestamp'] = loop_batch['timestamp'] + self.lead_time_hours * 3600  # increment timestamp by lead time in seconds
        print(
            "Updated timestamp to:", 
            pd.to_datetime(new_loop_batch['timestamp'].detach().cpu(), unit='s').tz_localize(None).strftime('%Y-%m-%d %H:%M:%S')
        )

        if "forcings" in loop_batch.keys():
            new_loop_batch['forcings'] = self.get_forcings(new_loop_batch['timestamp'])

        #if replace_land_grid_from_forcings and self.forcings is not None:
        """
        variables = self.dataset.variables
        vars_surface = variables['surface']
        sst_idx = vars_surface.index('sea_surface_temperature')
        sic_idx = vars_surface.index('sea_ice_cover')

        batch_sst = loop_batch['state']['surface'][:, sst_idx, :, :]
        batch_sic = loop_batch['state']['surface'][:, sic_idx, :, :]

        # get month from loop_batch timestamp  
        month = pd.to_datetime(loop_batch['timestamp'].detach().cpu(), unit='s').tz_localize(None).month - 1  # month is 0-indexed
        forcing_sst = self.monthly_forcings[:, month:month+1].to(loop_batch['state'].device).squeeze(1)
        forcing_sic = self.monthly_forcings[:, month:month+1].to(loop_batch['state'].device).squeeze(1)
        batch_sst[torch.isnan(self.land_sea_mask)] = forcing_sst[torch.isnan(self.land_sea_mask)]
        batch_sic[torch.isnan(self.land_sea_mask)] = forcing_sic[torch.isnan(self.land_sea_mask)]
        loop_batch['state']['surface'][:, sst_idx, :, :] = batch_sst
        loop_batch['state']['surface'][:, sic_idx, :, :] = batch_sic
        """

        return new_loop_batch
    
    @torch.no_grad()
    def _rollout(self, rollout_steps, batch):
        
        # Get loop batch
        loop_batch = batch.copy()
        loop_batch = TensorDict(
            {
                k: v.unsqueeze(0) if v is not None else None for k, v in loop_batch.items()
            },
            batch_size=[1]
        )
        loop_batch = loop_batch.to('cuda')

        # Initialize the list to store states
        states = []

        # If store_init is True, convert the initial state to xarray and append it
        if self.store_initial_condition:
            states.append(loop_batch['state'])
            
        for i, steps in enumerate(rollout_steps):
            # Print current timestamp of initial condition for control purpose as string
            timestamp = loop_batch['timestamp']
            init_timestamp = pd.to_datetime(
                loop_batch['timestamp'].detach().cpu(), unit='s').tz_localize(None)
            print(init_timestamp.strftime('%Y-%m-%d %H:%M:%S'), "\n---------------------------------------\n")
            
            #if i == 0 and self.store_initial_condition: 
            #    steps -= 1  # already stored initial conditions
               
            # Forward pass through the model
            if steps <= 0:
                pass
            else:
                output, loop_batch = self.module.forward_multistep(
                    loop_batch, 
                    iters=steps, 
                    return_format="list", 
                    return_loop_batch=True, 
                    update_fnc=self.update_batch
                )
                states.extend(output)

            # Dump the output to netcdf
            
            inc = 1 if i > 0 else 0
            self.dump_to_netcdf(states, timestamp + inc * self.lead_time_hours * 3600)

            # Clear states to save memory
            states.clear()

    def compute_number_of_rollout_steps(self, start, end):
        """
        Compute the number of steps required for the rollout based on the start and end times.
        """
        delta = (end - start).astype('timedelta64[h]')

        if delta < 0:
            raise ValueError("End time must be after start time.")
        elif delta == 0:
            raise ValueError("Start and end times must not be the same.")
        
        nsteps = int(delta.astype('timedelta64[h]') / np.timedelta64(self.lead_time_hours, 'h'))
        
        if nsteps <= 0:
            raise ValueError("The time range must be larger than the lead time.")
        
        print(
            "Number of steps for rollout:", nsteps,
            " yielding a total of ", (nsteps * self.lead_time_hours) // 24, "days."
        )
        
        return nsteps
    
    

    def rollout(self, start_timestamp: np.datetime64, end_timestamp: np.datetime64):

        batch, rollout_steps = super().rollout(start_timestamp, end_timestamp)

        print("######## Rollout ArchesWeather ############")
                
        print(
            "Starting rollout from", start_timestamp, 
            "to", end_timestamp, "with", sum(rollout_steps), "steps."
        )
        
        # Perform the deterministic rollout
        self._rollout(rollout_steps, batch)

        print('Done.\n#######################################\n')
