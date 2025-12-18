import numpy as np
import pandas as pd

import torch
from tensordict import TensorDict
from modules.base import AAIMIPRollout
from geoarches.lightning_modules.forecast import ForecastModuleWithCond


class DeterministicAAIMIPRollout(AAIMIPRollout):
    def __init__(
        self,
        module,
        dataset,
        aimip_name,
        storage_path,
        storage_type="monthly",
        member=1,
        grid_type="gn",
        init_method=1,
        physics=1,
        forcings=1,
        sst_scenario="",
        continue_rollout=True,
        replace_land_grid_from_forcings=True,
        store_initial_condition=True,
        ablate_forcings=False,
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
            replace_land_grid_from_forcings=replace_land_grid_from_forcings,
            store_initial_condition=store_initial_condition,
            ablate_forcings=ablate_forcings,
        )

    def update_batch(self, loop_batch, pred):
        # Update the batch with the model's output
        new_loop_batch = {k: v for k, v in loop_batch.items()}
        new_loop_batch["prev_state"] = loop_batch["state"]
        new_loop_batch["state"] = pred
        new_loop_batch["timestamp"] = (
            loop_batch["timestamp"] + self.lead_time_hours * 3600
        )  # increment timestamp by lead time in seconds

        print(
            "Updated timestamp to:",
            pd.to_datetime(new_loop_batch["timestamp"].detach().cpu(), unit="s")
            .tz_localize(None)
            .strftime("%Y-%m-%d %H:%M:%S"),
        )

        if "forcings" in loop_batch.keys():
            new_loop_batch["forcings"] = self.get_forcings(
                new_loop_batch["timestamp"]
            ).unsqueeze(0)

        if self.replace_land_grid_from_forcings and self.forcings is not None:
            variables = self.dataset.variables
            vars_surface = variables["surface"]

            # Gather indices for sst and sic in state and forcings
            sst_idx = vars_surface.index("sea_surface_temperature")
            sic_idx = vars_surface.index("sea_ice_cover")
            frc_sst_idx = self.dataset.forcing_vars.index("sea_surface_temperature")
            frc_sic_idx = self.dataset.forcing_vars.index("sea_ice_cover")

            # Replace land grid points in sst and sic with values from forcings
            batch_sst = new_loop_batch["state"]["surface"][:, sst_idx, ...]
            batch_sic = new_loop_batch["state"]["surface"][:, sic_idx, ...]
            forcing_sst = new_loop_batch["forcings"][:, frc_sst_idx, ...]
            forcing_sic = new_loop_batch["forcings"][:, frc_sic_idx, ...]

            # Make data compatible for masking
            lsm = self.land_sea_mask.unsqueeze(0).unsqueeze(
                0
            )  # add batch and channel dimensions
            forcing_sst = forcing_sst.unsqueeze(1)  # add channel dimension
            forcing_sic = forcing_sic.unsqueeze(1)  # add channel dimension
            print(batch_sst.shape, forcing_sst.shape, lsm.shape)
            batch_sst[lsm == 1] = forcing_sst[lsm == 1]
            batch_sic[lsm == 1] = forcing_sic[lsm == 1]
            new_loop_batch["state"]["surface"][:, sst_idx, ...] = batch_sst
            new_loop_batch["state"]["surface"][:, sic_idx, ...] = batch_sic

        return new_loop_batch

    @torch.no_grad()
    def _rollout(self, rollout_steps, batch):
        # Get loop batch
        loop_batch = batch.copy()
        loop_batch = TensorDict(
            {
                k: v.unsqueeze(0) if v is not None else None
                for k, v in loop_batch.items()
            },
            batch_size=[1],
        )
        loop_batch = loop_batch.to("cuda")

        # Initialize the list to store states
        states = []

        # If store_init is True, convert the initial state to xarray and append it
        if self.store_initial_condition:
            states.append(loop_batch["state"])

        for i, steps in enumerate(rollout_steps):
            # Print current timestamp of initial condition for control purpose as string
            timestamp = loop_batch["timestamp"]
            init_timestamp = pd.to_datetime(
                loop_batch["timestamp"].detach().cpu(), unit="s"
            ).tz_localize(None)
            print(
                init_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "\n---------------------------------------\n",
            )

            if i == 0 and self.store_initial_condition:
                inc = 0
            else:
                inc = 1

            # Forward pass through the model
            if steps <= 0:
                pass
            else:
                output, loop_batch = self.module.forward_multistep(
                    loop_batch,
                    iters=steps,
                    return_format="list",
                    return_loop_batch=True,
                    update_fnc=self.update_batch,
                )
                states.extend(output)

            # Dump the output to netcdf

            self.dump_to_netcdf(states, timestamp + inc * self.lead_time_hours * 3600)

            # Clear states to save memory
            states.clear()

    def rollout(self, start_timestamp: np.datetime64, end_timestamp: np.datetime64):
        batch, rollout_steps = super().rollout(start_timestamp, end_timestamp)

        print("######## Rollout ArchesWeather ############")

        print(
            "Starting rollout from",
            start_timestamp,
            "to",
            end_timestamp,
            "with",
            sum(rollout_steps),
            "steps.",
        )

        # Perform the deterministic rollout
        self._rollout(rollout_steps, batch)

        print("Done.\n#######################################\n")


class DeterministicClimateProjector(ForecastModuleWithCond):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset

    def get_forcings(self, timestamp):
        # Retrieve forcings for the given timestamp
        forcings = self.dataset.load_forcings(timestamp)

        return forcings

    def update_batch(self, pred, loop_batch):
        # Update the batch with the model's output
        new_loop_batch = {k: v for k, v in loop_batch.items()}
        new_loop_batch["prev_state"] = loop_batch["state"]
        new_loop_batch["state"] = pred
        new_loop_batch["timestamp"] = (
            loop_batch["timestamp"] + self.lead_time_hours * 3600
        )  # increment timestamp by lead time in seconds
        print(
            "Updated timestamp to:",
            pd.to_datetime(new_loop_batch["timestamp"].detach().cpu(), unit="s")
            .tz_localize(None)
            .strftime("%Y-%m-%d %H:%M:%S"),
        )

        if "forcings" in loop_batch.keys():
            new_loop_batch["forcings"] = self.get_forcings(new_loop_batch["timestamp"])

        return new_loop_batch

    def simulate(self, batch: TensorDict, target_date: str) -> TensorDict:
        """Simulate the next state given the current batch."""
        # Directly use the model's forward method for deterministic simulation

        current_date = pd.to_datetime(
            batch["timestamp"].detach().cpu(), unit="s"
        ).tz_localize(None)
        target_date = pd.to_datetime(target_date).tz_localize(None)

        predictions = []
        predictions.append(batch["state"])
        while current_date != target_date:
            next_state = self.forward(batch)
            predictions.append(next_state)

            batch = self.update_batch(next_state, batch)
            current_date = pd.to_datetime(
                batch["timestamp"].detach().cpu(), unit="s"
            ).tz_localize(None)

            if current_date.month == 12 and current_date.day == 31:
                print(
                    "Reached end of year:", current_date.strftime("%Y-%m-%d %H:%M:%S")
                )
                xr_datasets = [self.dataset.convert_to_xarray(p) for p in predictions]
                # xr_datasets = xr.concat(xr_datasets, dim='time')
                self.dump_to_netcdf(
                    xr_datasets, batch["timestamp"] - self.lead_time_hours * 3600
                )
                predictions.clear()
