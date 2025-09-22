

import os
import glob

from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr

import torch

from calendar import isleap

from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.dataloaders.era5_constants import arches_default_pressure_levels

# set mixed precision for torch matmul
torch.set_float32_matmul_precision('medium')

def load_batch_from_prediction(xrds: xr.Dataset, dataset: Era5Forecast):
    """
    Load the data from a xarray dataset. 
    Select the last two days of a year as an input.
    The variables are put into a tensordict with keys
    state and prev_state. Each key contains a tensordict with keys
    level and surface.
    args:
    variables: Dictionary of variables to select from the dataset, with keys
    surface and level. 
    levels: List of pressure levels to select from the dataset.
    """
    state = xrds.isel(time=-1)
    prev_state = xrds.isel(time=-2)

    state = dataset.convert_to_tensordict(state)
    state = dataset.normalize(state)
    prev_state = dataset.convert_to_tensordict(prev_state)
    prev_state = dataset.normalize(prev_state)

    batch = {
        "state": state,
        "prev_state": prev_state,
        "next_state": state
    }

    timestamp = torch.tensor(
        xrds.time.values[-1].astype('datetime64[ns]').item() // 10**9, dtype=torch.float64)
    
    batch['timestamp'] = timestamp
    batch['lead_time_hours'] = torch.tensor(dataset.lead_time_hours, dtype=torch.int64)

    return batch, xrds.time.values[-1]


class AAIMIPRollout:
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
            replacement_method='daily', # daily, monthly forcings, running mean
            store_initial_condition=True,
        ): 

        """
        Initialize the AAIMIPRollout class.
        Args:
            module: The model module to be used for the rollout.
            dataset: The dataset from which to draw data for the rollout.
            aimip_name: Name of the model for the aimip submission.
            storage_path: Path where the output files will be stored.
            storage_type: Type of storage_type, either 'monthly' or 'daily'.
            member: Member number for the model.
            grid_type: Type of grid used in the model.
            init_method: Initialization method used in the model.
            physics: Physics configuration used in the model.
            forcings: Forcings configuration used in the model.
        """
        super().__init__()
        
        self.aimip_name = aimip_name
        self.module = module.cuda()
        self.module.eval()  # Set the module to evaluation mode
        self.dataset: Era5Forecast = dataset
        self.timestamps = [t[-1] for t in self.dataset.timestamps]
        self.lead_time_hours = self.dataset.lead_time_hours
        self.continue_rollout = continue_rollout

        # Storage configuration
        if storage_type not in ["monthly", "daily"]:
            raise ValueError("storage_type type must be either 'monthly' or 'daily'.")
        self.storage_type = storage_type
        self.storage_path = os.path.join(storage_path, f"member_{member:02d}", "data")
        os.makedirs(self.storage_path, exist_ok=True)

        os.makedirs(self.storage_path, exist_ok=True)

        # File name specific parameters
        if not isinstance(member, int) or member < 0:
            raise ValueError("Member must be a non-negative integer.")
        
        if grid_type not in ['gn', 'gr']:
            raise ValueError("Grid type must be one of 'gn', or 'gr'.")
        
        self.member = member
        self.grid_type = grid_type
        self.init_method = init_method
        self.physics = physics  
        self.forcings = forcings  
        self.sst_scenario = sst_scenario
        self.replace_nans_from_state = replace_nans_from_state
        self.replacement_method = replacement_method
        self.store_initial_condition = store_initial_condition

        if self.dataset.forcings_ds is not None:
            print('Loading monthly forcings from dataset ... ', end='')
            timestamps = [np.datetime64("1979-01").astype('datetime64[M]') + np.timedelta64(i, 'M') for i in range(12)]
            forcings = []
            for t in timestamps:
                forcings.append(self.dataset.load_forcings(t))
            self.monthly_forcings = torch.stack(forcings, dim=0).unsqueeze(0).cuda()  # already in batch format
            self.land_sea_mask = self.dataset.forcings_ds['land_sea_mask'].to_numpy()  # (1, lat, lon)
            self.land_sea_mask = torch.tensor(self.land_sea_mask, dtype=torch.float32).cuda()  # (1, 1, lat, lon)
            print(self.monthly_forcings.shape)

            print('Done.')
        else:
            self.monthly_forcings = None
            warn("No forcings dataset found in the dataloader. Make sure this is intended.")

    def get_file_name(self, timestamp, member=None):
        """
        Generate a file name based on the timestamp.
        """
        timestamp = pd.to_datetime(timestamp.cpu(), unit='s').tz_localize(None).to_numpy()
        month = timestamp.astype('datetime64[M]').astype(int) % 12 + 1
        year = timestamp.astype('datetime64[Y]').astype(int) + 1970
        if isinstance(year, np.ndarray) or isinstance(year, list):
            year = year[0]  # Use th e first year if it's an array or list
            print(year)
        member = member or self.member
        
        if self.storage_type == "monthly":
            return f"Amon_{self.aimip_name}_aimip_r{member}i{self.init_method}p{self.physics}f{self.forcings}_{self.grid_type}_{year}_{month}.nc"
        elif self.storage_type == "daily":
            return f"day_{self.aimip_name}_aimip_r{member}i{self.init_method}p{self.physics}f{self.forcings}_{self.grid_type}_{year}.nc"
        else:
            raise ValueError("Storage type must be either 'monthly' or 'daily'.")

    def timestamp_to_batch(self, timestamp):
        idx = self.timestamps.index(timestamp)
        idx -= self.dataset.lead_time_hours // self.dataset.timedelta # account for prev_timestamp
        if idx < 0:
            raise ValueError("Timestamp is too early for the dataset.")
        elif idx >= len(self.timestamps):
            raise ValueError("Timestamp is too late for the dataset.")
        
        batch = self.dataset[idx]
        batch = {k: v.float() if 'state' in k else v for k, v in batch.items()}
        print(batch.keys())

        return batch
    
    def update_batch(self, loop_batch, prediction):

        raise NotImplementedError(
            "This is a base class. " \
            "Use a subclass for specific update_batch implementations."
        )

    def get_initial_batch(self, start_timestamp):
        files = glob.glob(f"{self.storage_path}/*.nc")

        if self.continue_rollout and len(files) > 0:

            print("Resuming Rollout ...")
            files.sort()
            f = files[-1]

            print(f"Loading batch from {f} to continue rollout ...")
            xrds = xr.load_dataset(f)
            batch, timestamp = load_batch_from_prediction(xrds, self.dataset)

            print("Batch loaded from timestamp: ", timestamp)
            start_timestamp = timestamp + np.timedelta64(self.lead_time_hours, 'h')

            print("Continuing rollout from timestamp: ", start_timestamp)
            self.store_initial_condition = False  # Do not store initial condition if continuing rollout
        else:
            if start_timestamp not in self.timestamps:
                raise ValueError(f"Start timestamp {start_timestamp} not found in dataset timestamps.")
            print("Preparing initial batch for timestamp: ", start_timestamp, end=' ... ')
            batch = self.timestamp_to_batch(start_timestamp)
            print("Done.")

        return batch
    
    def get_forcings(self, timestamp):
        assert self.monthly_forcings is not None, "No forcings dataset found in the dataloader."
        
        # get month of the current timestamp 
        month = pd.to_datetime(timestamp.cpu(), unit='s').tz_localize(None).month
        
        # gather the forcings for the current month
        forcings = self.monthly_forcings[:, month - 1, ...].squeeze(1)  # (1, C, lat, lon)

        return forcings

    def compute_steps_per_years(self, start_timestamp, end_timestamp):
        """
        Computes the number of rollout steps per year between start and end timestamps.
        If the time range is less than one year, it computes the number of days.
        1 year = 365 or 366 days

        Args:
            start_timestamp (np.datetime64): The start timestamp.
            end_timestamp (np.datetime64): The end timestamp.
        Returns:
            List[int]: A list containing the number of days in each year for the rollout.
        """
        nyears = (end_timestamp - start_timestamp).astype('timedelta64[Y]').astype(int)
        if nyears <= 0:
            rollout_steps = (end_timestamp - start_timestamp).astype('timedelta64[D]').astype(int)
            if rollout_steps <= 0:
                raise ValueError("The time range must be larger than one day.")
            else:
                rollout_steps = [rollout_steps]
        else:
            # first check if the first timestamp is at the beginning of a year
            # If not, the first year will be partial and we need to compute the number of days in that year
            if start_timestamp.astype('datetime64[M]').astype(int) % 12 != 0 or start_timestamp.astype('datetime64[D]').astype(int) % 31 != 0:
                last_day_of_first_year = (start_timestamp.astype('datetime64[Y]') + np.timedelta64(1, 'Y') - np.timedelta64(1, 'D')).astype('datetime64[D]')
                days_in_first_year = (last_day_of_first_year - start_timestamp.astype('datetime64[D]')).astype('timedelta64[D]').astype(int)
                print("Days in the first year: ", days_in_first_year)
                rollout_steps = [days_in_first_year]
                rollout_steps += [366 if isleap((start_timestamp.astype('datetime64[Y]') + np.timedelta64(i, 'Y')).astype(int) + 1970) 
                    else 365 for i in range(1, nyears)]
            else:
                rollout_steps = [366 if isleap((start_timestamp + np.timedelta64(i, 'Y')).astype('datetime64[Y]').astype(int) + 1970) 
                    else 365 for i in range(nyears)
                ]

        print("Computed rollout steps per year (in days): \n", rollout_steps)

        return rollout_steps
    
    def dump_to_netcdf(self, output, timestamp):
        # Concat list of xarrays along time dimension

        xarrays = [self.dataset.convert_to_xarray(
            self.dataset.denormalize(o), 
            timestamp + self.lead_time_hours * 3600 * i, 
            levels=arches_default_pressure_levels)
            for i, o in enumerate(output)
        ]

        xarrays = xr.concat(xarrays, dim=self.dataset.time_dim_name)

        # Generate file name
        file_name = self.get_file_name(timestamp)

        # Save the xarray to a netcdf file
        if self.storage_type == "monthly":
            xarrays = xarrays.mean(dim=self.dataset.time_dim_name)
            xarrays[self.dataset.time_dim_name] = pd.to_datetime(timestamp.astype('datetime64[M]')).tz_localize(None)
            xarrays = xarrays.set_coords(self.dataset.time_dim_name)
            xarrays.to_netcdf(f"{self.storage_path}/{file_name}")
        elif self.storage_type == "daily":
            xarrays.to_netcdf(f"{self.storage_path}/{file_name}")
        else:
            raise ValueError("Storage type must be either 'monthly' or 'daily'.")
        
    def rollout(self, start_timestamp, end_timestamp):
        batch = self.get_initial_batch(start_timestamp)
        steps_per_year = self.compute_steps_per_years(start_timestamp, end_timestamp)

        return batch, steps_per_year