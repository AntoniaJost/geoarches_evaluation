

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
    if not all([v[0] in xrds.dims.keys() for v in dataset.dimension_indexers.values()]):#
        # Respect dimension indexing of the loaded dataset (depends on dim names in dataset)
        name_map = {
            k: v[0] for k, v in dataset.dimension_indexers.items() if v[0] not in xrds.dims.keys()
        }  # only rename dimensions that are not already named correctly

        xrds = xrds.rename_dims(name_map)
        xrds = xrds.rename_vars(name_map)

    state = xrds.isel({name_map['time']: -1})
    prev_state = xrds.isel({name_map['time']: -2})

    state = dataset.convert_to_tensordict(state)
    state = dataset.normalize(state)
    prev_state = dataset.convert_to_tensordict(prev_state)
    prev_state = dataset.normalize(prev_state)

    batch = {
        "state": state,
        "prev_state": prev_state,
        "next_state": state # just a surrogate for forecasting
    }

    # access time coordinate via string
    time = xrds.coords[name_map['time']].values[-1].astype('datetime64[ns]').item()
    timestamp = torch.tensor(
        time // 10**9, dtype=torch.float64)

    batch['timestamp'] = timestamp
    batch['lead_time_hours'] = torch.tensor(dataset.lead_time_hours, dtype=torch.int64)

    return batch, xrds.coords[name_map['time']].values[-1]


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
            sst_scenario='',  # either '0', '2' or '4' indicating the sst scenario
            continue_rollout=True,
            replace_land_grid_from_forcings=True,
            replacement_method='daily', # daily, monthly forcings, running mean
            store_initial_condition=True,
            ablate_forcings=False
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
        self.ablate_forcings = ablate_forcings

        files = glob.glob(self.dataset.path + "/*.nc")  
        files = sorted(files)
        files = [f for f in files if "12h" in f]
        self.era5_ds = xr.open_mfdataset(files)  # to avoid multiprocessing issues with xarray

        # Storage configuration
        if storage_type not in ["monthly", "daily"]:
            raise ValueError("storage_type type must be either 'monthly' or 'daily'.")
        self.storage_type = storage_type
        member_id = f"{member:02d}" if isinstance(member, int) else member
        if ablate_forcings:
            self.storage_path = os.path.join(storage_path, f"member_{member_id}", "ablate_forcings", "data")
        else:
            self.storage_path = os.path.join(storage_path, f"member_{member_id}", "data")

        
        os.makedirs(self.storage_path, exist_ok=True)

        os.makedirs(self.storage_path, exist_ok=True)
        
        if grid_type not in ['gn', 'gr']:
            raise ValueError("Grid type must be one of 'gn', or 'gr'.")
        
        self.member = member
        self.grid_type = grid_type
        self.init_method = init_method
        self.physics = physics  
        self.forcings = forcings  
        self.sst_scenario = sst_scenario
        self.replace_land_grid_from_forcings = replace_land_grid_from_forcings
        self.replacement_method = replacement_method
        self.store_initial_condition = store_initial_condition

        if self.dataset.forcings_ds is not None:
            self.update_forcings = True
            print('Loading monthly forcings from dataset ... ', end='')
            timestamps = [np.datetime64("1979-01").astype('datetime64[M]') + np.timedelta64(i, 'M') for i in range(12)]
            forcings = []
            for t in timestamps:
                forcings.append(self.dataset.load_forcings(t))
            self.monthly_forcings = torch.stack(forcings, dim=0) # already in batch format
            self.land_sea_mask = self.dataset.forcings_ds['land_sea_mask'].to_numpy()  # (1, lat, lon)
            self.land_sea_mask = torch.tensor(self.land_sea_mask, dtype=torch.float32).cuda() # (1, 1, lat, lon)
            
            if self.sst_scenario in ['2', '4']:
                # Apply sst scenario adjustment to forcings

                sst_index = self.dataset.forcing_vars.index('sea_surface_temperature')
                self.monthly_forcings[:, sst_index, ...] = self.monthly_forcings[:, sst_index, ...]  * self.dataset.forcings_std[None, sst_index, None, None]
                self.monthly_forcings[:, sst_index, ...] = self.monthly_forcings[:, sst_index, ...] + self.dataset.forcings_mean[None, sst_index, None, None]
                print(f"Applying SST scenario p{self.sst_scenario}K to forcings ...", end='')
                self.monthly_forcings[:, :, sst_index, ...] += float(self.sst_scenario)
                # re-normalize
                self.monthly_forcings[:, sst_index, ...] = \
                    (self.monthly_forcings[:, sst_index, ...] - self.dataset.forcings_mean[None, sst_index, None, None]) 
                self.monthly_forcings[:, sst_index, ...] /= self.dataset.forcings_std[None, sst_index, None, None]
            self.monthly_forcings = self.monthly_forcings.cuda().unsqueeze(0)
            print('Done.')
        else:
            self.monthly_forcings = None
            with xr.open_dataset(self.dataset.files[0], **self.dataset.xr_options) as obs:
                self.land_sea_mask = obs['land_sea_mask'].to_numpy()[0]  # (1, lat, lon)
                self.land_sea_mask = torch.tensor(self.land_sea_mask, dtype=torch.float32).cuda()
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
        
        if self.sst_scenario in ['2', '4']:
            sst_string = f"p{self.sst_scenario}"
        elif self.sst_scenario == '0':
            sst_string = ''

        if self.storage_type == "monthly":
            return f"A{sst_string}mon_{self.aimip_name}_aimip_r{member}i{self.init_method}p{self.physics}f{self.forcings}_{self.grid_type}_{year}_{month}.nc"
        elif self.storage_type == "daily":
            return f"day{sst_string}_{self.aimip_name}_aimip_r{member}i{self.init_method}p{self.physics}f{self.forcings}_{self.grid_type}_{year}.nc"
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

        if self.sst_scenario in ['2', '4']:
            print(f"Applying SST scenario p{self.sst_scenario}K to state and prev state...", end='')
            sst_index = self.dataset.variables['surface'].index('sea_surface_temperature')
            sst_val = float(self.sst_scenario)
            print(sst_index, sst_val)
            batch = self.dataset.denormalize(batch)  # denormalize first
            batch['state']['surface'][sst_index, ...] += float(self.sst_scenario)
            batch['prev_state']['surface'][sst_index, ...] += float(self.sst_scenario)
            batch = self.dataset.normalize(batch)
            print('Done.')
            
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

            # add forcings to batch if module requires it
            if self.module.embedder.forcings_embedding is not None:
                forcings = self.get_forcings(batch['timestamp'])
                batch['forcings'] = forcings.squeeze(0)  # remove batch dimension
                batch['future_forcings'] = self.monthly_forcings  # placeholder, not used in current models

            print("Batch loaded from timestamp: ", timestamp)
            start_timestamp = timestamp

            print("Continuing rollout from timestamp: ", start_timestamp)
            self.store_initial_condition = False  # Do not store initial condition if continuing rollout
        else:
            if start_timestamp not in self.timestamps:
                raise ValueError(f"Start timestamp {start_timestamp} not found in dataset timestamps.")
            print("Preparing initial batch for timestamp: ", start_timestamp, end=' ... ')
            batch = self.timestamp_to_batch(start_timestamp)
            print("Done.")

        return batch, start_timestamp
    
    def get_forcings(self, timestamp):
        assert self.monthly_forcings is not None, "No forcings dataset found in the dataloader."
        
        if self.ablate_forcings:
            # return zeros of the same shape as forcings
            forcings = self.monthly_forcings[:, 0, ...].squeeze(1)
            return forcings
        else:
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
            # use pd.to_datetime to convert to datetime64[M] and datetime64[D]
            
            day = pd.to_datetime(start_timestamp.astype('datetime64[D]')).day
            month = pd.to_datetime(start_timestamp.astype('datetime64[M]')).month
            if day != 31 and month != 12:
                print("First year is partial.")
                last_day_of_first_year = (start_timestamp.astype('datetime64[Y]') + np.timedelta64(1, 'Y') - np.timedelta64(1, 'D')).astype('datetime64[D]')
                days_in_first_year = (last_day_of_first_year - start_timestamp.astype('datetime64[D]')).astype('timedelta64[D]').astype(int)
                print("Days in the first year: ", days_in_first_year)
                rollout_steps = [days_in_first_year]
                rollout_steps += [366 if isleap((start_timestamp.astype('datetime64[Y]') + np.timedelta64(i, 'Y')).astype(int) + 1970) 
                    else 365 for i in range(1, nyears)]
            else:
                print("First year is full.")
                rollout_steps = [366 if isleap((start_timestamp.astype('datetime64[Y]') + np.timedelta64(i, 'Y')).astype(int) + 1970) 
                    else 365 for i in range(nyears)
                ]


        print('#### Rollout Steps #### \n', rollout_steps)
        print("Computed rollout steps per year (in days): \n", rollout_steps)

        return rollout_steps
    
    def dump_to_netcdf(self, output, timestamp):
        # Concat list of xarrays along time dimension

        #if 'sea_surface_temperature' in self.dataset.variables['surface']:
        #    # Replace land grid points with nans where lsm == 1
        #    sst_index = self.dataset.variables['surface'].index('sea_surface_temperature')
        #    for i in range(len(output)):
        #        sst = output[i]['surface'][:, sst_index, ...]
        #        sst[:, :, self.land_sea_mask == 1] = torch.tensor(np.nan, dtype=sst.dtype).cuda()
        #        output[i]['surface'][:, sst_index, ...] = sst


        xarrays = [self.dataset.convert_to_xarray(
            self.dataset.denormalize(o), 
            timestamp + self.lead_time_hours * 3600 * i, 
            levels=arches_default_pressure_levels)
            for i, o in enumerate(output)
        ]

        xarrays = xr.concat(xarrays, dim=self.dataset.time_dim_name)

        # Concatente land sea mask to xarray from cached file in self.dataset
        #lsm_data = xr.DataArray(
        #    self.land_sea_mask.cpu().numpy(), 
        #   dims=(self.dataset.dimension_indexers['latitude'], 
        #         self.dataset.dimension_indexers['longitude']
        #          )
        #)
        
        #lsm_data = lsm_data.expand_dims({self.dataset.time_dim_name: xarrays[self.dataset.time_dim_name]})
        #lsm_data = lsm_data.rename('land_sea_mask')
        #xarrays = xr.merge([xarrays, lsm_data])
        
        # Rename time dimension to 'time' if it's not already named 'time'
        if self.dataset.time_dim_name != 'time':
            xarrays = xarrays.rename_dims({self.dataset.time_dim_name: 'time'})
            xarrays = xarrays.rename_vars({self.dataset.time_dim_name: 'time'})
            xarrays = xarrays.set_coords('time')

        # Rename level dimension to 'level' if it's not already named 'level'
        if self.dataset.level_dim_name != 'level' and self.dataset.level_dim_name in xarrays.dims:
            xarrays = xarrays.rename_dims({self.dataset.level_dim_name: 'level'})
            xarrays = xarrays.rename_vars({self.dataset.level_dim_name: 'level'})
            xarrays = xarrays.set_coords('level')

        # Generate file name
        file_name = self.get_file_name(timestamp)

        # Save the xarray to a netcdf file
        if self.storage_type == "monthly":
            xarrays = xarrays.mean(dim='time')
            xarrays['time'] = pd.to_datetime(timestamp.astype('datetime64[M]')).tz_localize(None)
            xarrays = xarrays.set_coords('time')
            xarrays.to_netcdf(f"{self.storage_path}/{file_name}")
        elif self.storage_type == "daily":
            xarrays.to_netcdf(f"{self.storage_path}/{file_name}")
        else:
            raise ValueError("Storage type must be either 'monthly' or 'daily'.")
        
    def rollout(self, start_timestamp, end_timestamp):
        batch, start_timestamp = self.get_initial_batch(start_timestamp)
        steps_per_year = self.compute_steps_per_years(start_timestamp, end_timestamp)

        return batch, steps_per_year