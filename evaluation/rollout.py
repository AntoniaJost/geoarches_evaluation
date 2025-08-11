'''from ducc0 import sht
import xarray as xr
import numpy as np
from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.lightning_modules import load_module
from geoarches.lightning_modules.diffusion import DiffusionModule
from typing import Union
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class SphericalHarmonics:
    def __init__(self, n_theta, n_phi, geometry='CC', spin=0, l_trunc=None):
        self.lat_spacing = n_theta
        self.lon_spacing = n_phi
        self.phi = np.linspace(0, 360, self.lon_spacing)
        self.theta = np.linspace(-180, 180, self.lat_spacing)

        self.spin = spin 
        self.geometry = geometry
        self.lmax = sht.maximum_safe_l(self.geometry, self.lat_spacing)
        self.l_trunc = l_trunc
    
    def analysis(self, field_arr):
        assert field_arr.ndim == 3, f'Expected dimension of field array is 3, but given array has {field_arr.ndim}'
        
        return sht.analysis_2d(map=field_arr, spin=self.spin, lmax=self.lmax, geometry=self.geometry)
    
    def synthesis(self, coeff_arr):
        assert coeff_arr.ndim == 2, f'Expected dimension of coeff array is 2, but given array has {coeff_arr.ndim}'
        
        return sht.synthesis_2d(alm=coeff_arr, spin=self.spin, lmax=self.lmax, geometry=self.geometry, ntheta=self.lat_spacing, nphi=self.lon_spacing)

    def coeffs_to_ltm(self, coeff_arr):
        assert coeff_arr.ndim == 2, f'Expected dimension of coeff array is 2, but given array has {coeff_arr.ndim}'

        coeff_mat = np.zeros((self.lmax, self.lmax), dtype=coeff_arr.dtype)
        for k in range(0, self.lmax):
            if k == 0:
                start = 0
                end = self.lmax 
            else:
                start = end
                end = start + (self.lmax - k)
                
            coeff_mat[k:, k] = coeff_arr[0, start:end]
        
        return coeff_mat
    
    def truncated_reconstruction(self, l_trunc=None, field_arr=None, coeffs=None):
        assert map is not None or coeffs is not None, 'Either a field array with values or complex coefficients have to be given'

        if l_trunc is None and self.l_trunc is None: 
            l_trunc = self.lmax // 2
        elif self.l_trunc is not None:
            l_trunc = self.l_trunc

        if coeffs is None:
            coeffs = self.analysis(field_arr)

            #coeffs_mat = self.coeffs_to_ltm(coeffs)
            #print(coeffs_mat)

        if coeffs.shape[-2] == self.lmax:
            # LT Matrix form
            coeffs[..., l_trunc:] = 0. + 0.j

        else:
            #midx = 0
            
            #for k in range(m_lim):
            #    midx += (self.lmax - k)

            #    coeffs[:, midx:] = 0. + 0.j

            start = 0
            end = 0
            for k in range(self.lmax):
                if (self.lmax - k) < l_trunc:
                    end += self.lmax - k

                    coeffs[end:] = 0. + 0.j
                    break
                end += self.lmax - k
                start = end - (self.lmax - l_trunc)
                coeffs[:, start:end] = 0. + 0.j

        return self.synthesis(coeffs)


    def plot_truncated_error(self, trunc_limits, trunc_errors=None, field_arr=None, title='', fpath=None):
        assert trunc_errors is None or field_arr is None

        if field_arr is not None:
            trunc_recs = [self.truncated_reconstruction(m_lim=m_lim, field_arr=field_arr) for m_lim in trunc_limits]

            trunc_errors = [np.square(np.linalg.norm(rec - field_arr)) / np.square(np.linalg.norm(field_arr)) for rec in trunc_recs]
        
        xticks = [self.lmax - m_lim for m_lim in trunc_limits]

        plt.figure(dpi=150)
        plt.title(title)
        plt.grid()
        plt.plot(trunc_errors)
        plt.xticks(list(range(len(trunc_errors))), labels=xticks)
        plt.xlabel('No. Truncated Harmonics')
        plt.yscale('log')
        plt.savefig(fpath)
        plt.close()
    
    def compute_radial_spectra(self, coeffs):
        """
        Computes the radial spectra from the spherical harmonic coefficients.
        The coefficients should be in the form of a 2D array with shape (n_coeffs, lmax).
        """
        if coeffs.shape[-2] != coeffs.shape[-1]:
            coeffs = self.coeffs_to_ltm(coeffs)

        # Compute the radial spectra by summing over the coefficients
        radial_spectra = np.sum(np.abs(coeffs) + 1.0e-7, axis=0)
        
        return radial_spectra

    def plot_radial_spectra(self, radial_spectra, title, labels, fpath, cmap, **kwargs):

        
        plt.figure()
        plt.grid()
        plt.title(title)
        for r, c, l in zip(radial_spectra, cmap, labels):         
            plt.plot(r, label=l, c=c, **kwargs)
        
        plt.yscale('log')
        plt.legend()    
        plt.savefig(fpath)
        plt.close()


    def plot_coeff_matrix(self, coeffs: np.array, fpath: str):
        if coeffs.shape[-2] != coeffs.shape[-1]:
            coeffs = self.coeffs_to_ltm(coeffs)

        plt.figure()
        plt.imshow(coeffs)
        plt.savefig(fpath)
        plt.show()


class Era5SHT(SphericalHarmonics):
    def __init__(self, n_theta, n_phi, geometry='CC', spin=0, l_trunc=None, variables=None, levels=None):
        super().__init__(n_theta=n_theta, n_phi=n_phi, geometry=geometry, spin=spin, l_trunc=l_trunc)
        self.lmax = sht.maximum_safe_l(ntheta=self.lat_spacing, geometry=self.geometry)

    def analyze_variables(self, signals: Union[torch.Tensor, xr.Dataset, np.array]) -> np.ndarray:
        pass

        
"""
ds = xr.load_dataset("data/era5_240/full/era5_240_2000_0h.nc")
signal = ds.isel(time=0)['2m_temperature'].to_numpy().T[None]
geometry = 'CC'
ntheta = 121
nphi = 240
l = sht.maximum_safe_l(ntheta=ntheta, geometry=geometry)
print("signal.shape:", signal.shape)
coeffs = sht.analysis_2d(map=signal, spin=0, lmax=l, geometry=geometry)
rec_signal = sht.synthesis_2d(alm=coeffs, spin=0, ntheta=ntheta, nphi=nphi, lmax=l, geometry=geometry)
print(rec_signal.shape)
print("Relative RMSE:", np.sqrt(np.mean((signal - rec_signal) ** 2)) / np.sqrt(np.mean(signal ** 2)))
print("Relative MSE:", np.mean((signal - rec_signal) ** 2) / np.mean(signal ** 2))
print(f"Relative Max Norm: {np.max(np.abs(signal - rec_signal)) / np.max(np.abs(signal)):.7f}")
print(f"Absolute error: {np.mean(np.abs(signal - rec_signal)):.10f}")
"""

# Dataset 
era5_ds = Era5Forecast(path="data/era5_240/full", domain="test") 
# run model on a sample
seed = 0
num_steps = 25  # if not provided to model.sample, model will use the default value (25)
scale_input_noise = 1.05

# loading ArchesWeatherFlow
device = "cuda:0"

# load_module will look in modelstore/
gen_model, gen_config = load_module("archesweathergen")

gen_model = gen_model.to(device)

batch = era5_ds[0]

batch = {k: v[None].to(device) for k, v in batch.items()}

"""final_sample, trajectory = gen_model.sample(
    batch, seed=seed, num_steps=num_steps, scale_input_noise=scale_input_noise, return_trajectory=True
)"""


def timestring_to_npdatetime(timestring):
    datetime = np.datetime64(timestring)

    return datetime.astype('datetime64[ns]')

class ArchesGenRollout:
    def __init__(self, model, device, dataset: Era5Forecast):
        self.dataset = dataset
        self.model = model
        self.device = device

    def get_init_sample(self, timestring):
        timestamps = [t[-1] for t in self.dataset.timestamps]
        datetime = timestring_to_npdatetime(timestring)
        idx = timestamps.index(datetime)
        idx -= self.dataset.lead_time_hours // self.dataset.timedelta

        return self.dataset[idx]



trajectory = gen_model.sample_rollout(
    batch, iterations=100
)

print(trajectory['surface'][0, :, 2, 0].shape)

"fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(25, 15))"       
spectra = []
geometry = 'CC'
ntheta = 121
nphi = 240
sph_op = SphericalHarmonics(n_theta=ntheta, n_phi=nphi, geometry=geometry, spin=0, l_trunc=None)
spectra = []
stride = 10
for i in range(stride):
    for j in range(stride):

        signal = trajectory['surface'][0, i * stride + j, 2].cpu().numpy()
        if (i * stride + j) % stride == 0:
            c = sph_op.analysis(signal)
            spectra.append(sph_op.compute_radial_spectra(c))
        """        
        axs[i, j].imshow(signal[0], cmap='coolwarm')
        axs[i, j].axis('off')
        """
    
"""plt.tight_layout()
plt.savefig("trajectory_surface_t2m.png", dpi=300)"""
cmap = plt.get_cmap('Blues')
cmap = cmap(np.linspace(0.1, 1., len(spectra)))

sph_op.plot_radial_spectra(
    radial_spectra=spectra,
    title='Radial Spectra of Surface 2m Temperature',
    labels=[f'Sample Step {(i * stride)}' for i in range(len(spectra))],
    fpath='radial_spectra_surface_t2m.png',
    cmap=cmap
)'''

import os

import numpy as np
import pandas as pd
import xarray as xr

import hydra
from omegaconf import OmegaConf

import torch
from tensordict import TensorDict


from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.dataloaders.era5 import arches_default_pressure_levels
from geoarches.lightning_modules import load_module
from geoarches.lightning_modules.diffusion import DiffusionModule
from geoarches.dataloaders.era5 import arches_default_pressure_levels


class AAIMIPRollout:
    def __init__(
            self, 
            module, 
            dataset, 
            model_name, 
            storage_path, 
            storage="monthly", 
            member=1, 
            grid_type='gn', 
            init_method=1, 
            physics=1, 
            forcings=1,
            sst_scenario=''
        ):

        """
        Initialize the AAIMIPRollout class.
        Args:
            module: The model module to be used for the rollout.
            dataset: The dataset from which to draw data for the rollout.
            model_name: Name of the model.
            storage_path: Path where the output files will be stored.
            storage: Type of storage, either 'monthly' or 'daily'.
            member: Member number for the model.
            grid_type: Type of grid used in the model.
            init_method: Initialization method used in the model.
            physics: Physics configuration used in the model.
            forcings: Forcings configuration used in the model.
        """
        super().__init__()
        
        self.model_name = model_name
        self.module = module.cuda()
        self.module.eval()  # Set the module to evaluation mode
        self.dataset: Era5Forecast = dataset
        self.timestamps = [t[-1] for t in self.dataset.timestamps]
        self.lead_time_hours = self.dataset.lead_time_hours

        # Storage configuration
        if storage not in ["monthly", "daily"]:
            raise ValueError("Storage type must be either 'monthly' or 'daily'.")
        self.storage = storage
        self.storage_path = storage_path

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

    def update_batch(self, loop_batch, output):
        # Update the batch with the model's output
        loop_batch['timestamp'] = loop_batch['timestamp'] + self.lead_time_hours * 3600
        loop_batch['prev_state'] = loop_batch['state']
        
        # Replace values in output if they are NaN in "next_state"
        # The value for NaN replacement is taken from the previous state according 
        # to the NaN values in "next_state"
        next_state = loop_batch['next_state']
        for key in output.keys():
            if torch.isnan(next_state[key]).any():
                output[key][torch.isnan(next_state[key])] = loop_batch['prev_state'][key][torch.isnan(next_state[key])]
        
        loop_batch['state'] = output

        return loop_batch
    
    def get_file_name(self, timestamp):
        """
        Generate a file name based on the timestamp.
        """
        month = timestamp.astype('datetime64[M]').astype(int) % 12 + 1
        year = timestamp.astype('datetime64[Y]').astype(int) + 1970
        
        if self.storage == "monthly":
            return f"Amon_{self.model_name}_aimip_r{self.member}i{self.init_method}p{self.physics}f{self.forcings}_{self.grid_type}_{year}_{month}.nc"
        elif self.storage == "daily":
            return f"day_{self.model_name}_aimip_r{self.member}i{self.init_method}p{self.physics}f{self.forcings}_{self.grid_type}_{year}.nc"
        else:
            raise ValueError("Storage type must be either 'monthly' or 'daily'.")
    
    def deterministic_rollout(self, start_timestamp, end_timestamp, rollout_steps, batch, store_init):
        # Get loop batch
        loop_batch = batch.copy()
        loop_batch = TensorDict({k: v.unsqueeze(0) for k, v in loop_batch.items()}, batch_size=[1])
        loop_batch = loop_batch.to('cuda')

        # Initialize the list to store xarrays
        xarrays = []

        # If store_init is True, convert the initial state to xarray and append it
        if store_init:
            xarrays.append(
                self.dataset.convert_to_xarray(
                    self.dataset.denormalize(loop_batch['state']), 
                    loop_batch['timestamp'], 
                    levels=arches_default_pressure_levels
                )
            )
            
        current_timestamp = start_timestamp
        for _ in range(rollout_steps):
            print(pd.to_datetime(loop_batch['timestamp'].detach().cpu(), unit='s').tz_localize(None))
            next_timestamp = current_timestamp + np.timedelta64(self.lead_time_hours, 'h')
            if next_timestamp > end_timestamp:
                print("Reached end timestamp during rollout, stopping.")
                break
            
            print("Next month: ", next_timestamp.astype('datetime64[M]'), "Current month: ", current_timestamp.astype('datetime64[M]'))
            print("Next year: ", next_timestamp.astype('datetime64[Y]'), "Current year: ", current_timestamp.astype('datetime64[Y]'))
            if next_timestamp.astype('datetime64[Y]') != current_timestamp.astype('datetime64[Y]'):
                xarrays = xr.concat(xarrays, dim='time')

                file_name = self.get_file_name(current_timestamp)
                if self.storage == "monthly":
                    xarrays = xarrays.mean(dim='time')
                    xarrays['time'] = pd.to_datetime(current_timestamp.astype('datetime64[M]')).tz_localize(None)
                    xarrays = xarrays.set_coords('time')
                    xarrays.to_netcdf(f"{self.storage_path}/{file_name}")
                elif self.storage == "daily":
                    xarrays.to_netcdf(f"{self.storage_path}/{file_name}")
                xarrays = []
            
            # Forward pass through the model
            with torch.no_grad():
                output = self.module.forward(loop_batch)
            
            xarray = self.dataset.convert_to_xarray(
                self.dataset.denormalize(output), 
                loop_batch['timestamp'].detach().cpu() + self.lead_time_hours * 3600, 
                levels=arches_default_pressure_levels)
            
            xarrays.append(xarray)

            # Update the batch with the model's output
            loop_batch = self.update_batch(loop_batch, output)
            current_timestamp = next_timestamp

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
        print("Number of steps for rollout:", nsteps, " yielding a total of ", (nsteps * self.lead_time_hours) // 24, "days.")
        
        return nsteps
    
    def timestamp_to_batch(self, timestamp):
        idx = self.timestamps.index(timestamp)
        idx -= self.dataset.lead_time_hours // self.dataset.timedelta # account for prev_timestamp
        if idx < 0:
            raise ValueError("Timestamp is too early for the dataset.")
        elif idx >= len(self.timestamps):
            raise ValueError("Timestamp is too late for the dataset.")
        
        batch = self.dataset[idx]

        return batch

    def rollout(self, start: np.datetime64, end: np.datetime64, store_initial_condition: bool = True):
        nsteps = self.compute_number_of_rollout_steps(start, end)
        print("Starting rollout from", start, "to", end, "with", nsteps, "steps.")

        if 'archesgen' in self.model_name:
            raise NotImplementedError("Rollout for ArchesGen models is not implemented yet.")
        else:
            # For ArchesWeather models, we need to get the initial batch based on the start timestamp
            if start not in self.timestamps:
                raise ValueError(f"Start timestamp {start} not found in dataset timestamps.")
            
            batch = self.timestamp_to_batch(start)
            print("Initial batch prepared for timestamp: ", start)

            # Perform the deterministic rollout
            self.deterministic_rollout(start, end, nsteps, batch, store_initial_condition)

        

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

    # Run the rollout detection
    print("Loading dataset for rollout ...", end=' ')
    loaded_config.dataloader.dataset.domain = "all"
    dataset = hydra.utils.instantiate(loaded_config.dataloader.dataset, loaded_config.stats)
    print("Done.")

    print("Creating storage path if it does not exist ...", end=' ')
    os.makedirs(cfg.storage.path, exist_ok=True)
    print("Done.")

    # Create the rollout module
    print("Creating rollout module ...", end=' ')
    rollout_module = AAIMIPRollout(
        model_name=cfg.aimip.name,
        module=module,
        dataset=dataset,
        storage_path=cfg.storage.path,
        storage=cfg.storage.type,
        member=cfg.aimip.member,
        grid_type=cfg.aimip.grid_type,
        init_method=cfg.aimip.init_method,
        physics=cfg.aimip.physics,
        forcings=cfg.aimip.forcings,
        sst_scenario=cfg.aimip.sst_scenario
    )
    print("Done.")
    
    start_timestamp = np.datetime64(cfg.start_timestamp).astype('datetime64[ns]')
    end_timestamp = np.datetime64(cfg.end_timestamp).astype('datetime64[ns]')

    print("Start timestamp: ", start_timestamp)
    print("End timestamp: ", end_timestamp)

    rollout_module.rollout(
        start=start_timestamp,
        end=end_timestamp,
        store_initial_condition=cfg.storage.store_initial_condition
    )


if __name__ == "__main__":
    main()