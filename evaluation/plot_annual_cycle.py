
import xarray as xr 
from glob import glob 
import matplotlib.pyplot as plt
import numpy as np
from plot.timeseries import plot_timeseries
from plot.timeseries import get_xlabel_multiplier
from metrics.utils import compute_annual_cycle

units = {
        'sea_surface_temperature': '[K]',
        '2m_temperature': '[K]',
        'mean_sea_level_pressure': '[Pa]',
        'specific_humidity': '[kg/kg]',
        'geopotential': '[m^2/s^2]',
        'u_component_of_wind': '[m/s]',
        'v_component_of_wind': '[m/s]',
        '10m_u_component_of_wind': '[m/s]',
        '10m_v_component_of_wind': '[m/s]',
        'sea_ice_cover': '[fraction]',
        'temperature': '[K]',
        'vertical_velocity': '[m/s]',

    }

# reference data path
file_name_filter = '12h'
ref_data_path = 'data/era5_1x1/full'
ref_files = glob(f'{ref_data_path}/*{file_name_filter}*.nc')
ref_ds = xr.open_mfdataset(ref_files)

ref_ds = compute_annual_cycle(ref_ds)


# model data path
model_data = [
    'evalstore/AW-M-1-aimip-w_forcings-interpolgt/1980-01-01T12:00/sst_0/daily/member_01/climeval/annual_cycle/data.nc'
]

model_data = [xr.open_dataset(f) for f in model_data]

model_labels = [
    'AWM1-igt-sst_sic-0k'
]

# create colormap as sequence of grays
cmap = plt.get_cmap("Greys")
colors = cmap(np.linspace(0.15, 0.95, len(model_data)))

for var in [
    'sea_surface_temperature',
    '2m_temperature',
    'sea_ice_cover'
    ]:


    fig = plt.figure(figsize=(10,6), dpi=150)
    ax = fig.add_subplot(1,1,1)
    for d, label, c in zip(model_data, model_labels, colors):
        if var in d.data_vars:
            ax.plot(d[var].mean(dim='time'), color=c, label=label)
        else:
            print(f"Variable {var} not in dataset {label}, skipping...")


    ax.set_title(f"Annual Cycle - {var}")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{units[var]}")
    ax.legend()
    plt.tight_layout()
    plt.show()  

