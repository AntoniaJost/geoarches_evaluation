import os
import glob
import time
import warnings

from itertools import product

from matplotlib import colors
import numpy as np
import xarray as xr


import hydra
from omegaconf import ListConfig, OmegaConf

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


from scipy.stats import linregress



from geoarches.dataloaders.era5 import surface_variables_short, level_variables_short
from geoarches.metrics.metric_base import compute_lat_weights, compute_lat_weights_weatherbench

from metrics.utils import compute_anomaly, compute_mean, detrend_data
from metrics.frequency_domain import compute_radial_spectrum, welch_psd
from metrics import timeseries as eval_timeseries

#from metrics.kernel_density_estimation import kde_variability_plot
from plot.timeseries import plot_timeseries, get_xlabel_multiplier
from plot.spatial import plot_variable, plot_temperature_with_geopotential_contours, plot_anomalies
from plot.spectra import plot_radial_spectrum

#climate_stats_path = importlib.resources.files(climate_stats)


def plot_all_variables(fnc, data, variables, xticks, output_path, reference=None, std=None):
    for var, lvl in variables:
        print(f"Processing {var} at level {lvl}...")
        x = data.sel(level=lvl)[var].to_numpy()
        x_std = std.sel(level=lvl)[var].to_numpy() if std is not None else None
        x_ref = reference.sel(level=lvl)[var].to_numpy() if reference is not None else None
        var_name = surface_variables_short[var] if var in surface_variables_short.keys() else level_variables_short[var] + str(lvl)
        
        plot_timeseries(
            time=time, 
            xticks=xticks,
            data=x,                 
            ref=x_ref,
            std=x_std,
            output_path=output_path, 
            variable_name=var_name, 
        )

def evaluate_ensemble(fnc):
    def eval_ensemble(*args, **kwargs):
        data = kwargs.pop("data", None)  # Remove 'data' from kwargs if it exists

        # Compute mean and std and pass it to the function
        if 'member' in data.dims:
            # Also fnc result to ensemble average
            print('Eval ensemble mean ...', end=' ')
            mean = data.mean(dim='member')
            std = data.std(dim='member')
            fnc(args[0], data=mean, std=std, **kwargs)
            print("Done.")
        else:
            fnc(args[0], data=data, **kwargs)

    return eval_ensemble


class ModelContainer:
    def __init__(self, path, label, dimension_indexers=None):
        self.path = path
        self.label = label
        self.dimension_indexers = dimension_indexers

    def _load_data(self):
        fpaths = glob.glob(self.path + "/*.nc")
        fpaths.sort()
        print("Opening data from:", self.path, " ...", end=" ")
        data = xr.open_mfdataset(fpaths, combine="by_coords")
        print('Done')

        if self.data.latitude[0] < self.data.latitude[-1]:  # if latitude is descending
            data['latitude'] = data.latitude[::-1]
        data = data.roll(longitude=-len(data.longitude) // 2, roll_coords=False)  # Roll longitude to match data

        if self.dimension_indexers is not None:
            data = data.rename(**self.dimension_indexers)

        self.data = data


class TimeSeries:
    def __init__(self, figsize=(10, 6), dpi=150):
        self.figsize = figsize
        self.dpi = dpi

    def visualize_on_ax(self, ax, data):
        pass 

    def create_figure(self):
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        return fig, ax
    
    def extract_min_max_time(self, data):
        time_min = data.time.min().values
        time_max = data.time.max().values
        return time_min, time_max
    
    def get_time_limits(self, data):
        min_times = []
        max_times = []
        for model_label, annual_cycle in data.items():
            time_min, time_max = self.extract_min_max_time(annual_cycle)
            min_times.append(time_min)
            max_times.append(time_max)
        min_time = min(min_times)
        max_time = max(max_times)

        return min_time, max_time

    def create_time_labels(self, min_time, max_time):
        time_range = np.arange(np.datetime64(min_time, 'M'), np.datetime64(max_time, 'M') + np.timedelta64(1, 'M'), np.timedelta64(1, 'M'))
        xtick_labels = [str(t)[:7] for t in time_range]
        xtick_ids = list(range(len(xtick_labels)))

        return xtick_labels, xtick_ids

    
class AnnualCycle(TimeSeries):
    def __init__(self, figsize=(10, 6), dpi=150, linewidth=2):
        self.figsize = figsize
        self.dpi = dpi
        self.linewidth = linewidth

    def visualize_to_ax(self, data, output_path):
        pass

    def compute_annual_cycle(self, data):
        eval_timeseries.compute_annual_cycle(data)

    def compute(self, model_containers):
        annual_cycles = {}
        for model_data in model_containers:
            annual_cycle = self.compute_annual_cycle(model_data.data)
            annual_cycles[model_data.label] = annual_cycle

        return annual_cycles


    
    def evaluate(self, model_containers):
        annual_cycles = self.compute(model_containers)

        min_times, max_times = [], []



        xtick_labels = []
        xtick_ids = []
        


class GeoClimate:
    def __init__(self, models):

        self.model_containers = []
        for model in models:
            container = ModelContainer(
                path=model['path'],
                label=model['label'],
                dimension_indexers=model.get('dimension_indexers')
            )
            container._load_data()
            self.model_containers.append(container)

    def evaluate(self):
        pass 
    
class ClimateEvaluator:
    """
    A class to evaluate climate data.
    """
    R = 6.371e6  # Radius of the Earth in meters
    g = 9.81     # Acceleration due to gravity in m/s^2
    pi = 3.14159
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



    short_names = {**surface_variables_short, **level_variables_short}

    def __init__(
            self, path, variables, levels, 
            reference_path=None, base_period=(1995, 2014), 
            output_path=None, spinup_years=0, 
            start_stamp=None, end_stamp=None, 
            target_dimension_indexers=None,
            reference_dimension_names=None,
            model_label='Model',
            use_lat_weighting='weatherbench'
        ):

        """
        Initialize the ClimateEvaluator with data.
        
        :param fpaths: List of netcdf or zarr to load.
        :type fpaths: list
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If the file is not a valid NetCDF or Zarr dataset.
        :raises RuntimeError: If the dataset cannot be opened.
        :raises Exception: For any other errors encountered.
        :return: None

        The loaded data is stored in the `data` attribute.
        and expected to have dimension time, level, latitude, longitude.

        """

        self.model_label = model_label
        if isinstance(path, ListConfig):
            path = list(path)
            
            # list of members
            print("Opening prediction data from multiple members:", path, " ...", end=" ")
            datasets = [xr.open_mfdataset(sorted(glob.glob(p + "/*.nc")), combine="by_coords") for p in path]
            self.data = xr.concat(datasets, dim='member')
            self.data["member"] = np.arange(0, len(path))  # Add member dimension
            #self.data = self.data.set_coords("member")
            print("Done.")
        else:
            fpaths = glob.glob(path + "/*.nc")
            fpaths.sort()
            print("Opening prediction data from:", path, " ...", end=" ")
            self.data = xr.open_mfdataset(fpaths, combine="by_coords")
            print('Done')

        print(self.data.time.values)
        if self.data.latitude[0] < self.data.latitude[-1]:  # if latitude is descending
            self.data['latitude'] = self.data.latitude[::-1]
        self.data = self.data.roll(longitude=-len(self.data.longitude) // 2, roll_coords=False)  # Roll longitude to match data


        # Get lat weighting function and make them a data array with latitudes
        if use_lat_weighting == 'weatherbench':
            self.lat_weights = compute_lat_weights_weatherbench(len(self.data.latitude))
            self.lat_weights = self.lat_weights.squeeze(-1)
            self.lat_weights = xr.DataArray(self.lat_weights, coords=[self.data.latitude], dims=["latitude"])
        elif use_lat_weighting == 'standard':
            self.lat_weights = compute_lat_weights(len(self.data.latitude))
            self.lat_weights = self.lat_weights.squeeze(-1)
            self.lat_weights = xr.DataArray(self.lat_weights, coords=[self.data.latitude], dims=["latitude"])
        else:
            self.lat_weights = None

        

        # Open reference data
        if reference_path is not None:
            reference_files = glob.glob(reference_path + "/*.nc")
            reference_files.sort()

            # extract hour from self.data time dimension and only keep files that match the hour
            hour = int(self.data.time.dt.hour[0])
            reference_files = [f for f in reference_files if f"{hour:02d}" in f]
            if not reference_files:
                raise FileNotFoundError(f"No reference files found for hour {hour} in {reference_path}")

            # preprocess files such that time dimension has datetime64[day] type
            # This avoids issues with non-monotonic time dimension being datetime64[ns]
            # in reference files
            def preprocess(ds, reference_dimension_names=reference_dimension_names):
                if reference_dimension_names is not None:
                    ds = ds.rename({v: k for k, v in reference_dimension_names.items()})
                ds['time'] = ds['time'].astype('datetime64[D]')
                if ds.latitude[0] > ds.latitude[-1]:  # if latitude is descending
                    ds = ds.reindex(latitude=list(reversed(ds.latitude)))
                return ds


            self.reference = xr.open_mfdataset(reference_files, combine='by_coords', preprocess=preprocess)
            self.reference = self.reference.transpose('time', 'level', 'latitude', 'longitude')
            self.reference = self.reference.reindex(latitude=list(reversed(self.reference.latitude)))
            # Select only variables that are in self.data

            # Roll data
            self.reference = self.reference.roll(longitude=-len(self.reference.longitude) // 2, roll_coords=False)  # Roll longitude to match data
            self.land_sea_mask = self.reference['land_sea_mask'].to_numpy()[0]
            self.reference = self.reference[list(self.data.data_vars.keys())]
            
        else:
            self.reference = None

        if start_stamp is not None or end_stamp is not None:
            self.data = self.data.sel(time=slice(start_stamp, end_stamp))

            if self.reference is not None:
                self.reference = self.reference.sel(time=slice(start_stamp, end_stamp))

        self.base_period = base_period
        self.output_path = output_path
        self.variables = [(var, slice(None)) for var in variables['surface']]
        self.variables.extend(product(variables['level'], levels)) 

    def get_variable_short_name(self, var, level=None):
        """ Get short name for variable """
        if var in surface_variables_short.keys():
            return surface_variables_short[var]
        elif var in level_variables_short.keys():
            return level_variables_short[var] + str(level)
        else:
            return var + (str(level) if level is not None else '')
        
    def select_reference_data(self, x, detrend=False):
            """ 
            Select time reference data for a given dataset x.

            """
            reference = self.reference
            reference = reference.sel(time=(reference.time >= x.time[0]) & (reference.time <= x.time[-1]))

            if detrend:
                reference = detrend_data(reference, self.base_period, ['sea_surface_temperature', '2m_temperature'])

            return reference

    def instant_spatial_map(self, data, timestamp):
        """year, month = str(timestamp).split('-')[:2]
        season = self.get_season_from_month(int(month))
        for var, lvl in self.variables:
            print(f"Processing {var} at level {lvl} ... ", end='')
            x = data.sel(level=lvl).sel(time=timestamp)[var].to_numpy()

            if var in ['sea_surface_temperature', '2m_temperature']:
                mask = self.land_sea_mask.astype(bool)
            else:
                mask = None

            reference = None  # do not plot reference for bias maps

            #if var in ['sea_surface_temperature', 'sea_ice_cover']:
                
            #    x = np.where(mask == 0, x, np.nan)

            var_name = self.get_variable_short_name(var, lvl)
            title = f"Bias Map {var_name} {year}" + (f" {season}" if season is not None else '')
            fname = f"bias_map_{var_name}_{year}" + (f"_{season}" if season is not None else '') + ".png"
            plot_variable(
                x=x,
                fname=fname,
                output_path=path,
                title=title,
                cbar_label=self.units[var],
                cmap='coolwarm',
                norm=colors.TwoSlopeNorm(vcenter=0),

            )"""
        pass

    def bias_map(self, data, year=2020, season=None, **kwargs):

        print("Compute bias map ... ", end='')

        assert self.reference is not None, "Reference data is required for bias map computation."
        reference = self.select_reference_data(data)

        if 'sea_surface_temperature' in reference.data_vars:
                mean = reference['sea_surface_temperature'].mean('longitude')
                reference['sea_surface_temperature'] = reference['sea_surface_temperature'].fillna(mean).ffill(dim='latitude')
        if 'sea_ice_cover' in reference.data_vars:
                reference['sea_ice_cover'] = reference['sea_ice_cover'].fillna(0)

        if season is not None:
            data = self.select_seasonal_data(year, season)
        else:
            data = data.sel(time=data.time.dt.year == year)
            reference = reference.sel(time=(reference.time >= data.time[0].astype('datetime64[D]')) & (reference.time <= data.time[-1].astype('datetime64[D]')))
            print(data)
            print(reference)
            data = data.mean('time')
            data['longitude'] = np.linspace(0, 360, len(data.longitude), endpoint=False)
            reference = reference.mean('time')
            bias = data - reference

        print('Done')

        # Specify output path
        path = f"{self.output_path}/climeval/bias_map"
        os.makedirs(path, exist_ok=True)

        print('Plot bias maps ... ', end='')
        for var, lvl in self.variables:
            print(f"Processing {var} at level {lvl} ... ", end='')
            if var in ['sea_surface_temperature', '2m_temperature']:
                mask = self.land_sea_mask.astype(bool)
            else:
                mask = None

            reference = None  # do not plot reference for bias maps

            x = bias.sel(level=lvl)[var]
            x = x.to_numpy()
            print(x.shape)
            #if var in ['sea_surface_temperature', 'sea_ice_cover']:
                
            #    x = np.where(mask == 0, x, np.nan)

            var_name = self.get_variable_short_name(var, lvl)
            title = f"Bias Map {var_name} {year}" + (f" {season}" if season is not None else '')
            fname = f"bias_map_{var_name}_{year}" + (f"_{season}" if season is not None else '') + ".png"
            plot_variable(
                x=x,
                fname=fname,
                output_path=path,
                title=title,
                cbar_label=self.units[var],
                cmap='coolwarm',
                norm=colors.TwoSlopeNorm(vcenter=0),

            )
            print('Done')

    def monthly_anomalies(self, data, detrend=False, year=2020):


        print("Calculate anomalies ... ", end='')

        anomaly = compute_anomaly(
            data,
            reduce_dims=['time'],
            groupby=['time.year', 'time.month'],
            mean_groupby=['time.month'],
            base_period=self.base_period,
            detrend=detrend
        )

        print('Done')

        
        # Get group from year arg 
        print(f"Select anomalies for year {year} and precompute ... ", end='')

        if year == 'all':
            anomaly = anomaly.mean("year")
        else:
            anomaly = anomaly.sel(year=year)
            anomaly = anomaly.drop_vars('year')


        anomaly = anomaly.compute()

        if self.reference is not None:
            mask = self.reference.isel(
                **{self.time_dim_name: 0}, drop=True)['sea_surface_temperature'].to_numpy()
            print(mask.shape)
            mask = np.isnan(mask)
        else: 
            mask = None

        # Reorder dimensions
        anomaly = anomaly.transpose('month', 'level', 'latitude', 'longitude')

        # Store anomalies on disk
        path = f"{self.output_path}/climeval/monthly_anomalies"
        os.makedirs(path, exist_ok=True)

        #anomaly.to_netcdf(f"{path}/monthly_anomalies_base_{self.base_period[0]}_{self.base_period[1]}.nc")
        print('Plot anomalies ... ', end='')
        plot_anomalies(anomaly, variables=self.variables, levels=self.levels, output_path=path, anomaly_type='Monthly', mask=mask)
        print('Done')

    @evaluate_ensemble
    def mean_climate(self, data, against_reference=False, **kwargs):
        data = detrend_data(
            data=data, 
            base_period=self.base_period, 
            variables_to_detrend=['2m_temperature', 'sea_surface_temperature']
        )

        mean_pred = compute_mean(data, reduce_dims=['time', 'latitude', 'longitude'], base_period=None, groupby=['time.dayofyear'])
        mean_pred = mean_pred.compute()
        if against_reference and self.reference is not None:
            reference = self.reference.sel(time=(self.reference.time >= data.time[0]) & (self.reference.time <= data.time[-1]))
            reference = detrend_data(
                data=reference, 
                base_period=self.base_period,
                variables_to_detrend=['2m_temperature', 'sea_surface_temperature']
            )
            reference = compute_mean(reference, reduce_dims=['time', 'latitude', 'longitude'], groupby=['time.dayofyear'])
            reference = reference.compute()
        elif against_reference and self.reference is None:
            warnings.warn("Reference data not provided or not available. Skipping reference comparison.")
            reference = None
        else:
            reference = None


        # Make time labels like month-dd and assume 365 days
        months = list(range(1, 13))
        days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        labels = [f"{m:02d}-01" for m in months]

        if "std" in kwargs.keys():
            std = kwargs["std"]
            std = compute_mean(std, reduce_dims=['time', 'latitude', 'longitude'], groupby=['time.dayofyear'])
            std = std.compute()

            std = None

        output_path = os.path.join(self.output_path, "climeval", "monthly_mean", "plots")
        os.makedirs(output_path, exist_ok=True)

        for v, lvl in self.variables:
            plot_timeseries(
                x=mean_pred.sel(level=lvl)[v], 
                ref=reference.sel(level=lvl)[v] if reference is not None else None, 
                xticks=[days, labels],
                title="Monthly Mean " + level_variables_short[v] + str(lvl),
                xlabel='Time',
                ylabel=self.units[v],
                output_path=os.path.join(output_path, f"monthly_mean_{level_variables_short[v]}_{lvl}.png"),
                std=std.sel(level=lvl)[v] if std is not None else None
            )

    
    def _annual_cycle(self, data, detrend):

        print('Compute annual cycle ... ', end='')
        annual_cycle = eval_timeseries.compute_annual_cycle(data, detrend, self.base_period)
        print('Done')

        return annual_cycle


    @evaluate_ensemble
    def annual_cycle(self, data, detrend=False, against_reference=False, **kwargs):
        
        # Specify output path
        output_path = f"{self.output_path}/climeval/annual_cycle"
        os.makedirs(output_path, exist_ok=True)

        print('Compute annual cycle ... ', end='')
        time = data.time
        time = time.dt.strftime('%Y-%m').values
        unique_time = sorted(list(set(time)))
        if against_reference and self.reference is not None:
            if 'sea_surface_temperature' in data.data_vars:
                data['sea_surface_temperature'] = xr.where(self.land_sea_mask == 0, data['sea_surface_temperature'], np.nan)
            
            if 'sea_ice_cover' in data.data_vars:
                data['sea_ice_cover'] = xr.where(self.land_sea_mask == 0, data['sea_ice_cover'], 0)

        if self.lat_weights is not None:
            data = data.weighted(self.lat_weights).mean(dim='latitude')

        annual_cycle = eval_timeseries.compute_annual_cycle(data, detrend, self.base_period)

            
        #annual_cycle.to_netcdf(f"{output_path}/data.nc")

        if 'std' in kwargs.keys():  # if ensemble
                print('Standard deviation of annual cycle ... ', end='')
                std_annual_cycle = eval_timeseries.compute_annual_cycle(kwargs['std'], detrend)
                std_annual_cycle.to_netcdf(f"{output_path}/std_annual_cycle.nc")
                print('Done')

        
        print('Done')

        # reference has to start at the same time as self.data
        if against_reference:
            print('Compute reference annual cycle ... ', end='')
            reference = self.select_reference_data(data, detrend=False)
            if self.lat_weights is not None:
                reference = reference.weighted(self.lat_weights).mean(dim='latitude')

            ref_annual_cycle = self._annual_cycle(reference, detrend=detrend)
            #ref_annual_cycle.to_netcdf(f"{output_path}/reference_annual_cycle.nc")
            print('Done')

        if 'std' in kwargs.keys():  # if ensemble
            print('Standard deviation of annual cycle ... ', end='')
            std_annual_cycle = self._annual_cycle(kwargs['std'], detrend)
            #std_annual_cycle.to_netcdf(f"{output_path}/std_annual_cycle.nc")
            print('Done')
        else:
            std = None

        for v, lvl in self.variables:
            var_name = surface_variables_short[v] if v in surface_variables_short.keys() else level_variables_short[v] + str(lvl)
            xtick_labels = [t for t in unique_time]
            xtick_ids = list(range(len(xtick_labels)))
            mult = get_xlabel_multiplier(len(xtick_ids))
            xticks = [xtick_ids[::mult], xtick_labels[::mult]]
            plot_timeseries(
                x=annual_cycle.sel(level=lvl)[v],
                ref=ref_annual_cycle.sel(level=lvl)[v] if against_reference else None,
                xticks=xticks,
                title='Annual Cycle ' + var_name,
                xlabel='Time',
                ylabel=self.units[v],
                output_path=os.path.join(output_path, f'annual_cycle_{var_name}.png'),
                std=std_annual_cycle.sel(level=lvl)[v] if std is not None else None,
                label=self.model_label,
                ref_label='ERA5' if against_reference else None,
            )
        return

    def select_seasonal_data(self, year, season):
        """
        Select seasonal data for a given year and season.
        
        :param year: Year to filter the data.
        :param season: Season to filter the data (e.g., 'DJF', 'MAM', 'JJA', 'SON').
        :return: Seasonal data as an xarray Dataset.
        """
        seasons = {
            'DJF': slice(f"{year}-12-01", f"{year+1}-02-28"),
            'MAM': slice(f"{year}-03-01", f"{year}-05-31"),
            'JJA': slice(f"{year}-06-01", f"{year}-08-31"),
            'SON': slice(f"{year}-09-01", f"{year}-11-30")
        }
        
        if season not in seasons:
            raise ValueError("Invalid season. Choose from 'DJF', 'MAM', 'JJA', or 'SON'.")
        
        return self.data.sel(time=seasons[season])

    def compute_mass_flux(self, data):

        """     
        psi =  (2 * R * pi * cos(lat) / g) * integral_0^p v(lat, p') dp'
        where R is the radius of the Earth, g is the acceleration due to gravity,
        and v(lat, p') is the meridional wind component at latitude lat and pressure
        level p'.
        Compute the mass flux for the Hadley Cell circulation.
        :param data: xarray Dataset containing climate data.
        :return: xarray Dataset with mass flux computed.
        """

        # Lat component 
        lat = np.cos(data.latitude)

        # compute zonal-mean meridional wind component
        data = data.mean(dim='longitude')

        # Constants
        const = 2. * self.R * self.pi * lat/ self.g

        # Compute the integral over pressure levels
        integral = data.v.integrate(coords='level')
        mass_flux = const * integral

        return mass_flux

    def hadley_cell_circulation(self, year, season):
        """
        Calculate the Hadley Cell circulation for a given year and season.

        :param year: Year to filter the data.
        :param season: Season to filter the data (e.g., 'DJF', 'MAM', 'JJA', 'SON').
        :return: Hadley Cell circulation data as an xarray Dataset.
        """
        seasonal_data = self.select_seasonal_data(year, season)
        
        self.compute_mass_flux(seasonal_data)

    def compute_webster_yang_index(self, year=None, month=None):
        """
        Compute the Webster-Yang index for a given year and month.
        If year and month are provided, filter the data accordingly.
        If not provided, use the entire dataset.

        Index is calculated as:
        index = (mean(u850(lat=[0°, 20°N], lon=[40°, 110°E]) - mean(u200(lat=[0°, 20°N], lon=[40°, 110°E]))

        :param year: Year to filter the data (optional).
        :param month: Month to filter the data (optional).
        :return: Webster-Yang index.
        """
        data = self.data.sel(time=slice(f"{year}-{month}-01", f"{year}-{month}-31")) if year and month else self.data
        
        # Webster-Yang index
        u850 = data.u850.sel(latitude=slice(0, 20), longitude=slice(40, 110)).mean(dim=['latitude', 'longitude'])
        u200 = data.u200.sel(latitude=slice(0, 20), longitude=slice(40, 110)).mean(dim=['latitude', 'longitude'])
        
        index = u850 - u200

        return index
    
    def southern_oscillation_index(self, data, plot=True):
        """
        Compute the Southern Oscillation Index (SOI) for a given year and month.
        If year and month are provided, filter the data accordingly.
        If not provided, use the entire dataset.

        :param year: Year to filter the data (optional).
        :param month: Month to filter the data (optional).
        :return: SOI index.
        """

        
        # Extract time dimension and ensure it is in datetime format month
        time = data['time'].to_numpy()

        # Get the ERA5 climatology data for tahiti and darwin
        soi = eval_timeseries.southern_oscillation_index(
            data['mean_surface_level_pressure'], 
            base_period=self.base_period, 
            detrend=True
        )

        # Get tahiti and darwin mean sea level pressure  
        if plot:
            output_path = f"{self.output_path}/climeval/soi/plots"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plot_timeseries(
                soi, 
                xticks=time.astype('datetime64[M]'), 
                output_path=os.path.join(output_path, "soi_plot.png"),
                title='SOI',
                xlabel='Time',
                ylabel='°C',
                fill='positive_negative',
                label=None
            )

        # store the SOI data
        #soi.to_netcdf(f"{self.output_path}/soi.nc")

        return 0
    
    def _compute_indian_monsoon_index(self, data, period, kinetic_energy=False):
        u850_1 = data.sel(longitude=slice(220, 260), latitude=slice(15, 5), level=850)['u_component_of_wind']
        u850_2 = data.sel(longitude=slice(250, 270), latitude=slice(30, 20), level=850)['u_component_of_wind']

        if period:
            u850_1 = u850_1.sel(time=slice(period[0], period[1]))
            u850_2 = u850_2.sel(time=slice(period[0], period[1]))

        if kinetic_energy:
            v850_1 = data.sel(longitude=slice(220, 260), latitude=slice(15, 5), level=850)['v_component_of_wind']
            v850_2 = data.sel(longitude=slice(250, 270), latitude=slice(30, 20), level=850)['v_component_of_wind']

            ke850_1 = 0.5 * (u850_1 ** 2 + v850_1 ** 2).mean(['latitude', 'longitude'])
            ke850_2 = 0.5 * (u850_2 ** 2 + v850_2 ** 2).mean(['latitude', 'longitude'])

            ke850 = ke850_1 + ke850_2
            ke850 = ke850.groupby('time.dayofyear').mean(dim=['time'])
            return ke850
        else:
            imd_index = u850_1.mean(['latitude', 'longitude']) - u850_2.mean(['latitude', 'longitude'])

            imd_index = imd_index.groupby('time.dayofyear').mean(dim=['time'])

            print('imd_index: ', imd_index)
        
            return imd_index
    
    @evaluate_ensemble
    def indian_monsoon_index(self, data, kinetic_energy=False, period=None, against_reference=False, ref='', **kwargs):

        imd_index = self._compute_indian_monsoon_index(data, period, kinetic_energy=kinetic_energy)

        #if 'std' in kwargs.keys():  # if ensemble
        #    std = kwargs['std']
        #   std_imd_index = self._compute_indian_monsoon_index(std, period)
        #    std_imd_index = std_imd_index.groupby('time.dayofyear').mean(dim=['time', 'latitude', 'longitude'])

        if against_reference and self.reference is not None:
            reference = self._compute_indian_monsoon_index(self.reference, period, kinetic_energy=kinetic_energy)
        
        xtick_labels = ['January', 'February', 'March', 'April', 
                'May', 'June', 'July', 'August', 'September', 
                'October', 'November', 'December'
                ]

        if len(imd_index.dayofyear) == 366:
            xtick_ids = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        else:
            xtick_ids = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

        output_path = f"{self.output_path}/climeval/imd/plots"
        os.makedirs(output_path, exist_ok=True)

        if kinetic_energy:
            title = f"{period[0]}-{period[1]} IMD Kinetic Energy"
            fname = f"{period[0]}-{period[1]}_imd_kinetic_energy_timeseries.png"
        else:
            title = f"{period[0]}-{period[1]} Indian Monsoon Index"
            fname = f"{period[0]}-{period[1]}_imd_timeseries.png"

        plot_timeseries(
            x=imd_index,
            ref=reference if against_reference and self.reference is not None else None,
            std=None, # if 'std' in kwargs.keys() else None,
            xticks=[xtick_ids, xtick_labels],
            title=title,
            xlabel='Time',
            ylabel='[m/s]',
            output_path=os.path.join(output_path, fname),
            linewidth=2,
            ref_label='ERA5' if against_reference else None,
            label=self.model_label
        )

    def regression_analysis(self, data, index):
        """
        This method performs regression analysis on against a given climate index like ONI or Webster-Yang.
        """

        print("Perform regression analysis ... ", end='')
        lat = data.latitude
        lon = data.longitude

        data = data.compute().to_numpy()
        index = index.to_numpy()  
        regression_map = np.empty((len(lat), len(lon)))  # Regression map to store slopes

        for i in range(len(lat)):
            for j in range(len(lon)):
                x = data[:, i, j]
                if np.all(np.isfinite(x)):
                    slope, _, r, _, _ = linregress(index, x)
                    regression_map[i, j] = slope
                else:
                    regression_map[i, j] = np.nan

        print('Done')
        return regression_map
    
    def _oni(self, data, period=None, power_spectrum=False, regression_analysis=False):
        assert 'sea_surface_temperature' in list(self.data.data_vars.keys()), \
            "Missing sea_surface_temperature variable"

        if period:
            data = data.sel(time=slice(period[0], period[1]))

        print(f"Compute Oceanic Nino Index for period {period} ... ", end='')

        nino34, sst = eval_timeseries.compute_oni_index(
            data=data, 
            base_period=self.base_period, 
            detrend=False
        )

        print('Done')

        # make time multindex normal index
        nino34 = nino34.reset_index('time')
        sst = sst.reset_index('time')

        print(nino34.values)

        if power_spectrum:
            print("Compute power spectrum ... ", end='')
            np_nino34 = nino34.to_numpy()

            spec_tuple = welch_psd(np_nino34)  # frequency, power_spectrum
            print('Done')
        else:
            spec_tuple = None

        if regression_analysis:
            regression_map = self.regression_analysis(sst, nino34)
        else:
            regression_map = None
        
        return nino34, spec_tuple, regression_map 
    
    #@evaluate_ensemble
    def oceanic_nino_index(self, data, against_reference=False, power_spectrum=False, regression_analysis=False, ref='', period=None, **kwargs):

    
        if period is None:
            output_path = f"{self.output_path}/climeval/oceanic_nino_index/plots/full"
        else:
            output_path = f"{self.output_path}/climeval/oceanic_nino_index/{period[0]}_{period[1]}"

        os.makedirs(output_path, exist_ok=True)
        
        print('### Model Data ###')
        nino34, psd, reg_map = self._oni(
            data, period=period, 
            power_spectrum=power_spectrum, 
            regression_analysis=regression_analysis
        )
        nino34.to_netcdf(f"{output_path}/data.nc")

        if against_reference and self.reference is not None:
            print("### Reference data ###")
            reference = self.select_reference_data(data, detrend=True)
            nino34_ref, psd_ref, ref_reg_map = self._oni(
                reference, 
                period=period, 
                power_spectrum=power_spectrum, 
                regression_analysis=regression_analysis
            )
            
            nino34_ref.to_netcdf(f"{output_path}/reference_data.nc")

        print('### Plotting ###')
        print('Plot ONI timeseries ... ', end='')
        plot_timeseries(
            nino34, 
            xticks=nino34.time.values.astype('datetime64[M]'),
            xlabel=r'Time',
            ylabel=r'°C',
            output_path=os.path.join(output_path, 'nino34_timeseries.png'),
            label=self.model_label, 
            fill='positive_negative', 
        )    

        if reference is not None:
            plot_timeseries(
                nino34_ref, 
                xticks=nino34.time.values.astype('datetime64[M]'),
                xlabel=r'Time',
                ylabel=r'°C',
                output_path=os.path.join(output_path, 'era5_nino34_timeseries_ref.png'),
                label='ERA5', 
                fill='positive_negative', 
            )
        print('Done')


        if power_spectrum:
            # make xticks only 2 decimals 
            #xtick_labels = [f"{x :.2f}" for x in psd[0]]
            print('Plot ONI power spectrum ... ', end='')
            plot_timeseries(
                psd[1], 
                #xticks=xtick_labels, 
                title='PSD of ONI',
                xlabel='Frequency (1/month)', 
                ylabel='Power', 
                output_path=f"{output_path}/psd.png",
                linewidth=2, 
                marker='o', 
                label=self.model_label,
                ref=psd_ref[1], 
                ref_label='ERA5' if against_reference else None,
            )
            print('Done')


        if regression_analysis:    
            print('Plot ONI regression map ... ', end='')
            plot_variable(
                reg_map, 
                extent=(-120, 130, 50, -50), 
                title='Nino3.4 Regression Map', 
                cmap='coolwarm', cbar_label='K/(K)', 
                fname='nino34_regression_map.png', 
                output_path=output_path, 
                central_longitude=180, 
                global_projection='Robinson', 
                #patch_kwargs=dict(
                #    xy=(lon_ids[0] + len(lon) // 2, lat_ids[0]),
                #    width=len(lon_ids), 
                #    height=len(lat_ids),
                #    fill=False,
                #    edgecolor='black', 
                #    linewidth=1.0),
                #    vmin=vmin, vmax=vmax

                )
            print('Done')


    def covariance(self, variable1, variable2, era5_base=False,  plot=True):
        """
        Compute the covariance between two variables in the dataset.
        
        :param variable1: First variable to compute covariance.
        :param variable2: Second variable to compute covariance.
        :return: Covariance as an xarray DataArray.
        
        pass
        """
        variable1_name = variable1.name
        variable1_level = variable1.level
        variable2_name = variable2.name
        variable2_level = variable2.level
        
        if variable1_level is not None:
            if variable1_level not in self.data.level.values:
                raise ValueError(f"Level {variable1_level} for variable {variable1_name} not found in the dataset. Available levels: {self.data.level.values}")
            data = self.data.sel(level=variable1_level)
            era5 = self.era5.sel(level=variable1_level) if era5_base and self.era5 is not None else None

        if variable2_level is not None:
            if variable2_level not in self.data.level.values:
                raise ValueError(f"Level {variable2_level} for variable {variable2_name} not found in the dataset. Available levels: {self.data.level.values}")
            data = self.data.sel(level=variable2_level)
            era5 = self.era5.sel(level=variable2_level) if era5_base and self.era5 is not None else None

     
        # Use the dataset's own climatology for covariance calculation
        print(f"Computing covariance between {variable1_name} and {variable2_name} at level {variable1_level} and {variable2_level} ...", end=" ")
        cov = xr.cov(data[variable1_name], data[variable2_name], dim='time')   
        print("Done.")

        # If era5_base is True, use ERA5 climatology as a base for covariance calculation       
        if era5_base and era5 is not None:
            # Use ERA5 climatology as a base for covariance calculation
            
            # select only base period for ERA5
            era5 = era5.sel(time=slice(f"{self.base_period[0]}-01-01", f"{self.base_period[1]}-12-31"))
            era5 = era5.fillna(value=era5.mean(dim=["latitude", "longitude"], skipna=True))
            
            # compute covariance
            print(f"Computing covariance between {variable1_name} and {variable2_name} at level {variable1_level} and {variable2_level} for ERA5 ...", end=" ")    
            era5_cov = xr.cov(era5[variable1_name], era5[variable2_name], dim='time')
            print("Done.")

        if plot:
            output_path = f"{self.output_path}/covariance/plots"
            os.makedirs(output_path, exist_ok=True)
            fname = f"covariance_{variable1_name}_{variable2_name}"
            title = f"{variable1_name} and {variable2_name} at level {variable1_level} and {variable2_level}"
            plot_variable(cov.compute(), fname=fname, output_path=output_path, title=title, cbar_label="Covariance")

            if era5_base:
                fname = f"covariance_{variable1_name}_{variable2_name}_era5"
                title = f"{variable1_name} and {variable2_name} (ERA5 {self.base_period[0]}-{self.base_period[1]})"
                plot_variable(era5_cov.compute(), fname=fname, output_path=output_path, title=title, cbar_label="Covariance")

        return cov

    
    def correlation(self, data, variable1, variable2, against_reference=True, colormap='coolwarm', ref=''):
        # as covariance, but compute correlation instead

        """        Compute the correlation between two variables in the dataset.
        :param variable1: First variable to compute correlation.
        :param variable2: Second variable to compute correlation.
        :return: Correlation as an xarray DataArray.
        """ 

        variable1_name = variable1['name']
        variable1_level = variable1['level']

        variable2_name = variable2['name']
        variable2_level = variable2['level']
        if variable1_level is not None:
            if variable1_level not in self.data.level.values:
                raise ValueError(f"Level {variable1_level} for variable {variable1_name} not found in the dataset. Available levels: {self.data.level.values}")
            data = data.sel(level=variable1_level)
            #era5 = self.era5.sel(level=variable1_level) if era5_base and self.era5 is not None else None

        if variable2_level is not None:
            if variable2_level not in self.data.level.values:
                raise ValueError(f"Level {variable2_level} for variable {variable2_name} not found in the dataset. Available levels: {self.data.level.values}")
            data = data.sel(level=variable2_level)
            
            #era5 = self.era5.sel(level=variable2_level) if era5_base and self.era5 is not None else None
        
        # Use the dataset's own climatology for correlation calculation
        print(f"Computing correlation between {variable1_name} and {variable2_name} at level {variable1_level} and {variable2_level} ...", end=" ")
        corr = xr.corr(data[variable1_name], data[variable2_name], dim='time')
        print("Done.")
        
        output_path = f"{self.output_path}/correlation/plots"
        os.makedirs(output_path, exist_ok=True)
        # plot the correlation
        fname = f"{ref}correlation_{variable1_name}_{variable2_name}"

        if ref != '':
            title_prefix = "ERA5 "
        else:
            title_prefix = ''

        # Get variable short names
        var1_short = self.get_variable_short_name(variable1_name, variable1_level)
        var2_short = self.get_variable_short_name(variable2_name, variable2_level)
        title = f"{title_prefix}{var1_short} and {var2_short}"
        plot_variable(corr.compute(), fname=fname, output_path=output_path, title=title, cbar_label="Correlation", cmap=colormap)

        if against_reference and self.reference is not None:
            self.correlation(
                data=self.reference, 
                variable1=variable1, 
                variable2=variable2, 
                against_reference=False, 
                colormap=colormap,
                ref='era5_'
            )
    
    def avg_temperature_map(self, level):
        """
        Plot 2m temperature with geopotential contours.

        :param level: Geopotential level to plot.
        """
        output_path = f"{self.output_path}/avg_temperature/plots"
        os.makedirs(output_path, exist_ok=True)

        for month in range(1, 13):
            temp = self.data['2m_temperature'].sel(time=self.data.time.dt.month == month).mean(dim='time')
            geopotential = self.data['geopotential'].sel(level=level, time=self.data.time.dt.month == month).mean(dim='time')

            plot_temperature_with_geopotential_contours(temp, geopotential, level, output_path, f"2m Temperature with Geopotential Heights at {level}hpa{month:02d}")
  
    #@evaluate_ensemble
    def latitude_time_plot(self, data, against_reference=False, ref_label="", **kwargs):
        """
        Plot a latitude-time plot for a given variable.
        
        :param variable: Variable to plot.
        :param level: Level to plot (optional).
        """
        output_path = f"{self.output_path}/climeval/latitude_time_plots"
        os.makedirs(output_path, exist_ok=True)

        print('Select data from time period ... ', end='')
        if 'time_period' in kwargs.keys():
            data = data.sel(time=slice(kwargs['time_period'][0], kwargs['time_period'][1]))
        print('Done')
        
        # Mask sea variables
        if ref_label == "":
            if 'sea_surface_temperature' in list(data.data_vars.keys()) and self.land_sea_mask is not None:
                print('Mask sea surface temperature over land ... ', end='')
                data['sea_surface_temperature'] = xr.where(self.land_sea_mask > 0.5, np.nan, data['sea_surface_temperature'])
        
        print('Interpolate missing latitudes ... ', end='')
        lat = data.latitude.to_numpy()
        data['latitude'] = lat[::-1]  # to interpolate, lat has to be monotonically increasing
        data = data.interpolate_na(dim='longitude', method='linear', fill_value="extrapolate").ffill(dim='latitude')
        data['latitude'] = lat  # restore original latitudes
        print('Done')

        print('Compute mean over time and longitude ... ', end='')
        data = compute_mean(data, reduce_dims=['time', 'longitude'], groupby=['time.year', 'time.month'])
        data = data.stack(time=['year', 'month'])
        print('Done')


        for var, lvl in self.variables:
            var_name = surface_variables_short[var] if var in surface_variables_short.keys() else level_variables_short[var] + str(lvl) 

            print(f"Plot {var_name} ... ", end='')
            plt.figure(figsize=(10, 6), dpi=300)
            plt.rcParams.update({'font.size': 12})
            plt.imshow(data.sel(level=lvl)[var], aspect="auto", interpolation='bilinear', cmap='coolwarm')
            plt.yticks(ticks=np.arange(0, data.latitude.size, 10), labels=data.latitude.values[::10].round(2))
            plt.xticks(ticks=np.arange(0, data.time.size, 12), labels=[f"{y}" for y in range(data['year'].values[0], data['year'].values[-1]+1)], rotation=90)
            plt.colorbar(label=self.units[var], orientation='vertical', pad=0.05, extend='both', shrink=0.9)
            plt.title(f'{var_name}')
            plt.xlabel('Time')
            plt.ylabel('Latitude')
            plt.savefig(f"{output_path}/{ref_label}_latitude_time_{var_name}.png", bbox_inches='tight')
            plt.close()

            print('Done')

        if against_reference and self.reference is not None:
            print('\n####### REFERENCE DATA #######\n')

            self.latitude_time_plot(data=self.reference, against_reference=False, ref_label="era5", **kwargs)

    @evaluate_ensemble
    def annual_anomalies(self, data, against_reference=True, plot=True):
        """
        Plot anomalies of a variable against time.
        
        :param variable: Variable to plot.
        :param level: Level to plot (optional).
        :param plot: Whether to plot the anomalies.
        """
        output_path = f"{self.output_path}/annual_anomalies/plots"
        os.makedirs(output_path, exist_ok=True)

        anomaly = eval_timeseries.compute_annual_anomaly(
            data, True, self.base_period)
        
        if against_reference:
            ref_anomaly = eval_timeseries.compute_annual_anomaly(
                self.reference, True, self.base_period)
        else:
            ref_anomaly = None


        if plot:    
            xticks = np.arange(0, anomaly.time.size)
            xtick_labels = [f"{y}-{m:2d}" for (y, m) in anomaly.time.values]
            mult = get_xlabel_multiplier(len(xtick_labels))
            for var, lvl in self.variables:
                var_name = surface_variables_short[var] if var in surface_variables_short.keys() else level_variables_short[var] + str(lvl) 
                plot_timeseries(
                    x=anomaly[var], 
                    ref=ref_anomaly[var] if against_reference else None,
                    var=var_name,
                    xticks=[xticks[::mult], xtick_labels[::mult]],
                    xlabel='Time',
                    ylabel=f'Anomaly [{self.units[var]}]',
                    title=f'Yearly Anomalies of {var_name}',
                    output_path=os.path.join(output_path, f'annual_anomalies_{var_name}.png')
                )

    """def variability_kde_timeseries(
            self,
            variable="temperature",
            level=850,                 # use None/null for surface variables
            label_model="ArchesWeather",
            plot_era5=True,
            bandwidth="scott",
            output_fname="variability_kde_timeseries",
            detrend=False,
        ):
   
        kde_variability_plot(
            data=self.data,
            era5=self.era5 if plot_era5 else None,
            variable=variable,
            level=level,
            base_period=self.base_period,
            output_path=self.output_path,
            label_model=label_model,
            include_era5=plot_era5,
            bandwidth=bandwidth,
            output_fname=output_fname,
            detrend=detrend,
        )
    """
    @evaluate_ensemble
    def radial_spectrum(self, data, groupby=['time.year', 'time.month'], projection_years=[2020], ref_year=2020, against_reference=False, **kwargs):
        """
        Compute the radial spectrum of the given data.
        """


        # Check if reference data is available
        if against_reference:
            print("Load reference data and compute reduction ... ", end="")
            
            reference = self.select_reference_data(data)
            reference = reference.fillna(value=reference.mean(dim=["latitude", "longitude"], skipna=True))
            reference = reference.sel(time=reference.time.dt.year.isin([ref_year]))
            reference = reference.compute()

            print("Done")
        else:
            print("No reference data provided. Skipping reference comparison.")
            reference = None

        print("Load data and compute reduction ... ", end="")

        data = data.fillna(value=data.mean(dim=["longitude"], skipna=True)).ffill(dim="latitude").bfill(dim="latitude")
        data = data.sel(time=data.time.dt.year.isin([ref_year]))
        data = data.compute()

        print("Done")

        output_path = self.output_path + "/climeval" + "/radial_spectrum" + "/plots"
        os.makedirs(output_path, exist_ok=True)

        def _get_var_spectrum(data, var, level=None):
            if data is None:
                return None
            
            x = data.sel(level=level)[var].to_numpy()
            spec = np.stack([compute_radial_spectrum(x=xi) for xi in x], axis=0)
            spec = np.mean(spec, axis=0) 
            
            return spec

        spectra = {"ref": {}, "model": {}}
        for var, lvl in self.variables:
            var_name = surface_variables_short[var] if var in surface_variables_short.keys() else level_variables_short[var] + str(lvl)
            print(f"{var_name} ... ", end="")
            print("Compute ... ", end="")
            ref_spec = _get_var_spectrum(reference, var, level=lvl)
            spec = _get_var_spectrum(data, var, level=lvl)  
            spectra["ref"][var_name] = ref_spec
            spectra["model"][var_name] = spec                       
            print("Plot ... ", end="")
            plot_radial_spectrum(spec, var_name, output_path=output_path + f"/radial_spectrum_{var_name}_{ref_year}.png", ref_spec=ref_spec)
            print("Done")
        

        # Dump spectra to a ref and a model npz file
        np.savez(
            os.path.join(output_path, f"model_radial_spectra_{ref_year}.npz"), 
            **spectra["model"]
        )

        np.savez(
            os.path.join(output_path, f"ref_radial_spectra_{ref_year}.npz"), 
            **spectra["ref"]
        )

    def iter_variables(self, fnc, data, reference=None, stddev=None, variables=None, **kwargs):
        """
        Iterate over the variables and levels specified in the configuration
        or over a provided list of variables and levels, applying the given function
        to each variable-level pair. 
        
        :return: None
        """
        if variables is not None:
            vars_to_process = variables
        else:
            vars_to_process = self.variables

        for var, lvl in vars_to_process:
            var_name = self.get_variable_short_name(var, lvl)
            print(f"Processing {var_name} ...")
            fnc(data, var_name=var_name, reference=reference, stddev=stddev, **kwargs)


    def evaluate(self, eval_metrics, **kwargs):
        """
        Evaluate the climate data.
        
        :return: Evaluation results.
        """
        # Placeholder for evaluation logic
        
        for k, v in kwargs.items():
            if k not in eval_metrics:
                print(f"Skipping {k} as it is not in eval_metrics.")
                continue

            if eval_metrics[k]:

                if hasattr(self, k):
                    method = getattr(self, k)
                    if callable(method):
                        print(f"\nEvaluating {k}\n---------------------------\n")
                        
                        v = OmegaConf.to_container(v)
                        method(data=self.data, **v)
                        print("\n---------------------------\n")
                    else:
                        print(f"{k} is not callable.")
                else:
                    print(f"{k} is not a valid method of ClimateEvaluator.")
            else:
                print(f"Skipping {k} as it is set to False in eval_metrics.")
        


@hydra.main(version_base=None, config_path="configs", config_name="eval_config")
def main(cfg):
    """
    Main function to run the climate evaluation.
    
    :param cfg: Configuration object from Hydra.
    :return: None
    """

    # Load the data
    evaluator = ClimateEvaluator(**cfg['evaluator'])
    evaluator.evaluate(eval_metrics=cfg['eval_metrics'], **cfg['evaluate'])

if __name__ == "__main__":

    main()
