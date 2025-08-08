import glob
import os
import numpy as np
import xarray as xr
# importlib module for retrieving module path
import hydra
import importlib 
#from geoclim import stats as climate_stats

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd 
from cartopy.crs import PlateCarree, Robinson
from cartopy import feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from omegaconf import ListConfig, DictConfig
from omegaconf import OmegaConf

from geoarches.dataloaders.era5 import surface_variables_short, level_variables_short
from metrics.southern_oscillation_index import calculate_southern_oscillation_index

#climate_stats_path = importlib.resources.files(climate_stats)

def plot_soi(soi_data, time, output_path):
    # function that plots the Southern Oscillation Index (SOI) data
    # values larger than 0 are red and values smaller than 0 are blue
    plt.figure(figsize=(15, 5), dpi=150)
    time = list(set(time.astype('datetime64[M]').to_numpy()))  # Ensure time is in datetime format
    time.sort()
    ids = list(range(0, len(time)))
    plt.plot(ids, soi_data, label="SOI", color='black')

    # Highlight positive and negative values
    positive = soi_data.where(soi_data > 0, drop=False)
    negative = soi_data.where(soi_data < 0, drop=False)
    pos_ids = (soi_data > 0).to_numpy().astype(np.int32).tolist()
    neg_ids = (soi_data < 0).to_numpy().astype(np.int32).tolist()
    pos_ids = [ids[i] for i, pidx in enumerate(pos_ids) if pidx == 1]
    neg_ids = [ids[i] for i, nidx in enumerate(neg_ids) if nidx == 1]
    print(len(pos_ids), len(neg_ids), len(ids))
    plt.fill_between(ids, 0, positive, color='orange')
    plt.fill_between(ids, 0, negative, color='paleturquoise')

    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)

    years = list(set([t.astype('datetime64[Y]') for t in time]))
    years.sort()
    print(years)
    plt.xticks(ticks=list(range(0, len(time), 12)), labels=years, rotation=45, ha='right')
    
    plt.grid()
    
    plt.title('Southern Oscillation Index (SOI)')
    plt.xlabel('Time')
    plt.ylabel('SOI')
    
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/soi_plot.png')

def plot_variable(x, fname, output_path, title=None, ax=None, cbar_label=None, cmap='viridis'):
    # Plot a xarray DataArray with cartopy projection 
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': Robinson()})
    else:
        fig = ax.figure

    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, edgecolor='black')

    ax.set_title(title)
    img = ax.imshow(
        x, transform=PlateCarree(), cmap=cmap, vmin=x.min(), vmax=x.max()
    )

    # Colorbar with triangular ends and smaller size
    cbar = fig.colorbar(
        img, ax=ax, orientation='horizontal',
        pad=0.1, extend='both', shrink=0.7, aspect=30
    )
    cbar.set_label(cbar_label if cbar_label else "")

    ax.gridlines(
        draw_labels=True, 
        dms=True, 
        x_inline=False, 
        y_inline=False
    )

    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

    plt.tight_layout()

    if output_path:
        plt.savefig(f"{output_path}/{fname}.png", dpi=300, bbox_inches='tight')

    plt.close(fig)  # Close the figure to free memory

def plot_anomalies(data, mean, variables, levels, output_path, anomaly_type="monthly"):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Plot surface variables first
    for var in variables["surface"]:
        varx = data[var].to_numpy()
        meanx = mean[var].to_numpy()
        print(varx.shape, meanx.shape)
        assert varx.shape[0] == 12, "Surface variable should have 12 months"
        for i in range(varx.shape[0]):
            title = f"{anomaly_type} Anomalies {months[i]} {surface_variables_short[var]} "
            anomaly = varx[i] - meanx[i]
            plot_variable(anomaly, fname=f"{anomaly_type}_anomaly_{months[i]}_{surface_variables_short[var]}", output_path=output_path, title=title)

    # Plot level variables
    for var in variables["level"]:
        for lvl in levels:
            varx = data[var].sel(level=lvl).to_numpy()
            meanx = mean[var].sel(level=lvl).to_numpy()
            assert varx.shape[0] == 12, "Level variable should have 12 months"
            for i in range(varx.shape[0]):
                title = f"{anomaly_type} Anomalies {months[i]} {level_variables_short[var]}{lvl}"
                plot_variable(
                    varx[i] - meanx[i], 
                    fname=f"{anomaly_type}_anomaly_{months[i]}_{level_variables_short[var]}{lvl}", 
                    output_path=output_path, 
                    title=title
                )

def plot_temperature_with_geopotential_contours(temp, geopotential, level, output_path, title):

    # Use cartopy to plot temperature with geopotential contours
    fig, ax = plt.subplots(subplot_kw={'projection': Robinson()})
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')

    # Plot temperature and geopotential
    temp.plot(ax=ax, transform=Robinson(), cmap='coolwarm', vmin=temp.min(), vmax=temp.max())
    geopotential.plot.contour(ax=ax, transform=PlateCarree(), levels=4, cmap='gray', linewidths=1.)
    ax.set_title(title)

    plt.savefig(f"{output_path}/2m_temp_geopotential_{level}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_annual_oscillation(data, output_path, variable_name, add_linear_trend=True, ref=None):
    """
    Plot the annual oscillation of a variable.
    
    :param data: The data to plot.
    :param variable_name: The name of the variable to plot.
    :return: None
    """
    plt.figure(figsize=(15, 5), dpi=150)
    plt.plot(data, label="Prediction", color='blue')

    if ref is not None:
        plt.plot(ref, label=f'ERA5 {variable_name}', linestyle='--', color='orange')
    plt.xticks(rotation=45)
    plt.grid()
    plt.title(f'Annual Oscillation of {variable_name}')
    plt.xlabel('Time')

    # find first occurence of years 1980, 1985, 1990 ... in data['time']
    time = data['time'].values
    years = pd.date_range(start=time[0], end=time[-1], freq='YS')
    indices = [np.where(time == year)[0][0] for year in years if year in time]

    if add_linear_trend:
        # Add a linear trend line
        z = np.polyfit(np.arange(len(data)), data, 1)
        p = np.poly1d(z)
        plt.plot(np.arange(len(data)), p(np.arange(len(data))), label='Trend', linestyle='--', color='black')

    xticklabels = np.datetime_as_string(time[indices], unit="Y")
    plt.xticks(ticks=indices, labels=xticklabels, rotation=90)
    plt.ylabel(variable_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/annual_oscillation_{variable_name}.png')

class ClimateEvaluator:
    """
    A class to evaluate climate data.
    """
    R = 6.371e6  # Radius of the Earth in meters
    g = 9.81     # Acceleration due to gravity in m/s^2
    pi = 3.14159

    def __init__(self, path, variables=None, levels=None, era5_clim_path=None, clim_stats=None, era5_path=None, base_period=(1995, 2014), output_path=None):
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
        fpaths = glob.glob(path + "/*.nc")
        fpaths.sort()
        # Load prediction data
        print("Opening prediction data from:", path, " ...", end=" ")
        self.data = xr.open_mfdataset(fpaths, combine="by_coords")
        self.data = self.data.roll(longitude=-len(self.data.longitude) // 2, roll_coords=False)  # Roll longitude to match data

        print("Done.")

        # Open climatology data

        self.era5_clim = xr.load_dataset(era5_clim_path) if era5_clim_path else None

        # Open era5 
        if era5_path is not None:
            era5_files = glob.glob(era5_path + "/*.nc")
            era5_files.sort()

            # extract hour from self.data time dimension and only keep files that match the hour
            hour = int(self.data.time.dt.hour[0])
            era5_files = [f for f in era5_files if f"{hour:02d}" in f]
            if not era5_files:
                raise FileNotFoundError(f"No ERA5 files found for hour {hour} in {era5_path}")
            
            # preprocess files such that time dimension has datetime64[day] type  
            # This avoids issues with non-monotonic time dimension being datetime64[ns] 
            # in ERA5 files
            def preprocess_time(ds):
                ds['time'] = ds['time'].astype('datetime64[D]')
                return ds
            
            self.era5 = xr.open_mfdataset(era5_files, combine='by_coords', preprocess=preprocess_time)
            self.era5 = self.era5[list(self.data.data_vars.keys())]
            self.era5 = self.era5.transpose('time', 'level', 'latitude', 'longitude')
            self.era5 = self.era5.reindex(latitude=list(reversed(self.era5.latitude)))
                    # Roll data
            self.era5 = self.era5.roll(longitude=-len(self.era5.longitude) // 2, roll_coords=False)  # Roll longitude to match data

        else:
            self.era5 = None
            
        self.base_period = base_period
        self.output_path = output_path

        if variables is None:
            self.variables = {
                "surface": self.data.variables.intersection(surface_variables_short.keys()).tolist(),
                "level": self.data.variables.intersection(level_variables_short.keys()).tolist()
            }
        else:
            self.variables = variables 

        if levels is None:
            self.levels = self.data.level.values.tolist()
        else:
            self.levels = levels

    def anomalies(self, type="monthly", year=2015, plot=False):
        era5 = self.era5.sel(time=slice(f"{self.base_period[0]}-01-01", f"{self.base_period[1]}-12-31"))        
        era5 = era5.fillna(value=era5.mean(dim=["latitude", "longitude"], skipna=True))
        if type == "monthly":
            mean = era5.groupby('time.month').mean('time', skipna=True)
            data = self.data.sel(time=self.data.time.dt.year==year)
            data = data.groupby(['time.month']).mean('time')   
        elif type == "daily":
            mean = era5.groupby('time.dayofyear').mean('time')
            data = self.data.sel(time=self.data.time.dt.year==year)
            data = data.groupby(['time.dayofyear'])
        else:
            raise ValueError("Invalid type. Choose 'monthly' or 'daily'.")
        
        # Store anomalies on disk
        os.makedirs(f"{self.output_path}/anomalies/{type}", exist_ok=True)
        
        if self.output_path:
            output_path = f"{self.output_path}/anomalies/{type}"
            
            #anomalies.to_netcdf(f"{output_path}/anomalies_{type}_{self.base_period[0]}_{self.base_period[1]}.nc")
       
        if plot:
            os.makedirs(f"{self.output_path}/anomalies/{type}/plots", exist_ok=True)
            output_path = f"{self.output_path}/anomalies/{type}/plots"

            plot_anomalies(data, mean, self.variables, self.levels, output_path, anomaly_type=type) 

        return 0

    def annual_cycle(self, plot=True, plot_against_era5=False):

        data =  self.data.mean(['latitude', 'longitude'])
        output_path = f"{self.output_path}/annual_oscillation"

        os.makedirs(output_path, exist_ok=True)
        
        # era5 has to start at the same time as self.data
        if self.era5 is not None:
            era5 = self.era5.fillna(value=self.era5.mean(dim=["latitude", "longitude"], skipna=True))
            era5 = era5.sel(time=slice(data.time.min(), data.time.max()))

        if plot:
            op = f"{self.output_path}/annual_oscillation/plots"
            os.makedirs(op, exist_ok=True)
            for var in self.variables['surface']:
                x = data[var]
                plot_annual_oscillation(
                    x, op, surface_variables_short[var], 
                    ref=era5[var].mean(['latitude', 'longitude']).to_numpy() if plot_against_era5 else None)

            for var in self.variables['level']:
                for lvl in self.levels:
                    x = data[var].sel(level=lvl)
                    plot_annual_oscillation(
                        x, op, level_variables_short[var] + str(lvl), 
                        ref=era5[var].mean(['latitude', 'longitude']).sel(level=lvl).to_numpy() if plot_against_era5 else None)
        
        # store the annual oscillation data
        x.to_netcdf(f"{output_path}/annual_oscillations.nc")

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

    def madden_julian_oscillation(self, year=None, month=None):
        """
        Calculate the Madden-Julian Oscillation (MJO) index.
        If year and month are provided, filter the data accordingly.
        If year and month are not provided, use the entire dataset and
        produce a time series of MJO indices.

        year: Year to filter the data (optional).
        month: Month to filter the data (optional).

        :return: MJO index.
        """

        data = self.data.sel(time=slice(f"{year}-{month}-01", f"{year}-{month}-31")) if year and month else self.data

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
    
    def southern_oscillation_index(self, plot=True, month=None, year=None):
        """
        Compute the Southern Oscillation Index (SOI) for a given year and month.
        If year and month are provided, filter the data accordingly.
        If not provided, use the entire dataset.

        :param year: Year to filter the data (optional).
        :param month: Month to filter the data (optional).
        :return: SOI index.
        """

        data = self.data.sel(time=slice(f"{year}-{month}-01", f"{year}-{month}-31")) if year and month else self.data        
        era5 = self.era5.sel(time=slice(f"{self.base_period[0]}-01-01", f"{self.base_period[1]}-12-31")) if self.era5 is not None else None
        
        # Extract time dimension and ensure it is in datetime format month
        time = data['time']

        # Get the ERA5 climatology data for tahiti and darwin
        data = data.groupby(['time.year','time.month']).mean('time')

        soi = calculate_southern_oscillation_index(data, era5)

        # Get tahiti and darwin mean sea level pressure
        """mslp_tahiti = data['mean_sea_level_pressure'].sel(latitude=-17.65, longitude=-149.57 + 180, method='nearest')
        mslp_darwin = data['mean_sea_level_pressure'].sel(latitude=-12.46, longitude=130.84 + 180, method='nearest')

        mslp_tahiti_era5 = tahiti_era5['mean_sea_level_pressure']
        mslp_darwin_era5 = darwin_era5['mean_sea_level_pressure']

        # Calculate the SOI
        pdiff = mslp_tahiti - mslp_darwin
        pdiff_avg = (mslp_tahiti_era5  - mslp_darwin_era5).groupby(['time.month']).mean(dim='time')
        pdiff_std = (mslp_tahiti_era5 - mslp_darwin_era5).groupby(['time.month']).std(dim='time')

        soi = 10 * (pdiff - pdiff_avg) / pdiff_std
        soi = soi.compute()
        soi = soi.stack(time=('year', 'month'), create_index=True).transpose('time', ...)"""
        
        if plot:
            output_path = f"{self.output_path}/soi/plots"
            os.makedirs(output_path, exist_ok=True)
            plot_soi(soi, time=time, output_path=output_path)

        # store the SOI data
        #soi.to_netcdf(f"{self.output_path}/soi.nc")

        return 0
    
    def covariance(self, variable1, variable2, era5_base=False,  plot=True):
        """
        Compute the covariance between two variables in the dataset.
        
        :param variable1: First variable to compute covariance.
        :param variable2: Second variable to compute covariance.
        :return: Covariance as an xarray DataArray.
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
    
    def correlation(self, variable1, variable2, era5_base=False, plot=True, colormap='coolwarm'):
        # as covariance, but compute correlation instead

        """        Compute the correlation between two variables in the dataset.
        :param variable1: First variable to compute correlation.
        :param variable2: Second variable to compute correlation.
        :return: Correlation as an xarray DataArray.
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
        
        # Use the dataset's own climatology for correlation calculation
        print(f"Computing correlation between {variable1_name} and {variable2_name} at level {variable1_level} and {variable2_level} ...", end=" ")
        corr = xr.corr(data[variable1_name], data[variable2_name], dim='time')
        print("Done.")

        # If era5_base is True, use ERA5 climatology as a base for correlation calculation
        if era5_base and era5 is not None:
            # select only base period for ERA5
            era5 = era5.sel(time=slice(f"{self.base_period[0]}-01-01", f"{self.base_period[1]}-12-31"))
            era5 = era5.fillna(value=era5.mean(dim=["latitude", "longitude"], skipna=True))
            
            # compute correlation
            print(f"Computing correlation between {variable1_name} and {variable2_name} at level {variable1_level} and {variable2_level} for ERA5 ...", end=" ")    
            era5_corr = xr.corr(era5[variable1_name], era5[variable2_name], dim='time')
            print("Done.")
        
        if plot:
            output_path = f"{self.output_path}/correlation/plots"
            os.makedirs(output_path, exist_ok=True)
            # plot the correlation
            fname = f"correlation_{variable1_name}_{variable2_name}"
            title = f"{variable1_name} and {variable2_name} at level {variable1_level} and {variable2_level}"
            plot_variable(corr.compute(), fname=fname, output_path=output_path, title=title, cbar_label="Correlation", cmap=colormap)
            if era5_base:
                fname = f"correlation_{variable1_name}_{variable2_name}_era5"
                title = f"{variable1_name} and {variable2_name} (ERA5 {self.base_period[0]}-{self.base_period[1]})"
                plot_variable(era5_corr.compute(), fname=fname, output_path=output_path, title=title, cbar_label="Correlation", cmap=colormap)


        return corr
    
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

    def latitude_time_plot(self, variable, level=None):
        """
        Plot a latitude-time plot for a given variable.
        
        :param variable: Variable to plot.
        :param level: Level to plot (optional).
        """
        output_path = f"{self.output_path}/latitude_time_plots"
        os.makedirs(output_path, exist_ok=True)

        if level is not None:
            data = self.data[variable].sel(level=level)
        else:
            data = self.data[variable]

        # Create latitude-time plot
        plt.figure(figsize=(15, 5), dpi=150)
        data.plot(x='time', y='latitude', cmap='viridis')
        plt.title(f"Latitude-Time Plot of {variable} at Level {level}" if level else f"Latitude-Time Plot of {variable}")
        plt.xlabel('Time')
        plt.ylabel('Latitude')
        plt.colorbar(label=variable)
        plt.tight_layout()
        plt.savefig(f"{output_path}/{variable}_latitude_time_plot.png")

    def anomaly_over_years(self, variable, level=None, plot=True):
        """
        Plot anomalies of a variable against time.
        
        :param variable: Variable to plot.
        :param level: Level to plot (optional).
        :param plot: Whether to plot the anomalies.
        """
        output_path = f"{self.output_path}/anomalies_against_time"
        os.makedirs(output_path, exist_ok=True)

        if level is not None:
            data = self.data[variable].sel(level=level)
            era5 = self.era5[variable].sel(level=level) 
        else:
            data = self.data[variable]
            era5 = self.era5[variable]

        # Calculate monthly anomalies over years
        monthly_mean = era5.groupby('time.month').mean(['time', 'latitude', 'longitude'])
        anomalies = data.groupby(['time.year', 'time.month']).mean(['time', 'latitude', 'longitude']) - monthly_mean
        anomalies = anomalies.mean(dim="month")
        anomalies = anomalies.rename(year='time')
        anomalies = anomalies.reset_index('time')

        if plot:
            plt.figure(figsize=(15, 5), dpi=150)
            plt.plot(anomalies.compute(), label='Model Anomalies', color='blue', linewidth=2)
            plt.title(f"Anomalies of {variable} at Level {level}" if level else f"Anomalies of {variable}")
            plt.xlabel('Time')
            plt.ylabel('Anomaly')
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{output_path}/{variable}_anomalies.png")

        return anomalies

    def evaluate(self):
        """
        Evaluate the climate data.
        
        return: Evaluation results.
        """
        # Placeholder for evaluation logic
        return {"status": "Evaluation complete", "data": self.data}
    


@hydra.main(version_base=None, config_path="configs", config_name="eval_config")
def main(cfg):
    """
    Main function to run the climate evaluation.
    
    :param cfg: Configuration object from Hydra.
    :return: None
    """

    # Load the data
    evaluator = ClimateEvaluator(**cfg['evaluator'])

    for method, params in cfg.evaluate.items():
        getattr(evaluator, method)(**params)

    # Perform evaluation
    #results = evaluator.evaluate()

    # Print results
    
    #print("Evaluation Results:", results)

if __name__ == "__main__":

    main()
