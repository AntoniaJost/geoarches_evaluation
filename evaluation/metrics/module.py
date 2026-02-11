import os
import itertools

from matplotlib.colors import CenteredNorm
from sympy import use, var
from metrics import functional

from metrics.functional import timeseries, spectral, kernel_density_estimation, utils
from plot.functional.timeseries import timerseries_to_ax, get_xlabel_multiplier
from plot.functional.spatial import lambert_conformal_projection_plot
from plot.modules import EarthPlotter, SpatialPlotter, TimeseriesPlotter



from geoarches.metrics.metric_base import (
    compute_lat_weights,
    compute_lat_weights_weatherbench,
)

import xarray as xr
import numpy as np

import pandas as pd


def annual_mean(data, time_dim='time', year=None):
    if year is not None:
        data = data.sel({f'{time_dim}.year': year})
        return data.mean(time_dim)
    else:
        return data.mean(time_dim)

def seasonal_mean(data,  season=None, time_dim='time'):
    if season is not None:
        data = data.groupby(f"{time_dim}.season").mean(time_dim).sel(season=season)
        return data
    else:
        return data.mean(time_dim)
    
def instantaneous(data, time, time_dim='time'):
    return data.sel({time_dim: time}, nearest=True, drop=True)

def compute_anomaly(
        data: xr.DataArray | xr.Dataset, 
        baseline_period, 
        mean_groups, 
        baseline_mean_groups,
        standardize: bool = False,
    ):

    baseline_data = data.sel(
        time=(data.time.values >= np.datetime64(baseline_period[0], 'ns')) & 
             (data.time.values <= np.datetime64(baseline_period[1], 'ns')), method="nearest"
    )

    if baseline_mean_groups is None:
        baseline_mean = baseline_data.mean(dim='time')
    else:
        baseline_mean = baseline_data.groupby(baseline_mean_groups).mean(dim='time')

    anomalies = data.groupby(mean_groups).mean(dim='time') - baseline_mean

    if standardize:
        if baseline_mean_groups is None:
            baseline_std = baseline_data.std(dim='time')
        else: 
            baseline_std = baseline_data.groupby(baseline_mean_groups).std(dim='time')
        anomalies = anomalies / baseline_std

    return anomalies

def compute_bias(data, reference_data):
    bias = data - reference_data
    return bias

class XYMaps:
    def __init__(self, variables, xdim, ydim, temporal_selection: list = ['annual'], plotter_kwargs: dict = {}):
        self.variables = list(variables)
        self.xdim = xdim
        self.ydim = ydim

        self.temporal_selection = temporal_selection

        print("Initializing XYMaps Plotter")
        self.plotter = SpatialPlotter(xdim=self.xdim, ydim=self.ydim, **plotter_kwargs)
        
    def select_by_time(self, data, temporal_dim):
        if temporal_dim == 'annual':
            data = annual_mean(data)
        elif temporal_dim in ['DJF', 'MAM', 'JJA', 'SON']:
            data = seasonal_mean(data, season=temporal_dim)
        else: 
            data = instantaneous(data, time=temporal_dim)   

        return data
    
    def compute(self, data_container, temporal_dim, var: dict, frequency: str = "monthly"):
        """

        Compute spatially averaged data for given temporal dimension and variable.

        Args:
            data_container (CMORDataContainer): _description_
            temporal_dim (_type_): _description_
            var (dict): _description_
            frequency (str, optional): _description_. Defaults to "monthly".

        Returns:
            _type_: _description_
        """
        # Load data of the variable given a frequency
        data = data_container.get_variable_data(**var, frequency=frequency)

        # Select by time (i.e. select the time range / season / year / instance)
        data = self.select_by_time(data, temporal_dim)

        # Average over all dimensions except the spatial ones
        data = data.mean(
            dim=[d for d in data.dims if d not in [self.xdim, self.ydim]]
        )

        return data
    
    def evaluate(self, data_containers):

        for ts in self.temporal_selection:
            print(f"Computing {self.xdim}-{self.ydim} map for temporal selection: {ts}")
            print("-" * 72)
            # Do computation here
            for variable in self.variables:
                name, pressure_level = variable['name'], variable['pressure_level']
                if pressure_level is not None:
                    print(
                        f"--> Processing variable: {name}" 
                        f" at pressure level {pressure_level} Pa")
                else:
                    print(f"--> Processing variable: {name}")
                
                for data_container in data_containers:
                    data = self.compute(
                        data_container,
                        temporal_dim=ts,
                        var=variable,
                        frequency='monthly'
                    )

                    # ensure that data has dimensions in order [ydim, xdim]
                    data = data.transpose(self.ydim, self.xdim)
                    x = data[name].values 

                    var_name = f"{name}_{pressure_level}Pa" if pressure_level is not None else name
                    # Add temporal selection to output path for plotting
                    output_path = os.path.join(self.plotter.output_path, ts)

                    self.plotter.plot(
                        x=x,
                        variable_name=var_name,
                        title="",
                        model_label=data_container.model_label,
                        style="imshow",
                        output_path=output_path
                    )
                          

class XYBiasMaps(XYMaps):
    def compute(self, data_container, temporal_dim, variable_name: str, frequency: str = "monthly"):
        """

        Compute spatially averaged bias data for given temporal dimension and variable.

        Args:
            data_container (CMORDataContainer): _description_
            temporal_dim (_type_): _description_
            variable_name (str): _description_
            frequency (str, optional): _description_. Defaults to "monthly".

        Returns:
            _type_: _description_
        """

        # Get Model data
        data = data_container.get_variable_data(variable_name, frequency)
        data = self.select_by_time(data, temporal_dim, variable_name, frequency)
        data = data.mean(
            dim=[d for d in data.dims if d not in [self.xdim, self.ydim]]
        )

        # Get Reference data
        reference_data = self.reference_data.get_variable_data(variable_name, frequency)
        reference_data = self.select_by_time(reference_data, temporal_dim, variable_name, frequency)
        reference_data = reference_data.mean(
            dim=[d for d in reference_data.dims if d not in [self.xdim, self.ydim]]
        )
      
        bias_data = compute_bias(data, reference_data)

        return bias_data
    
    def evaluate(self, data_containers):
        idx_ref = [i for i, dc in enumerate(data_containers) if dc.is_reference]
        self.reference_data = data_containers[idx_ref[0]] if idx_ref else None
        assert self.reference_data is not None, "No reference data container found."

        return super().evaluate(data_containers)
    
    
class XYAnomalyMaps(XYMaps):
    def __init__(self, variables, xdim, ydim, temporal_selection = ['annual'], baseline_period = ('1981-01-01T00', '2010-12-31T00')):
        super().__init__(variables, xdim, ydim, temporal_selection)
        self.baseline_period = baseline_period

    def compute(self, data_container, temporal_dim, variable_name: str, frequency: str = "monthly"):
        """

        Compute spatially averaged anomaly data for given temporal dimension and variable.

        Args:
            data_container (CMORDataContainer): _description_
            temporal_dim (_type_): _description_
            variable_name (str): _description_
            frequency (str, optional): _description_. Defaults to "monthly".

        Returns:
            _type_: _description_
        """

        # Get Model data
        data = data_container.get_variable_data(variable_name, frequency)
        data = self.select_by_time(data, temporal_dim, variable_name, frequency)
        data = data.mean(
            dim=[d for d in data.dims if d not in [self.xdim, self.ydim]]
        )
      
        anomaly_data = compute_anomaly(data, baseline_period=self.baseline_period)

        return anomaly_data

class Timeseries:
    def __init__(
            self, 
            variables: list[str] = None, 
            detrend: bool = False, 
            compute_anomalies: bool = False, 
            mean_groups: list = None,
            baseline_period: tuple = None, 
            baseline_mean_groups: list = None,
            plotter_kwargs: dict = None):
        super().__init__()

        # Variables to evaluate
        self.variables = variables

        # Mean and baseline groups 
        self.mean_groups = mean_groups

        # Baseline mean groups for anomaly calculation
        self.baseline_mean_groups = baseline_mean_groups
        
        # If to compute anomalies and how to compute them
        self.compute_anomalies = compute_anomalies
        if self.compute_anomalies and baseline_period is None:
            raise ValueError("A baseline period must be provided when computing anomalies.")
        
        # Baseline period for anomaly calculation
        self.baseline_period = baseline_period

        # Whether to detrend data before analysis
        self.detrend = detrend

        # Initialize plotter
        output_path = plotter_kwargs.get("output_path", ".")
        if compute_anomalies:
            output_path = os.path.join(output_path, f"{baseline_period[0]}-{baseline_period[1]}")
        if detrend:
            output_path = os.path.join(output_path, "detrended")
        plotter_kwargs["output_path"] = output_path 
        self.timeseries_plotter = TimeseriesPlotter(**(plotter_kwargs or {}))

    def xlabels_from_time(self, time: xr.DataArray):
        if "month" in time.dims and "year" in time.dims:
            YM = itertools.product(time['year'].values, time['month'].values)
            return [f"{y}-{m:02d}" for y, m in YM]
        elif "month" in time.dims:
            return time['month'].values
        elif "year" in time.dims:
            return time['year'].values
        else:
            raise ValueError("Time coordinate must have 'month' and/or 'year' dimensions.")

    def compute(self, data_container, variable: str):
        data = data_container.get_variable_data(
                         variable_name=variable,
                         frequency='monthly'
                     )
        
    def spectrum(self, data: np.ndarray | xr.DataArray, fs: float = 1.0):
        fx, fy = spectral.welch_psd(data, fs=fs)

        return fx, fy
        
    def compute_latitude_weighted_weights(self, latitude): 
        return compute_lat_weights_weatherbench(latitude)
    
    def evaluate(self, data_containers, variables: list[str]):
        
        for data_container in data_containers:
            for variable in variables:
                self.compute(data_container=data_container, variable=variable)


def detrend_data(data: np.ndarray | xr.DataArray) -> np.ndarray | xr.DataArray:
    coeffs = np.polyfit(range(len(data)), data, deg=1)
    coeffs[-1] = 0  # Set intercept to zero to preserve mean
    fit = np.polyval(coeffs, range(len(data)))
    data.values = data.values - fit
    
    return data 


class SeasonalCycles(Timeseries):
    def __init__(
            self, variables: list[str], mean_groups: list = ["year"], linear_trend: bool = False,
            detrend=False, compute_anomalies: bool = False, baseline_period: tuple = None,
            baseline_mean_groups: list = None, plotter_kwargs: dict = None):
        
        super().__init__(variables,
                        detrend=detrend, 
                        compute_anomalies=compute_anomalies, 
                        baseline_period=baseline_period,
                        baseline_mean_groups=baseline_mean_groups,
                        mean_groups=mean_groups,
                        plotter_kwargs=plotter_kwargs)
        
        if linear_trend and detrend:
            raise ValueError("Cannot apply both linear trend and detrending. Choose one.")
        
        self.linear_trend = linear_trend


    def compute(self, data_container, variable: str):
        data = data_container.get_variable_data(**variable, frequency='monthly')
        
        # Reduce to spatial average
        data = data.mean(dim=["lat", "lon"])

        # Detrend data if specified
        if self.detrend:
            data[variable["name"]].values = detrend_data(data[variable["name"]])

        # Compute seasonal cycle
        if self.compute_anomalies:
            seasonal_cycle = compute_anomaly(
                data=data, baseline_period=self.baseline_period, 
                mean_groups=self.mean_groups,
                baseline_mean_groups=self.baseline_mean_groups
            )
        else:
            seasonal_cycle = data.groupby(self.mean_groups).mean(dim=["time"])

        time = self.xlabels_from_time(seasonal_cycle)
        seasonal_cycle = seasonal_cycle.stack(
            time=[g.split(".")[-1] for g in self.mean_groups]).reset_index("time")
        seasonal_cycle["time"] = time
        seasonal_cycle = seasonal_cycle.dropna(dim="time", how="any")

        return seasonal_cycle

    def evaluate(self, data_containers):
        
        for variable in self.variables:
                name, pressure_level = variable['name'], variable['pressure_level']
                cycles = {}
                for data_container in data_containers:
                    cycle = self.compute(
                        data_container=data_container, variable=variable  
                    )

                    cycles[data_container.model_label] = \
                        (cycle.time.values, cycle[name].values)
                
                # Colors for each model
                colors = {
                    data_container.model_label: data_container.model_color for data_container in data_containers
                }
                
                if self.linear_trend:
                    trends = {}
                    for label, (x, y) in cycles.items():
                        coeffs = np.polyfit(range(len(x)), y, deg=1)
                        trend_values = np.polyval(coeffs, range(len(x)))
                        # Coeff is per month slope, convert to per decade
                        m = coeffs[0] * 12 * 10
                        trends[label] = (trend_values, m)
                else:
                    trends = {}

                var_name = f"{name}_{pressure_level}Pa" if pressure_level is not None else name
                
                self.timeseries_plotter.plot(
                    model_data=cycles,
                    linear_trend=trends,
                    colors=colors,
                    title="",
                    variable_name=name,
                    xlabel="Time",
                    ylabel=f"{self.timeseries_plotter.cmor_units[name]}",
                    xticks=None,
                    fname=f"{var_name}.png"
                )

def compute_soi(data, base_period, detrend=False):


    if data.lon.max() > 180:
        data = data.assign_coords(
            lon=(data.lon - 180)
        )
    
    # Get Tahiti and Darwin grid points
    lat_tahiti = -17.65
    lon_tahiti = -149.42 
    lat_darwin = -12.46
    lon_darwin = 130.84

    tahiti_data = data.sel(lat=lat_tahiti, lon=lon_tahiti, method="nearest")
    darwin_data = data.sel(lat=lat_darwin, lon=lon_darwin, method="nearest")

    tahiti_anomaly = compute_anomaly(
        tahiti_data, baseline_period=base_period, 
        mean_groups=["time.year", "time.month"], 
        baseline_mean_groups=["time.month"], 
        standardize=True)
    
    darwin_anomaly = compute_anomaly(
        darwin_data, baseline_period=base_period, 
        mean_groups=["time.year", "time.month"], 
        baseline_mean_groups=["time.month"], 
        standardize=True)
    
    soi_index = tahiti_anomaly - darwin_anomaly 

    # Standardize by dividing by the standard deviation of the difference during the baseline period
    soi_index = soi_index / soi_index.std(dim="year")
    
    return soi_index

class SouthernOscillationIndex(Timeseries):
    def __init__(
            self, variables: list[str] = ["psl"], detrend=False, 
            compute_anomalies: bool = True, baseline_period: tuple = None, 
            spectrum: bool = False, plotter_kwargs: dict = None):
        super().__init__(
            variables, 
            detrend=detrend, 
            compute_anomalies=compute_anomalies, 
            baseline_period=baseline_period, 
            plotter_kwargs=plotter_kwargs)
        
        self.spectrum = spectral.welch_psd if spectrum else None

    def compute(self, data_container):
        data = data_container.get_variable_data(name="psl", frequency="monthly", pressure_level=None)
        soi = compute_soi(data, base_period=self.baseline_period, detrend=self.detrend)
        time = [f"{y}-{m:02d}" for y, m in itertools.product(soi.year.values, soi.month.values)]
        soi = soi.stack(time=("year", "month")).reset_index("time")
        # Convert time to datetime objects for plotting
        soi["time"] = time
        return soi

    
    def evaluate(self, data_containers):
        soi_indices = {}
        spectra = {}
        for data_container in data_containers:
            soi_index = self.compute(data_container=data_container)
            
            if self.spectrum:
                fx, fy = self.spectrum(soi_index["psl"].values)
                spectra[data_container.model_label] = (fx, fy)

            soi_indices[data_container.model_label] = (
                soi_index.time.values, soi_index["psl"].values
            )
        
        for label, soi_data in soi_indices.items():
            self.timeseries_plotter.plot(
                model_data={label: soi_data},
                fill="sign",
                colors={label: "k"},
                linewidths={label: 0.75},
                title="",
                variable_name="",
                xlabel="Time",
                ylabel="SOI Index",
                xticks=None,
                fname=f"SOI_{label}.png"
            )

        if self.spectrum:
            self.timeseries_plotter.plot(
                model_data=spectra,
                fname="SOI_Spectrum.png",
                title="",
                variable_name="",
                xlabel="Frequency (1/months)",
                ylabel="Power Spectral Density",
                colors={
                    data_container.model_label: 
                    data_container.model_color 
                    for data_container in data_containers
                },
            )

            

def compute_correlation_matrix(data1: xr.DataArray, data2: xr.DataArray):
    data1 = data1 - data1.mean()
    data2 = data2 - data2.mean()

    correlation = (data1 * data2) / np.sqrt((data1 ** 2).sum() * (data2 ** 2).sum())

    return correlation.values
    
def compute_eof(data: xr.DataArray, n_modes: int = 1,  var_name: str = "psl"):
    # Compute covariance matrix
    data = data - data.mean(dim="time")

    # Flatten lat and lon
    data = data.transpose("time", "lat", "lon", ...)
    x = data[var_name].values

    x = x.reshape(x.shape[0], -1)

    # compute covariance matrix across time dimension
    covariance_matrix = np.cov(x)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select leading modes
    eof_modes = eigenvectors[:, :n_modes] 

    return np.abs(eof_modes)


class ModesOfAnnualVariability(Timeseries):
    def __init__(
            self, variables: list[str], detrend=False, 
            compute_anomalies: bool = False, baseline_period: tuple = None, plotter_kwargs: dict = None):
        super().__init__(variables, detrend=detrend, compute_anomalies=compute_anomalies, baseline_period=baseline_period, plotter_kwargs=plotter_kwargs)


class NorthernAnnularMode:
    def __init__(self, method="Hurrell", plotter_kwargs: dict = None, baseline_period: tuple = ('1981-01-01', '2010-12-31')):
        self.method = method
        self.plotter_kwargs = plotter_kwargs or {}
        self.baseline_period = baseline_period
        

        self.timeseries_plotter = TimeseriesPlotter(**self.plotter_kwargs)

    def compute(self, data_container):
        zg100000 = data_container.get_variable_data(name="zg", frequency="monthly", pressure_level=100000)

        # Select grid points based on method    
        if self.method == "Hurrell":
            if zg100000.lat[0] > zg100000.lat[-1]:  # Check if latitudes are in ascending order
                zg100000_sel = zg100000.sel(lon=slice(90, 220))
            else:
                zg100000_sel = zg100000.sel(lon=slice(90, 220))
            # Average over time to get mean spatial pattern
            #psl_grouped = psl.groupby(["time.year", "time.month"]).mean(dim="time")  
            #psl_grouped = psl_grouped.stack(time=["year", "month"]).reset_index("time")
            


            # loading pattern is obtained from baseline period anomaly
            zg100000_sel_base = zg100000_sel.sel(time=slice(self.baseline_period[0], self.baseline_period[1]))
            zg100000_anomaly = compute_anomaly(
                data=zg100000_sel_base, baseline_period=self.baseline_period,
                mean_groups=["time.year"],
                baseline_mean_groups=None,
                standardize=True
            )
            zg100000_anomaly = zg100000_anomaly.rename({"year": "time"}).reset_index("time")
            time = zg100000_anomaly.time.values

            # create time as years for numpy compatibility
            eof_modes = compute_eof(zg100000_anomaly)


            # Project data onto leading EOF mode to get NAOI index
            # -> first compute anomaly map of psl
            psl_anomaly = compute_anomaly(
                data=psl, baseline_period=self.baseline_period,
                mean_groups=["time.year"],
                baseline_mean_groups=None,
                standardize=True
            )

            psl_anomaly = psl_anomaly.rename({"year": "time"}).reset_index("time")

            # Initialize empty array for NAOI index
            nao_index = np.empty((len(psl_anomaly.lat), len(psl_anomaly.lon)))
            psl_anomaly_x = psl_anomaly.psl.values
            for i in range(len(psl_anomaly.lat)):
                for j in range(len(psl_anomaly.lon)):
                    grid_point_data = psl_anomaly_x[:, i, j]
                    nao_index[i, j] = np.dot(grid_point_data, eof_modes[:, 0])
                    # Normalize by leading eigenvalue to get standardized index
                    nao_index[i, j] = nao_index[i, j] / (eof_modes[:, 0].T @ eof_modes[:, 0])
                   
            # Contour f plot with cartopy projection to the lat lon map of the NAOI index
            output_path = self.plotter_kwargs.get("output_path", ".")
            fpath = f"{output_path}/{data_container.model_label}_NAOI_{self.method}.png"
            lambert_conformal_projection_plot(
                data=xr.DataArray(nao_index, coords=[psl_anomaly.lat, psl_anomaly.lon], dims=["lat", "lon"]),
                central_latitude=55,
                central_longitude=-25,
                extent=(-90, 40, 20, 90),
                fpath=fpath
            )

    
        else:
            raise NotImplementedError(
                f"{self.method} not implemented for NAOI index calculation."
            )

        return time, eof_modes

    def evaluate(self, data_containers):
        eof_modes = {}
        for data_container in data_containers:
            time, eof_mode = self.compute(data_container)
            eof_modes[data_container.model_label] = (range(len(eof_mode)),  eof_mode)

        colors = {
            data_container.model_label: data_container.model_color for data_container in data_containers
        }

        self.timeseries_plotter.plot(
            model_data=eof_modes,
            colors=colors,
            title=f"NAO Index - Method: {self.method}",
            variable_name="",
            xlabel="Time",
            ylabel="EOF Mode 1 Amplitude",
            xticks=time,
            fname=f"{self.method}_eof_timeseries.png"
        )
    

        
class NorthernAtlanticOscillationIndex:
    def __init__(self, method="Hurrell", plotter_kwargs: dict = None, baseline_period: tuple = ('1981-01-01', '2010-12-31')):
        self.method = method
        self.plotter_kwargs = plotter_kwargs or {}
        self.baseline_period = baseline_period


        self.timeseries_plotter = TimeseriesPlotter(**self.plotter_kwargs)

    def compute(self, data_container):
        psl = data_container.get_variable_data(name="psl", frequency="monthly", pressure_level=None)
        # Select grid points based on method    
        if self.method == "Hurrell":
            if psl.lat[0] > psl.lat[-1]:  # Check if latitudes are in ascending order
                psl_sel = psl.sel(lat=slice(80, 20), lon=slice(90, 220))
            else:
                psl_sel = psl.sel(lat=slice(20, 80), lon=slice(90, 220))

            # Average over time to get mean spatial pattern
            #psl_grouped = psl.groupby(["time.year", "time.month"]).mean(dim="time")  
            #psl_grouped = psl_grouped.stack(time=["year", "month"]).reset_index("time")

            psl_sel_anomaly = compute_anomaly(
                data=psl_sel, baseline_period=self.baseline_period,
                mean_groups=["time.year"],
                baseline_mean_groups=None,
                standardize=True
            )
            
            psl_sel_anomaly = psl_sel_anomaly.rename({"year": "time"}).reset_index("time")
            time = psl_sel_anomaly.time.values

            # create time as years for numpy compatibility
            eof_modes = compute_eof(psl_sel_anomaly)


            # Project data onto leading EOF mode to get NAOI index
            # -> first compute anomaly map of psl
            psl_anomaly = compute_anomaly(
                data=psl, baseline_period=self.baseline_period,
                mean_groups=["time.year"],
                baseline_mean_groups=None,
                standardize=True
            )

            psl_anomaly = psl_anomaly.rename({"year": "time"}).reset_index("time")

            # Initialize empty array for NAOI index
            nao_index = np.empty((len(psl_anomaly.lat), len(psl_anomaly.lon)))
            psl_anomaly_x = psl_anomaly.psl.values
            for i in range(len(psl_anomaly.lat)):
                for j in range(len(psl_anomaly.lon)):
                    grid_point_data = psl_anomaly_x[:, i, j]
                    nao_index[i, j] = np.dot(grid_point_data, eof_modes[:, 0])
                    # Normalize by leading eigenvalue to get standardized index
                    nao_index[i, j] = nao_index[i, j] 
                   
            # Contour f plot with cartopy projection to the lat lon map of the NAOI index
            output_path = self.plotter_kwargs.get("output_path", ".")
            fpath = f"{output_path}/{data_container.model_label}_NAOI_{self.method}.png"
            lambert_conformal_projection_plot(
                data=xr.DataArray(nao_index, coords=[psl_anomaly.lat, psl_anomaly.lon], dims=["lat", "lon"]),
                central_latitude=55,
                central_longitude=-25,
                extent=(-90, 40, 20, 90),
                fpath=fpath
            )

    
        else:
            raise NotImplementedError(
                f"{self.method} not implemented for NAOI index calculation."
            )

        return time, eof_modes

    def evaluate(self, data_containers):
        eof_modes = {}
        for data_container in data_containers:
            time, eof_mode = self.compute(data_container)
            eof_modes[data_container.model_label] = (range(len(eof_mode)),  eof_mode)

        colors = {
            data_container.model_label: data_container.model_color for data_container in data_containers
        }

        self.timeseries_plotter.plot(
            model_data=eof_modes,
            colors=colors,
            title=f"NAO Index - Method: {self.method}",
            variable_name="",
            xlabel="Time",
            ylabel="EOF Mode 1 Amplitude",
            xticks=time,
            fname=f"{self.method}_eof_timeseries.png"
        )
        

################################################################################

'''
class ClimateMetric:
    """
    Base class for any climate metric
    """

    fontdict = {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "ytick.labelsize": 12,
        "xtick.labelsize": 12,
    }
    linestyles = ["-", "--", "-.", ":"]
    markerstyles = ["o", "s", "^", "D", "v", "x", "*"]
    units = units = {
        "sea_surface_temperature": "[K]",
        "2m_temperature": "[K]",
        "mean_sea_level_pressure": "[Pa]",
        "specific_humidity": "[kg/kg]",
        "geopotential": "[m^2/s^2]",
        "u_component_of_wind": "[m/s]",
        "v_component_of_wind": "[m/s]",
        "10m_u_component_of_wind": "[m/s]",
        "10m_v_component_of_wind": "[m/s]",
        "sea_ice_cover": "[fraction]",
        "temperature": "[K]",
        "vertical_velocity": "[m/s]",
    }

    def __init__(
        self,
        variables,
        use_lat_weighting="weatherbench",
        latitude=None,
        figsize=(10, 6),
        dpi=150,
        base_period=None,
        output_path=".",
        fontdict=None
    ):  
        
        if fontdict is not None:
            self.fontdict.update(fontdict)
            
        self.variables = variables
        self.use_lat_weighting = use_lat_weighting
        self.latitude = latitude
        self.figsize = figsize
        self.dpi = dpi
        self.base_period = base_period

        self.lat_weights = compute_lat_weights_weatherbench(self.latitude)
        self.lat_weights = self.lat_weights.squeeze(-1)
        self.lat_weights = xr.DataArray(
            self.lat_weights,
            coords=[np.linspace(90, -90, self.latitude)],
            dims=["latitude"],
        )

        self.output_path = output_path

    def rmse(self, data1, data2):
        rmse =  np.sqrt(((data1 - data2) ** 2).mean()).item()
        print(rmse)
        return rmse

    def compute(self, data):
        pass


class SpatialMetric(ClimateMetric):
    """
    Base class for climate metrics that involve spatial data
    """

    def __init__(
        self,
        variables,
        time="average_time",
        x="latitude",
        y="longitude",
        use_lat_weighting="weatherbench",
        latitude=None,
        figsize=(10, 6),
        dpi=150,
        base_period=None,
        output_path=".",
        fontdict=None
    ):
        super().__init__(
            variables,
            use_lat_weighting,
            latitude,
            figsize,
            dpi,
            base_period,
            output_path,
            fontdict=fontdict
        )

        self.time = time
        self.x = x
        self.y = y

    def select_time(self, data, time):
        if time == "annual":
            data = data.mean(dim="time")
        elif time in ["DJF", "MAM", "JJA", "SON"]:
            data = data.sel(time=data["time.season"] == time)
            # take average over selected season
            data = data.mean(dim="time")
        else:
            data = data.sel(time=pd.to_datetime(time), method="nearest")
        return data


class XYPlot(SpatialMetric):
    """
    Base class for climate
    metrics that involve XY plots, e.g. pressure_level over latitude
    """

    def __init__(
        self,
        variables,
        plot_type="contour",
        time="annual",
        x="latitude",
        y="longitude",
        use_lat_weighting="weatherbench",
        latitude=None,
        figsize=(10, 6),
        dpi=150,
        diverging_cmap="bwr",
        sequential_cmap="PuBu",
        base_period=None,
        output_path=".",
        fontdict=None
    ):
        super().__init__(
            variables,
            time=time,
            x=x,
            y=y,
            use_lat_weighting=use_lat_weighting,
            latitude=latitude,
            figsize=figsize,
            dpi=dpi,
            base_period=base_period,
            output_path=output_path,
            fontdict=fontdict
        )
        self.plot_type = plot_type
        self.output_path = output_path + f"/{x}_{y}_plots"
        self.sequential_cmap = sequential_cmap
        self.diverging_cmap = diverging_cmap
        os.makedirs(self.output_path, exist_ok=True)
    

    def visualize_on_ax(
        self, fig, ax, data, variable_name, time, 
        model_label, cmap, norm=None, infotext=""
    ):
        # transpose data such that x is on the horizontal axis and y on the vertical axis
        data = data.transpose(self.y, self.x)

        if self.plot_type == "imshow":
            ax.set_xlabel(self.x, fontsize=self.fontdict["axes.labelsize"])
            ax.set_ylabel(self.y, fontsize=self.fontdict["axes.labelsize"])

            im = ax.imshow(data, cmap="coolwarm",
                           aspect="auto", origin="lower")
            cbar = plt.colorbar(
                im,
                ax=ax,
                label=self.units[variable_name] if variable_name in self.units else "",
            )

            # fontsize for colorbar label
            cbar.ax.yaxis.label.set_size(self.fontdict["axes.labelsize"])
            # cbar tick label size
            cbar.ax.tick_params(labelsize=self.fontdict["axes.xticklabelsize"])

        elif self.plot_type == "variable":
            plot_variable(
                data,
                fname=f"{self.y}_{self.x}_plot_{time}_{variable_name}_{model_label}.png",
                output_path=self.output_path,
                title="",#f"{model_label} - {time}, Variable: {variable_name}",
                ax=None,
                cbar_label=self.units[variable_name]
                if variable_name in self.units
                else "",
                cmap=cmap,
                fontdict=self.fontdict,
                infotext=infotext,
                norm=norm,
            )

            return
        elif self.plot_type == "contour":
                
            CS = ax.contourf(
                data[self.x],
                data[self.y],
                data,
                cmap="coolwarm",
                interpolation="bilinear",
                levels=20,
            )

            CS2 = ax.contour(CS, levels=CS.levels, colors="k")
            plt.gca().set_aspect("auto")
            if self.y == "level":
                ax.invert_yaxis()
                # also invert the data for contour lines
                data.values = data.values[-1::-1, :]
                
            cbar = plt.colorbar(
                CS,
                ax=ax,
                label=self.units[variable_name] if variable_name in self.units else "",
            )
            # fontsize for colorbar label
            cbar.ax.yaxis.label.set_size(self.fontdict["axes.labelsize"])
            # cbar tick label size
            cbar.ax.tick_params(labelsize=self.fontdict["xtick.labelsize"])

            time_title = (
                time if time in ["average_time", "DJF", "MAM", "JJA", "SON"] else "Average"
            )
            ax.set_yticks(data[self.y].values)
            ax.set_xticks(data[self.x].values[::30])
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=self.fontdict["xtick.labelsize"],
            )
        #ax.set_title(
        #    model_label + " - " + time_title + ", Variable: " + variable_name,
        #    fontsize=self.fontdict["axes.titlesize"],
        #)
        output_path = self.output_path + f"/{self.y}_{self.x}_plot_{time}_{variable_name}_{model_label}.png"
        
        plt.savefig(
            output_path,
            bbox_inches="tight",
        )

        plt.close()

    def create_figure(self):
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        return fig, ax

    def evaluate(self, model_containers):
        if any ([model.is_reference for model in model_containers]):
            ground_truth_data_index = [
                i
                for i, model in enumerate(model_containers)
                if model.is_reference
            ][0]
            ground_truth_data = model_containers[ground_truth_data_index].data
            model_reference_label = model_containers[ground_truth_data_index].label
        else:
            ground_truth_data = None
        self.time = self.time if isinstance(self.time, ListConfig) else list(self.time)
        for time in self.time:
            for model_data in model_containers:
                data = model_data.data

                data = self.select_time(data, time)

                # Mean data over all dims not equal to x or y

                dims_to_mean = [
                    dim
                    for dim in data.dims
                    if dim not in [self.x, self.y] and dim != "level"
                ]
                data = data.mean(dim=dims_to_mean)

                if model_data.label.lower() != model_reference_label.lower() and ground_truth_data is not None:
                    print(
                        f"Difference Map {model_data.label} using {model_reference_label}...")
                    gt_data = self.select_time(ground_truth_data, time)
                    gt_data = gt_data.mean(dim=dims_to_mean)
                    diff_data = data - gt_data
                    
                else:
                    diff_data = None

                for var in self.variables:
                    variable_name, lvl = var
                    short_var_name = (
                        variable_name if lvl is None else f"{variable_name}_{lvl}"
                    )
                    print(f"Visualizing XY plot for variable: {short_var_name}")
                    fig, ax = self.create_figure()
                    if lvl is None:
                        plot_data = data[variable_name]
                    else:
                        print(data)
                        plot_data = data[variable_name].sel(level=lvl)
                    self.visualize_on_ax(
                        fig, ax, plot_data, variable_name, time, model_label=model_data.label, cmap=self.sequential_cmap
                    )

                    if diff_data:
                        print(
                            f"Visualizing Difference Map XY plot for variable: {short_var_name}")
                        fig, ax = self.create_figure()
                        if lvl is None:
                            plot_data = diff_data[variable_name]
                            rmse = self.rmse(
                                data[variable_name].values,
                                gt_data[variable_name].values
                            )
                        else:
                            plot_data = diff_data[variable_name].sel(level=lvl)
                            rmse = self.rmse(
                                data[variable_name].sel(level=lvl).values,
                                gt_data[variable_name].sel(level=lvl).values
                            )

                        norm = CenteredNorm(vcenter=0)

                        self.visualize_on_ax(
                            fig,
                            ax,
                            plot_data,
                            short_var_name,
                            time,
                            cmap=self.diverging_cmap,
                            model_label=model_data.label + " diffmap",
                            norm=norm,
                            infotext="Mean=" +
                            f"{float(plot_data.mean().values):.2f}, RMSE={rmse:.2f}",
                        )


class TimeSeries(ClimateMetric):
    """
    Base class for climate metrics that involve time series data
    """

    def __init__(
        self,
        variables,
        use_lat_weighting="weatherbench",
        latitude=121,
        base_period=None,
        figsize=(10, 6),
        dpi=150,
        linewidth=2,
        output_path=".",
    ):
        super().__init__(
            variables,
            use_lat_weighting=use_lat_weighting,
            latitude=latitude,
            base_period=base_period,
            figsize=figsize,
            dpi=dpi,
            output_path=output_path,
        )
        self.linewidth = linewidth

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
        for _, annual_cycle in data.items():
            time_min, time_max = self.extract_min_max_time(annual_cycle)
            min_times.append(time_min)
            max_times.append(time_max)
        min_time = min(min_times)
        max_time = max(max_times)

        return min_time, max_time

    def get_xtick_ids(self, min_time, max_time):
        time_range = pd.date_range(start=min_time, end=max_time, freq="MS")
        xtick_labels = [time.strftime("%b %Y") for time in time_range]
        xtick_ids = list(range(len(xtick_labels)))
        return xtick_ids


class AnnualCycle(TimeSeries):
    """
    Class to compute and visualize the annual cycle of a variable
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_path = self.output_path + "/annual_cycle"
        os.makedirs(self.output_path, exist_ok=True)

    def visualize_to_ax(self, ax, data, label, xtick_labels, color):
        ax.plot(
            data.time.values,
            data.values,
            label=label,
            linewidth=self.linewidth,
            c=color,
        )

    def compute_annual_cycle(self, data):
        return functional.timeseries.compute_annual_cycle(data)

    def compute_kde_timeseries(self, annomalies):
        ygrid_bonds = kernel_density_estimation._get_ygrid_bounds(annomalies)
        kdes = {}
        for model_name, anomalies in annomalies.items():
            kdes[model_name] = kernel_density_estimation.compute_1d_pdf_over_grid(anomalies)

        return kdes
    

    def compute(self, model_containers):
        annual_cycles = {}
        for model_data in model_containers:
            data = model_data.data
            if self.lat_weights is not None:
                print("Applying latitude weighting...")
                data = data.weighted(self.lat_weights).mean(dim="latitude")
            annual_cycle = self.compute_annual_cycle(model_data.data)
            annual_cycles[model_data.label] = annual_cycle


        return annual_cycles

    def evaluate(self, model_containers):
        annual_cycles = self.compute(model_containers)

        xtick_ids = self.get_xtick_ids(*self.get_time_limits(annual_cycles))

        for var in self.variables:
            variable_name, lvl = var
            short_var_name = variable_name if lvl is None else f"{variable_name}_{lvl}"
            print(f"Visualizing annual cycle for variable: {short_var_name}")
            fig, ax = self.create_figure()

            for i, (lbl, annual_cycle) in enumerate(annual_cycles.items()):
                print(f"Plotting annual cycle for model: {lbl}")
                ac = (
                    annual_cycle[variable_name]
                    if lvl is None
                    else annual_cycle[variable_name].sel(level=lvl)
                )
                self.visualize_to_ax(
                    ax,
                    ac,
                    label=model_containers[i].label,
                    xtick_labels=annual_cycle.time.values,
                    color=model_containers[i].data_color,
                )
                ax.set_xlabel("Time")
                ax.set_ylabel(f"{self.units.get(variable_name, '')}")
            plt.grid()
            plt.legend(fontsize=self.fontdict["axes.labelsize"])
            plt.savefig(self.output_path + f"/annual_cycle_{short_var_name}.png")

class AnomalyKDE(TimeSeries):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_path = self.output_path + "/AnomalyKDE"
        os.makedirs(self.output_path, exist_ok=True)

    def compute_kde_timeseries(self, anomalies):
        ygrid_bonds = kernel_density_estimation._get_ygrid_bounds(anomalies.values())
        kdes = {}
        for model_name, anomalies in anomalies.items():
            kdes[model_name] = kernel_density_estimation.compute_1d_pdf_over_grid(
                anomalies, ygrid_bounds=ygrid_bonds)

        return kdes
    
    def compute_anomaly(self, data_containers, var, lvl):
        anomalies = {}
        for container in data_containers:
            data = container.data
            print(data)
            data = data.sel(level=lvl)[var] if lvl is not None else data[var]
            anomalies[container.label] = functional.utils.compute_anomaly(
                data, groupby=["time.year", "time.month"], base_period=self.base_period, 
                detrend=True, mean_groupby=["time.month"], reduce_dims=["time", "latitude", "longitude"]
            )

        return anomalies
    
    def create_figure(self):
        fig = plt.figure(figsize=(10, 6), dpi=self.dpi)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)

        return fig, gs

    def evaluate(self, model_containers):
        

        for var in self.variables:
            variable_name, lvl = var
            anomalies = self.compute_anomaly(model_containers, var, lvl)

            kdes = self.compute_kde_timeseries(
                {
                    n: ano[variable_name]
                    for n, ano in anomalies.items()
                }
            )
        
            short_var_name = variable_name if lvl is None else f"{variable_name}_{lvl}"
            print(f"Visualizing annual cycle KDE for variable: {short_var_name}")

            fig, gs = self.create_figure()
            ax0 = fig.add_subplot(gs[0])
            for i, (lbl, pdf) in enumerate(kdes.items()):
                print(f"Plotting annual cycle KDE for model: {lbl}")
                color = model_containers[i].data_color
                ax0.plot(
                    pdf,
                    self.lat_weights,
                    label=lbl,
                    color=color,
                    linewidth=self.linewidth,
                )
                #ax0.set_title(f"KDE - {short_var_name}")
                ax0.set_xlabel("Density")

            ax1 = fig.add_subplot(gs[1])
            for i, (lbl, ano) in enumerate(anomalies.items()):
                print(f"Plotting annual cycle for model: {lbl}")
                ac = (
                    ano[variable_name]
                    if lvl is None
                    else ano[variable_name].sel(level=lvl)
                )
                self.visualize_to_ax(
                    ax1,
                    ac,
                    label=model_containers[i].label,
                    xtick_labels=ano.time.values,
                    color=model_containers[i].data_color,
                )
            #ax1.set_title(f"Annual Cycle - {short_var_name}")
            ax1.set_xlabel("Time")
            ax1.grid()
            ax1.legend()
            plt.savefig(self.output_path + f"/annual_cycle_kde_{short_var_name}.png")
    

class RadialSpectrum(ClimateMetric):
    def __init__(self, reference_years: list, linewidth=2, **kwargs):
        super().__init__(**kwargs)
        self.reference_years = reference_years
        self.linewidth = linewidth
        self.output_path = self.output_path + "/radial_spectrum"
        os.makedirs(self.output_path, exist_ok=True)

    def visualize_to_ax(
        self, ax, radial_spec, label, color, linestyle, marker, linewidth
    ):
        """
        Plots the radial spectrum of a variable.
        """

        l = np.linspace(1, radial_spec.shape[0], radial_spec.shape[0])
        wavelength = 2 * np.pi * 6371.0 / np.sqrt(l * (l + 1))
        wavelength = list(wavelength)
        ax.loglog(
            wavelength,
            radial_spec,
            linewidth=linewidth,
            color=color,
            label=label,
            linestyle=linestyle,
            marker=marker,
            markevery=20,
        )
        ax.invert_xaxis()
        ax.set_xlabel("Wavelength (km)")
        # Mirror graph 
        ax.grid(which="both", linestyle="-.", linewidth=0.2)

    def compute_radial_spectrum(self, data):
        radial_spectra = {}
        for year in self.reference_years:
            print(f"Computing radial spectrum for year: {year}")
            yearly_data = data.sel(
                time=data.time.dt.year == year, method="nearest")
            yearly_data = yearly_data.values

            radial_spectra[year] = (
                sum(
                    [
                        functional.frequency_domain.compute_radial_spectrum(x)
                        for x in yearly_data
                    ]
                )
                / yearly_data.shape[0]
            )

        return radial_spectra

    def compute(self, model_containers, variable_name, lvl):
        radial_spectra = {}
        for model_data in model_containers:
            data = model_data.data
            data = (
                data[variable_name]
                if lvl is None
                else data[variable_name].sel(level=lvl)
            )

            radial_spectrum = self.compute_radial_spectrum(data)
            radial_spectra[model_data.label] = radial_spectrum

        return radial_spectra

    def evaluate(self, model_containers):
        for var in self.variables:
            variable_name, lvl = var
            radial_spectra = self.compute(model_containers, variable_name, lvl)

            short_var_name = (
                variable_name if lvl is None else f"{variable_name} ({lvl}hPa)"
            )
            print(
                f"Visualizing radial spectrum for variable: {short_var_name}")
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            plt.rcParams.update(self.fontdict)
            for i, (lbl, spectra) in enumerate(radial_spectra.items()):
                for j, (year, spectrum) in enumerate(spectra.items()):
                    color = list(model_containers[i].data_color)
                    # Slightly lighten the color for better visibility
                    linestyle = self.linestyles[j % len(self.linestyles)]
                    marker = self.markerstyles[j % len(self.markerstyles)]
                    self.visualize_to_ax(
                        ax,
                        spectrum,
                        label=f"{lbl} ({year})",
                        color=color,
                        linewidth=self.linewidth,
                        linestyle=linestyle,
                        marker=marker,
                    )
            ax.set_xlabel("Wavenumber")
            ax.set_ylabel("Power Spectral Density")
            ax.legend(fontsize=self.fontdict["axes.labelsize"])
            output_path = self.output_path + \
                f"/radial_spectrum_{short_var_name}.png"
            plt.savefig(output_path)
            plt.close()


class OceanicNinoIndex(TimeSeries):
    def __init__(
        self,
        variables,
        analyze_spectrum=False,
        use_lat_weighting="weatherbench",
        latitude=121,
        base_period=None,
        figsize=(10, 6),
        dpi=150,
        linewidth=2,
    ):
        super().__init__(
            variables, use_lat_weighting, latitude, base_period, figsize, dpi, linewidth
        )
        self.analyze_spectrum = analyze_spectrum
        self.output_path = self.output_path + "/nino_index"
        os.makedirs(self.output_path, exist_ok=True)

    def compute_nino_index(self, data):
        return functional.timeseries.compute_oni_index(
            data, base_period=self.base_period
        )


class SouthernOscillationIndex(TimeSeries):
    def __init__(
        self,
        variables,
        analyze_spectrum=False,
        use_lat_weighting="weatherbench",
        latitude=121,
        base_period=None,
        figsize=(10, 6),
        dpi=150,
        linewidth=2,
        output_path=".",
    ):
        super().__init__(
            variables, use_lat_weighting,
            latitude, base_period, figsize,
            dpi, linewidth, output_path
        )
        self.analyze_spectrum = analyze_spectrum
        self.output_path = self.output_path + "/soi_index"
        os.makedirs(self.output_path, exist_ok=True)

    def compute_soi_index(self, data):
        return functional.timeseries.compute_southern_oscillation_index(
            data, base_period=self.base_period
        )

    def evaluate(self, model_containers):
        soi_indices = {}
        for model_data in model_containers:
            soi_index = self.compute_soi_index(model_data.data)
            soi_indices[model_data.label] = soi_index

            output_path = self.output_path + \
                "/southern_oscillation_index_{}.png".format(
                    model_data.label
                )
            
            xticks_mult = get_xlabel_multiplier(len(soi_index.time.values))
            xtick_labels = soi_index.time.values[::xticks_mult]
            xtick_labels = [f"{year}-{month}" for year, month in xtick_labels]
            xticks = [
                list(range(0, len(soi_index.time.values), xticks_mult)),
                xtick_labels,
            ]
            plot_timeseries(
                x=soi_index,
                xticks=xticks,
                title="",
                ylabel="SOI Index",
                xlabel="Time",
                label=model_data.label,
                figsize=self.figsize,
                dpi=self.dpi,
                linewidth=self.linewidth,
                output_path=output_path,
                fill="positive_negative",
                fontdict=self.fontdict,
            )

        if self.analyze_spectrum:
            frequencies = {}
            for model_label, soi_index in soi_indices.items():
                print(f"Analyzing SOI PSD for model: {model_label}")
                freq_domain = functional.frequency_domain.welch_psd(soi_index)
                frequencies[model_label] = freq_domain

            output_path = self.output_path \
                + "/southern_oscillation_index_spectra.png"

            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            plt.rcParams.update(self.fontdict)
            for i, (lbl, freq_domain) in enumerate(frequencies.items()):
                color = list(model_containers[i].data_color)
                ax.plot(
                    freq_domain[0],
                    freq_domain[1],
                    label=lbl,
                    color=color,
                    linewidth=self.linewidth,
                )
            ax.set_xlabel("Frequency (1/months)")
            ax.set_ylabel("Power Spectral Density")
            ax.legend()
            ax.grid()
            plt.savefig(output_path)
            plt.close()


class CorrelationAnalysis(ClimateMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_correlation(self, data_x, data_y):
        return data_x.corr(data_y, dim=['latitude', 'longitude']) 

    def evaluate(self, model_containers):
        print("-" * 72)
        print()

        output_path = self.output_path + "/correlation_analysis"
        os.makedirs(output_path, exist_ok=True)

        for model in model_containers:
            print(f"### Processing model: {model.label} ###")
            data = model.data

            for var_x, var_y in self.variables:
                print(f"-> [{var_x} / {var_y}]")
                data_x = data[var_x].mean(dim="time")
                data_y = data[var_y].mean(dim="time")

                correlation_coeffs = self.compute_correlation(data_x, data_y)

                print(f"-> Visualizing correlation for [{var_x} / {var_y}]")
                fig, ax = self.create_figure()
                plot_variable(
                    correlation_coeffs,
                    fname=f"correlation_{var_x}_vs_{var_y}_{model.label}.png",
                    output_path=output_path,
                    title=f"",
                    ax=ax,
                    cbar_label="Correlation Coefficient",
                    cmap="coolwarm",
                    fontdict=self.fontdict,
                )
                print("")


class RegressionAnalysis(ClimateMetric):
    def __init__(
            self, 
            variables,
            use_lat_weighting="weatherbench", 
            latitude=None, 
            figsize=(10, 6), 
            dpi=150, 
            base_period=None, 
            output_path="."
    ):
        """_summary_

        Args:
            variables (_type_): Pair of variables to regress.
            use_lat_weighting (str, optional): _description_. Defaults to "weatherbench".
            latitude (_type_, optional): _description_. Defaults to None.
            figsize (tuple, optional): _description_. Defaults to (10, 6).
            dpi (int, optional): _description_. Defaults to 150.
            base_period (_type_, optional): _description_. Defaults to None.
            output_path (str, optional): _description_. Defaults to ".".
        """
        super().__init__(variables, use_lat_weighting, latitude, figsize,       
                         dpi, base_period, output_path)
        

    def create_figure(self):
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        return fig, ax
        
    def regress_variables(self, data_x, data_y):
        return utils.compute_regression_coefficients_2d(data_x, data_y)

    def evaluate(self, model_containers):
        print("-" * 72)
        print()

        output_path = self.output_path + "/regression_analysis"
        os.makedirs(output_path, exist_ok=True)

        for model in model_containers:
            print(f"### Processing model: {model.label} ###")
            data = model.data

            for var_x, var_y in self.variables:
                print(f"-> [{var_x} / {var_y}]")
                data_x = data[var_x].mean(dim=["time"])
            
                data_y = data[var_y].mean(dim=["time"])

                regression_coeffs = self.regress_variables(data_x, data_y)

                print(f"-> Visualizing regression slope for [{var_x} / {var_y}]")

            infotext = f"slope={regression_coeffs['slope'].mean():.2f}\n" + \
                       f"intercept={regression_coeffs['intercept'].mean():.2f}\n" + \
                       f"p_value={regression_coeffs['p_value'].mean():.2f}\n" + \
                       f"r_value={regression_coeffs['r_value'].mean():.2e}"
            
            norm = TwoSlopeNorm(vcenter=0)
            plot_variable(
                regression_coeffs['slope'],
                fname=f"regression_{var_x}_vs_{var_y}_{model.label}.png",
                output_path=output_path,
                title="",
                ax=None,
                cbar_label="Regression Slope",
                cmap="coolwarm",
                fontdict=self.fontdict,
                infotext=infotext,
                norm=norm,
            )
            print("")
        '''