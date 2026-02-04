from cProfile import label
from typing import List
import xarray as xr
from glob import glob
from hydra.utils import instantiate
import numpy as np
from metrics import module as metric_modules

from plot.modules import SpatialPlotter

class GeoClimate:
    def __init__(self, data, metric_cfgs, output_path=".",  container_type="cmorized"):
        """
        Initializes the GeoClimate evaluation module with data containers and metrics.
        Parameters:
        data (list): List of dictionaries containing data container configurations.
        metrics (list): List of metric configuration dictionaries.
        """

        if container_type == "cmorized":
            self.container_class = CMORDataContainer
        else:
            self.container_class = DataContainer

        self.output_path = output_path
        self._init_data(data)
        self._init_metrics(metric_cfgs=metric_cfgs)


    def _init_data(self, data):
        self.data_containers = []
        for d_name, v in data.items():

            print("Adding data for:", d_name)
        
            container = self.container_class(**v)

            container._load_data_from_paths()
            self.data_containers.append(container)

    def _init_metrics(self, metric_cfgs):
        self.metric_objects = {}

        metric_cfgs = metric_cfgs["metrics"]
        for metric_name, metric_cfg in metric_cfgs.items():
            print("Adding metric:", metric_name)
            #metric = getattr(metric_modules, metric_cfg["target"])(
            #    output_path=self.output_path,
            #    **metric_cfg["params"]
            #)
            metric = instantiate(metric_cfg, output_path=self.output_path)
            self.metric_objects[metric_name] = metric


    def evaluate(self, target_metrics=None):
        if target_metrics is None:
            metrics = self.metric_objects
        else:
            print(target_metrics)
            print(self.metric_objects.keys())
            metrics = {m: self.metric_objects[m] for m in target_metrics}

        for metric in metrics.values():
            print(f"Evaluating {metric.__class__.__name__}")
            metric.evaluate(self.data_containers)


class CMORDataContainer:
    
    variable_names = {   
            "hus": "specific_humidity",
            "psl": "sea_level_pressure",
            "ta:": "air_temperature",
            "tas": "surface_air_temperature",
            "tos": "sea_surface_temperature",
            "ua": "eastward_wind",
            "uas": "eastward_near_surface_wind",
            "vas": "northward_near_surface_wind",
            "va": "northward_wind",
            "zg": "geopotential_height",
            "siconc": "sea_ice_cover",
            "wap": "vertical_velocity",
    }
    

    def __init__(
            self, model_label, path_to_monthly_data=None, path_to_daily_data=None, 
            variable_names=None):
        """
        This container loads cmorized data and provides functionality 
        to load and yield data. 

        Returns:
            _type_: _description_
        """

        print("#" * 72)
        print("Initializing CmorDataContainer for ", model_label)
        assert path_to_monthly_data is not None or path_to_daily_data is not None, \
        "At least one of path_to_monthly_data or path_to_daily_data must be provided."

        if path_to_daily_data is not None:
            self.load_daily_data()
        if path_to_monthly_data is not None:
            self.load_monthly_data()

        if variable_names is not None:
            self.variable_names = variable_names
        
        self.path_to_monthly_data = path_to_monthly_data
        self.path_to_daily_data = path_to_daily_data

        self.model_label = model_label
        print("Initialized CmorDataContainer for ", self.model_label)
        print("#" * 72)


    def load_monthly_data(self):
        """
            Loads monthly data.
        """
        assert self.path_to_monthly_data is not None, \
        "path_to_monthly_data must be provided to load monthly data."

        print("--> Loading monthly data from:", self.path_to_monthly_data)


        fpaths = glob(self.path_to_monthly_data + "/*.nc", recursive=True)
        fpaths.sort()

        # Data is given per variable
        data_vars = {}
        for var_short, var_name in self.variable_names.items():
            var_fpaths = [f for f in fpaths if f"/{var_short}/" in f]
            print(f"... {var_name} from:", var_fpaths, " ...", end=" ")
            data_vars[var_name] = xr.open_mfdataset(var_fpaths, combine="by_coords")
            print("Done")
        print("--> Finished loading monthly data.")
        
        
        self.monthly_data = data_vars

    def load_daily_data(self):
        """
            Loads daily data.
        """

        assert self.path_to_daily_data is not None, \
        "path_to_daily_data must be provided to load daily data."

        print("--> Loading daily data from:", self.path_to_daily_data)
        fpaths = glob(self.path_to_daily_data + "/*.nc", recursive=True)
        fpaths.sort()

        # Data is given per variable
        data_vars = {}
        for var_short, var_name in self.variable_names.items():
            var_fpaths = [f for f in fpaths if f"/{var_short}/" in f]
            print(f"... {var_name} from:", var_fpaths, " ...", end=" ")
            data_vars[var_name] = xr.open_mfdataset(var_fpaths, combine="by_coords")
            print("Done")

        print("--> Finished loading daily data.")
        self.daily_data = data_vars

    def get_variable_data(self, variable_name, frequency="monthly"):
        """
        Yields data for the specified variable name and frequency.
        Parameters:
        variable_name (str): Name of the variable to retrieve.
        frequency (str): Frequency of the data ("monthly" or "daily").
        Returns:
        xarray.DataArray: Data for the specified variable.
        """

        if frequency == "monthly":
            assert hasattr(self, "monthly_data"), \
            "Monthly data not loaded. Please load monthly data first."
            return self.monthly_data[variable_name]
        elif frequency == "daily":
            assert hasattr(self, "daily_data"), \
            "Daily data not loaded. Please load daily data first."
            return self.daily_data[variable_name]
        else:
            raise ValueError("Frequency must be either 'monthly' or 'daily'.")
    


# Class that calculates temporally averaged data from CMORized NetCDF files
# Further the data is averaged over spatial dimensions that are not specified 
# as plotting dimensions


def annual_mean(data, time_dim='time', year=None):
    if year is not None:
        data = data.sel({f'{time_dim}.year': year})
        return data.mean(time_dim)
    else:
        return data.mean(time_dim)

def seasonal_mean(data,  season=None, time_dim='time'):
    if season is not None:
        data = data.sel({f'{time_dim}.season': season})
        return data.mean(time_dim)
    else:
        return data.mean(time_dim)
    
def instantaneous(data, time, time_dim='time'):
    return data.sel({time_dim: time}, nearest=True, drop=True)

class XYMaps:
    def __init__(self, xdim, ydim, temporal_selection: list = ['annual']):
        self.xdim = xdim
        self.ydim = ydim

        self.temporal_selection = temporal_selection

        print("Initializing XYMaps Plotter")
        from module import SpatialPlotter
        self.plotter = SpatialPlotter(xdim=self.xdim, ydim=self.ydim)
        
    def compute(self, data_container: CMORDataContainer, temporal_dim, variable_name: str, frequency: str = "monthly"):
        """

        Compute spatially averaged data for given temporal dimension and variable.

        Args:
            data_container (CMORDataContainer): _description_
            temporal_dim (_type_): _description_
            variable_name (str): _description_
            frequency (str, optional): _description_. Defaults to "monthly".

        Returns:
            _type_: _description_
        """
        data = data_container.get_variable_data(variable_name, frequency)

        if temporal_dim == 'annual':
            data = annual_mean(data)
        elif temporal_dim in ['DJF', 'MAM', 'JJA', 'SON']:
            data = seasonal_mean(data, season=temporal_dim)
        else: 
            data = instantaneous(data, time=temporal_dim)    
        
        # Take spatial means
        data = data.mean(
            dim=[d for d in data.dims if d not in [self.xdim, self.ydim]]
        )

        return data
    
    def evaluate(self, data_containers: CMORDataContainer):
        for ts in self.temporal_selection:
            print(f"Computing {self.xdim}-{self.ydim} map for temporal selection: {ts}")
            print("-" * 72)
            # Do computation here
            for data_container in data_containers:
                data = self.compute(
                    data_container,
                    temporal_dim=ts,
                    variable_name='surface_air_temperature',
                    frequency='monthly'
                )


    
                
            
        


class DataContainer:
    """
    Class to load and hold climate data from NetCDF files
    """

    def __init__(
        self,
        path_list,
        label,
        data_color=np.zeros(
            3,
        )
        + np.random.rand(
            3,
        ),
        filename_filters=None,
        dimension_indexers=None,
        time_range=None,
        assign_coords: dict =None,
        is_reference: bool = False,
    ):
        """
        Initializes the DataContainer with the given parameters.
        Parameters:
        path (str): Path to the directory containing NetCDF files.
        label (str): Label for the dataset.
        filename_filters (list): List of strings to filter filenames.
        data_color (str): Color associated with the dataset for e.g. line plots.
        dimension_indexers (dict, optional): Dictionary to rename dimensions in the dataset.
        """

        self.path_list = list(path_list)
        self.label = label
        self.filename_filters = filename_filters
        self.data_color = data_color
        self.dimension_indexers = dimension_indexers
        self.time_range = time_range
        self.assign_coords = assign_coords
        self.is_reference = is_reference

    def _load_data_from_path(self, path):
        fpaths = glob(path + "/*.nc")
        fpaths.sort()
        if self.filename_filters is not None:
            fpaths = [f for f in fpaths if any(nf in f for nf in self.filename_filters)]

        print("Opening data from:", path, " ...", end=" ")
        data = xr.open_mfdataset(fpaths, combine="by_coords")
        print("Done")

        if data.latitude[0] < data.latitude[-1]:  # if latitude is descending
            data["latitude"] = data.latitude[::-1]

        data = data.roll(
            longitude=-len(data.longitude) // 2, roll_coords=False
        )  # Roll longitude to match data

        if self.dimension_indexers is not None:
            print(data)
            data = data.rename(**self.dimension_indexers)
            print(data)

        if self.assign_coords is not None:
            for key, value in self.assign_coords.items():
                print(f"Assigning coords {key}: {value}")
                data[key] = value

        if self.time_range is not None:
            data = data.sel(
                time=slice(self.time_range["start"], self.time_range["end"])
            )

        return data
    
    def _load_data_from_paths(self):
        datasets = [self._load_data_from_path(p) for p in self.path_list]

        if len(datasets) == 1:
            self.data = datasets[0]
            return
        else:
            print("Calculating member mean from multiple paths...")
            self.data = xr.concat(datasets, dim="member").mean("member")
            print("Calculating member std from multiple paths...")
            self.std = xr.concat(datasets, dim="member").std("member")
            return