from cProfile import label
from typing import List
from omegaconf import OmegaConf
import xarray as xr
from glob import glob
from hydra.utils import instantiate
import numpy as np
from metrics import module as metric_modules

from plot.modules import SpatialPlotter

class GeoClimate:
    def __init__(self, data, metric_cfgs, output_path="."):
        """
        Initializes the GeoClimate evaluation module with data containers and metrics.
        Parameters:
        data (list): List of dictionaries containing data container configurations.
        metrics (list): List of metric configuration dictionaries.
        """

        self.output_path = output_path
        self._init_data(data["models"])
        self._init_metrics(metric_cfgs=metric_cfgs)


    def _init_data(self, data):
        self.data_containers = []
        for d_name, v in data.items():

            print("Adding data for:", d_name)
        
            container = CMORDataContainer(**v)

            self.data_containers.append(container)

    def _init_metrics(self, metric_cfgs):
        self.metric_objects = {}

        metric_cfgs = metric_cfgs["metrics"]
        for metric_name, metric_cfg in metric_cfgs.items():
            print("Adding metric:", metric_name)

            # preprend global output path to metric output path
            metric_cfg["plotter_kwargs"]["output_path"] = \
                self.output_path + "/" + metric_cfg["plotter_kwargs"].get("output_path", "")
            metric_cfg = OmegaConf.to_container(cfg=metric_cfg, resolve=True)
            metric = instantiate(metric_cfg)
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
            "ta": "air_temperature",
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
            self, model_label, grid_type='gn', path_to_monthly_data=None, path_to_daily_data=None, 
            variable_names=None, is_reference: bool = False, roll_longitude: bool = True, color='k',
            period=None
        ):
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

        self.period = period
        self.grid_type = grid_type
        self.roll_longitude = roll_longitude
        if path_to_daily_data is not None:
            if not isinstance(path_to_daily_data, str) and path_to_daily_data is not None:
                self.path_to_daily_data = list(path_to_daily_data)
            else:
                self.path_to_daily_data = path_to_daily_data

            self.load_daily_data()

        if path_to_monthly_data is not None:
            if not isinstance(path_to_monthly_data, str) and path_to_monthly_data is not None:
                self.path_to_monthly_data = list(path_to_monthly_data)
            else:
                self.path_to_monthly_data = path_to_monthly_data
            self.load_monthly_data()

        if variable_names is not None:
            self.variable_names = variable_names

        self.is_reference = is_reference

        self.model_label = model_label
        self.model_color = color

        print("Initialized CmorDataContainer for ", self.model_label)
        print("#" * 72)

    def load_data(self, path, var_name):
        if isinstance(path, list):
            fpaths = []
            for p in path:
                fpaths += glob(p + f"/{var_name}/" + "/**/*.nc", recursive=True)
        else:
            fpaths = glob(path + f"/{var_name}/" + "/**/*.nc", recursive=True)
            
        fpaths.sort()
        if fpaths == []:
            print(f"!!! No files found for variable {var_name} in path {path}/{var_name}/")
            return None
        
        print(f"... {var_name} from:", fpaths, " ...", end=" ")
        var = []
        for f in fpaths:
            var.append(xr.open_mfdataset(f, combine="by_coords"))
        if len(var) > 1:
            var = xr.concat(var, dim="member")
            var_mean = var.mean("member")
            var_std = var.std("member")
            # Combine into a single dataset with mean and std
            var = xr.concat([var_mean, var_std], dim="stat").assign_coords(stat=["mean", "std"])

        else:
            var = var[0]

        if self.period is not None:
            var = var.sel(
                time=slice(self.period[0], self.period[1])
            )
        if self.roll_longitude:
            var = var.roll(
                lon=-len(var.lon) // 2, roll_coords=False
            )  # Roll longitude to match data
        print("Done")
        return var

    def load_monthly_data(self):
        """
            Loads monthly data.
        """
        assert self.path_to_monthly_data is not None, \
        "path_to_monthly_data must be provided to load monthly data."

        print("--> Loading monthly data from:", self.path_to_monthly_data)

        # Data is given per variable
        data_vars = {}
        for var_short, _ in self.variable_names.items():
            data_vars[var_short] = self.load_data(self.path_to_monthly_data, var_short)

        print("--> Finished loading monthly data.")
        
        
        self.monthly_data = data_vars

    def load_daily_data(self):
        """
            Loads daily data.
        """

        assert self.path_to_daily_data is not None, \
        "path_to_daily_data must be provided to load daily data."

        print("--> Loading daily data from:", self.path_to_daily_data)

        # Data is given per variable
        data_vars = {}
        for var_short, _ in self.variable_names.items():

            data_vars[var_short] = self.load_data(self.path_to_daily_data, var_short)
  
        print("--> Finished loading daily data.")
        self.daily_data = data_vars

    def yield_data(self, data_arr):
        if "stat" in data_arr.coords and "mean" in data_arr.coords["stat"]:
            return data_arr.sel(stat="mean", drop=True)
        else:
            return data_arr

    def get_variable_data(self, name, pressure_level=None, frequency="monthly"):
        """
        Yields data for the specified variable name and frequency.
        Parameters:
        name (str): Name of the variable to retrieve.
        frequency (str): Frequency of the data ("monthly" or "daily").
        Returns:
        xarray.DataArray: Data for the specified variable.
        """

        if frequency == "monthly":
            assert hasattr(self, "monthly_data"), \
            "Monthly data not loaded. Please load monthly data first."
            data = self.yield_data(self.monthly_data[name])
        elif frequency == "daily":
            assert hasattr(self, "daily_data"), \
            "Daily data not loaded. Please load daily data first."
            data = self.yield_data(self.daily_data[name])
        else:
            raise ValueError("Frequency must be either 'monthly' or 'daily'.")
        
        if pressure_level is not None:
            data = data.sel(plev=pressure_level)

        return data


# Class that calculates temporally averaged data from CMORized NetCDF files
# Further the data is averaged over spatial dimensions that are not specified 
# as plotting dimensions


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