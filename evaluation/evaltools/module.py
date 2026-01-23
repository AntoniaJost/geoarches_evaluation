import xarray as xr
from glob import glob
from hydra.utils import instantiate
import numpy as np
from metrics import module as metric_modules


class GeoClimate:
    def __init__(self, data, metric_cfgs, output_path="."):
        """
        Initializes the GeoClimate evaluation module with data containers and metrics.
        Parameters:
        data (list): List of dictionaries containing data container configurations.
        metrics (list): List of metric configuration dictionaries.
        """

        self.output_path = output_path
        self._init_data(data)
        self._init_metrics(metric_cfgs=metric_cfgs)

    def _init_data(self, data):
        self.data_containers = []
        for d_name, v in data.items():

            print("Adding data for:", d_name)
        
            container = DataContainer(**v)

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


class CmorDataContainer:
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
            "zg": "geopotential_height"}
    

    def __init__(
            self, label, path_to_monthly_data=None, path_to_daily_data=None, 
            variable_names=None):
        """
        This container loads cmorized data and provides functionality 
        to load and yield data. 

        Returns:
            _type_: _description_
        """

        print("#" * 72)
        print("Initializing CmorDataContainer for ", label)
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

        print("Initialized CmorDataContainer for ", label)
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