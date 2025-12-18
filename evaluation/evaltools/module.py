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

            container._load_data()
            self.data_containers.append(container)

    def _init_metrics(self, metric_cfgs):
        self.metric_objects = {}

        metric_cfgs = metric_cfgs["metrics"]
        for metric_name, metric_cfg in metric_cfgs.items():
            print("Adding metric:", metric_name)
            metric = getattr(metric_modules, metric_cfg["target"])(
                output_path=self.output_path,
                **metric_cfg["params"]
            )
            self.metric_objects[metric_name] = metric

    def evaluate(self, target_metrics=None):
        if target_metrics is None:
            metrics = self.metric_objects
        else:
            print(self.metric_objects)
            metrics = {m: self.metric_objects[m] for m in target_metrics}

        for metric in metrics.values():
            print(f"Evaluating {metric.__class__.__name__}")
            metric.evaluate(self.data_containers)


class DataContainer:
    """
    Class to load and hold climate data from NetCDF files
    """

    def __init__(
        self,
        path,
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

        self.path = path
        self.label = label
        self.filename_filters = filename_filters
        self.data_color = data_color
        self.dimension_indexers = dimension_indexers
        self.time_range = time_range

    def _load_data(self):
        fpaths = glob(self.path + "/*.nc")
        fpaths.sort()
        if self.filename_filters is not None:
            fpaths = [f for f in fpaths if any(nf in f for nf in self.filename_filters)]

        print("Opening data from:", self.path, " ...", end=" ")
        if "era5" in self.path.lower() and len(self.filename_filters) > 1:
            # The times are not monotonically increasing in ERA5 data
            data = xr.open_mfdataset(
                fpaths,
                combine="nested",
            )
        else:
            data = xr.open_mfdataset(fpaths, combine="by_coords")
        print("Done")

        if data.latitude[0] < data.latitude[-1]:  # if latitude is descending
            data["latitude"] = data.latitude[::-1]

        data = data.roll(
            longitude=-len(data.longitude) // 2, roll_coords=False
        )  # Roll longitude to match data

        if self.dimension_indexers is not None:
            data = data.rename(**self.dimension_indexers)

        if self.time_range is not None:
            data = data.sel(
                time=slice(self.time_range["start"], self.time_range["end"])
            )

        self.data = data
