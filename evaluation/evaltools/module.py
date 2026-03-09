from omegaconf import OmegaConf
import xarray as xr
from glob import glob
from hydra.utils import instantiate
import numpy as np

# ---------------------------------------------------------------------------
# Dask chunk sizes per temporal frequency.
# Chunks are chosen so each chunk fits comfortably in memory and enables
# parallel I/O across files.  Spatial dimensions are left unchunked because
# all metrics operate on the full lat/lon grid at once.
# ---------------------------------------------------------------------------
_CHUNK_SIZES = {
    "monthly": {"time": 120},   # ~10 years of monthly data
    "daily":   {"time": 365},   # ~1 year of daily data
}


def _normalize_path(path):
    """Return *path* as a list when it is not already a string."""
    return path if isinstance(path, str) else list(path)


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
            period=None, nlat=181, nlon=360
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
        self.nlat = nlat
        self.nlon = nlon

        # Per-variable lazy caches: populated on first access in get_variable_data.
        # Use None sentinel so missing variables can be distinguished from unloaded ones.
        self._monthly_cache = {}
        self._daily_cache = {}

        # Store paths only – data is loaded on demand (see get_variable_data).
        self.path_to_daily_data = None
        self.path_to_monthly_data = None

        if path_to_daily_data is not None:
            self.path_to_daily_data = _normalize_path(path_to_daily_data)

        if path_to_monthly_data is not None:
            self.path_to_monthly_data = _normalize_path(path_to_monthly_data)

        if variable_names is not None:
            self.variable_names = variable_names

        self.is_reference = is_reference
        self.model_label = model_label
        self.model_color = color

        print("Initialized CmorDataContainer for ", self.model_label)
        print("#" * 72)

    def load_data(self, path, var_name, frequency="monthly"):
        """
        Open NetCDF files for *var_name* as a dask-backed lazy dataset.

        Each file under *path*/{var_name}/**/*.nc is treated as one ensemble
        member.  When multiple members exist, the dataset is concatenated along
        a new ``stat`` dimension containing the member mean and std.

        The returned dataset is **lazy** (dask-backed): no data is read from
        disk until an explicit ``.compute()`` or ``.values`` access occurs.
        """

        fpaths = []
        if isinstance(path, list):
            for p in path:
                # Appending as each path is globbed allows for multiple ensemble members across different paths.
                fpaths.append(glob(p + f"/{var_name}/" + "/**/*.nc", recursive=True))
        else:
            fpaths.append(glob(path + f"/{var_name}/" + "/**/*.nc", recursive=True))

  
        if not fpaths:
            print(f"!!! No files found for variable {var_name} in path {path}/{var_name}/")
            return None

        chunks = _CHUNK_SIZES.get(frequency, {"time": 120})
        print(f"... {var_name}: {len(fpaths)} file(s), chunks={chunks} ...", end=" ")

        # Open each file lazily with dask chunks.
        # Each file is one ensemble member; open_mfdataset handles multi-file
        # members internally via combine="by_coords".
        member_datasets = [
            xr.open_mfdataset(f, combine="by_coords", chunks=chunks)
            for f in fpaths
        ]

        if len(member_datasets) > 1:
            # Concatenate members lazily, then compute mean + std along that dim.
            var = xr.concat(member_datasets, dim="member")
            if var_name == "tos":
                var[var_name] = var[var_name] + 273.15
            # These reductions stay lazy (dask graph nodes).
            var_mean = var.mean("member")
            var_std  = var.std("member")
            var = xr.concat([var_mean, var_std], dim="stat").assign_coords(
                stat=["mean", "std"]
            )
        else:
            var = member_datasets[0]
            if var_name == "tos":
                var[var_name] = var[var_name] + 273.15

        # Period selection and longitude roll are both lazy operations.
        if self.period is not None:
            var = var.sel(time=slice(self.period[0], self.period[1]))

            #if "MPI-ESM" in self.model_label:
            #    print(var[var_name].values)
        if self.roll_longitude:
            var = var.roll(lon=-len(var.lon) // 2, roll_coords=False)

        if var.lat[0] < var.lat[-1]:
            # If latitudes are in ascending order, flip to descending (common in CMIP data).
            var = var.reindex(lat=var.lat[::-1])

        # Make sure that the variable is interpolated to the expected grid (if not already).
        # I.e. interpolate lat and lon to self.nlat x self.nlon grid.  
        # This ensures that all variables are on the same grid, which is important for some metrics.
        # This is a no-op if the data is already on the target grid.
        var = self._interpolate_to_target_grid(var) 

        print("Done (lazy)")
        return var

    def _interpolate_to_target_grid(self, ds):
        """
        Interpolate *ds* so that its ``lat`` and ``lon`` dimensions have exactly
        ``self.nlat`` and ``self.nlon`` points respectively.

        If the dataset already has the correct number of points on both axes the
        dataset is returned unchanged (no-op).  Otherwise linear interpolation
        is applied via :func:`xarray.Dataset.interp`.

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset with ``lat`` and ``lon`` coordinate dimensions.

        Returns
        -------
        xr.Dataset
            Dataset on the target grid.
        """
        current_nlat = ds.sizes.get("lat", None)
        current_nlon = ds.sizes.get("lon", None)

        if current_nlat == self.nlat and current_nlon == self.nlon:
            return ds  # already on target grid – nothing to do

        # Build target coordinate arrays that span the same range as the
        # source coordinates so we do not extrapolate.
        target_lat = np.linspace(float(ds.lat[0]), float(ds.lat[-1]), self.nlat)
        target_lon = np.linspace(float(ds.lon[0]), float(ds.lon[-1]), self.nlon)

        ds = ds.interp(lat=target_lat, lon=target_lon, method="linear")
        return ds

    def _load_single_variable(self, path, var_name, frequency):
        """Load *var_name* lazily and store in the appropriate cache."""
        ds = self.load_data(path, var_name, frequency=frequency)
        return ds

    def preload_all(self, frequency="monthly"):
        """
        Eagerly preload all known variables for *frequency*.

        This is equivalent to the behaviour of the old ``load_monthly_data`` /
        ``load_daily_data`` methods.  Calling it is optional; variables are
        also loaded automatically on the first ``get_variable_data`` call.
        """
        path = self.path_to_monthly_data if frequency == "monthly" else self.path_to_daily_data
        assert path is not None, f"No {frequency} data path configured."
        print(f"--> Pre-loading all {frequency} variables from: {path}")
        for var_short in self.variable_names:
            self.get_variable_data(var_short, frequency=frequency)
        print(f"--> Finished pre-loading {frequency} data.")

    # Keep old method names for backward compatibility.
    def load_monthly_data(self):
        """Pre-load all monthly variables (optional - variables load lazily by default)."""
        self.preload_all("monthly")

    def load_daily_data(self):
        """Pre-load all daily variables (optional - variables load lazily by default)."""
        self.preload_all("daily")


    def get_variable_data(self, name, pressure_level=None, frequency="monthly"):
        """
        Return a (dask-backed, lazy) DataArray for *name* at the given frequency.

        The variable is loaded from disk on the **first** call and cached for
        subsequent calls.  No data is computed until an explicit ``.values`` or
        ``.compute()`` is triggered by a metric.

        Parameters
        ----------
        name : str
            CMOR short name of the variable (e.g. ``"tas"``, ``"ua"``).
        pressure_level : int or None
            If given, select this pressure level from the ``plev`` dimension.
        frequency : str
            ``"monthly"`` or ``"daily"``.

        Returns
        -------
        xr.DataArray
        """
        if frequency == "monthly":
            cache = self._monthly_cache
            path  = self.path_to_monthly_data
        elif frequency == "daily":
            cache = self._daily_cache
            path  = self.path_to_daily_data
        else:
            raise ValueError("frequency must be 'monthly' or 'daily'.")

        if path is None:
            raise ValueError(
                f"No {frequency} data path configured for {self.model_label}."
            )

        # Load and cache lazily on first access.
        if name not in cache:
            cache[name] = self._load_single_variable(path, name, frequency)

        data = cache[name]
        if data is None:
            raise KeyError(
                f"Variable '{name}' not found under {frequency} path for "
                f"{self.model_label}."
            )

        if pressure_level is not None:
            data = data.sel(plev=pressure_level, method="nearest")

        return data[name]


