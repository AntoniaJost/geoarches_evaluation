"""
metrics/base.py
===============
Shared utility functions and base classes for all metric evaluators.

Hierarchy
---------
BaseMetric
├── SpatialMetric       – lat/lon map outputs (XYMaps, XYBiasMaps, …)
└── TimeseriesMetric    – time-series outputs (SeasonalCycles, SOI, …)
"""

import itertools
import os

import numpy as np
import xarray as xr
from scipy import stats
from scipy.sparse.linalg import svds as _truncated_svds

# ---------------------------------------------------------------------------
# Temporal-selection helpers
# ---------------------------------------------------------------------------

def annual_mean(data: xr.DataArray, time_dim: str = "time", year=None) -> xr.DataArray:
    """Return annual (or single-year) mean."""
    if year is not None:
        data = data.sel({f"{time_dim}.year": year})
    return data.mean(time_dim)


def seasonal_mean(data: xr.DataArray, season: str = None, time_dim: str = "time") -> xr.DataArray:
    """Return seasonal mean or full time mean when *season* is None."""
    if season is not None:
        return data.groupby(f"{time_dim}.season").mean(time_dim).sel(season=season)
    return data.mean(time_dim)


def instantaneous(data: xr.DataArray, time, time_dim: str = "time") -> xr.DataArray:
    """Select the closest time step to *time*."""
    return data.sel({time_dim: time}, method="nearest", drop=True)


def select_by_time(data: xr.DataArray, temporal_dim: str) -> xr.DataArray:
    """Dispatch to annual-mean, seasonal-mean, or instantaneous selection."""
    if temporal_dim == "annual":
        return annual_mean(data)
    if temporal_dim in ("DJF", "MAM", "JJA", "SON"):
        return seasonal_mean(data, season=temporal_dim)
    return instantaneous(data, time=temporal_dim)


# ---------------------------------------------------------------------------
# Data-processing helpers
# ---------------------------------------------------------------------------

def compute_latitude_weights(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Return sqrt-cosine latitude weights shaped (1, lat, lon).

    Uses NumPy broadcasting instead of ``np.repeat`` to avoid allocating an
    intermediate copy of each latitude row.
    """
    w = np.sqrt(np.cos(np.deg2rad(latitude)))  # (lat,)
    # Broadcast (lat, 1) against a unit (1, lon) row -> (lat, lon), add batch dim
    return (w[:, np.newaxis] * np.ones((1, len(longitude))))[np.newaxis, ...]


def compute_anomaly(
    data: xr.DataArray,
    mean_groups,
    baseline_mean_groups,
    baseline_period=None,
    standardize: bool = False,
    weights=None,
) -> xr.DataArray:
    """Compute anomaly (optionally standardised) relative to a baseline period.

    When *baseline_period* is ``None`` the full dataset is the baseline, so we
    avoid the redundant O(T) nearest-neighbour lookup and use *data* directly.
    """
    if baseline_period is None:
        # All time steps belong to the baseline – no selection needed.
        baseline_data = data
    else:
        baseline_data = data.sel(
            time=(data.time.values >= np.datetime64(baseline_period[0], "ns"))
            & (data.time.values <= np.datetime64(baseline_period[1], "ns")),
            method="nearest",
        )

    if baseline_mean_groups is None:
        baseline_mean = baseline_data.mean(dim="time")
    else:
        baseline_mean = baseline_data.groupby(baseline_mean_groups).mean(dim="time")

    if mean_groups is None:
        anomalies = data - baseline_mean
    else:
        anomalies = data.groupby(mean_groups).mean(dim="time") - baseline_mean

    if standardize:
        if baseline_mean_groups is None:
            baseline_std = baseline_data.std(dim="time")
        else:
            baseline_std = baseline_data.groupby(baseline_mean_groups).std(dim="time")
        anomalies = anomalies / baseline_std

    if weights is not None:
        anomalies = anomalies.weighted(weights)

    return anomalies


def compute_bias(data: xr.DataArray, reference_data: xr.DataArray) -> xr.DataArray:
    return data - reference_data


def detrend_data(data: xr.DataArray) -> xr.DataArray:
    """Remove a linear trend in-place (preserves the mean)."""
    coeffs = np.polyfit(range(len(data)), data, deg=1)
    coeffs[-1] = 0  # keep mean
    fit = np.polyval(coeffs, range(len(data)))
    data.values = data.values - fit
    return data


def compute_eof(data: xr.DataArray, n_modes: int = 1):
    """Compute leading *n_modes* EOF patterns via **truncated** SVD.

    For large daily grids, ``scipy.sparse.linalg.svds`` computes only the
    *k* leading singular vectors instead of the full decomposition, giving
    O(T * N * k) complexity vs O(min(T,N)^2 * max(T,N)) for
    ``numpy.linalg.svd``.

    Returns ``(eof_modes, U, S, Vt)`` with the same shapes as before:
    ``U`` and ``Vt`` contain only the *n_modes* columns / rows.
    """
    data = data.transpose("time", "lat", "lon", ...)
    x = data.values.reshape(data.values.shape[0], -1)  # (T, N)
    # scipy.sparse.linalg.svds returns singular values in *ascending* order
    # and requires k < min(T, N).
    if n_modes is None: 
        k = n_modes = min(x.shape) - 1
    else:
        k = min(n_modes, min(x.shape) - 1)
    if k >= 1:
        U_k, S_k, Vt_k = _truncated_svds(x, k=k)
        # Reverse to descending order (matching np.linalg.svd convention).
        U_k  = U_k[:, ::-1]
        S_k  = S_k[::-1]
        Vt_k = Vt_k[::-1, :]
    else:
        # Fallback for degenerate matrices.
        U_k, S_k, Vt_k = np.linalg.svd(x, full_matrices=False)
        U_k  = U_k[:, :n_modes]
        S_k  = S_k[:n_modes]
        Vt_k = Vt_k[:n_modes, :]

    eof_modes = U_k[:, :n_modes].copy()
    for i in range(n_modes):
        mode = eof_modes[:, i]
        if np.corrcoef(x[:, 0], mode)[0, 1] > 0:
            eof_modes[:, i] = -mode
    return eof_modes, U_k, S_k, Vt_k


def compute_correlation_matrix(data1: xr.DataArray, data2: xr.DataArray) -> np.ndarray:
    data1 = data1 - data1.mean()
    data2 = data2 - data2.mean()
    return ((data1 * data2) / np.sqrt((data1 ** 2).sum() * (data2 ** 2).sum())).values


def compute_soi(data: xr.DataArray, base_period, detrend: bool = False) -> xr.DataArray:
    """Compute the Southern Oscillation Index (Tahiti minus Darwin)."""
    if data.lon.max() > 180:
        data = data.assign_coords(lon=(data.lon - 180))

    tahiti = data.sel(lat=-17.65, lon=-149.42, method="nearest")
    darwin = data.sel(lat=-12.46, lon=130.84, method="nearest")

    kwargs = dict(
        mean_groups=["time.year", "time.month"],
        baseline_mean_groups=["time.month"],
        standardize=True,
    )
    tahiti_anom = compute_anomaly(tahiti, baseline_period=base_period, **kwargs)
    darwin_anom = compute_anomaly(darwin, baseline_period=base_period, **kwargs)

    soi = tahiti_anom - darwin_anom
    soi = soi / soi.std(dim="year")
    return soi


# ---------------------------------------------------------------------------
# Private helpers used by multiple metric classes
# ---------------------------------------------------------------------------

def _format_var_name(name: str, pressure_level) -> str:
    """Append pressure level in Pa when present."""
    return f"{name}_{pressure_level}Pa" if pressure_level is not None else name


def _get_model_colors(data_containers) -> dict:
    """Return ``{model_label: model_color}`` for every data container."""
    return {dc.model_label: dc.model_color for dc in data_containers}


def _get_reference_container(data_containers):
    """Return ``(label, container)`` for the first reference, otherwise ``(None, None)``."""
    for dc in data_containers:
        if dc.is_reference:
            return dc.model_label, dc
    return None, None


def _log_variable_info(name: str, pressure_level) -> None:
    if pressure_level is not None:
        print(f"--> Processing variable: {name} at pressure level {pressure_level} Pa")
    else:
        print(f"--> Processing variable: {name}")


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class BaseMetric:
    """
    Common interface and shared helpers for **all** metric evaluators.

    Subclasses must implement :py:meth:`evaluate`.

    Parameters
    ----------
    variables:
        List of variable dicts ``{"name": str, "pressure_level": int}``.
        Pass an empty list for metrics that operate on fixed variables (e.g.
        ``AnnularModes``).
    frequency:
        Temporal resolution of the source data (``"monthly"`` or ``"daily"``).
    plotter_kwargs:
        Passed verbatim to the plotter that the subclass constructs.
    """

    def __init__(
        self,
        variables: list = None,
        frequency: str = "monthly",
        plotter_kwargs: dict = None,
    ) -> None:
        self.variables = list(variables) if variables else []
        self.frequency = frequency
        self.plotter_kwargs = dict(plotter_kwargs) if plotter_kwargs else {}

    # -- Interface -----------------------------------------------------------

    def evaluate(self, data_containers) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement evaluate()")

    # -- Convenience wrappers ------------------------------------------------

    def _reference(self, data_containers):
        """Return ``(label, container)`` for the first reference container."""
        return _get_reference_container(data_containers)

    def _colors(self, data_containers) -> dict:
        """Return colour dict keyed by model label."""
        return _get_model_colors(data_containers)

    def _fmt_var(self, name: str, pressure_level) -> str:
        return _format_var_name(name, pressure_level)

    def _log(self, name: str, pressure_level) -> None:
        _log_variable_info(name, pressure_level)


class SpatialMetric(BaseMetric):
    """
    Base class for metrics that produce spatial (lat/lon) map outputs.

    Constructs a :class:`~plot.modules.SpatialPlotter` and exposes helpers
    for temporal selection and reference-data retrieval.
    """

    def __init__(
        self,
        variables: list,
        xdim: str,
        ydim: str,
        temporal_selection: list = None,
        frequency: str = "monthly",
        plotter_kwargs: dict = None,
    ) -> None:
        super().__init__(variables, frequency, plotter_kwargs)
        self.xdim = xdim
        self.ydim = ydim
        self.temporal_selection = temporal_selection or ["annual"]

        # Deferred import to avoid circular imports at module load time.
        from plot.modules import SpatialPlotter
        self.plotter = SpatialPlotter(xdim=self.xdim, ydim=self.ydim, **self.plotter_kwargs)

    # -- Helpers -------------------------------------------------------------

    def select_by_time(self, data: xr.DataArray, temporal_dim: str) -> xr.DataArray:
        return select_by_time(data, temporal_dim)

    def get_reference_data(self, data_containers) -> None:
        """Set ``self.reference_label`` and ``self.reference_data`` from containers."""
        self.reference_label, self.reference_data = self._reference(data_containers)


class TimeseriesMetric(BaseMetric):
    """
    Base class for metrics that produce time-series outputs.

    Constructs a :class:`~plot.modules.TimeseriesPlotter`, adjusting the
    output path when anomalies or detrending are requested.  Also provides
    the :py:meth:`xlabels_from_time` helper used by all timeseries subclasses.
    """

    def __init__(
        self,
        variables: list = None,
        frequency: str = "monthly",
        detrend: bool = False,
        compute_anomalies: bool = False,
        mean_groups: list = None,
        baseline_period: tuple = None,
        baseline_mean_groups: list = None,
        plotter_kwargs: dict = None,
    ) -> None:
        super().__init__(variables, frequency, plotter_kwargs)

        if compute_anomalies and baseline_period is None:
            raise ValueError("baseline_period is required when compute_anomalies=True.")

        self.mean_groups = mean_groups
        self.baseline_mean_groups = baseline_mean_groups
        self.compute_anomalies = compute_anomalies
        self.baseline_period = baseline_period
        self.detrend = detrend

        # Build output path taking anomaly / detrend flags into account.
        output_path = self.plotter_kwargs.get("output_path", ".")
        if compute_anomalies and baseline_period:
            output_path = os.path.join(output_path, f"{baseline_period[0]}-{baseline_period[1]}")
        if detrend:
            output_path = os.path.join(output_path, "detrended")
        kw = dict(self.plotter_kwargs)
        kw["output_path"] = output_path

        from plot.modules import TimeseriesPlotter
        self.timeseries_plotter = TimeseriesPlotter(**kw)

    # -- Helpers -------------------------------------------------------------

    def xlabels_from_time(self, time: xr.DataArray):
        """Convert a time coordinate with year/month dims to string labels."""
        if "month" in time.dims and "year" in time.dims:
            return [
                f"{y}-{m:02d}"
                for y, m in itertools.product(time["year"].values, time["month"].values)
            ]
        if "month" in time.dims:
            return time["month"].values
        if "year" in time.dims:
            return time["year"].values
        raise ValueError("Time coordinate must have 'month' and/or 'year' dimensions.")
