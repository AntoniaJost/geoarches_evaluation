"""
metrics/module.py
=================
Metric-evaluator classes for climate diagnostics.

All shared utility functions and the three abstract base classes live in
:mod:`metrics.base`.  The concrete evaluators here simply inherit from the
appropriate base and provide domain-specific ``compute``/``evaluate`` logic.

Class hierarchy
---------------
BaseMetric
├─ SpatialMetric
│   ├─ XYMaps
│   │   ├─ LatTimeMap
│   │   ├─ TimeLongitudeMap
│   │   ├─ PressureLatMap
│   │   ├─ XYBiasMaps
│   │   ├─ XYAnomalyMaps
│   │   └─ XYTrendMaps
│   └─ VariableCorrelationMaps
├─ TimeseriesMetric
│   └─ Timeseries
│       ├─ SeasonalCycles
│       └─ SouthernOscillationIndex
├─ MonsoonIndices
├─ AnnularModes
│   ├─ NorthernAnnularMode
│   └─ NorthernAtlanticOscillationIndex
├─ RadialSpectrum
├─ Distribution
│   ├─ Histogram
│   ├─ AnimatedHistogram
│   └─ ReturnPeriods
├─ TropicalCycloneFrequency
└─ QuantitativeBaseline

EOF is a standalone computational helper (not a metric evaluator).
"""

import logging
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
import xarray as xr
import matplotlib as mpl
from matplotlib.colors import CenteredNorm
from scipy import stats

from metrics.functional import spectral
from metrics.functional.return_periods import (
    annual_block_maxima,
    domain_max_series,
    empirical_return_periods,
    fit_gev,
    return_levels,
)

mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["font.family"] = "DejaVu Sans"  # for non-math text, e.g. axis labels and legends
mpl.rcParams["axes.titlesize"] = 8
mpl.rcParams["axes.labelsize"] = 7
mpl.rcParams["xtick.labelsize"] = 7
mpl.rcParams["ytick.labelsize"] = 7
mpl.rcParams["legend.fontsize"] = 6

from metrics.base import (
    BaseMetric,
    SpatialMetric,
    TimeseriesMetric,
    _format_var_name,
    _get_model_colors,
    _get_reference_container,
    _log_variable_info,
    select_by_time,
    compute_anomaly,
    compute_eof,
    compute_soi,
    detrend_data,
)
from geoarches.metrics.metric_base import compute_lat_weights_weatherbench
from plot.modules import (
    A4_WIDTH,
    EarthPlotter,
    FrequencyPlotter,
    SOIFrequencyPlotter,
    TimeseriesPlotter,
    TaylorDiagramPlotter,
    LatitudinalProfilePlotter,
)
from plot.projections import CartopyProjectionPlotter
from evaltools.module import CMORDataContainer as _CMORDataContainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level formatting helper
# ---------------------------------------------------------------------------

def _fmt_scalar(v: float, sig: int = 4) -> str:
    """Format *v* using *sig* significant figures.

    Switches to scientific notation automatically when the magnitude falls
    outside the range ``[0.001, 10 000)`` so that very small values such as
    specific-humidity RMSE (e.g. 2.3 × 10⁻⁵) are never rounded to zero.
    Numbers within that range are shown in plain decimal form; scientific
    notation is used elsewhere.

    Examples
    --------
    >>> _fmt_scalar(0.000023)   # '2.300e-05'
    >>> _fmt_scalar(1.23456)    # '1.235'
    >>> _fmt_scalar(-12.7)      # '-12.70'
    """
    if v == 0:
        return "0"
    abs_v = abs(v)
    if 1e-3 <= abs_v < 1e4:
        # Fixed decimal: enough places so that *sig* significant digits show.
        decimals = max(0, sig - 1 - int(math.floor(math.log10(abs_v))))
        return f"{v:.{decimals}f}"
    return f"{v:.{sig - 1}e}"


# ---------------------------------------------------------------------------
# Module-level helper: human-readable variable label
# ---------------------------------------------------------------------------
_VARIABLE_LONG_NAMES: dict = _CMORDataContainer.variable_names


def _long_var_label(short_name: str, pressure_level=None) -> str:
    """Return a human-readable label for *short_name*, optionally with level.

    Parameters
    ----------
    short_name:
        CMOR short name (e.g. ``"ua"``).
    pressure_level:
        Pressure level **in Pascal** (e.g. 70000).  Divided by 100 to produce
        the hPa value shown in the label.  Pass ``None`` to omit the level.
    """
    long_name = _VARIABLE_LONG_NAMES.get(short_name, short_name)
    if pressure_level is None:
        return long_name
    hpa = int(round(pressure_level / 100))
    return f"{long_name} ({hpa} hPa)"


def _temporal_title_suffix(ts) -> str:
    """Return a title suffix for temporal selection.

    Suppress the default annual label and only append explicit seasonal
    selections such as DJF/MAM/JJA/SON.
    """
    if ts is None:
        return ""
    ts_str = str(ts).strip()
    if ts_str.lower() == "annual":
        return ""
    return f" ({ts_str})"


# ============================================================================
# Spatial map metrics
# ============================================================================

class XYMaps(SpatialMetric):
    """
    Global or regional lat/lon mean maps.

    Produces one image per variable × temporal-selection × model, optionally
    computing RMSE and bias against a reference dataset.
    """

    def __init__(
        self,
        variables: list,
        xdim: str,
        ydim: str,
        temporal_selection: list = None,
        frequency: str = "monthly",
        per_member: bool = False,
        plotter_kwargs: dict = None,
        compute_rmse: bool = False,
        p95=True,
        stack_models_in_rows: bool = False,
    ) -> None:
        super().__init__(
            variables=variables,
            xdim=xdim,
            ydim=ydim,
            temporal_selection=temporal_selection,
            frequency=frequency,
            per_member=per_member,
            plotter_kwargs=plotter_kwargs,
        )
        self.compute_rmse = compute_rmse
        self.p95 = p95
        self.stack_models_in_rows = stack_models_in_rows

    def eval_against_reference(
        self, data: xr.DataArray, reference_data: xr.DataArray
    ) -> tuple:
        """Return latitude-weighted (RMSE, bias) against *reference_data*.

        Works directly on NumPy arrays to avoid creating redundant xarray
        copies (which would each trigger a dask graph materialisation).
        """#
        latitude = data.lat.values
        diff = data.values - reference_data.values
        lat_weights = (
            compute_lat_weights_weatherbench(latitude_resolution=len(latitude))
            .cpu()
            .numpy()
            .squeeze()
        )
        w = np.broadcast_to(lat_weights[:, np.newaxis], diff.shape)

        valid = ~np.isnan(diff)
        if not np.any(valid):
            return np.nan, np.nan

        w_valid = w[valid]
        diff_valid = diff[valid]
        rmse = float(np.sqrt(np.sum((diff_valid ** 2) * w_valid) / np.sum(w_valid)))
        bias = float(np.sum(diff_valid * w_valid) / np.sum(w_valid))
        return rmse, bias

    def compute(
        self,
        data: xr.DataArray,
        temporal_dim: str,
        var: dict,
        frequency: str = "monthly",
    ) -> xr.DataArray:
        """Return the spatial map for *temporal_dim*, averaged over non-spatial dims."""
        if "stat" in data.dims:
            data = data.sel(stat="mean", drop=True)
        data = self.select_by_time(data, temporal_dim)
        data = data.mean(dim=[d for d in data.dims if d not in [self.xdim, self.ydim]])
        return data

    def evaluate(self, data_containers) -> None:
        self.get_reference_data(data_containers)

        for ts in self.temporal_selection:
            logger.info(f"--> XYMaps: computing {self.xdim}-{self.ydim} map (temporal selection: {ts})")
            for variable in self.variables:
                name, pressure_level = variable["name"], variable["pressure_level"]
                _log_variable_info(name, pressure_level)

                var_name = _format_var_name(name, pressure_level)
                output_path = os.path.join(self.plotter.output_path, ts, var_name)
                os.makedirs(output_path, exist_ok=True)

                # ── cache lookup ──────────────────────────────────────────────
                nc_paths = {
                    dc.model_label: self._nc_path(output_path, f"{dc.model_label}_{var_name}")
                    for dc in data_containers
                }

                # maps: {model_label: (data, output_path, var_name, rmse, bias)}
                maps = {}

                if self._all_cached(list(nc_paths.values())):
                    logger.info(f"    Loading cached results from '{output_path}'")
                    for dc in data_containers:
                        ds = self._load_nc(nc_paths[dc.model_label])
                        data = ds["data"].transpose(self.ydim, self.xdim)
                        rmse = float(ds.attrs["rmse"]) if "rmse" in ds.attrs else None
                        bias = float(ds.attrs["bias"]) if "bias" in ds.attrs else None
                        maps[dc.model_label] = (data, output_path, var_name, rmse, bias)
                        ds.close()
                else:
                    # ── compute maps ──────────────────────────────────────────
                    raw_maps = {}
                    for dc in data_containers:
                        data = dc.get_variable_data(**variable, frequency=self.frequency)
                        data = self.compute(data, temporal_dim=ts, var=variable, frequency=self.frequency)
                        data = data.transpose(self.ydim, self.xdim)
                        raw_maps[dc.model_label] = data

                    ref_map = raw_maps.get(self.reference_label)

                    for dc in data_containers:
                        data = raw_maps[dc.model_label]
                        if ref_map is not None and dc.model_label != self.reference_label:
                            if self.compute_rmse:
                                rmse, bias = self.eval_against_reference(data, ref_map)
                                logger.info(f"    RMSE against reference for {dc.model_label}: {_fmt_scalar(rmse)}")
                                logger.info(f"    Bias against reference for {dc.model_label}: {_fmt_scalar(bias)}")
                            else:
                                rmse, bias = None, None
                        else:
                            rmse, bias = None, None
                            logger.info(f"    No reference data available for {dc.model_label}.")
                        # ── save to cache ─────────────────────────────────────
                        self._save_nc(
                            data, nc_paths[dc.model_label],
                            attrs={"rmse": rmse, "bias": bias},
                        )
                        maps[dc.model_label] = (data, output_path, var_name, rmse, bias)

                # Compute shared colorbar limits across all models so the
                # figures are directly comparable.
                all_values = np.concatenate([d.values.ravel() for d, _, _, _, _ in maps.values()])
                shared_vmin = float(np.nanmin(all_values))
                shared_vmax = float(np.nanmax(all_values))
                cbar_label = self.plotter.cmor_units.get(name, "")

                if self.stack_models_in_rows and len(maps) > 1:
                    stacked_maps = []
                    row_info_right = {}
                    for model_label, (data, _, _, rmse, _) in maps.items():
                        stacked_maps.append((model_label, data))
                        row_info_right[model_label] = (
                            f"RMSE: {_fmt_scalar(rmse)}" if rmse is not None else ""
                        )
                    self.plotter.plot_stacked_maps(
                        maps=stacked_maps,
                        variable_name=var_name,
                        output_path=output_path,
                        fname=f"{var_name}_all_models.pdf",
                        title=f"{_long_var_label(name, pressure_level)}{_temporal_title_suffix(ts)}",
                        cbar_label=cbar_label,
                        vmin=shared_vmin,
                        vmax=shared_vmax,
                        cbar_orientation="horizontal",
                        row_info_right=row_info_right,
                    )
                    logger.info(
                        "    Saved stacked spatial map: %s",
                        os.path.join(output_path, f"{var_name}_all_models.pdf"),
                    )
                else:
                    for model_label, (data, output_path, var_name, rmse, bias) in maps.items():
                        self.plotter.plot(
                            x=data,
                            variable_name=var_name,
                            title="",
                            model_label=model_label,
                            style="imshow",
                            output_path=output_path,
                            vmin=shared_vmin,
                            vmax=shared_vmax,
                            cbar_label=cbar_label,
                            infotext_topright=f"RMSE: {_fmt_scalar(rmse)}" if rmse is not None else "",
                            infotext_left=f"{model_label}",
                            infotext_topleft=f"{_long_var_label(name, pressure_level)}",
                            cbar_orientation="horizontal",
                        )


class LatTimeMap(XYMaps):
    """Hovmöller-style latitude-time diagram."""

    def compute(
        self,
        data: xr.DataArray,
        temporal_dim,
        var: dict,
        frequency: str = "monthly",
    ):
        if "stat" in data.dims:
            data = data.sel(stat="mean", drop=True)
        data = data.mean(dim="lon").transpose("lat", "time")
        return data.time.values, data.lat.values, data.values

    def evaluate(self, data_containers) -> None:
        logger.info(f"--> LatTimeMap: computing {self.xdim}-{self.ydim} Hovmöller diagram")
        for variable in self.variables:
            name, pressure_level = variable["name"], variable["pressure_level"]
            _log_variable_info(name, pressure_level)

            var_name = _format_var_name(name, pressure_level)
            output_path = os.path.join(self.plotter.output_path, var_name)
            os.makedirs(output_path, exist_ok=True)

            # ── cache lookup ──────────────────────────────────────────────────
            nc_paths = {
                dc.model_label: self._nc_path(output_path, f"{dc.model_label}_{var_name}_lattime")
                for dc in data_containers
            }

            # maps: {model_label: (time_vals, lat_vals, z, output_path, var_name)}
            maps = {}

            if self._all_cached(list(nc_paths.values())):
                logger.info(f"    Loading cached Hovmöller results from '{output_path}'")
                for dc in data_containers:
                    ds = self._load_nc(nc_paths[dc.model_label])
                    da = ds["data"]
                    time_vals = da.time.values
                    lat_vals  = da.lat.values
                    z         = da.values
                    maps[dc.model_label] = (time_vals, lat_vals, z, output_path, var_name)
                    ds.close()
            else:
                for dc in data_containers:
                    data_array = dc.get_variable_data(**variable, frequency=self.frequency)
                    time_vals, lat_vals, z = self.compute(
                        data_array, temporal_dim=None, var=variable, frequency=self.frequency
                    )
                    # ── save to cache ─────────────────────────────────────────
                    da = xr.DataArray(z, coords={"lat": lat_vals, "time": time_vals}, dims=["lat", "time"])
                    self._save_nc(da, nc_paths[dc.model_label])
                    maps[dc.model_label] = (time_vals, lat_vals, z, output_path, var_name)

            # Compute shared colorbar limits across all models so the
            # Hovmöller diagrams are directly comparable.
            all_z_values = np.concatenate([z.ravel() for _, _, z, _, _ in maps.values()])
            shared_vmin = float(np.nanmin(all_z_values))
            shared_vmax = float(np.nanmax(all_z_values))

            for model_label, (time_vals, lat_vals, z, output_path, var_name) in maps.items():
                fig = plt.figure(figsize=(6.7, 3.5), dpi=150)
                plt.imshow(z, cmap="coolwarm", vmin=shared_vmin, vmax=shared_vmax)
                _cbar = plt.colorbar(
                    shrink=0.6,
                    extend="both",
                )
                _cbar.set_label(self.plotter.cmor_units.get(name, ""), fontsize=10)
                time_labels = [pd.to_datetime(t).year for t in time_vals]
                time_labels_clean = sorted(set(time_labels))
                ticks = [time_labels.index(t) for t in time_labels_clean][::4]
                time_labels_clean = [str(t) for t in time_labels_clean][::4]
                plt.xticks(ticks=ticks, labels=time_labels_clean, rotation=45)
                plt.yticks(
                    ticks=np.linspace(0, len(lat_vals), 7),
                    labels=np.linspace(90, -90, 7),
                    rotation=45,
                )
                plt.title(f"Hovmöller: {_long_var_label(name, pressure_level)} – {model_label}", pad=10)
                plt.savefig(
                    os.path.join(output_path, f"{model_label}_{var_name}.pdf"),
                    bbox_inches="tight",
                )
                plt.close(fig)


class TimeLongitudeMap(XYMaps):
    """Longitude–time Hovmöller diagram.

    Averages the data over a configurable latitude band, yielding a
    ``(time, lon)`` diagram analogous to :class:`LatTimeMap`.

    Parameters
    ----------
    lat_band : tuple of (south_lat, north_lat)
        Latitude range over which to average before plotting.
        Defaults to the full globe ``(-90, 90)``.
    """

    def __init__(
        self,
        variables: list,
        xdim: str = "lon",
        ydim: str = "time",
        temporal_selection: list = None,
        lat_band: tuple = (-90, 90),
        frequency: str = "monthly",
        plotter_kwargs: dict = None,
        per_member: bool = False,
    ) -> None:
        super().__init__(
            variables=variables, xdim=xdim, ydim=ydim, temporal_selection=temporal_selection,
            frequency=frequency, plotter_kwargs=plotter_kwargs, per_member=per_member)
        self.lat_band = lat_band

    def compute(
        self,
        data: xr.DataArray,
        temporal_dim,
        var: dict,
        frequency: str = "monthly",
    ):
        if "stat" in data.dims:
            data = data.sel(stat="mean", drop=True)
        # Lat coords are stored descending (90 → -90); slice accordingly.
        lat_min, lat_max = min(self.lat_band), max(self.lat_band)
        data = data.sel(lat=slice(lat_max, lat_min))
        data = data.mean(dim="lat").transpose("time", "lon")
        return data.time.values, data.lon.values, data.values

    def evaluate(self, data_containers) -> None:
        lat_str = f"{min(self.lat_band)}°–{max(self.lat_band)}°"
        logger.info(
            f"--> TimeLongitudeMap: computing time–lon Hovmöller diagram (lat band {lat_str})"
        )
        for variable in self.variables:
            name, pressure_level = variable["name"], variable["pressure_level"]
            _log_variable_info(name, pressure_level)

            var_name = _format_var_name(name, pressure_level)
            output_path = os.path.join(self.plotter.output_path, var_name)
            os.makedirs(output_path, exist_ok=True)

            # ── cache lookup ──────────────────────────────────────────────────
            nc_paths = {
                dc.model_label: self._nc_path(output_path, f"{dc.model_label}_{var_name}_timelon")
                for dc in data_containers
            }

            # maps: {model_label: (time_vals, lon_vals, z, output_path, var_name)}
            maps = {}

            if self._all_cached(list(nc_paths.values())):
                logger.info(f"    Loading cached time–lon results from '{output_path}'")
                for dc in data_containers:
                    ds = self._load_nc(nc_paths[dc.model_label])
                    da = ds["data"]
                    time_vals = da.time.values
                    lon_vals  = da.lon.values
                    z         = da.values
                    maps[dc.model_label] = (time_vals, lon_vals, z, output_path, var_name)
                    ds.close()
            else:
                for dc in data_containers:
                    data_array = dc.get_variable_data(**variable, frequency=self.frequency)
                    time_vals, lon_vals, z = self.compute(
                        data_array, temporal_dim=None, var=variable, frequency=self.frequency
                    )
                    # ── save to cache ─────────────────────────────────────────
                    da = xr.DataArray(z, coords={"time": time_vals, "lon": lon_vals}, dims=["time", "lon"])
                    self._save_nc(da, nc_paths[dc.model_label])
                    maps[dc.model_label] = (time_vals, lon_vals, z, output_path, var_name)

            all_z_values = np.concatenate([z.ravel() for _, _, z, _, _ in maps.values()])
            shared_vmin = float(np.nanmin(all_z_values))
            shared_vmax = float(np.nanmax(all_z_values))
            unit = self.plotter.cmor_units.get(name, "")

            for model_label, (time_vals, lon_vals, z, output_path, var_name) in maps.items():
                os.makedirs(output_path, exist_ok=True)
                fig = plt.figure(figsize=(6.7, 3.5), dpi=150)
                plt.imshow(
                    z, aspect="auto",
                    cmap="coolwarm", vmin=shared_vmin, vmax=shared_vmax,
                    origin="upper",
                )
                _cbar = plt.colorbar(shrink=0.6, extend="both")
                _cbar.set_label(unit, fontsize=10)

                # x-axis: longitude ticks
                n_lon = len(lon_vals)
                lon_tick_step = max(1, n_lon // 6)
                plt.xticks(
                    ticks=np.arange(0, n_lon, lon_tick_step),
                    labels=[f"{v:.0f}°" for v in lon_vals[::lon_tick_step]],
                    rotation=45,
                )
                # y-axis: time ticks (every 4 years)
                time_years = [pd.to_datetime(t).year for t in time_vals]
                unique_years = sorted(set(time_years))
                ticks = [time_years.index(y) for y in unique_years][::4]
                tick_labels = [str(y) for y in unique_years][::4]
                plt.yticks(ticks=ticks, labels=tick_labels)

                plt.xlabel("Longitude", fontsize=self.plotter.fontdict.get("axes.labelsize", 12))
                plt.ylabel("Time", fontsize=self.plotter.fontdict.get("axes.labelsize", 12))
                plt.title(
                    f"Hovmöller (lon–time): {_long_var_label(name, pressure_level)} "
                    f"[{lat_str}] – {model_label}",
                    fontsize=self.plotter.fontdict.get("axes.titlesize", 14),
                    pad=10,
                )
                plt.tight_layout()
                fpath = os.path.join(output_path, f"{model_label}_{var_name}_timelon.pdf")
                plt.savefig(fpath, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"    Saved time–lon Hovmöller: {fpath}")


_WIND_VARIABLE_NAMES: frozenset = frozenset({
    "ua", "va", "uas", "vas", "wap",
    "u_component_of_wind", "v_component_of_wind",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "vertical_velocity",
})

_SPECIFIC_HUMIDITY_VARIABLE_NAMES: frozenset = frozenset({
    "hus", "huss", "q", "specific_humidity",
})

_TEMPERATURE_VARIABLE_NAMES: frozenset = frozenset({
    "tas", "ta",
})


class LatitudeProfile(BaseMetric):
    """Latitudinal line profile (latitude on x-axis, magnitude on y-axis).

    For each selected variable/time/latitude-band combination, all models are
    drawn in a single line plot.

    When the variable is a wind variable (see ``_WIND_VARIABLE_NAMES``) and a
    reference model is present, the latitude of the absolute maximum of each
    model profile is compared to that of the reference profile.  The resulting
    offset in degrees is annotated in the bottom-right corner of the figure.
    """

    def __init__(
        self,
        variables: list,
        temporal_selection: list = None,
        latitude_band=None,
        longitude_band=None,
        frequency: str = "monthly",
        plotter_kwargs: dict = None,
        per_member: bool = False,
    ) -> None:
        super().__init__(
            variables=variables,
            frequency=frequency,
            plotter_kwargs=plotter_kwargs,
            per_member=per_member,
        )
        self.temporal_selection = temporal_selection or ["annual"]
        # Class-level defaults used as fallback when a variable does not
        # specify its own latitude_band / longitude_band.
        _default_lat = _normalize_band_list(latitude_band)
        self._default_lat_band = _default_lat[0] if _default_lat else None
        _default_lon = _normalize_band_list(longitude_band)
        self._default_lon_band = _default_lon[0] if (_default_lon and _default_lon != [None]) else None
        self.plotter = LatitudinalProfilePlotter(**self.plotter_kwargs)

    def compute(
        self,
        data: xr.DataArray,
        temporal_dim: str,
        lat_band,
        lon_band=None,
    ) -> xr.DataArray:
        if "stat" in data.dims:
            data = data.sel(stat="mean", drop=True)

        data = select_by_time(data, temporal_dim)

        if lat_band is not None:
            lat_min, lat_max = min(lat_band), max(lat_band)
            if data.lat.values[0] > data.lat.values[-1]:
                data = data.sel(lat=slice(lat_max, lat_min))
            else:
                data = data.sel(lat=slice(lat_min, lat_max))

        if lon_band is not None and "lon" in data.dims:
            lon_min, lon_max = min(lon_band), max(lon_band)
            data = data.sel(lon=slice(lon_min, lon_max))

        avg_dims = [d for d in data.dims if d != "lat"]
        if avg_dims:
            data = data.mean(dim=avg_dims)

        return data.sortby("lat")

    @staticmethod
    def _peak_latitude(profile: xr.DataArray) -> float:
        """Return the latitude of the absolute maximum of *profile*."""
        idx = int(np.argmax(np.abs(profile.values)))
        return float(profile.lat.values[idx])

    @staticmethod
    def _mean_magnitude(profile: xr.DataArray) -> float:
        """Return the mean profile magnitude across latitude."""
        return float(np.nanmean(np.abs(profile.values)))

    @staticmethod
    def _warming_kelvin_from_label(model_label: str) -> float | None:
        """Extract warming amount X from a model label token like ``pXk``."""
        match = re.search(r"p([0-9]+(?:\.[0-9]+)?)k", model_label.lower())
        if not match:
            return None
        warming = float(match.group(1))
        return warming if warming > 0 else None

    def evaluate(self, data_containers) -> None:
        import math

        colors = _get_model_colors(data_containers)
        reference_label, _ = _get_reference_container(data_containers)

        for ts in self.temporal_selection:
            parent_output_path = os.path.join(self.plotter.output_path, ts)
            os.makedirs(parent_output_path, exist_ok=True)

            all_var_data = []

            for variable in self.variables:
                name, pressure_level = variable["name"], variable["pressure_level"]
                _log_variable_info(name, pressure_level)

                # Per-variable bands override class-level defaults.
                raw_lat = variable.get("latitude_band", None)
                lat_band = _normalize_band_list(raw_lat)[0] if raw_lat is not None else self._default_lat_band
                raw_lon = variable.get("longitude_band", None)
                lon_band = _normalize_band_list(raw_lon)[0] if raw_lon is not None else self._default_lon_band

                is_wind = name in _WIND_VARIABLE_NAMES
                is_specific_humidity = name in _SPECIFIC_HUMIDITY_VARIABLE_NAMES
                is_temperature = name in _TEMPERATURE_VARIABLE_NAMES
                var_name = _format_var_name(name, pressure_level)
                unit = self.plotter.cmor_units.get(name, "")
                ylabel = f"{unit}"

                latlon_dir = _latlon_dir(lat_band, lon_band)
                output_path = os.path.join(parent_output_path, latlon_dir, var_name)
                os.makedirs(output_path, exist_ok=True)

                nc_cache_tag = f"{latlon_dir}_{ts}_latprofile"
                nc_paths = {
                    dc.model_label: self._nc_path(
                        output_path,
                        f"{dc.model_label}_{var_name}_{nc_cache_tag}",
                    )
                    for dc in data_containers
                }

                profiles = {}
                if self._all_cached(list(nc_paths.values())):
                    logger.info(
                        "    Loading cached latitude profiles from '%s'",
                        output_path,
                    )
                    for dc in data_containers:
                        ds = self._load_nc(nc_paths[dc.model_label])
                        profiles[dc.model_label] = ds["data"].sortby("lat")
                        ds.close()
                else:
                    for dc in data_containers:
                        data = dc.get_variable_data(
                            name=name,
                            pressure_level=pressure_level,
                            frequency=self.frequency,
                        )
                        profile = self.compute(
                            data=data,
                            temporal_dim=ts,
                            lat_band=lat_band,
                            lon_band=lon_band,
                        )
                        self._save_nc(profile, nc_paths[dc.model_label])
                        profiles[dc.model_label] = profile

                # Build infobox region string.
                if lat_band is not None and lon_band is not None:
                    infobox_region = (
                        f"Lat: {min(lat_band):g}° to {max(lat_band):g}°, "
                        f"Lon: {min(lon_band):g}° to {max(lon_band):g}°"
                    )
                elif lat_band is not None:
                    infobox_region = f"Latitude band: {min(lat_band):g}° to {max(lat_band):g}°"
                elif lon_band is not None:
                    infobox_region = f"Longitude band: {min(lon_band):g}° to {max(lon_band):g}°"
                else:
                    infobox_region = "Latitude band: global"

                # Build per-model comparison annotation against the
                # reference model when applicable.
                comparison_annotation = ""
                if is_wind and reference_label is not None and reference_label in profiles:
                    ref_peak = self._peak_latitude(profiles[reference_label])
                    offset_lines = []
                    for model_label, prof in profiles.items():
                        if model_label == reference_label:
                            continue
                        model_peak = self._peak_latitude(prof)
                        offset = model_peak - ref_peak
                        sign = "+" if offset >= 0 else ""
                        offset_lines.append(
                            f"{model_label}: {sign}{offset:.1f}°"
                        )
                        logger.info(
                            "    Wind jet offset  %s vs %s: %+.1f°",
                            model_label, reference_label, offset,
                        )
                    if offset_lines:
                        comparison_annotation = (
                            f"Jet offset vs {reference_label}\n"
                            + "\n".join(offset_lines)
                        )
                elif (
                    (is_specific_humidity or is_temperature)
                    and reference_label is not None
                    and reference_label in profiles
                ):
                    ref_mean = self._mean_magnitude(profiles[reference_label])
                    if is_specific_humidity and np.isclose(ref_mean, 0.0):
                        logger.warning(
                            "    Specific-humidity reference mean is ~0 for '%s'; "
                            "skipping %%/K annotation.",
                            reference_label,
                        )
                    else:
                        change_lines = []
                        for model_label, prof in profiles.items():
                            if model_label == reference_label:
                                continue
                            model_mean = self._mean_magnitude(prof)
                            warming_k = self._warming_kelvin_from_label(model_label)
                            if warming_k is None:
                                logger.warning(
                                    "    Could not parse warming token 'pXk' from '%s'; "
                                    "skipping normalized change annotation for this model.",
                                    model_label,
                                )
                                continue

                            if is_specific_humidity:
                                pct_change = ((model_mean - ref_mean) / abs(ref_mean)) * 100.0
                                change_per_k = pct_change / warming_k
                                sign = "+" if change_per_k >= 0 else ""
                                change_lines.append(
                                    f"{model_label}: {sign}{change_per_k:.2f}%/K"
                                )
                                logger.info(
                                    "    Specific-humidity change %s vs %s: %+.2f%%/K (warming %.3g K)",
                                    model_label,
                                    reference_label,
                                    change_per_k,
                                    warming_k,
                                )
                            else:
                                temp_change_per_k = (model_mean - ref_mean) / warming_k
                                sign = "+" if temp_change_per_k >= 0 else ""
                                change_lines.append(
                                    f"{model_label}: {sign}{temp_change_per_k:.3f} K/K"
                                )
                                logger.info(
                                    "    Temperature change %s vs %s: %+.3f K/K (warming %.3g K)",
                                    model_label,
                                    reference_label,
                                    temp_change_per_k,
                                    warming_k,
                                )

                        if change_lines:
                            heading = (
                                f"Specific humidity vs {reference_label} (%/K)"
                                if is_specific_humidity
                                else f"Temperature vs {reference_label} (K/K)"
                            )
                            comparison_annotation = (
                                f"{heading}\n" + "\n".join(change_lines)
                            )

                vd = dict(
                    model_profiles=profiles,
                    ylabel=ylabel,
                    infobox_topleft=f"{_long_var_label(name, pressure_level)}",
                    infobox_topright=f"{ts}" if ts != "annual" else "",
                    infobox_bottomleft=infobox_region,
                    infobox_bottomright=comparison_annotation or None,
                )
                all_var_data.append(vd)

                # Single-variable PDF in the per-variable subdirectory
                self.plotter.plot(
                    model_profiles=profiles,
                    colors=colors,
                    ylabel=ylabel,
                    fname=f"{var_name}_latprofile.pdf",
                    output_path=output_path,
                    infobox_topleft=vd["infobox_topleft"],
                    infobox_topright=vd["infobox_topright"],
                    infobox_bottomleft=vd["infobox_bottomleft"],
                    infobox_bottomright=vd["infobox_bottomright"],
                )
                logger.info(
                    "    Saved single latitude profile: %s",
                    os.path.join(output_path, f"{var_name}_latprofile.pdf"),
                )

            # Plot all variables in chunks of 3 (one multi-panel PDF per ts)
            n_chunks = math.ceil(len(all_var_data) / 3) if all_var_data else 0
            for chunk_idx in range(n_chunks):
                chunk = all_var_data[chunk_idx * 3:(chunk_idx + 1) * 3]
                suffix = f"_part{chunk_idx + 1}" if n_chunks > 1 else ""
                fname = f"latprofile{suffix}.pdf"
                self.plotter.plot_multi(
                    var_data=chunk,
                    colors=colors,
                    output_path=parent_output_path,
                    fname=fname,
                )
                logger.info(
                    "    Saved multi-panel latitude profile: %s",
                    os.path.join(parent_output_path, fname),
                )


class PressureLatMap(XYMaps):
    """Zonal-mean pressure–latitude cross-section.

    Averages the data over *all* longitudes and the chosen *temporal_dim*,
    yielding a ``(plev, lat)`` map drawn with a logarithmic, top-to-bottom
    pressure y-axis.

    Because the full vertical profile is required, *pressure_level* in each
    variable dict is ignored – the 3-D field ``(time, plev, lat, lon)`` is
    always fetched and level selection is never applied.
    """

    def __init__(
        self,
        variables,
        xdim,
        ydim,
        temporal_selection=None,
        frequency="monthly",
        per_member=False,
        plotter_kwargs=None,
        compute_rmse=True,
        levels=None,
        stack_models_in_rows=False,
    ):
        super().__init__(
            variables,
            xdim,
            ydim,
            temporal_selection,
            frequency,
            per_member,
            plotter_kwargs,
            compute_rmse,
            stack_models_in_rows=stack_models_in_rows,
        )
        self.levels = levels
    def compute(
        self,
        data: xr.DataArray,
        temporal_dim: str,
        var: dict,
        frequency: str = "monthly",
    ) -> xr.DataArray:
        if "stat" in data.dims:
            data = data.sel(stat="mean", drop=True)
        # Zonal mean first, then temporal selection, then collapse remaining time dims.
        data = data.mean(dim="lon")
        data = self.select_by_time(data, temporal_dim)
        extra = [d for d in data.dims if d not in ("lat", "plev")]
        if extra:
            data = data.mean(dim=extra)
        return data.transpose("plev", "lat")

    def evaluate(self, data_containers) -> None:
        for ts in self.temporal_selection:
            logger.info(
                f"--> PressureLatMap: computing pressure–lat cross-section (temporal selection: {ts})"
            )
            for variable in self.variables:
                name, pressure_level = variable["name"], variable["pressure_level"]
                _log_variable_info(name, pressure_level)

                var_name = _format_var_name(name, pressure_level)
                output_path = os.path.join(self.plotter.output_path, ts, var_name)
                os.makedirs(output_path, exist_ok=True)

                # ── cache lookup ──────────────────────────────────────────────
                nc_paths = {
                    dc.model_label: self._nc_path(output_path, f"{dc.model_label}_{var_name}_plat")
                    for dc in data_containers
                }

                # maps: {model_label: section DataArray (plev, lat)}
                maps = {}

                if self._all_cached(list(nc_paths.values())):
                    logger.info(f"    Loading cached pressure–lat results from '{output_path}'")
                    for dc in data_containers:
                        ds = self._load_nc(nc_paths[dc.model_label])
                        maps[dc.model_label] = ds["data"].transpose("plev", "lat")
                        ds.close()
                else:
                    for dc in data_containers:
                        raw = dc.get_variable_data(
                            name=name, pressure_level=None, frequency=self.frequency
                        )
                        section = self.compute(raw, temporal_dim=ts, var=variable, frequency=self.frequency)
                        # ── save to cache ─────────────────────────────────────
                        self._save_nc(section, nc_paths[dc.model_label])
                        maps[dc.model_label] = section

                all_values = np.concatenate([s.values.ravel() for s in maps.values()])
                shared_vmin = float(np.nanmin(all_values))
                shared_vmax = float(np.nanmax(all_values))
                unit = self.plotter.cmor_units.get(name, "")

                for model_label, section in maps.items():
                    lat_vals  = section.lat.values
                    plev_vals = section.plev.values
                    z         = section.values  # (plev, lat)

                    if float(np.nanmin(z)) < 0 and float(np.nanmax(z)) > 0:
                        cmap = "bwr"
                        norm = CenteredNorm(vcenter=0)
                        vmin = vmax = None
                    else:
                        cmap = "coolwarm"
                        norm = None
                        vmin, vmax = shared_vmin, shared_vmax

                    fig, ax = plt.subplots(figsize=(6.7, 3.5), dpi=150)
                    cnt = ax.contourf(
                        lat_vals, plev_vals, z,
                        cmap=cmap, norm=norm,
                        vmin=vmin, vmax=vmax,
                        extend="both",
                    )
                    ax.set_yscale("log")
                    ax.invert_yaxis()
                    ax.set_xlabel(
                        "Latitude",
                        fontsize=self.plotter.fontdict.get("axes.labelsize", 12),
                    )
                    ax.set_ylabel(
                        r"Pressure ($\mathrm{Pa}$)",
                        fontsize=self.plotter.fontdict.get("axes.labelsize", 12),
                    )
                    ax.set_title(
                        f"{_long_var_label(name, None)}{_temporal_title_suffix(ts)} – {model_label}",
                        fontsize=self.plotter.fontdict.get("axes.titlesize", 14),
                    )
                    cbar = fig.colorbar(
                        cnt, ax=ax, orientation="vertical",
                        pad=0.05, shrink=0.8, extend="both",
                    )
                    cbar.set_label(unit, fontsize=10)
                    plt.tight_layout()
                    fpath = os.path.join(output_path, f"{model_label}_{var_name}_plat.pdf")
                    plt.savefig(fpath, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    logger.info(f"    Saved pressure–lat map: {fpath}")

class PressureLatBiasMap(PressureLatMap):
    """Pressure–latitude bias map (model minus reference)."""
    
    def eval_against_reference(self, model_section: xr.DataArray, ref_section: xr.DataArray) -> tuple:
        
        rmse = float(np.sqrt(np.nanmean((model_section.values - ref_section.values) ** 2)))
        bias = float(np.nanmean(model_section.values - ref_section.values))
        return rmse, bias

    def evaluate(self, data_containers) -> None:
        self.get_reference_data(data_containers)
        for ts in self.temporal_selection:
            logger.info(
                f"--> PressureLatBiasMap: computing pressure–lat bias map (temporal selection: {ts})"
            )
            for variable in self.variables:
                name, pressure_level = variable["name"], variable["pressure_level"]
                _log_variable_info(name, pressure_level)

                var_name = _format_var_name(name, pressure_level)
                output_path = os.path.join(self.plotter.output_path, ts, var_name)
                os.makedirs(output_path, exist_ok=True)

                # Non-reference model labels only
                non_ref_dcs = [dc for dc in data_containers if dc.model_label != self.reference_label]

                # ── cache lookup ──────────────────────────────────────────────
                nc_paths = {
                    dc.model_label: self._nc_path(output_path, f"{dc.model_label}_{var_name}_plat_bias")
                    for dc in non_ref_dcs
                }

                # maps: {model_label: (bias_section, output_path, var_name, rmse, bias_val)}
                maps = {}

                if self._all_cached(list(nc_paths.values())):
                    logger.info(f"    Loading cached pressure–lat bias results from '{output_path}'")
                    for dc in non_ref_dcs:
                        ds = self._load_nc(nc_paths[dc.model_label])
                        bias_section = ds["data"].transpose("plev", "lat")
                        rmse = float(ds.attrs["rmse"]) if "rmse" in ds.attrs else None
                        bias_val = float(ds.attrs["bias_val"]) if "bias_val" in ds.attrs else None
                        maps[dc.model_label] = (bias_section, output_path, var_name, rmse, bias_val)
                        ds.close()
                else:

                    for dc in data_containers:
                        
                        if dc.model_label == self.reference_label:
                            logger.info(f"    Skipping reference model ({dc.model_label}) for bias map.")
                            continue
                        data = dc.get_variable_data(**variable, frequency=self.frequency)

                        ref_data = self.reference_data.get_variable_data(
                            **variable, frequency=self.frequency
                        )

                        data = data.sel(plev=ref_data.plev, method="nearest", tolerance=100)  # Ensure levels match for bias calculation; tolerance avoids KeyError if exact match not found

                        model_section = self.compute(
                            data=data,
                            temporal_dim=ts,
                            var=variable,
                            frequency=self.frequency,
                        ).transpose("plev", "lat")
                        ref_section = self.compute(
                            data=ref_data,
                            temporal_dim=ts,
                            var=variable,
                            frequency=self.frequency,
                        ).transpose("plev", "lat")
                        
                        bias_section = model_section - ref_section
                        rmse, bias_val = self.eval_against_reference(model_section, ref_section)
                        logger.info(f"    {dc.model_label} – RMSE: {_fmt_scalar(rmse)}, Bias: {_fmt_scalar(bias_val)}")
                        # ── save to cache ─────────────────────────────────────
                        self._save_nc(
                            bias_section, nc_paths[dc.model_label],
                            attrs={"rmse": rmse, "bias_val": bias_val},
                        )
                        maps[dc.model_label] = (bias_section, output_path, var_name, rmse, bias_val)

                if not maps:
                    continue

                all_values = np.concatenate([bm.values.ravel() for bm, _, _, _, _ in maps.values()])
                abs_max = float(np.nanmax(np.abs(all_values)))
                p95 = float(np.nanpercentile(np.abs(all_values), 95))
                abs_max = min(abs_max, p95)  # Avoid outliers dominating the colorbar range; but if the data is very uniform, the 95th percentile may be the same
                shared_vmin = -abs_max
                shared_vmax = abs_max
                cbar_label = self.plotter.cmor_units.get(name, "")

                if self.stack_models_in_rows and len(maps) > 1:
                    fig_h = max(2.0 * len(maps), 4.0)
                    fig, axes = plt.subplots(
                        len(maps),
                        1,
                        figsize=(6.7, fig_h),
                        dpi=self.plotter.dpi,
                        squeeze=False,
                    )
                    axes = axes[:, 0]

                    mappable = None
                    for i, (model_label, (bias_section, _, _, rmse, bias_val)) in enumerate(maps.items()):
                        ax = axes[i]
                        lat_vals = bias_section.lat.values
                        plev_vals = bias_section.plev.values
                        z = bias_section.values
                        cnt = ax.contourf(
                            lat_vals,
                            plev_vals,
                            z,
                            cmap="bwr",
                            vmin=shared_vmin,
                            vmax=shared_vmax,
                            extend="both",
                        )
                        mappable = cnt
                        ax.set_yscale("log")
                        ax.invert_yaxis()
                        if i == len(maps) - 1:
                            ax.set_xlabel(
                                "Latitude",
                                fontsize=self.plotter.fontdict.get("axes.labelsize", 12),
                            )
                        ax.set_ylabel(
                            r"Pressure ($\mathrm{Pa}$)",
                            fontsize=max(self.plotter.fontdict.get("axes.labelsize", 12) - 1, 8),
                        )
                        ax.text(
                            -0.06,
                            0.5,
                            model_label,
                            transform=ax.transAxes,
                            ha="right",
                            va="center",
                            fontsize=self.plotter.fontdict.get("axes.labelsize", 12),
                            fontweight="bold",
                        )
                        info_txt = (
                            f"RMSE / Bias: {_fmt_scalar(rmse)} / {_fmt_scalar(bias_val)}"
                            if rmse is not None else ""
                        )
                        if info_txt:
                            ax.text(
                                1.01,
                                0.5,
                                info_txt,
                                transform=ax.transAxes,
                                ha="left",
                                va="center",
                                fontsize=max(self.plotter.fontdict.get("axes.labelsize", 12) - 2, 8),
                            )

                    fig.suptitle(
                        f"{_long_var_label(name, pressure_level)}{_temporal_title_suffix(ts)}",
                        fontsize=self.plotter.fontdict.get("axes.titlesize", 14),
                        fontweight="bold",
                        y=0.995,
                    )
                    if mappable is not None:
                        cbar = fig.colorbar(
                            mappable,
                            ax=axes,
                            orientation="horizontal",
                            pad=0.05,
                            shrink=0.95,
                            extend="both",
                        )
                        cbar.set_label(cbar_label, fontsize=10)

                    plt.tight_layout()
                    fpath = os.path.join(output_path, f"{var_name}_plat_bias_all_models.pdf")
                    plt.savefig(fpath, dpi=self.plotter.dpi, bbox_inches="tight")
                    plt.close(fig)
                    logger.info(f"    Saved stacked pressure–lat bias map: {fpath}")
                else:
                    for model_label, (bias_section, output_path, var_name, rmse, bias_val) in maps.items():
                        os.makedirs(output_path, exist_ok=True)
                        self.plotter.plot(
                            x=bias_section,
                            variable_name=f"{var_name}_bias",
                            title="",
                            infobox_topright=f"RMSE / Bias: {_fmt_scalar(rmse)} / {_fmt_scalar(bias_val)}" if rmse is not None else "",
                            model_label=model_label,
                            infobox_left=f"{model_label}",
                            infobox_topleft=f"{_long_var_label(name, pressure_level)}",
                            style="imshow",
                            output_path=output_path,
                            vmin=shared_vmin,
                            vmax=shared_vmax,
                            cbar_label=cbar_label,
                            cbar_orientation="horizontal",
                        )
                        fpath = os.path.join(output_path, f"{model_label}_{var_name}_plat_bias.pdf")
                        logger.info(f"    Saved pressure–lat bias map: {fpath}")

class XYBiasMaps(XYMaps):
    """Spatial bias maps (model minus reference)."""

    def evaluate(self, data_containers) -> None:
        self.get_reference_data(data_containers)
        for ts in self.temporal_selection:
            all_var_data = []  # list of per-variable dicts for the grid plotter

            for variable in self.variables:
                name, pressure_level = variable["name"], variable["pressure_level"]
                _log_variable_info(name, pressure_level)

                var_name = _format_var_name(name, pressure_level)
                output_path = os.path.join(self.plotter.output_path, ts, var_name)
                os.makedirs(output_path, exist_ok=True)

                # Non-reference model labels only
                non_ref_dcs = [dc for dc in data_containers if dc.model_label != self.reference_label]

                # ── cache lookup ──────────────────────────────────────────────
                nc_paths = {
                    dc.model_label: self._nc_path(output_path, f"{dc.model_label}_{var_name}_bias")
                    for dc in non_ref_dcs
                }

                # maps: {model_label: (bias_map, output_path, var_name, rmse, bias_val)}
                maps = {}

                if self._all_cached(list(nc_paths.values())):
                    logger.info(f"    Loading cached bias results from '{output_path}'")
                    for dc in non_ref_dcs:
                        ds = self._load_nc(nc_paths[dc.model_label])
                        bias_map = ds["data"].transpose(self.ydim, self.xdim)
                        rmse = float(ds.attrs["rmse"]) if "rmse" in ds.attrs else None
                        bias_val = float(ds.attrs["bias_val"]) if "bias_val" in ds.attrs else None
                        maps[dc.model_label] = (bias_map, output_path, var_name, rmse, bias_val)
                        ds.close()
                else:
                    for dc in data_containers:
                        if dc.model_label == self.reference_label:
                            logger.info(f"    Skipping reference model ({dc.model_label}) for bias map.")
                            continue
                        data = dc.get_variable_data(**variable, frequency=self.frequency)
                        ref_data = self.reference_data.get_variable_data(
                            **variable, frequency=self.frequency
                        )
                        model_map = self.compute(
                            data=data,
                            temporal_dim=ts,
                            var=variable,
                            frequency=self.frequency,
                        ).transpose(self.ydim, self.xdim)
                        ref_map = self.compute(
                            data=ref_data,
                            temporal_dim=ts,
                            var=variable,
                            frequency=self.frequency,
                        ).transpose(self.ydim, self.xdim)
                        bias_map = model_map - ref_map
                        if self.compute_rmse:
                            rmse, bias_val = self.eval_against_reference(model_map, ref_map)
                            logger.info(f"    {dc.model_label} – RMSE: {_fmt_scalar(rmse)}, Bias: {_fmt_scalar(bias_val)}")
                        else:
                            rmse, bias_val = None, None
                        self._save_nc(
                            bias_map, nc_paths[dc.model_label],
                            attrs={"rmse": rmse, "bias_val": bias_val},
                        )
                        maps[dc.model_label] = (bias_map, output_path, var_name, rmse, bias_val)

                if not maps:
                    continue

                all_values = np.concatenate([bm.values.ravel() for bm, _, _, _, _ in maps.values()])
                abs_max = float(np.nanmax(np.abs(all_values)))
                abs_min = float(np.nanmin(all_values))
                if self.p95:
                    p95 = float(np.nanpercentile(np.abs(all_values), 97.5))
                else:
                    p95 = abs_max
                abs_max = min(abs_max, p95)
                abs_min = max(abs_min, float(np.nanpercentile(all_values, 2.5)))
                shared_vmin = abs_min
                shared_vmax = abs_max

                cbar_label = self.plotter.cmor_units.get(name, "")
                all_var_data.append({
                    "title": f"{_long_var_label(name, pressure_level)}{_temporal_title_suffix(ts)}",
                    "cbar_label": cbar_label,
                    "vmin": shared_vmin,
                    "vmax": shared_vmax,
                    "maps": maps,
                })

            if not all_var_data:
                continue

            # ── CSV: RMSE and bias per model × variable ───────────────────────
            ts_output_path = os.path.join(self.plotter.output_path, ts)
            os.makedirs(ts_output_path, exist_ok=True)
            csv_rows = []
            for vd in all_var_data:
                for model_label, (_, _, var_name_entry, rmse, bias_val) in vd["maps"].items():
                    csv_rows.append({
                        "model": model_label,
                        "variable": vd["title"],
                        "rmse": rmse,
                        "bias": bias_val,
                    })
            if csv_rows:
                import pandas as _pd
                csv_path = os.path.join(ts_output_path, "bias_metrics.csv")
                _pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
                logger.info("    Saved bias metrics CSV: %s", csv_path)

            # ── Grid plots: models in rows, variables in columns (≤4 per figure)
            model_labels = list(all_var_data[0]["maps"].keys())
            ts_output_path = os.path.join(self.plotter.output_path, ts)
            os.makedirs(ts_output_path, exist_ok=True)
            n_chunks = math.ceil(len(all_var_data) / 4)
            for chunk_idx in range(n_chunks):
                chunk = all_var_data[chunk_idx * 4:(chunk_idx + 1) * 4]
                suffix = f"_part{chunk_idx + 1}" if n_chunks > 1 else ""
                fname = f"bias_all_vars{suffix}.pdf"
                self.plotter.plot_multi_var_bias_grid(
                    var_data=chunk,
                    model_labels=model_labels,
                    output_path=ts_output_path,
                    fname=fname,
                )
                logger.info(
                    "    Saved multi-variable bias grid: %s",
                    os.path.join(ts_output_path, fname),
                )


class XYAnomalyMaps(XYMaps):
    """Spatial anomaly maps relative to a configurable baseline period."""

    def __init__(
        self,
        variables: list,
        xdim: str,
        ydim: str,
        temporal_selection: list = None,
        baseline_period: tuple = ("1981-01-01T00", "2010-12-31T00"),
        frequency: str = "monthly",
        plotter_kwargs: dict = None,
        per_member: bool = False,
    ) -> None:
        super().__init__(
            variables=variables, xdim=xdim, ydim=ydim, temporal_selection=temporal_selection,
            frequency=frequency, plotter_kwargs=plotter_kwargs, per_member=per_member)
        self.baseline_period = baseline_period

    def compute(
        self,
        data: xr.DataArray,
        temporal_dim: str,
        var: dict,
        frequency: str = "monthly",
    ) -> xr.DataArray:
        if "stat" in data.dims:
            data = data.sel(stat="mean", drop=True)
        data = self.select_by_time(data, temporal_dim)
        data = data.mean(dim=[d for d in data.dims if d not in [self.xdim, self.ydim]])
        return compute_anomaly(
            data,
            mean_groups=None,
            baseline_mean_groups=None,
            baseline_period=self.baseline_period,
        )
    


        

# ---------------------------------------------------------------------------
# Season-to-month lookup used by XYTrendMaps
# ---------------------------------------------------------------------------
_SEASON_MONTHS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}


def _unit_per_decade(unit: str) -> str:
    """Return a TeX unit label for a per-decade trend.

    If *unit* is already a ``$...$``-wrapped mathtext string the inner content
    is extended with ``\\,\\mathrm{decade}^{-1}`` before re-wrapping so the
    whole expression sits inside one math group.  Plain text units fall back
    to a ``/decade`` suffix and an empty unit yields a bare
    ``$\\mathrm{decade}^{-1}$`` label.
    """
    if not unit:
        return r"$\mathrm{decade}^{-1}$"
    if unit.startswith("$") and unit.endswith("$"):
        inner = unit[1:-1]
        return f"${inner}\\,\\mathrm{{decade}}^{{-1}}$"
    return f"{unit}/decade"


def _build_selection_annotation(
    lat_band=None,
    lon_band=None,
    time_range=None,
) -> str:
    """Build a human-readable annotation string describing active selections.

    Used by :class:`RadialSpectrum` and :class:`Distribution` subclasses to
    annotate plots when a spatial or temporal restriction was applied before
    the computation.

    Parameters
    ----------
    lat_band : tuple (south_lat, north_lat) or None
    lon_band : tuple (west_lon, east_lon) or None
    time_range : list/tuple [start_str, end_str] or None

    Returns an empty string when no selection is active.
    """
    def _fmt_lat(v: float) -> str:
        return f"{abs(v):.0f}\u00b0{'N' if v >= 0 else 'S'}"

    def _fmt_lon(v: float) -> str:
        return f"{abs(v):.0f}\u00b0{'E' if v >= 0 else 'W'}"

    parts = []
    if time_range is not None:
        # If time range starts on the first of January and ends on the 31st of December, just show the years.
        start, end = time_range
        if start.endswith("-01-01") and end.endswith("-12-31"):
            start = start[:4]
            end = end[:4]
        parts.append(f"{start} \u2013 {end}")
    if lat_band is not None or lon_band is not None:
        if lon_band is not None:
            lons = sorted(lon_band)
            lon_str = f"{_fmt_lon(lons[0])}\u2013{_fmt_lon(lons[1])}"
        else:
            lon_str = "0\u00b0\u2013360\u00b0E"
        if lat_band is not None:
            lats = sorted(lat_band, reverse=True)  # north first, then south
            lat_str = f"{_fmt_lat(lats[0])}\u2013{_fmt_lat(lats[1])}"
        else:
            lat_str = "90\u00b0N\u201390\u00b0S"
        parts.append(f"Region: [{lon_str}, {lat_str}]")
    return "\n".join(parts)


def _normalize_band_list(band):
    """Normalize a band parameter to a list of band tuples.

    Accepts:
    - ``None``                        → ``[None]``       (single pass, no restriction)
    - ``(a, b)`` / ``[a, b]`` where *a*, *b* are numbers → ``[(a, b)]``
    - ``[(a1, b1), (a2, b2), …]``     → list of tuples  (multiple passes)

    This mirrors how ``time`` works in :class:`Distribution` – a single value
    causes one evaluation pass; a list causes one pass per element.
    """
    if band is None:
        return [None]
    # Flat pair: [a, b] where a and b are numbers.
    if len(band) == 2 and all(isinstance(x, (int, float)) for x in band):
        return [tuple(band)]
    # List of pairs.
    return [tuple(b) if b is not None else None for b in band]


def _latlon_dir(lat_band, lon_band) -> str:
    """Return a filesystem-safe directory-name segment for a lat/lon selection.

    Used to organise output into ``time-range / lat-lon / var`` sub-trees.
    Returns ``"global"`` when neither restriction is active.

    Examples
    --------
    >>> _latlon_dir((-30, 30), None)
    'lat-30_30'
    >>> _latlon_dir((-30, 30), (-180, 0))
    'lat-30_30_lon-180_0'
    >>> _latlon_dir(None, None)
    'global'
    """
    if lat_band is None and lon_band is None:
        return "global"
    parts = []
    if lat_band is not None:
        lat_band = list(lat_band)
        print(lat_band)
        s, n = min(lat_band), max(lat_band)
        parts.append(f"lat{s}_{n}")
    if lon_band is not None:
        lon_band = list(lon_band)
        w, e = min(lon_band), max(lon_band)
        parts.append(f"lon{w}_{e}")
    return "_".join(parts)


def _normalize_time_ranges(time_range) -> list:
    """Normalise *time_range* to a list of ``(start, end)`` string tuples.

    ``None``                          → ``[(None, None)]``  – no filtering
    ``["start", "end"]``             → ``[("start", "end")]``
    ``[["s1","e1"], ["s2","e2"]]``   → list of tuples
    """
    if time_range is None:
        return [(None, None)]
    if isinstance(time_range[0], str):
        return [(str(time_range[0]), str(time_range[1]))]
    return [(str(r[0]), str(r[1])) for r in time_range]


def _time_range_dir(start, end) -> str | None:
    """Filesystem-safe subdir name, or *None* when no range is active."""
    if start is None:
        return None
    return f"{start}_{end}"


def _normalize_season_list(season) -> list:
    """Normalise *season* to a list of season strings (or ``[None]``).

    ``None``          → ``[None]``            – no filtering (all months)
    ``"JJA"``         → ``["JJA"]``           – single season
    ``["DJF","JJA"]`` → ``["DJF", "JJA"]``   – multiple passes
    """
    if season is None:
        return [None]
    if isinstance(season, str):
        return [season.upper()]
    return [str(s).upper() for s in season]


class XYTrendMaps(XYMaps):
    """Spatial map of the linear-regression slope (per decade) at every grid point.

    For each variable the full time series (optionally restricted to
    *trend_period*) is regressed against a fractional-year time axis using
    vectorised OLS.  The resulting ``(lat, lon)`` map shows the local rate of
    change in units of ``[variable unit] / decade``.

    When *temporal_dim* (from ``temporal_selection``) is one of the four
    standard seasons (``"DJF"``, ``"MAM"``, ``"JJA"``, ``"SON"``), only the
    months belonging to that season are included before computing the trend.
    When it is ``"annual"``, yearly means are computed first.  Any other value
    (or ``None``) uses the raw monthly time series.
    """

    def __init__(
        self,
        variables: list,
        xdim: str,
        ydim: str,
        temporal_selection: list = None,
        trend_period: tuple = None,
        frequency: str = "monthly",
        plotter_kwargs: dict = None,
        per_member: bool = False,
    ) -> None:
        super().__init__(
            variables=variables, xdim=xdim, ydim=ydim, temporal_selection=temporal_selection,
            frequency=frequency, plotter_kwargs=plotter_kwargs, per_member=per_member)
        self.trend_period = trend_period

    def compute(
        self,
        data: xr.DataArray,
        temporal_dim: str,
        var: dict,
        frequency: str = "monthly",
    ) -> xr.DataArray:
        """Return a ``(lat, lon)`` DataArray of OLS slope per decade."""
        if "stat" in data.dims:
            data = data.sel(stat="mean", drop=True)

        # Optional time window restriction.
        if self.trend_period is not None:
            t = data.time.values
            mask = (
                (t >= np.datetime64(self.trend_period[0], "ns"))
                & (t <= np.datetime64(self.trend_period[1], "ns"))
            )
            data = data.isel(time=mask)

        # Aggregate to seasonal or annual means before fitting when requested.
        if temporal_dim in _SEASON_MONTHS:
            months = _SEASON_MONTHS[temporal_dim]
            data = data.isel(time=data.time.dt.month.isin(months))
            data = data.resample(time="YE").mean()
        elif temporal_dim == "annual":
            data = data.resample(time="YE").mean()
        # else: raw monthly (or whatever frequency was loaded)

        # Drop time steps where all grid points are NaN (e.g. incomplete years
        # after resampling).
        data = data.dropna(dim="time", how="all")

        # Time axis in fractional years, centred for numerical stability.
        times_days = data.time.values.astype("datetime64[D]").astype(np.float64)
        times_years = times_days / 365.25
        times_years -= times_years.mean()

        # Vectorised OLS: beta = cov(t, y) / var(t)  for every grid point.
        data_vals = data.values  # (T, lat, lon)
        t = times_years           # (T,)
        t_mean = t.mean()
        y_mean = data_vals.mean(axis=0)                                       # (lat, lon)
        cov = (
            (t[:, np.newaxis, np.newaxis] - t_mean) * (data_vals - y_mean)
        ).mean(axis=0)
        var_t = float(((t - t_mean) ** 2).mean())
        beta = cov / var_t if var_t > 0 else np.zeros_like(cov)              # (lat, lon)
        slope_per_decade = beta * 10.0

        return xr.DataArray(
            slope_per_decade,
            coords={"lat": data.lat, "lon": data.lon},
            dims=["lat", "lon"],
        )

    def evaluate(self, data_containers) -> None:
        ts_list = self.temporal_selection if self.temporal_selection else ["full"]
        for ts in ts_list:
            logger.info(
                f"--> XYTrendMaps: computing trend map (temporal selection: {ts})"
            )
            for variable in self.variables:
                name, pressure_level = variable["name"], variable["pressure_level"]
                _log_variable_info(name, pressure_level)

                var_name = _format_var_name(name, pressure_level)
                output_path = os.path.join(self.plotter.output_path, f"trend_{ts}", var_name)
                os.makedirs(output_path, exist_ok=True)

                # ── cache lookup ──────────────────────────────────────────────
                nc_paths = {
                    dc.model_label: self._nc_path(output_path, f"{dc.model_label}_{var_name}_trend")
                    for dc in data_containers
                }

                # maps: {model_label: (trend_map, output_path, var_name)}
                maps = {}

                if self._all_cached(list(nc_paths.values())):
                    logger.info(f"    Loading cached trend results from '{output_path}'")
                    for dc in data_containers:
                        ds = self._load_nc(nc_paths[dc.model_label])
                        maps[dc.model_label] = (
                            ds["data"].transpose(self.ydim, self.xdim),
                            output_path, var_name,
                        )
                        ds.close()
                else:
                    for dc in data_containers:
                        raw = dc.get_variable_data(**variable, frequency=self.frequency)
                        trend_map = self.compute(
                            raw, temporal_dim=ts, var=variable, frequency=self.frequency
                        )
                        trend_map = trend_map.transpose(self.ydim, self.xdim)
                        # ── save to cache ─────────────────────────────────────
                        self._save_nc(trend_map, nc_paths[dc.model_label])
                        maps[dc.model_label] = (trend_map, output_path, var_name)

                # Symmetric colorbar centred on zero (98th percentile of |values|).
                all_values = np.concatenate(
                    [m.values.ravel() for m, _, _ in maps.values()]
                )
                abs_max = float(np.nanpercentile(np.abs(all_values[~np.isnan(all_values)]), 98))
                shared_vmin = -abs_max
                shared_vmax = abs_max
                unit = self.plotter.cmor_units.get(name, "")
                cbar_label = _unit_per_decade(unit)

                for model_label, (trend_map, output_path, var_name) in maps.items():
                    self.plotter.plot(
                        x=trend_map,
                        variable_name=f"{var_name}_trend",
                        title=f"Trend ({ts}) – {model_label}",
                        model_label=model_label,
                        style="imshow",
                        output_path=output_path,
                        vmin=shared_vmin,
                        vmax=shared_vmax,
                        cbar_label=cbar_label,
                    )


# ============================================================================
# Variable-pair correlation maps
# ============================================================================

class VariableCorrelationMaps(SpatialMetric):
    """Point-wise Pearson correlation between pairs of variables.

    For each pair ``(var1, var2)`` and each model a ``(lat, lon)`` map of the
    Pearson correlation coefficient is computed over the full (or
    season-filtered) time axis and saved as a global map.

    Parameters
    ----------
    variable_pairs:
        List of two-element lists / tuples, each element being a variable
        dict ``{"name": str, "pressure_level": int | None}``.
        Example::

            variable_pairs:
              - [{name: tas, pressure_level: null},
                 {name: psl, pressure_level: null}]
    temporal_selection:
        One season label per evaluation run.  ``"annual"`` keeps all months;
        ``"DJF"`` / ``"MAM"`` / ``"JJA"`` / ``"SON"`` filter to those months
        before computing the correlation.  Defaults to ``["annual"]``.
    correlate_against_reference_var1:
        When *True*, the correlation for each non-reference model is computed
        between *var2* of that model and *var1* of the **reference** model
        (instead of *var1* of the same model).  Useful for measuring how well
        a model's response variable covaries with the observed forcing.
        The reference model name is appended to the top-right annotation box.
    frequency:
        Temporal resolution of the source data.
    plotter_kwargs:
        Forwarded to :class:`~plot.modules.SpatialPlotter`.
    """

    # Month numbers corresponding to each standard season label.
    _SEASON_MONTHS: dict = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }

    def __init__(
        self,
        variable_pairs: list,
        xdim: str = "lon",
        ydim: str = "lat",
        temporal_selection: list = None,
        correlate_against_reference_var1: bool = False,
        time_range=None,
        frequency: str = "monthly",
        plotter_kwargs: dict = None,
        per_member: bool = False,
    ) -> None:
        # Collect the unique variable specs so the parent stores them;
        # they are not used for actual data fetching (done pair-by-pair here).
        _seen, _vars = [], []
        for pair in variable_pairs:
            for v in pair:
                key = (v["name"], v.get("pressure_level"))
                if key not in _seen:
                    _seen.append(key)
                    _vars.append(dict(v))
        super().__init__(
            variables=_vars,
            xdim=xdim,
            ydim=ydim,
            temporal_selection=temporal_selection,
            frequency=frequency,
            plotter_kwargs=plotter_kwargs,
            per_member=per_member,
        )
        # Normalise pairs to list of (dict, dict).
        self.variable_pairs: list = [
            (dict(p[0]), dict(p[1])) for p in variable_pairs
        ]
        self.correlate_against_reference_var1 = correlate_against_reference_var1
        self.time_ranges = self._normalize_time_ranges(time_range)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _pearson_map(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Point-wise Pearson r over the leading (time) axis.

        Returns a float32 ``(lat, lon)`` array.  Grid cells where either
        input is all-NaN are returned as NaN.
        """
        a = arr1 - np.nanmean(arr1, axis=0, keepdims=True)
        b = arr2 - np.nanmean(arr2, axis=0, keepdims=True)
        num   = np.nansum(a * b, axis=0)
        denom = np.sqrt(np.nansum(a ** 2, axis=0) * np.nansum(b ** 2, axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(denom > 0, num / denom, np.nan)
        return r.astype(np.float32)

    @staticmethod
    def _season_filter(data: xr.DataArray, months: list) -> xr.DataArray:
        """Keep only time steps whose calendar month is in *months*."""
        return data.isel(time=data.time.dt.month.isin(months))

    @staticmethod
    def _short_var_label(name: str, pressure_level=None) -> str:
        """Short uppercase label, e.g. ``"TAS"`` or ``"UA (850 hPa)"``."""
        label = name.upper()
        if pressure_level is not None:
            hpa = int(round(pressure_level / 100))
            return f"{label} ({hpa} hPa)"
        return label

    @staticmethod
    def _normalize_time_ranges(time_range) -> list:
        """Normalise *time_range* to a list of ``(start, end)`` string tuples.

        ``None``                     → ``[(None, None)]``  – no filtering
        ``["s", "e"]``              → ``[("s", "e")]``    – single range
        ``[["s1","e1"],["s2","e2"]]`` → list of tuples  – multiple ranges
        """
        if time_range is None:
            return [(None, None)]
        if isinstance(time_range[0], str):
            return [(str(time_range[0]), str(time_range[1]))]
        return [(str(r[0]), str(r[1])) for r in time_range]

    @staticmethod
    def _time_range_dir(start: str | None, end: str | None) -> str | None:
        """Filesystem-safe subdirectory name, or *None* if no range."""
        if start is None:
            return None
        return f"{start}_{end}"

    @staticmethod
    def _time_range_label(start: str | None, end: str | None) -> str | None:
        """Human-readable annotation string, or *None* if no range."""
        if start is None:
            return None
        return f"{start} \u2013 {end}"

    @staticmethod
    def _prepare_stat_only(data: xr.DataArray) -> xr.DataArray:
        """Drop only the stat dimension; preserve the member dim if present."""
        if "stat" in data.dims:
            data = data.sel(stat="mean", drop=True)
        return data

    @staticmethod
    def _prepare(data: xr.DataArray) -> xr.DataArray:
        """Drop stat/member dimensions if present."""
        if "stat" in data.dims:
            data = data.sel(stat="mean", drop=True)
        if "member" in data.dims:
            data = data.mean("member")
        return data

    # ── evaluate ─────────────────────────────────────────────────────────────

    def _plot_corr_map(
        self,
        corr_da: xr.DataArray,
        output_path: str,
        variable_name: str,
        infotext_topleft: str,
        infotext_topright: str,
        infotext_left: str,
    ) -> None:
        """Render and save a single Pearson-r map."""
        self.plotter.plot(
            x=corr_da,
            variable_name=variable_name,
            title="",
            model_label=infotext_left,
            style="imshow",
            output_path=output_path,
            vmin=-1.0,
            vmax=1.0,
            cbar_label="Pearson r",
            cmap="bwr",
            infotext_topleft=infotext_topleft,
            infotext_topright=infotext_topright,
            infotext_left=infotext_left,
            cbar_orientation="horizontal",
        )

    def _compute_corr(
        self,
        dv1: xr.DataArray,
        dv2: xr.DataArray,
        season_months: list | None,
        time_range: tuple[str, str] | None = None,
    ) -> xr.DataArray:
        """Time-slice, season-filter, align and compute point-wise Pearson r map."""
        if time_range is not None:
            dv1 = dv1.sel(time=slice(*time_range))
            dv2 = dv2.sel(time=slice(*time_range))
        if season_months:
            dv1 = self._season_filter(dv1, season_months)
            dv2 = self._season_filter(dv2, season_months)
        dv1, dv2 = xr.align(dv1, dv2, join="inner")
        r = self._pearson_map(
            dv1.transpose("time", self.ydim, self.xdim).values,
            dv2.transpose("time", self.ydim, self.xdim).values,
        )
        return xr.DataArray(
            r,
            coords={self.ydim: dv1[self.ydim], self.xdim: dv1[self.xdim]},
            dims=[self.ydim, self.xdim],
        )

    def evaluate(self, data_containers) -> None:  # noqa: C901
        self.get_reference_data(data_containers)

        for ts in self.temporal_selection:
            logger.info(
                f"--> VariableCorrelationMaps: computing correlations "
                f"(temporal selection: {ts})"
            )
            season_months = self._SEASON_MONTHS.get(ts)  # None → all months

            for var1_spec, var2_spec in self.variable_pairs:
                var1_name = var1_spec["name"]
                var1_plev = var1_spec.get("pressure_level", None)
                var2_name = var2_spec["name"]
                var2_plev = var2_spec.get("pressure_level", None)

                # Top-left: short variable names, one per line to avoid clutter.
                lbl1 = self._short_var_label(var1_name, var1_plev)
                lbl2 = self._short_var_label(var2_name, var2_plev)
                infotext_topleft = f"{lbl1} /\n{lbl2}"

                pair_str = (
                    f"{_format_var_name(var1_name, var1_plev)}_vs_"
                    f"{_format_var_name(var2_name, var2_plev)}"
                )
                base_pair_dir = os.path.join(
                    self.plotter.output_path, ts, pair_str
                )
                os.makedirs(base_pair_dir, exist_ok=True)
                logger.info(f"    Pair: {pair_str}")

                # Pre-fetch var1 of the reference once per pair (reused across
                # all time ranges).  Season-filter is applied here; _compute_corr
                # will additionally apply the per-iteration time slice.
                ref_var1_arr: xr.DataArray | None = None
                if (
                    self.correlate_against_reference_var1
                    and self.reference_data is not None
                ):
                    _rv1 = self._prepare(
                        self.reference_data.get_variable_data(
                            **var1_spec, frequency=self.frequency
                        )
                    )
                    if season_months:
                        _rv1 = self._season_filter(_rv1, season_months)
                    ref_var1_arr = _rv1

                # ── loop over (optional) time ranges ──────────────────────────
                for tr_start, tr_end in self.time_ranges:
                    time_range  = (tr_start, tr_end) if tr_start is not None else None
                    time_dir    = self._time_range_dir(tr_start, tr_end)
                    time_label  = self._time_range_label(tr_start, tr_end)

                    output_dir = (
                        os.path.join(base_pair_dir, time_dir)
                        if time_dir else base_pair_dir
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    for dc in data_containers:
                        label = dc.model_label
                        use_ref_var1 = (
                            self.correlate_against_reference_var1
                            and ref_var1_arr is not None
                            and label != self.reference_label
                        )

                        # Top-right: season + optional time range + cross-model note.
                        tr_parts = [ts]
                        if time_label:
                            tr_parts.append(time_label)
                        if use_ref_var1:
                            tr_parts.append(f"(var1: {self.reference_label})")
                        infotext_tr = "\n".join(tr_parts)

                        if self.per_member:
                            # ── per-member path ───────────────────────────────────────
                            dv2_raw = self._prepare_stat_only(
                                dc.get_variable_data(
                                    **var2_spec, frequency=self.frequency, all_members=True
                                )
                            )
                            if use_ref_var1:
                                dv1_raw = ref_var1_arr  # already season-filtered
                            else:
                                dv1_raw = self._prepare_stat_only(
                                    dc.get_variable_data(
                                        **var1_spec, frequency=self.frequency, all_members=True
                                    )
                                )

                            has_members = "member" in dv2_raw.dims

                            if has_members:
                                model_dir = os.path.join(output_dir, label)
                                os.makedirs(model_dir, exist_ok=True)

                                member_vals = dv2_raw.member.values
                                member_corr_maps: list = []

                                for idx, m in enumerate(member_vals):
                                    dv2_m = dv2_raw.sel(member=m, drop=True)
                                    dv1_m = (
                                        dv1_raw.sel(member=m, drop=True)
                                        if "member" in dv1_raw.dims
                                        else dv1_raw
                                    )
                                    nc_path = self._nc_path(
                                        model_dir,
                                        f"{label}_{pair_str}_{ts}_corr_m{idx}",
                                    )
                                    if self._all_cached([nc_path]):
                                        ds = self._load_nc(nc_path)
                                        corr_da = ds["data"].transpose(self.ydim, self.xdim)
                                        ds.close()
                                    else:
                                        corr_da = self._compute_corr(
                                            dv1_m, dv2_m, season_months, time_range
                                        )
                                        self._save_nc(corr_da, nc_path)

                                    member_corr_maps.append(corr_da)
                                    self._plot_corr_map(
                                        corr_da=corr_da,
                                        output_path=model_dir,
                                        variable_name=f"{pair_str}_m{idx}",
                                        infotext_topleft=infotext_topleft,
                                        infotext_topright=infotext_tr,
                                        infotext_left=f"{label} (m{idx})",
                                    )
                                    logger.info(
                                        f"    Saved per-member corr map: {label} m{idx} – {pair_str}"
                                    )

                                # Member-mean = average of per-member correlation maps.
                                if member_corr_maps:
                                    nc_path_mean = self._nc_path(
                                        model_dir,
                                        f"{label}_{pair_str}_{ts}_corr_member_mean",
                                    )
                                    if self._all_cached([nc_path_mean]):
                                        ds = self._load_nc(nc_path_mean)
                                        corr_mean_da = ds["data"].transpose(self.ydim, self.xdim)
                                        ds.close()
                                    else:
                                        mean_r = np.nanmean(
                                            np.stack(
                                                [c.values for c in member_corr_maps], axis=0
                                            ),
                                            axis=0,
                                        )
                                        corr_mean_da = xr.DataArray(
                                            mean_r,
                                            coords=member_corr_maps[0].coords,
                                            dims=member_corr_maps[0].dims,
                                        )
                                        self._save_nc(corr_mean_da, nc_path_mean)

                                    tr_mean_parts = [ts]
                                    if time_label:
                                        tr_mean_parts.append(time_label)
                                    tr_mean_parts.append("member mean")
                                    if use_ref_var1:
                                        tr_mean_parts.append(f"(var1: {self.reference_label})")
                                    self._plot_corr_map(
                                        corr_da=corr_mean_da,
                                        output_path=model_dir,
                                        variable_name=f"{pair_str}_member_mean",
                                        infotext_topleft=infotext_topleft,
                                        infotext_topright="\n".join(tr_mean_parts),
                                        infotext_left=f"{label} (member mean)",
                                    )
                                    logger.info(
                                        f"    Saved member-mean corr map: {label} – {pair_str}"
                                    )
                                continue  # skip the standard path below

                            # No member dim – fall through to the standard path.
                            dv2_raw = self._prepare(dv2_raw)
                            if not use_ref_var1:
                                dv1_raw = self._prepare(dv1_raw)

                        # ── standard path ─────────────────────────────────────────
                        nc_path = self._nc_path(
                            output_dir,
                            f"{label}_{pair_str}_{ts}_corr",
                        )

                        if self._all_cached([nc_path]):
                            logger.info(
                                f"    Loading cached correlation '{label}' ← '{nc_path}'"
                            )
                            ds = self._load_nc(nc_path)
                            corr_da = ds["data"].transpose(self.ydim, self.xdim)
                            ds.close()
                        else:
                            if self.per_member:
                                # Already prepared above (no member dim case).
                                dv2 = dv2_raw
                                dv1 = dv1_raw if not use_ref_var1 else ref_var1_arr
                            else:
                                dv2 = self._prepare(
                                    dc.get_variable_data(**var2_spec, frequency=self.frequency)
                                )
                                if use_ref_var1:
                                    dv1 = ref_var1_arr
                                else:
                                    dv1 = self._prepare(
                                        dc.get_variable_data(
                                            **var1_spec, frequency=self.frequency
                                        )
                                    )
                            corr_da = self._compute_corr(dv1, dv2, season_months, time_range)
                            self._save_nc(corr_da, nc_path)
                            logger.info(f"    Computed and cached correlation for '{label}'")

                        self._plot_corr_map(
                            corr_da=corr_da,
                            output_path=output_dir,
                            variable_name=pair_str,
                            infotext_topleft=infotext_topleft,
                            infotext_topright=infotext_tr,
                            infotext_left=label,
                        )
                        logger.info(
                            f"    Saved correlation map: {label} – {pair_str} ({ts})"
                        )


# ============================================================================
# Timeseries metrics
# ============================================================================

class Timeseries(TimeseriesMetric):
    """
    Thin base class for time-series evaluators.

    Concrete work is done by :class:`SeasonalCycles` and
    :class:`SouthernOscillationIndex`.  The spectrum helper is available to
    all subclasses.
    """

    def compute(self, data_container, variable: str):
        data_container.get_variable_data(variable_name=variable, frequency="monthly")

    def spectrum(self, data: xr.DataArray, fs: float = 1.0):
        fx, fy = spectral.welch_psd(data, fs=fs)
        return fx, fy

    def evaluate(self, data_containers) -> None:
        pass


def _compute_linear_trend(x_values, y_values, time_step: str = "monthly"):
    """Return (trend_line, slope_per_step) from a 1-D time series.

    Parameters
    ----------
    time_step:
        ``"yearly"`` if each index step represents one year;
        ``"monthly"`` if each step represents one month.
        The returned slope is already expressed per the given step so the
        caller can label it as "per year" or "per month" directly.
    """
    coeffs = np.polyfit(range(len(x_values)), y_values, deg=1)
    trend = np.polyval(coeffs, range(len(x_values)))
    slope_per_step = float(coeffs[0])
    return trend, slope_per_step


class SeasonalCycles(Timeseries):
    """Global-mean seasonal cycle (or anomaly cycle) per variable.

    Extra options
    -------------
    lat_band : tuple or list of tuples, optional
        Latitude restriction applied before averaging.  A single pair
        ``(south, north)`` or a list of pairs for multiple passes.
        Defaults to the global average (``None``).
    lon_band : tuple or list of tuples, optional
        Longitude restriction, same convention as *lat_band*.
    compute_psd : bool, default False
        When ``True`` the power spectral density of each cycle is computed
        (using Welch's method) and saved as a separate ``*_psd.pdf`` /
        ``*_psd.nc`` alongside the cycle plot.
    """

    def __init__(
        self,
        variables: list,
        mean_groups: list = None,
        linear_trend: bool = False,
        detrend: bool = False,
        compute_anomalies: bool = False,
        baseline_period: tuple = None,
        baseline_mean_groups: list = None,
        lat_band=None,
        lon_band=None,
        compute_psd: bool = False,
        time_range=None,
        season=None,
        plotter_kwargs: dict = None,
        per_member: bool = False,
    ) -> None:
        super().__init__(
            variables=variables,
            detrend=detrend,
            compute_anomalies=compute_anomalies,
            baseline_period=baseline_period,
            baseline_mean_groups=baseline_mean_groups,
            mean_groups=mean_groups or ["year"],
            plotter_kwargs=plotter_kwargs,
            per_member=per_member,
        )
        if linear_trend and detrend:
            raise ValueError("Cannot apply both linear_trend and detrend simultaneously.")
        self.linear_trend = linear_trend
        # Spatial restriction – normalized to list form for iteration.
        self.lat_band = lat_band
        self.lon_band = lon_band
        self.compute_psd = compute_psd
        # Time range list – normalized to [(None, None)] when not specified.
        self.time_ranges = _normalize_time_ranges(time_range)
        self._current_time_range: tuple = (None, None)
        # Season filter – normalized to [None] (=all months) when not specified.
        self.seasons = _normalize_season_list(season)
        self._current_season: str | None = None

    def cosine_lat_weights(self, data: xr.DataArray) -> xr.DataArray:
        """Return a 1-D array of latitude weights for area-weighted mean."""
        lat_rad = np.deg2rad(data.lat)
        w = np.cos(lat_rad)
        w /= w.mean()  # normalise so the global mean is on the same scale as unweighted
        return w

    def compute(self, data: xr.DataArray) -> xr.DataArray:
        # Apply optional time range restriction.
        tr_start, tr_end = self._current_time_range
        if tr_start is not None:
            data = data.sel(time=slice(tr_start, tr_end))
        # Apply optional season filter.
        if self._current_season is not None:
            months = _SEASON_MONTHS.get(self._current_season, [])
            if months:
                data = data.isel(time=data.time.dt.month.isin(months))
        # Apply optional spatial restriction before the global mean.
        if self.lat_band is not None:
            lat_min, lat_max = min(self.lat_band), max(self.lat_band)
            if data.lat.values[0] > data.lat.values[-1]:  # descending
                data = data.sel(lat=slice(lat_max, lat_min))
            else:
                data = data.sel(lat=slice(lat_min, lat_max))
        if self.lon_band is not None:
            lon_min, lon_max = min(self.lon_band), max(self.lon_band)
            data = data.sel(lon=slice(lon_min, lon_max))

        w = self.cosine_lat_weights(data)
        data.values = data.values * w.values[np.newaxis, :, np.newaxis]
        data = data.mean(dim=["lat", "lon"])
        

        if self.detrend:
            data.values = detrend_data(data).values

        if self.compute_anomalies:
            seasonal_cycle = compute_anomaly(
                data=data,
                baseline_period=self.baseline_period,
                mean_groups=self.mean_groups,
                baseline_mean_groups=self.baseline_mean_groups,
            )
        else:
            seasonal_cycle = data.groupby(self.mean_groups).mean(dim=["time"])

        time = self.xlabels_from_time(seasonal_cycle)
        seasonal_cycle = seasonal_cycle.stack(
            time=[g.split(".")[-1] for g in self.mean_groups]
        ).reset_index("time")
        seasonal_cycle["time"] = time
        return seasonal_cycle.dropna(dim="time", how="any")

    def _monthly_spatial_mean(self, data: xr.DataArray) -> xr.DataArray:
        """Apply time-slice and spatial selection then return a monthly 1-D mean.

        Mirrors the first steps of :meth:`compute` (time slice + lat/lon
        selection + spatial mean) but skips the groupby step, so the result
        retains the full monthly time axis.  Used for PSD computation so that
        the frequency axis is always in cycles month⁻¹.
        """
        tr_start, tr_end = self._current_time_range
        if tr_start is not None:
            data = data.sel(time=slice(tr_start, tr_end))
        # Apply optional season filter.
        if self._current_season is not None:
            months = _SEASON_MONTHS.get(self._current_season, [])
            if months:
                data = data.isel(time=data.time.dt.month.isin(months))
        if self.lat_band is not None:
            lat_min, lat_max = min(self.lat_band), max(self.lat_band)
            if data.lat.values[0] > data.lat.values[-1]:  # descending grid
                data = data.sel(lat=slice(lat_max, lat_min))
            else:
                data = data.sel(lat=slice(lat_min, lat_max))
        if self.lon_band is not None:
            lon_min, lon_max = min(self.lon_band), max(self.lon_band)
            data = data.sel(lon=slice(lon_min, lon_max))
        return data.mean(dim=["lat", "lon"])

    def evaluate(self, data_containers):
        base_output_path = self.timeseries_plotter.output_path
        lat_bands = _normalize_band_list(self.lat_band)
        lon_bands = _normalize_band_list(self.lon_band)
        # Pre-compute all (time_range, season, lat_band, lon_band) combos.
        combos = [
            ((tr_start, tr_end), season, lat_b, lon_b)
            for tr_start, tr_end in self.time_ranges
            for season in self.seasons
            for lat_b in lat_bands
            for lon_b in lon_bands
        ]

        for variable in self.variables:
            name, pressure_level = variable["name"], variable["pressure_level"]
            _log_variable_info(name, pressure_level)

            var_name = _format_var_name(name, pressure_level)

            # Determine temporal resolution once (needed both for caching and
            # for the trend-slope correction).
            has_month = any("month" in g for g in self.mean_groups)
            is_monthly_only = self.mean_groups in [["time.month"], ["time.dayofyear"]]
            time_step = "monthly" if has_month else "yearly"
            steps_per_decade = 120 if time_step == "monthly" else 10
            trend_unit = "/decade"

            for (tr_start, tr_end), season, lat_b, lon_b in combos:
                # Make the active selection available to self.compute().
                self._current_time_range = (tr_start, tr_end)
                self._current_season = season
                time_dir = _time_range_dir(tr_start, tr_end)
                self.lat_band = lat_b
                self.lon_band = lon_b
                latlon_d = _latlon_dir(lat_b, lon_b)

                # Build subdirectory path: latlon / [time /] [season /] var
                sub_parts = [latlon_d]
                if time_dir:
                    sub_parts.append(time_dir)
                if season:
                    sub_parts.append(season)
                sub_dir = os.path.join(*sub_parts)

                output_path = os.path.join(base_output_path, sub_dir, var_name)
                os.makedirs(output_path, exist_ok=True)

                # Fname prefix reused for cycle plot and PSD plot.
                fname_base = f"{sub_dir}/{var_name}"

                # ── cache lookup ──────────────────────────────────────────
                nc_paths = {
                    dc.model_label: self._nc_path(output_path, f"{dc.model_label}_{var_name}_cycle")
                    for dc in data_containers
                }

                cycles: dict = {}
                standard_deviations: dict = {}
                members_per_model: dict = {}
                trends: dict = {}

                if self._all_cached(list(nc_paths.values())):
                    logger.info(f"    Loading cached seasonal-cycle results from '{output_path}'")
                    for dc in data_containers:                            
                        ds = self._load_nc(nc_paths[dc.model_label])
                        t_labels = ds["time_labels"].values.astype(str)
                        cycle = xr.DataArray(
                            ds["data"].values, coords={"time": t_labels}, dims=["time"]
                        )
                        cycles[dc.model_label] = cycle
                        if "std" in ds:
                            standard_deviations[dc.model_label] = xr.DataArray(
                                ds["std"].values, coords={"time": t_labels}, dims=["time"]
                            )
                        if is_monthly_only:
                            i = 0
                            while f"member_{i}" in ds:
                                members_per_model.setdefault(dc.model_label, []).append(
                                    xr.DataArray(ds[f"member_{i}"].values, coords={"time": t_labels}, dims=["time"])
                                )
                                i += 1
                        if "trend" in ds:
                            trend_da = xr.DataArray(
                                ds["trend"].values, coords={"time": t_labels}, dims=["time"]
                            )
                            slope = float(ds.attrs.get("trend_slope", 0.0))
                            trends[dc.model_label] = (trend_da, slope)
                        ds.close()
                else:
                    for dc in data_containers:
                        logger.info(f"--> SeasonalCycles: processing '{dc.model_label}'")
                        _freq = "daily" if any("dayofyear" in g for g in self.mean_groups) else "monthly"
                        data = dc.get_variable_data(
                            name=name, pressure_level=pressure_level,
                            frequency=_freq, all_members=True,
                        )

                        if "member" in data.dims:
                            cycle = self.compute(data.mean("member"))
                            mc = [
                                self.compute(data.sel(member=m, drop=True))
                                for m in data.member.values
                            ]
                            if is_monthly_only:
                                members_per_model[dc.model_label] = mc
                                logger.info(
                                    f"    Stored {len(mc)} individual member cycles for '{dc.model_label}'."
                                )
                            else:
                                spread_cycle = xr.concat(mc, dim="member").std("member")
                                standard_deviations[dc.model_label] = spread_cycle
                                logger.info(
                                    f"    Computed ensemble spread from {len(data.member.values)} members."
                                )
                        else:
                            cycle = self.compute(data)
                        cycles[dc.model_label] = cycle

                    # Compute trends before saving so they go into the cache too.
                    if self.linear_trend:
                        for label, da in cycles.items():
                            x_pos = np.arange(len(da))
                            trend_values, m = _compute_linear_trend(
                                x_pos, da.values, time_step=time_step)
                            trend_da = xr.DataArray(trend_values, coords=da.coords, dims=da.dims)
                            trends[label] = (trend_da, m * steps_per_decade)

                    # ── save to cache ─────────────────────────────────────
                    for dc in data_containers:
                        label = dc.model_label
                        cycle = cycles[label]
                        t_labels = np.array(cycle.time.values, dtype=str)
                        ds_vars: dict = {
                            "data": xr.DataArray(cycle.values, dims=["t_idx"]),
                            "time_labels": xr.DataArray(t_labels, dims=["t_idx"]),
                        }
                        save_attrs: dict = {}
                        if label in standard_deviations:
                            ds_vars["std"] = xr.DataArray(
                                standard_deviations[label].values, dims=["t_idx"]
                            )
                        if is_monthly_only and label in members_per_model:
                            for _i, _mc in enumerate(members_per_model[label]):
                                ds_vars[f"member_{_i}"] = xr.DataArray(_mc.values, dims=["t_idx"])
                        if label in trends:
                            trend_da, slope = trends[label]
                            ds_vars["trend"] = xr.DataArray(trend_da.values, dims=["t_idx"])
                            save_attrs["trend_slope"] = slope
                        self._save_nc(xr.Dataset(ds_vars), nc_paths[label], attrs=save_attrs)

                colors = _get_model_colors(data_containers)

                # If trends were not loaded from cache and linear_trend is set, compute now.
                if self.linear_trend and not trends:
                    for label, da in cycles.items():
                        x_pos = np.arange(len(da))
                        trend_values, m = _compute_linear_trend(
                            x_pos, da.values, time_step=time_step)
                        trend_da = xr.DataArray(trend_values, coords=da.coords, dims=da.dims)
                        trends[label] = (trend_da, m * steps_per_decade)

                # ── Warmed-ERA5 regression lines for SST=2K / SST=4K ─────
                _SURFACE_TEMP_VARS = {"tas", "tos"}
                _sst_warmings: set = set()
                for dc in data_containers:
                    if "2K" in dc.model_label:
                        _sst_warmings.add(2)
                    if "4K" in dc.model_label:
                        _sst_warmings.add(4)
                extra_linestyles: dict = {}
                if name in _SURFACE_TEMP_VARS and _sst_warmings:
                    ref_label, _ = self._reference(data_containers)
                    ref_cycle = cycles.get(ref_label)
                    if ref_cycle is not None:
                        warming_lines = ["-.", ":"]
                        for idk, delta_k in enumerate(sorted(_sst_warmings)):
                            warmed_cycle = ref_cycle + float(delta_k)
                            x_pos = np.arange(len(warmed_cycle))
                            trend_values, m = _compute_linear_trend(
                                x_pos, warmed_cycle.values, time_step=time_step
                            )
                            era_label = f"ERA5 +{delta_k}K"
                            cycles[era_label] = xr.DataArray(
                                trend_values, coords=ref_cycle.coords, dims=ref_cycle.dims
                            )
                            colors[era_label] = colors.get(ref_label, "gray")
                            extra_linestyles[era_label] = warming_lines[idk]
                            logger.info(
                                f"    Added warmed-ERA5 regression series '{era_label}' "
                                f"(trend slope: {m * steps_per_decade:.4g} {trend_unit})"
                            )

                # ── top-right infobox: time range (season) / region ────────
                # Placed above the axes so it never overlaps with data lines.
                def _fmt_lon(v: float) -> str:
                    return f"{abs(v):.0f}\u00b0{'E' if v >= 0 else 'W'}"

                def _fmt_lat(v: float) -> str:
                    return f"{abs(v):.0f}\u00b0{'N' if v >= 0 else 'S'}"

                tr_lines: list = []
                # First line: time range + optional season suffix
                if tr_start is not None and season is not None:
                    tr_lines.append(f"{tr_start} \u2013 {tr_end} ({season})")
                elif tr_start is not None:
                    tr_lines.append(f"{tr_start} \u2013 {tr_end}")
                elif season is not None:
                    tr_lines.append(season)
                # Second line: region
                if lat_b is not None or lon_b is not None:
                    loc_parts: list = []
                    if lon_b is not None:
                        w, e = min(lon_b), max(lon_b)
                        loc_parts.append(f"{_fmt_lon(w)} \u2013 {_fmt_lon(e)}")
                    if lat_b is not None:
                        s, n = min(lat_b), max(lat_b)
                        loc_parts.append(f"{_fmt_lat(n)} \u2013 {_fmt_lat(s)}")
                    if loc_parts:
                        tr_lines.append(" / ".join(loc_parts))
                infobox_tr = "\n".join(tr_lines) if tr_lines else None

                unit = self.timeseries_plotter.cmor_units.get(name, "")
                var_label = _long_var_label(name, pressure_level)

                # When grouping by dayofyear the time coordinate contains
                # integer day numbers (1-365).  Build explicit month-name
                # x-ticks so the axis is readable.
                xticks_arg = None
                if any("dayofyear" in g for g in self.mean_groups) and cycles:
                    _longest_cycle = max(cycles.values(), key=len)
                    doy_vals = np.array(_longest_cycle.time.values, dtype=float)
                    # Tick position at the first day of each month
                    _month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
                    _month_abbrs = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    positions = []
                    labels = []
                    for doy, lbl in zip(_month_starts, _month_abbrs):
                        # Find the index in the plotted x-range closest to this doy
                        idx = int(np.argmin(np.abs(doy_vals - doy)))
                        positions.append(idx)
                        labels.append(lbl)
                    xticks_arg = (positions, _month_abbrs)

                
                self.timeseries_plotter.plot(
                    model_data=cycles,
                    model_stds=standard_deviations,
                    model_members=members_per_model,
                    linear_trend=trends,
                    colors=colors,
                    linestyles=extra_linestyles,
                    title="",
                    variable_name=name,
                    xlabel="",
                    ylabel=unit,
                    xticks=xticks_arg,
                    year_ticks_only=(time_step == "monthly"),
                    trend_unit=trend_unit,
                    fname=f"{fname_base}/{var_name}.pdf",
                    infobox_topleft=f"{var_label}",
                    infobox_topright=None, #infobox_tr,
                    despine=True,
                )

                # ── PSD of the seasonal cycle ─────────────────────────────
                if self.compute_psd:
                    # PSD is always computed on the raw monthly spatial-mean
                    # time series (not the grouped cycle), so the freq axis is
                    # consistently in month⁻¹ regardless of mean_groups.
                    fs = 1.0   # 1 sample / month

                    model_psds: dict = {}
                    psd_nc_paths: dict = {}
                    for dc in data_containers:
                        label = dc.model_label
                        if label not in cycles:
                            continue
                        psd_nc_paths[label] = self._nc_path(
                            output_path, f"{label}_{var_name}_psd"
                        )

                    if self._all_cached(list(psd_nc_paths.values())):
                        logger.info(f"    Loading cached PSD results from '{output_path}'")
                        for label, nc_p in psd_nc_paths.items():
                            ds = self._load_nc(nc_p)
                            model_psds[label] = (ds["freq"].values, ds["psd"].values)
                            ds.close()
                    else:
                        for dc in data_containers:
                            label = dc.model_label
                            if label not in cycles:
                                continue
                            # Fetch the raw monthly data and reduce to a 1-D
                            # spatial-mean time series (time range + lat/lon
                            # restrictions applied, no groupby).
                            raw_data = dc.get_variable_data(
                                name=name, pressure_level=pressure_level,
                                frequency="monthly", all_members=True,
                            )
                            if "member" in raw_data.dims:
                                monthly_ts = self._monthly_spatial_mean(
                                    raw_data.mean("member")
                                )
                            else:
                                monthly_ts = self._monthly_spatial_mean(raw_data)
                            n = len(monthly_ts)
                            nperseg = max(min(n // 2, 256), min(n, 4))
                            noverlap = nperseg // 2
                            try:
                                fx, fy = spectral.welch_psd(
                                    monthly_ts, fs=fs,
                                    nperseg=nperseg, noverlap=noverlap
                                )
                                model_psds[label] = (fx, fy)
                                if label in psd_nc_paths:
                                    self._save_nc(
                                        xr.Dataset({
                                            "freq": xr.DataArray(fx, dims=["freq"]),
                                            "psd":  xr.DataArray(fy, dims=["freq"]),
                                        }),
                                        psd_nc_paths[label],
                                    )
                            except Exception as exc:
                                logger.warning(f"    PSD failed for '{label}': {exc}")

                    if model_psds:
                        psd_plotter = FrequencyPlotter(
                            output_path=self.timeseries_plotter.output_path,
                            figsize=self.plotter_kwargs.get("figsize", (6.7, 3.5)),
                            dpi=self.timeseries_plotter.dpi,
                            linewidth=getattr(
                                self.timeseries_plotter, "linewidth", 2.0),
                        )
                        xmax = 0.33 # 1 cycle per 3 months (seasonal cycle) – adjust as needed
                        # Select only the portion of the spectrum up to xmax.
                        for lbl, (fx, fy) in model_psds.items():
                            mask = fx <= xmax
                            model_psds[lbl] = (fx[mask], fy[mask])

                        psd_plotter.plot_psd(
                            model_spectra=model_psds,
                            colors=colors,
                            linestyles={lbl: "-" for lbl in model_psds},
                            title="",
                            ylabel="Power Spectral Density",
                            infotext_topright=var_label,
                            fname=f"{fname_base}/{var_name}_psd.pdf",
                            semilog=True,
                            annotation=infobox_tr,
                        )


class MonsoonIndices(BaseMetric):
    """
    Monsoon strength indices (Webster-Yang and others) and seasonal wind
    vector-field diagnostics.

    Inherits :class:`BaseMetric` to obtain shared helpers and consistent
    ``plotter_kwargs`` handling.  Uses :class:`TimeseriesPlotter` for index
    output and :class:`CartopyProjectionPlotter` for quiver plots.

    Parameters
    ----------
    wind_vector_fields : list[dict], optional
        Each entry configures one set of seasonal quiver plots.  Recognised
        keys per entry:

        ``pressure_level`` : int
            Pressure level in Pa (e.g. ``85000``).
        ``seasons`` : list[dict]
            Each dict has a ``label`` (short display name, e.g. ``"JJA"``)
            and a ``months`` list of month names (e.g.
            ``["June", "July", "August"]``).
        ``region`` : list[float], optional
            ``[lon_min, lon_max, lat_min, lat_max]`` bounding box.
            Defaults to the South-Asian monsoon region
            ``[40, 110, -10, 40]``.
        ``quiver_stride`` : int, optional
            Subsampling stride applied to lat/lon before plotting arrows.
            Defaults to ``3``.
    """

    # Full month-name → number mapping shared across methods.
    _MONTH_NUM: dict = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    # Default South-Asian monsoon region [lon_min, lon_max, lat_min, lat_max].
    _SA_MONSOON_REGION: list = [40, 110, -10, 40]

    def __init__(
        self,
        method: str = "webster_yang",
        frequency: str = "monthly",
        target_year: int = 2024,
        baseline_period: tuple = None,
        wind_vector_fields: list = None,
        plotter_kwargs: dict = None,
        per_member: bool = False,
    ) -> None:
        # Embed method name in output path before passing to base.
        pk = dict(plotter_kwargs or {})
        pk["output_path"] = os.path.join(pk.get("output_path", "."), method)
        super().__init__(variables=[], frequency=frequency, plotter_kwargs=pk, per_member=per_member)

        self.method = method
        self.target_year = target_year
        self.baseline_period = baseline_period
        self.wind_vector_fields = list(wind_vector_fields) if wind_vector_fields else []
        self.timeseries_plotter = TimeseriesPlotter(**self.plotter_kwargs)
        if self.wind_vector_fields:
            self._cartopy = CartopyProjectionPlotter(
                output_path=self.plotter_kwargs.get("output_path", "."),
                figsize=tuple(self.plotter_kwargs.get("figsize", (6.7, 4.0))),
                dpi=self.plotter_kwargs.get("dpi", 150),
            )

    # -- Wind vector-field helpers -------------------------------------------

    @staticmethod
    def _months_from_spec(months_spec) -> list:
        """Convert a list of month-name strings to integers."""
        mapping = MonsoonIndices._MONTH_NUM
        return [mapping[m] for m in months_spec]

    def compute_wind_vectors(
        self,
        data_container,
        pressure_level: int,
        months: list,
        region: list = None,
    ) -> tuple:
        """Return seasonal-mean (U, V) DataArrays at *pressure_level*.

        Parameters
        ----------
        data_container:
            Source data container.
        pressure_level:
            Pressure level in Pa (e.g. ``85000``).
        months:
            Integer month numbers to average over.
        region:
            ``[lon_min, lon_max, lat_min, lat_max]`` bounding box.
            Defaults to the South-Asian monsoon region.

        Returns
        -------
        u_mean, v_mean : xr.DataArray
            Seasonal-mean U and V on the requested spatial domain.
        """
        region = list(region) if region else self._SA_MONSOON_REGION
        lon_min, lon_max, lat_min, lat_max = region

        u = data_container.get_variable_data(name="ua", pressure_level=pressure_level, frequency="monthly")
        v = data_container.get_variable_data(name="va", pressure_level=pressure_level, frequency="monthly")

        if "stat" in u.dims:
            u = u.sel(stat="mean", drop=True)
        if "stat" in v.dims:
            v = v.sel(stat="mean", drop=True)
        if "member" in u.dims:
            u = u.mean(dim="member")
        if "member" in v.dims:
            v = v.mean(dim="member")

        # Filter to requested months and average.
        u = u.sel(time=u.time.dt.month.isin(months)).mean("time")
        v = v.sel(time=v.time.dt.month.isin(months)).mean("time")

        # Spatial subset – CMOR lat is often descending (90→−90), so use a
        # boolean mask instead of slice() to be order-independent.
        u = u.where((u.lat >= lat_min) & (u.lat <= lat_max) &
                    (u.lon >= lon_min) & (u.lon <= lon_max), drop=True)
        v = v.where((v.lat >= lat_min) & (v.lat <= lat_max) &
                    (v.lon >= lon_min) & (v.lon <= lon_max), drop=True)

        # Ensure lat is ascending for meshgrid / contourf consistency.
        if u.lat.values[0] > u.lat.values[-1]:
            u = u.isel(lat=slice(None, None, -1))
            v = v.isel(lat=slice(None, None, -1))

        return u, v

    def _plot_wind_vectors(
        self,
        u: xr.DataArray,
        v: xr.DataArray,
        model_label: str,
        season_label: str,
        pressure_level: int,
        region: list,
        quiver_stride: int,
        output_path: str,
    ) -> None:
        """Render and save a Plate-Carree quiver plot for (U, V).

        The season label is placed as a bold title at the top of the figure.
        """
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        lon = u.lon.values
        lat = u.lat.values
        u_arr = u.values
        v_arr = v.values

        # Sub-sample for legible arrows.
        s = max(1, int(quiver_stride))
        lon_s = lon[::s]
        lat_s = lat[::s]
        u_s = u_arr[::s, ::s]
        v_s = v_arr[::s, ::s]

        lon_min, lon_max, lat_min, lat_max = region

        fontdict = {
            "axes.titlesize": mpl.rcParams["axes.titlesize"],
            "axes.labelsize": mpl.rcParams["axes.labelsize"],
            "xtick.labelsize": mpl.rcParams["xtick.labelsize"],
            "ytick.labelsize": mpl.rcParams["ytick.labelsize"],
        }
        fontdict.update(self.plotter_kwargs.get("fontdict", {}))

        figsize = tuple(self.plotter_kwargs.get("figsize", (A4_WIDTH, 3.6)))

        fig = plt.figure(figsize=figsize, dpi=self.plotter_kwargs.get("dpi", 300))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
        ax.add_feature(cfeature.LAND, facecolor="lightgrey", alpha=0.4, edgecolor="black", linewidth=0.3)

        gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.6, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": fontdict["xtick.labelsize"]}
        gl.ylabel_style = {"size": fontdict["ytick.labelsize"]}

        # Wind speed as background shading.
        speed = np.sqrt(u_arr**2 + v_arr**2)
        lon2d, lat2d = np.meshgrid(lon, lat)
        cf = ax.contourf(
            lon2d, lat2d, speed,
            levels=12,
            cmap="YlOrRd",
            transform=ccrs.PlateCarree(),
            alpha=0.75,
            extend="both"
        )
        cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.03, shrink=0.7)
        cbar.set_label(r"Wind speed  $\mathrm{m\,s^{-1}}$", fontsize=fontdict["axes.labelsize"])
        cbar.ax.tick_params(labelsize=fontdict["ytick.labelsize"])

        # Quiver arrows.
        lon2d_s, lat2d_s = np.meshgrid(lon_s, lat_s)
        ax.quiver(
            lon2d_s, lat2d_s, u_s, v_s,
            transform=ccrs.PlateCarree(),
            scale=None,
            width=0.003,
            headwidth=4,
            headlength=4,
            color="black",
            alpha=0.85,
        )

        # Bold season label as title.
        hpa = int(round(pressure_level / 100))
        ax.set_title(
            f"{season_label}  |  {hpa} hPa  |  {model_label}",
            fontsize=12,
            pad=8,
        )

        os.makedirs(output_path, exist_ok=True)
        fname = f"wind_vectors_{model_label}_{season_label}_{hpa}hPa.pdf"
        fig.savefig(
            os.path.join(output_path, fname),
            bbox_inches="tight",
        )
        plt.close(fig)
        logger.info(f"Saved quiver plot → {os.path.join(output_path, fname)}")

    def _evaluate_wind_vectors(self, data_containers) -> None:
        """Compute and plot seasonal wind vector fields for all containers."""
        for vf_spec in self.wind_vector_fields:
            # Resolve configuration with sensible defaults.
            pressure_level = int(vf_spec["pressure_level"])
            seasons = list(vf_spec.get("seasons", []))
            region = list(vf_spec.get("region", self._SA_MONSOON_REGION))
            quiver_stride = int(vf_spec.get("quiver_stride", 3))

            hpa = int(round(pressure_level / 100))
            vf_output = os.path.join(
                self.plotter_kwargs.get("output_path", "."),
                f"wind_vectors_{hpa}hPa",
            )

            for season in seasons:
                season_label = str(season["label"])
                month_names = list(season["months"])
                months = self._months_from_spec(month_names)

                # Build a short month-abbrev label for the filename.
                _ABBR = {
                    "January": "Jan", "February": "Feb", "March": "Mar",
                    "April": "Apr", "May": "May", "June": "Jun",
                    "July": "Jul", "August": "Aug", "September": "Sep",
                    "October": "Oct", "November": "Nov", "December": "Dec",
                }
                "/".join(_ABBR.get(m, m[:3]) for m in month_names)
                # Display label shown on the plot: bold season name + month abbrevs.
                display_label = f"{season_label}"

                for dc in data_containers:
                    logger.info(
                        f"--> MonsoonIndices: wind vectors for '{dc.model_label}'"
                        f" season={display_label} level={pressure_level} Pa"
                    )
                    try:
                        u, v = self.compute_wind_vectors(dc, pressure_level, months, region)
                        self._plot_wind_vectors(
                            u=u,
                            v=v,
                            model_label=dc.model_label,
                            season_label=display_label,
                            pressure_level=pressure_level,
                            region=region,
                            quiver_stride=quiver_stride,
                            output_path=vf_output,
                        )
                    except Exception:
                        logger.exception(
                            f"    Failed to compute wind vectors for '{dc.model_label}'"
                            f" at {pressure_level} Pa / {display_label}"
                        )

    # -- Index computation methods ------------------------------------------

    def webster_yang(self, data_container, baseline: bool = False) -> xr.DataArray:
        """Webster-Yang 850-250 hPa wind shear index."""
        u850 = data_container.get_variable_data(name="ua", pressure_level=85000, frequency=self.frequency)
        u250 = data_container.get_variable_data(name="ua", pressure_level=25000, frequency=self.frequency)

        if "stat" in u850.dims:
            u850 = u850.sel(stat="mean", drop=True)
            u250 = u250.sel(stat="mean", drop=True)
        
        if "member" in u850.dims:
            u850 = u850.mean(dim="member")
            u250 = u250.mean(dim="member")

        u850 = u850.sel(lat=slice(20, 0), lon=slice(220, 290))
        u250 = u250.sel(lat=slice(20, 0), lon=slice(220, 290))

        if baseline:
            logger.info(f"    MonsoonIndices: selecting baseline period {self.baseline_period}")
            yr = data_container.get_variable_data  # noqa – used only for label
            u850 = u850.sel(
                time=(u850.time.dt.year >= int(self.baseline_period[0]))
                & (u850.time.dt.year <= int(self.baseline_period[1]))
            )
            u250 = u250.sel(
                time=(u250.time.dt.year >= int(self.baseline_period[0]))
                & (u250.time.dt.year <= int(self.baseline_period[1]))
            )
        else:
            u850 = u850.sel(time=u850.time.dt.year == self.target_year)
            u250 = u250.sel(time=u250.time.dt.year == self.target_year)

        index = (u850 - u250).mean(["lat", "lon"])
        if self.frequency == "monthly":
            index = index.groupby("time.month").mean("time").rename({"month": "time"})
        else:
            index = index.groupby("time.dayofyear").mean("time").rename({"dayofyear": "time"})
        return index

    def compute(self, data_container, baseline: bool = False) -> xr.DataArray:
        if self.method == "webster_yang":
            return self.webster_yang(data_container, baseline=baseline)
        raise ValueError(f"Unknown monsoon index method: {self.method}")

    def evaluate(self, data_containers) -> None:
        reference_label, _ = self._reference(data_containers)
        output_path = self.timeseries_plotter.output_path
        os.makedirs(output_path, exist_ok=True)

        # ── cache paths ───────────────────────────────────────────────────────
        nc_paths = {
            dc.model_label: self._nc_path(
                output_path, f"{dc.model_label}_{self.method}_monsoon_index"
            )
            for dc in data_containers
        }
        baseline_nc_path = (
            self._nc_path(output_path, f"{reference_label}_baseline_{self.method}_monsoon_index")
            if reference_label else None
        )
        all_paths = list(nc_paths.values())
        if baseline_nc_path:
            all_paths.append(baseline_nc_path)

        # indices: {label: xr.DataArray(time)} – may include the baseline entry
        indices: dict = {}
        colors = self._colors(data_containers)
        linestyles = {dc.model_label: "-" for dc in data_containers}

        def _nc_to_da(ds):
            t_labels = ds["time_labels"].values.astype(str)
            return xr.DataArray(ds["data"].values, coords={"time": t_labels}, dims=["time"])

        if self._all_cached(all_paths):
            logger.info(f"    Loading cached monsoon-index results from '{output_path}'")
            for dc in data_containers:
                ds = self._load_nc(nc_paths[dc.model_label])
                indices[dc.model_label] = _nc_to_da(ds)
                ds.close()
            if baseline_nc_path and os.path.exists(baseline_nc_path):
                b_label = (
                    f"{reference_label} ({self.baseline_period[0]}-{self.baseline_period[1]})"
                    if self.baseline_period else reference_label
                )
                ds = self._load_nc(baseline_nc_path)
                indices[b_label] = _nc_to_da(ds)
                colors[b_label] = colors[reference_label]
                linestyles[b_label] = "--"
                ds.close()
        else:
            for dc in data_containers:
                raw_index = self.compute(dc)
                t_vals = np.array(raw_index.time.values, dtype=str)
                index_da = xr.DataArray(raw_index.values, coords={"time": t_vals}, dims=["time"])
                indices[dc.model_label] = index_da
                logger.info(f"--> MonsoonIndices: computed index for '{dc.model_label}'")
                # ── save primary index ────────────────────────────────────────
                self._save_nc(
                    xr.Dataset({
                        "data": xr.DataArray(raw_index.values, dims=["t_idx"]),
                        "time_labels": xr.DataArray(t_vals, dims=["t_idx"]),
                    }),
                    nc_paths[dc.model_label],
                )

                if dc.model_label == reference_label and baseline_nc_path:
                    ref_index = self.compute(dc, baseline=True)
                    ref_t_vals = np.array(ref_index.time.values, dtype=str)
                    logger.info(f"    Computed baseline index for reference '{reference_label}'")
                    b_label = (
                        f"{reference_label} ({self.baseline_period[0]}-{self.baseline_period[1]})"
                        if self.baseline_period else reference_label
                    )
                    indices[b_label] = xr.DataArray(
                        ref_index.values, coords={"time": ref_t_vals}, dims=["time"]
                    )
                    colors[b_label] = colors[reference_label]
                    linestyles[b_label] = "--"
                    # ── save baseline index ───────────────────────────────────
                    self._save_nc(
                        xr.Dataset({
                            "data": xr.DataArray(ref_index.values, dims=["t_idx"]),
                            "time_labels": xr.DataArray(ref_t_vals, dims=["t_idx"]),
                        }),
                        baseline_nc_path,
                    )

        if self.frequency == "monthly":
            xticks = (
                range(0, 12),
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            )
        elif self.frequency == "daily":
            xticks = (
                [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            )
        else:
            xticks = None

        self.timeseries_plotter.plot(
            model_data=indices,
            colors=colors,
            title=f"{self.method.replace('_', ' ').title()} Index ({self.target_year})",
            xlabel="Time",
            ylabel="Index Value",
            xticks=xticks,
            fname=f"{self.method}_monsoon_index.pdf",
            title_fontweight="bold",
            linestyles=linestyles,
            legend_loc="best",
        )

        # ── Wind vector fields (optional) ─────────────────────────────────────
        if self.wind_vector_fields:
            self._evaluate_wind_vectors(data_containers)


class SouthernOscillationIndex(Timeseries):
    """Southern Oscillation Index (Tahiti minus Darwin standardised pressure).

    When *regress_sst* is ``True`` the class also regresses the computed SOI
    against sea-surface temperature (``tos``) at every grid point and saves a
    global map of the OLS regression slope (K per SOI unit) for each data
    container.  The SST time axis is restricted to exactly the (year, month)
    pairs present in the SOI before fitting so that the two time series are
    always perfectly aligned.
    """

    def __init__(
        self,
        detrend: bool = False,
        compute_anomalies: bool = True,
        baseline_period: tuple = None,
        spectrum: bool = False,
        regress_sst: bool = False,
        nino34_box: bool = True,
        plotter_kwargs: dict = None,
        per_member: bool = False,
    ) -> None:
        super().__init__(
            detrend=detrend,
            compute_anomalies=compute_anomalies,
            baseline_period=baseline_period,
            plotter_kwargs=plotter_kwargs,
            per_member=per_member,
        )
        self.spectrum_fn = spectral.welch_psd if spectrum else None
        self._freq_plotter = SOIFrequencyPlotter(**self.plotter_kwargs) if spectrum else None
        self.regress_sst = regress_sst
        self.nino34_box = nino34_box
        if regress_sst:
            self._cartopy = CartopyProjectionPlotter(
                output_path=self.plotter_kwargs.get("output_path", ".")
            )

    def compute(self, data_container) -> xr.DataArray:
        data = data_container.get_variable_data(name="psl", frequency="monthly", pressure_level=None)
        if "stat" in data.dims:
            data = data.sel(stat="mean", drop=True)
        soi = compute_soi(data, base_period=self.baseline_period, detrend=self.detrend)
        import itertools
        time = [f"{y}-{m:02d}" for y, m in itertools.product(soi.year.values, soi.month.values)]
        soi = soi.stack(time=("year", "month")).reset_index("time")

        soi = soi.assign_coords(time=time)
        print(soi)
        soi = soi.chunk({'time': 10})
        soi = soi.rolling(time=3, center=True).mean().dropna("time")  # 3-month running mean
        return soi

    def _compute_sst_regression(
        self, data_container, soi_da: xr.DataArray
    ) -> tuple:
        """Regress monthly SST anomalies against the SOI index.

        Parameters
        ----------
        data_container:
            Data container from which ``tos`` is fetched.
        soi_da:
            SOI ``DataArray`` with ``(year, month)`` dimensions as returned
            directly by :func:`metrics.base.compute_soi`.  The intersection
            of years and months between *soi_da* and the SST anomaly field is
            used so that both series always have the same number of time
            instances before regression.

        Returns
        -------
        tuple of (xr.DataArray, xr.DataArray)
            ``(regression_map, pvalue_map)`` – both shaped ``(lat, lon)``.
            *regression_map* is the OLS slope in K per SOI unit.
            *pvalue_map* contains the two-tailed p-value of the Pearson
            correlation at each grid point (t-test with n-2 degrees of freedom).
        """
        tos = data_container.get_variable_data(name="tos", frequency="monthly", pressure_level=None)
        if "stat" in tos.dims:
            tos = tos.sel(stat="mean", drop=True)
        if "member" in tos.dims:
            logger.info(
                f"    SouthernOscillationIndex: member dimension detected in tos "
                f"(size {tos.sizes['member']}), averaging over members before regression"
            )
            tos = tos.mean(dim="member")

        # Monthly anomalies → dimensions (year, month, lat, lon)
        tos_anom = compute_anomaly(
            tos,
            baseline_period=self.baseline_period,
            mean_groups=["time.year", "time.month"],
            baseline_mean_groups=["time.month"],
        )

        # Intersect (year, month) axes so both arrays have identical time coverage.
        common_years = np.intersect1d(soi_da.year.values, tos_anom.year.values)
        common_months = np.intersect1d(soi_da.month.values, tos_anom.month.values)
        tos_anom = tos_anom.sel(year=common_years, month=common_months)
        soi_aligned = soi_da.sel(year=common_years, month=common_months)

        # Stack to a single time dimension: (T, lat, lon) and (T,)
        tos_flat = tos_anom.stack(time=("year", "month")).transpose("time", "lat", "lon")
        soi_flat = soi_aligned.stack(time=("year", "month"))

        # Remove any time steps where the SOI value is NaN.
        valid = ~np.isnan(soi_flat.values)
        tos_flat = tos_flat.isel(time=valid)
        soi_flat = soi_flat.isel(time=valid)

        # ── Restrict to the plotted domain before computing regression ────────
        # Lat coords are stored descending (90 → -90); slice accordingly.
        # Keep the full longitude range but only the ±45° latitude band so
        # that the regression statistics are meaningful for the tropical
        # domain that will be displayed.  Outside this band the returned maps
        # are filled with NaN.
        full_lats = tos_flat.lat.values          # all latitudes (for reindex later)
        tos_flat_band = tos_flat.sel(lat=slice(45, -45))

        soi_vals = soi_flat.values                        # (T,)
        tos_vals = tos_flat_band.values                   # (T, lat_band, lon)
        T = soi_vals.shape[0]

        # Vectorised OLS slope: beta = cov(X, Y) / var(X)
        soi_mean = soi_vals.mean()
        tos_mean = tos_vals.mean(axis=0)            # (lat, lon)
        cov = (
            (soi_vals[:, np.newaxis, np.newaxis] - soi_mean) * (tos_vals - tos_mean)
        ).mean(axis=0)
        var_soi = float(((soi_vals - soi_mean) ** 2).mean())
        beta = cov / var_soi if var_soi > 0 else np.zeros_like(cov)   # (lat, lon)

        # Pearson r → t-statistic → two-tailed p-value (df = T - 2).
        soi_std = float(soi_vals.std())
        tos_std = tos_vals.std(axis=0)              # (lat, lon)
        denom = soi_std * tos_std
        r = np.where(denom > 0, cov / denom, 0.0)  # (lat, lon)
        r = np.clip(r, -1.0 + 1e-10, 1.0 - 1e-10)
        t_stat = r * np.sqrt(T - 2) / np.sqrt(1.0 - r ** 2)
        from scipy.stats import t as _t_dist
        pvalues = 2.0 * _t_dist.sf(np.abs(t_stat), df=max(T - 2, 1))  # (lat, lon)

        def _fix_lons(arr):
            """Correct roll_coords=False shift and sort longitude axis."""
            da = xr.DataArray(arr, coords={"lat": tos_flat_band.lat, "lon": tos_flat_band.lon}, dims=["lat", "lon"])
            n_lon = da.sizes["lon"]
            corrected_lons = np.roll(da.lon.values, -n_lon // 2)
            da = da.assign_coords(lon=corrected_lons).sortby("lon")
            # Reindex onto the full latitude grid, filling outside ±45° with NaN.
            return da.reindex(lat=full_lats, fill_value=np.nan)

        regression_map = _fix_lons(beta)
        pvalue_map = _fix_lons(pvalues)

        return regression_map, pvalue_map

    def evaluate(self, data_containers) -> None:
        soi_indices: dict = {}
        psd_spectra: dict = {}
        colors = self._colors(data_containers)
        output_path = self.timeseries_plotter.output_path
        os.makedirs(output_path, exist_ok=True)

        # ── cache paths for SOI timeseries ─────────────────────────────────
        soi_nc_paths = {
            dc.model_label: self._nc_path(output_path, f"SOI_{dc.model_label}")
            for dc in data_containers
        }
        # Cache paths for SST regression maps (in plotter output dir).
        reg_output_path = self.plotter_kwargs.get("output_path", ".")
        sst_nc_paths = {
            dc.model_label: self._nc_path(
                os.path.join(reg_output_path, dc.model_label),
                f"SOI_SST_regression_{dc.model_label}",
            )
            for dc in data_containers
        } if self.regress_sst else {}

        soi_cached = self._all_cached(list(soi_nc_paths.values()))
        sst_cached = (not self.regress_sst) or self._all_cached(
            [p for lbl, p in sst_nc_paths.items()
             if not ("mpi" in lbl.lower())]  # mpi models are explicitly skipped
        )

        def _nc_to_soi(ds):
            t_labels = ds["time_labels"].values.astype(str)
            return xr.DataArray(ds["data"].values, coords={"time": t_labels}, dims=["time"])

        regression_maps: dict = {}   # {label: (regression_map, pvalue_map)}

        if soi_cached and sst_cached:
            logger.info(f"    Loading cached SOI results from '{output_path}'")
            for dc in data_containers:
                ds = self._load_nc(soi_nc_paths[dc.model_label])
                soi = _nc_to_soi(ds)
                soi_indices[dc.model_label] = soi
                if self.spectrum_fn and "psd_freq" in ds and "psd_power" in ds:
                    psd_spectra[dc.model_label] = (ds["psd_freq"].values, ds["psd_power"].values)
                ds.close()
            if self.regress_sst:
                for dc in data_containers:
                    if "mpi" in dc.model_label.lower():
                        continue
                    p = sst_nc_paths[dc.model_label]
                    if os.path.exists(p):
                        ds = self._load_nc(p)
                        reg_map = ds["regression"].transpose("lat", "lon")
                        pval_map = ds["pvalue"].transpose("lat", "lon")
                        regression_maps[dc.model_label] = (reg_map, pval_map)
                        ds.close()
        else:
            # compute SOI timeseries 
            for dc in data_containers:
                soi = self.compute(dc)
                t_labels = np.array(soi.time.values, dtype=str)
                soi_indices[dc.model_label] = soi
                ds_vars: dict = {
                    "data": xr.DataArray(soi.values, dims=["t_idx"]),
                    "time_labels": xr.DataArray(t_labels, dims=["t_idx"]),
                }
                if self.spectrum_fn:
                    fx, fy = self.spectrum_fn(soi.values)
                    psd_spectra[dc.model_label] = (fx, fy)
                    ds_vars["psd_freq"]  = xr.DataArray(fx, dims=["freq"])
                    ds_vars["psd_power"] = xr.DataArray(fy, dims=["freq"])
                self._save_nc(xr.Dataset(ds_vars), soi_nc_paths[dc.model_label])

            if self.regress_sst:
                for dc in data_containers:
                    if any(x in dc.model_label.lower() for x in ["mpi", "hadgem", "cesm", "gfdl"]):
                        logger.warning(
                            f"    SouthernOscillationIndex: skipping SST regression for '{dc.model_label}' "
                            f"because 'tos' variable is not available"
                        )
                        continue
                  
                    psl = dc.get_variable_data(name="psl", frequency="monthly", pressure_level=None)
                    if "stat" in psl.dims:
                        psl = psl.sel(stat="mean", drop=True)
                    soi_raw = compute_soi(psl, base_period=self.baseline_period, detrend=self.detrend)

                    # Drop NaN values at the start and end of the time series that result from the rolling mean.
                    soi = soi.dropna(dim="time", how="all")
                    
                    logger.info(
                        f"--> SouthernOscillationIndex: computing SST regression for '{dc.model_label}'"
                    )
                    regression_map, pvalue_map = self._compute_sst_regression(dc, soi_raw)
                    regression_maps[dc.model_label] = (regression_map, pvalue_map)
                    # save SST regression to cache
                    self._save_nc(
                        xr.Dataset({
                            "regression": regression_map,
                            "pvalue": pvalue_map,
                        }),
                        sst_nc_paths[dc.model_label],
                    )

        # plot regression maps with Pearson r against reference 
        if self.regress_sst and regression_maps:
            ref_label, _ = self._reference(data_containers)

            shared_vabs = 1.0
            logger.info("    SOI–SST regression shared colour scale fixed to [-1.0, 1.0]")

            ref_map = regression_maps.get(ref_label, (None, None))[0]
            r_annotations = {}

            with mpl.rc_context({
                "axes.titlesize": 14,
                "axes.labelsize": 13,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 11,
            }):
                for label, (regression_map, pvalue_map) in regression_maps.items():
                    if ref_map is not None and label != ref_label:
                        a = regression_map.values.ravel()
                        b = ref_map.values.ravel()
                        valid = ~(np.isnan(a) | np.isnan(b))
                        if valid.sum() > 2:
                            r_val = float(np.corrcoef(a[valid], b[valid])[0, 1])
                        else:
                            r_val = float("nan")
                        r_annotation = f"r = {r_val:.3f}"
                        logger.info(
                            f"    Pearson r ({label} vs {ref_label}): {r_val:.3f}"
                        )
                    else:
                        r_annotation = ""
                    r_annotations[label] = r_annotation

                    self._cartopy.plate_carree(
                        x=regression_map.lon.values,
                        y=regression_map.lat.values,
                        z=regression_map.values,
                        fname=f"{label}/SOI_SST_regression_{label}.pdf",
                        cbar_label=r"$\mathrm{K}$ per SOI unit",
                        cmap="coolwarm",
                        vmin=-shared_vabs,
                        vmax=shared_vabs,
                        stipple_pvalues=pvalue_map.values,
                        stipple_lat=pvalue_map.lat.values,
                        stipple_lon=pvalue_map.lon.values,
                        stipple_alpha_level=0.05,
                        infotext_topright=r_annotation,
                        infotext_topleft=label,
                        extent=[-180, 180, -45, 45],
                        nino34_box=self.nino34_box,
                        dpi=150,
                    )
                    logger.info(f"Saved SOI–SST regression map for '{label}'")

                if len(regression_maps) > 1:
                    import cartopy.crs as ccrs
                    import cartopy.feature as cfeature
                    import matplotlib.patches as mpatches

                    n_rows = len(regression_maps)
                    fig = plt.figure(figsize=(8.5, max(2.6 * n_rows + 0.8, 4.8)), dpi=150)
                    gs = fig.add_gridspec(n_rows, 1, hspace=0.14)
                    axes = []
                    mappable = None
                    levels = np.linspace(-shared_vabs, shared_vabs, 21)

                    for i, (label, (regression_map, pvalue_map)) in enumerate(regression_maps.items()):
                        ax = fig.add_subplot(gs[i, 0], projection=ccrs.Robinson(central_longitude=180.0))
                        axes.append(ax)
                        ax.set_global()
                        ax.coastlines(linewidth=0.6)
                        ax.add_feature(cfeature.LAND, edgecolor="black", linewidth=0.3)

                        ax.gridlines(
                            draw_labels=False,
                            dms=True,
                            x_inline=False,
                            y_inline=False,
                            linewidth=0.4,
                            alpha=0.8,
                        )

                        cnt = ax.contourf(
                            regression_map.lon.values,
                            regression_map.lat.values,
                            regression_map.values,
                            transform=ccrs.PlateCarree(),
                            cmap="coolwarm",
                            vmin=-shared_vabs,
                            vmax=shared_vabs,
                            levels=levels,
                            extend="both",
                        )
                        mappable = cnt

                        lon2d, lat2d = np.meshgrid(pvalue_map.lon.values, pvalue_map.lat.values)
                        sig_mask = pvalue_map.values < 0.05
                        ax.scatter(
                            lon2d[sig_mask],
                            lat2d[sig_mask],
                            s=0.25,
                            color="k",
                            alpha=0.2,
                            transform=ccrs.PlateCarree(),
                            linewidths=0,
                            zorder=5,
                        )

                        if self.nino34_box:
                            nino34_rect = mpatches.Rectangle(
                                xy=(-170, -5),
                                width=50,
                                height=10,
                                linewidth=1.2,
                                edgecolor="black",
                                facecolor="none",
                                transform=ccrs.PlateCarree(),
                                zorder=6,
                            )
                            ax.add_patch(nino34_rect)

                        ax.set_extent([-180, 180, -45, 45], crs=ccrs.PlateCarree())
                        ax.text(
                            0.0,
                            1.02,
                            label,
                            fontsize=13,
                            fontweight="bold",
                            ha="left",
                            va="bottom",
                            transform=ax.transAxes,
                        )
                        r_annotation = r_annotations.get(label, "")
                        if r_annotation:
                            ax.text(
                                1.0,
                                1.02,
                                r_annotation,
                                fontsize=12,
                                ha="right",
                                va="bottom",
                                transform=ax.transAxes,
                            )

                    if mappable is not None:
                        cbar = fig.colorbar(
                            mappable,
                            ax=axes,
                            orientation="horizontal",
                            pad=0.035,
                            fraction=0.035,
                            aspect=45,
                            extend="both",
                        )
                        cbar.set_label(r"$\mathrm{K}$ per SOI unit", fontsize=13)
                        cbar.ax.tick_params(labelsize=11)

                    stacked_fpath = os.path.join(reg_output_path, "SOI_SST_regression_all_models.pdf")
                    plt.savefig(stacked_fpath, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    logger.info(f"Saved stacked SOI–SST regression maps: {stacked_fpath}")

        for label, soi_data in soi_indices.items():
            os.makedirs(os.path.join(output_path, label), exist_ok=True)
            self.timeseries_plotter.plot_stem(
                model_data={label: soi_data},
                title="",
                xlabel="",
                ylabel="Southern Oscillation Index (SOI)",
                xticks=None,
                year_ticks_only=True,
                fname=f"{label}/SOI_{label}.pdf",
            )

        if self.spectrum_fn and psd_spectra:
            self._freq_plotter.plot_soi_psd(
                model_spectra=psd_spectra,
                colors=colors,
                linestyles={label: "-" for label in psd_spectra},
                fname="SOI_Spectrum.pdf",
            )


# ============================================================================
# EOF and annular / circulation modes
# ============================================================================

class EOF:
    """
    Empirical Orthogonal Function decomposition via truncated SVD.

    This is a **computational helper**, not a metric evaluator.  It is used
    internally by :class:`AnnularModes` and its subclasses.
    """

    def __init__(self, n_modes: int = 1, weight_by_latitude: bool = False) -> None:
        self.n_modes = n_modes
        self.weight_by_latitude = weight_by_latitude

    def cosine_latitude_weights(self, latitudes: np.ndarray) -> np.ndarray:
        """Compute cosine latitude weights for a given array of latitudes."""
        lat_rad = np.radians(latitudes)
        weights = np.cos(lat_rad)
        return weights

    def compute_eofs(
        self,
        data: xr.DataArray,
        lat_slicer=None,
        lon_slicer=None,
    ) -> None:
        assert isinstance(data, xr.DataArray), "data must be an xr.DataArray."

        self._data = data
        # Check if NaNs are present, if yes, interpolate them before SVD to avoid errors.
        #if np.isnan(self._data.values).any():
            
        #    logger.warning("NaN values detected in data; applying nearest-neighbor interpolation before EOF computation.")
            # sort lat and lon to ensure proper interpolation
            #self._data = self._data.sortby("lat")
            #self._data = self._data.interpolate_na(method="linear", dim="lat", max_gap=None, limit=None )
            #self._data = self._data.sortby("lat", ascending=False)  # restore original order if needed
            #self._data = self._data.interpolate_na(method="linear", dim="lon", max_gap=None, limit=None)

        # time values set in format (yyyy-mm)
        self.time = list(set(self._data.time.values.astype("datetime64[M]").astype(str)))
        self._centred_data = compute_anomaly(
            self._data,
            baseline_period=None,
            mean_groups=["time.year", "time.month"],
            baseline_mean_groups=["time.month"],
        ).stack({"time": ["year", "month"]}).reset_index("time", drop=True)
        # (yyyy, mm) → yyyy-mm string format for easier plotting and interpretation.
        self._centred_data = self._centred_data.transpose(
            "time", "lat", "lon").dropna(dim="time", how="all") # all for SAM, any for NAM
        print(f"Data shape after stacking and dropping NaNs: {self._centred_data.shape}")

        # Normalise coordinate order so that lat/lon slicers always work
        # regardless of whether the input data is ascending or descending.
        self._centred_data = self._centred_data.sortby("lat").sortby("lon")


        # Keep an unweighted copy for regression/projection maps so that
        # polar grid points (where cos(lat) → 0) retain their physical
        # amplitude instead of being driven to zero by the area weight.
        self._centred_data_unweighted = self._centred_data.copy(deep=True)

        self._centred_data.values = self._centred_data.values * \
            self.cosine_latitude_weights(self._centred_data.lat.values)[:, np.newaxis]


        eof_modes, A, Lh, E = compute_eof(
            self._centred_data.sel(lat=lat_slicer, lon=lon_slicer), 
            n_modes=None
        )

        self.eof_modes = eof_modes
        self.norm_coeff = self.eof_modes.shape[0] - 1
        self._L = Lh * Lh / self.norm_coeff

        # Fraction of total variance explained by the leading mode.
        total_var = self._L.sum() 
        self.explained_variance_ratio = self._L[0] / total_var if total_var > 0 else 0.0
        logger.info(
            f"EOF: explained variance ratio of leading mode: \
                {self.explained_variance_ratio:.4f}"
            )


    def eigenvals_timeseries(self) -> np.ndarray:
        return self.eof_modes / np.std(self.eof_modes, axis=0, keepdims=True)

    def project_eofs(
        self,
        eof_modes: np.ndarray,
        lat_slicer=None,
        lon_slicer=None,
    ) -> xr.DataArray:
        """Project *anomaly* onto *eof_modes* with a significance mask.

        Projection and p-value computation are carried out **only within
        the requested lat/lon band** (``lat_slicer``, ``lon_slicer``).
        The result is then placed back onto the full lat/lon grid of
        *anomaly*, with ``NaN`` outside the requested band.  When no
        slicers are provided the full grid is used (original behaviour).

        Fully vectorised – no per-grid-point Python loops:

        * each grid-point projection is ``np.dot(gp, eof_modes[:, 0])``
        * the significance mask uses the two-tailed p-value from
          Pearson r; grid points with p >= 0.05 are set to 0.

        Returns
        -------
        xr.DataArray
            Shape ``(lat, lon)`` matching *anomaly*. Values are NaN
            outside the requested band and 0 where not significant inside.
        """
        self._sign_flipped = False
        full_lat = self._centred_data_unweighted.lat.values
        full_lon = self._centred_data_unweighted.lon.values

        # Restrict to the desired band for projection and p-value stats.
        # Use the *unweighted* centred anomaly so that the regression map
        # reflects real physical amplitudes at every latitude.  The area-
        # weighted data (self._centred_data) was only needed for the SVD
        # decomposition; projecting through it would suppress polar grid
        # points where sqrt(cos(lat)) → 0.
        lat_slicer = lat_slicer if lat_slicer is not None else slice(None)
        lon_slicer = lon_slicer if lon_slicer is not None else slice(None)

        data_band = self._centred_data_unweighted.sel(lat=lat_slicer, lon=lon_slicer)

        T = data_band.values.shape[0]
        n_lat_band = len(data_band.lat)
        n_lon_band = len(data_band.lon)
        print(f"Projecting onto band: {n_lat_band} lat x {n_lon_band} lon")

        y = eof_modes[:, 0] / np.sqrt(self._L[0])                                       # (T,)

        # Flatten spatial dims -> (T, lat_band * lon_band).
        anomaly_flat = data_band.values.reshape(T, -1)          # (T, N)

        # Raw projection: equivalent to np.dot(gp, y) for every grid point.
        proj_flat = anomaly_flat.T @ y                             # (N,)

        # Vectorised Pearson r (matches stats.linregress p-value).
        x_c = anomaly_flat - anomaly_flat.mean(axis=0, keepdims=True)  # (T, N)
        y_c = y - y.mean()                                             # (T,)

        r_num = x_c.T @ y_c                                           # (N,)
        r_den = np.sqrt((x_c ** 2).sum(axis=0)) * np.sqrt((y_c ** 2).sum())
        r = np.where(r_den > 0, r_num / r_den, 0.0)                  # (N,)

        # t = r*sqrt(T-2)/sqrt(1-r^2); two-tailed p from t-distribution.
        r_sq = np.clip(r ** 2, 0.0, 1.0 - 1e-12)
        t_stat = r * np.sqrt(T - 2) / np.sqrt(1.0 - r_sq)
        p = 2.0 * stats.t.sf(np.abs(t_stat), df=T - 2)              # (N,)

        # Apply significance mask; insignificant grid points become 0.
        proj_masked = np.where(p < 0.05, proj_flat, 0.0).reshape(n_lat_band, n_lon_band)

        proj_band = xr.DataArray(
            proj_masked,
            coords={"lat": data_band.lat.values, "lon": data_band.lon.values},
            dims=["lat", "lon"],
        )
        # Print proj values
        print(proj_band.values)

        proj_full = proj_band.reindex(lat=full_lat, lon=full_lon, fill_value=np.nan)
        
        # Enforce the standard NAM/SAM sign convention: positive index
        # corresponds to *lower* pressure over the pole.  If the mean
        # projection poleward of 60° (NH) or equatorward of -60° (SH) is
        # positive, the EOF sign came out opposite to the convention and we
        # flip.
        polar_lat = lat_slicer
        if polar_lat is not None and polar_lat != slice(None):
            lat_lo, lat_hi = polar_lat.start, polar_lat.stop
            if lat_hi is not None and lat_hi > 0:
                # Northern hemisphere – check high latitudes
                polar = proj_full.sel(lat=slice(60, 90))
            elif lat_lo is not None and lat_lo < 0:
                # Southern hemisphere – check high latitudes
                polar = proj_full.sel(lat=slice(-90, -60))
            else:
                polar = None
            if polar is not None:
                polar_mean = float(polar.mean(skipna=True))
                if polar_mean > 0:
                    proj_full = -proj_full
                    self._sign_flipped = True

        return proj_full


class AnnularModes(BaseMetric):
    """
    Base class for EOF-based annular mode / circulation indices.

    Subclasses override :py:meth:`compute` to select the relevant variable
    and :py:meth:`spatial_plot` to choose the appropriate map projection.
    """

    def __init__(
        self,
        method: str = "EOF",
        time=None,
        baseline_period: tuple = None,
        plotter_kwargs: dict = None,
        frequency: str = "monthly",
        latitude_bands: tuple = None,
        longitude_bands: tuple = None,
        per_member: bool = False,
        weight_by_latitude: bool = False,
    ) -> None:
        
        
        self.name = "Annular Mode"
        super().__init__(
            variables=[], 
            frequency=frequency, 
            plotter_kwargs=plotter_kwargs, 
            per_member=per_member,
        )

        self.method = method
        self.time = time
        self.baseline_period = baseline_period
        self.latitudes: np.ndarray = None
        self.longitudes: np.ndarray = None

        # Latitude / longitude slicers – always ascending so they work
        # correctly with the sortby("lat") / sortby("lon") applied in
        # EOF.compute_eofs.
        self.latitude_bands = latitude_bands
        self.latitude_slicer = (
            slice(min(latitude_bands), max(latitude_bands))
            if latitude_bands is not None
            else slice(None)
        )
        self.longitude_bands = longitude_bands
        self.longitude_slicer = (
            slice(min(longitude_bands), max(longitude_bands))
            if longitude_bands is not None
            else slice(None)
        )

        self.eof_op = EOF(
            n_modes=1, 
            weight_by_latitude=weight_by_latitude
        )

        self.timeseries_plotter = TimeseriesPlotter(**self.plotter_kwargs)
        self.taylor_plotter = TaylorDiagramPlotter(
            output_path=self.plotter_kwargs.get("output_path", "."),
        )

    # -- Helpers -------------------------------------------------------------
    def get_shared_vmin_vmax(self, projections: dict) -> tuple:
        """Determine shared colour scale limits across all projections."""
        all_vals = np.concatenate([p.values.ravel() for p in projections.values()])
        valid_vals = all_vals[~np.isnan(all_vals)]
        #v_abs_max = float(np.percentile(np.abs(valid_vals), 98))
        v_abs_max = np.abs(valid_vals).max() if valid_vals.size > 0 else 1.0
        return -v_abs_max, v_abs_max
    
    @staticmethod
    def _month_to_num(month_name: str) -> int:
        mapping = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12,
        }
        return mapping[month_name]

    def slice_time(self, data: xr.DataArray, time) -> xr.DataArray:
        """Select months or a season from *data*."""

        if isinstance(time, omegaconf.ListConfig) or isinstance(time, list):
            months = [self._month_to_num(m) for m in time]
            return data.sel(time=data.time.dt.month.isin(months) )
        if time in ("DJF", "MAM", "JJA", "SON"):
            return data.sel(time=data.time.dt.season == time)
        return data

    def _extract_data(self, data_container, all_members=False) -> xr.DataArray:
        """Return the raw DataArray (with member dim if present) for this metric.

        Subclasses must override this method.  The default raises
        :exc:`NotImplementedError` so that callers without a concrete
        override get a clear error rather than silent incorrect behaviour.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _extract_data() "
            "to support per_member evaluation."
        )

    def spatial_plot(self, projection, model_label: str, info_vals=None, member_suffix: str = "", vmin=None, vmax=None) -> None:
        """Override in subclasses to render the spatial EOF pattern."""

    def compute(self, data: xr.DataArray):
        """
        Compute the leading EOF pattern and its time series.

        Returns ``(time, projection, eigenvals)`` where *projection* is a
        2-D spatial map and *eigenvals* is the 1-D principal-component time
        series.
        """
        data = self.slice_time(data, self.time)

        self.latitudes = data.lat.values
        self.longitudes = data.lon.values

        self.eof_op.compute_eofs(
            data=data, 
            lat_slicer=self.latitude_slicer, 
            lon_slicer=self.longitude_slicer
        )

        eigenvals = self.eof_op.eigenvals_timeseries()

        self.latitudes = data.lat.values
        self.longitudes = data.lon.values
        projection = self.eof_op.project_eofs(
            eigenvals,
            lat_slicer=self.latitude_slicer,
            lon_slicer=self.longitude_slicer,
        )
        # eigenvals is (T, n_modes); extract only the leading mode → 1-D (T,)
        pc1 = eigenvals[:, 0]
        # If project_eofs flipped the spatial pattern to match the standard
        # sign convention, flip the PC time series consistently.
        if self.eof_op._sign_flipped:
            pc1 = -pc1
        return self.eof_op.time, projection, pc1, self.eof_op.explained_variance_ratio

    def regress_against_reference(
        self,
        model_projection: xr.DataArray,
        reference_projection: xr.DataArray,
    ) -> tuple:
        """Return (r-value, p-value, std_err) from linear regression.

        Regression is restricted to the lat/lon band used for projection.
        Points outside the band are NaN in both projections (set by
        :py:meth:`EOF.project_eofs`)  and are excluded automatically via a
        joint non-NaN mask – no manual index selection required.
        """
        model_flat = np.asarray(model_projection).ravel()
        ref_flat   = np.asarray(reference_projection).ravel()
        valid = ~(np.isnan(model_flat) | np.isnan(ref_flat))
        logger.debug(f"    Regressing over {valid.sum()} grid points within the lat/lon band.")
        lr = stats.linregress(model_flat[valid], ref_flat[valid])
        return lr.rvalue, lr.pvalue, lr.stderr

    def _compute_single(self, data: xr.DataArray):
        """Call the base :py:meth:`AnnularModes.compute` directly on a plain DataArray.

        Used by :py:meth:`evaluate` when ``per_member=True`` so that the
        member-loop bypasses the subclass override of ``compute`` (which
        expects a data container, not a raw DataArray).
        """
        return AnnularModes.compute(self, data)

    def _projection_params(self) -> dict:
        """Return kwargs for the azimuthal-equidistant multi-panel plot.

        Subclasses override this to supply ``central_latitude``, ``extent``,
        ``cbar_label``, and optionally ``wedge``.  Returning an empty dict
        suppresses the all-models panel plot.
        """
        return {}

    def spatial_plot_all_models(
        self,
        model_projections: dict,
        vmin: float,
        vmax: float,
        fname: str,
    ) -> None:
        """Render all models in a single multi-panel figure with a shared colorbar.

        Parameters
        ----------
        model_projections:
            ``{display_label: (projection_da, info_vals_dict)}``.
        vmin / vmax:
            Shared colour-scale bounds (from :meth:`get_shared_vmin_vmax`).
        fname:
            Output filename relative to the configured output path.
        """
        params = self._projection_params()
        if not params:
            return
        output_path = self.plotter_kwargs.get("output_path", ".")
        data_dict      = {lbl: proj for lbl, (proj, _)  in model_projections.items()}
        info_vals_dict = {lbl: iv   for lbl, (_, iv)    in model_projections.items()}
        self._cartopy.azimuthal_equidistant_multi(
            data_dict=data_dict,
            info_vals_dict=info_vals_dict,
            fname=os.path.join(output_path, fname) if not os.path.isabs(fname) else fname,
            vmin=vmin,
            vmax=vmax,
            **params,
        )

    def evaluate(self, data_containers) -> None:
        if self.per_member:
            results = self._evaluate_per_member(data_containers)
        else:
            results = self._evaluate_standard(data_containers)
        self._write_metrics_csv(results)

    def _evaluate_standard(self, data_containers) -> dict:
        """Standard evaluation (single result per data container)."""
        eof_modes: dict = {}
        projections: dict = {}
        explained_variances: dict = {}

        colors = self._colors(data_containers)
        ref_name, _ = self._reference(data_containers)
        output_path = self.plotter_kwargs.get("output_path", ".")
        os.makedirs(output_path, exist_ok=True)

        # ── cache paths (one subdir per model for NC files) ─────────────────
        for dc in data_containers:
            os.makedirs(os.path.join(output_path, dc.model_label), exist_ok=True)

        proj_nc_paths = {
            dc.model_label: self._nc_path(
                os.path.join(output_path, dc.model_label),
                f"{self.method}_{dc.model_label}_projection",
            )
            for dc in data_containers
        }
        ts_nc_paths = {
            dc.model_label: self._nc_path(
                os.path.join(output_path, dc.model_label),
                f"{self.method}_{dc.model_label}_timeseries",
            )
            for dc in data_containers
        }
        all_nc_paths = list(proj_nc_paths.values()) + list(ts_nc_paths.values())

        if self._all_cached(all_nc_paths):
            logger.info(f"    Loading cached EOF results from '{output_path}'")
            for dc in data_containers:
                ds_proj = self._load_nc(proj_nc_paths[dc.model_label])
                projection = ds_proj["data"].transpose("lat", "lon")
                expl_var = float(ds_proj.attrs.get("explained_variance", 0.0))
                projections[dc.model_label] = projection
                explained_variances[dc.model_label] = expl_var
                ds_proj.close()

                ds_ts = self._load_nc(ts_nc_paths[dc.model_label])
                t_labels = ds_ts["time_labels"].values.astype(str)
                time_da = xr.DataArray(
                    ds_ts["data"].values, coords={"time": t_labels}, dims=["time"]
                )
                eof_modes[dc.model_label] = time_da
                ds_ts.close()
        else:
            reference_projection = None
            ref_dc = next((dc for dc in data_containers if dc.model_label == ref_name), None)
            if ref_dc is not None:
                time, projection, eigenvals, expl_var = self.compute(ref_dc)
                time_da = xr.DataArray(
                    eigenvals,
                    coords={"time": np.array(time, dtype=str)},
                    dims=["time"],
                )
                eof_modes[ref_dc.model_label] = time_da
                projections[ref_dc.model_label] = projection
                explained_variances[ref_dc.model_label] = expl_var
                reference_projection = projection
                # ── save reference ────────────────────────────────────────────
                self._save_nc(projection, proj_nc_paths[ref_dc.model_label],
                              attrs={"explained_variance": expl_var})
                t_arr = np.array(time, dtype=str)
                self._save_nc(
                    xr.Dataset({
                        "data": xr.DataArray(eigenvals, dims=["t_idx"]),
                        "time_labels": xr.DataArray(t_arr, dims=["t_idx"]),
                    }),
                    ts_nc_paths[ref_dc.model_label],
                )

            for dc in data_containers:
                if dc.model_label == ref_name:
                    continue
                time, projection, eigenvals, expl_var = self.compute(dc)
                if reference_projection is not None:
                    r, _, _ = self.regress_against_reference(projection, reference_projection)
                    if r < 0:
                        logger.debug(f"    {dc.model_label}: sign flipped (r={r:.4f})")
                        projection = -projection
                        eigenvals = -eigenvals
                time_da = xr.DataArray(
                    eigenvals,
                    coords={"time": np.array(time, dtype=str)},
                    dims=["time"],
                )
                eof_modes[dc.model_label] = time_da
                projections[dc.model_label] = projection
                explained_variances[dc.model_label] = expl_var
                # ── save to cache ─────────────────────────────────────────────
                self._save_nc(projection, proj_nc_paths[dc.model_label],
                              attrs={"explained_variance": expl_var})
                t_arr = np.array(time, dtype=str)
                self._save_nc(
                    xr.Dataset({
                        "data": xr.DataArray(eigenvals, dims=["t_idx"]),
                        "time_labels": xr.DataArray(t_arr, dims=["t_idx"]),
                    }),
                    ts_nc_paths[dc.model_label],
                )

        _shared_vmin, _shared_vmax = self.get_shared_vmin_vmax(projections)

        # Collect Taylor-diagram statistics for all non-reference models.
        reference_projection = projections.get(ref_name)
        taylor_stats: dict = {}


        # Collect metrics for CSV
        metrics = []
        for label, projection in projections.items():
            info_vals = {f"{explained_variances[label]: .1%}"}
            row = {"model": label, "member": "", "explained_variance": explained_variances[label]}
            if label != ref_name and reference_projection is not None:
                r, p, std_err = self.regress_against_reference(projection, reference_projection)
                proj_vals = np.asarray(projection)
                ref_vals  = np.asarray(reference_projection)
                std_ratio = np.nanstd(proj_vals) / np.nanstd(ref_vals)
                pm = np.nanmean(proj_vals)
                rm = np.nanmean(ref_vals)
                centred_rmse = float(
                    np.sqrt(np.nanmean(((proj_vals - pm) - (ref_vals - rm)) ** 2))
                )
                row.update({
                    "centered_rmse": centred_rmse,
                    "r_value": r,
                    "std_ratio": std_ratio,
                })
                logger.debug(
                    f"    {label}: r={r:.4f}, p={p:.4e}, std_ratio={std_ratio:.4f}, "
                    f"centred_rmse={centred_rmse:.4f}"
                )
                taylor_stats[label] = (float(r), float(std_ratio))
            else:
                row.update({"centered_rmse": None, "r_value": None, "std_ratio": None})
            metrics.append(row)

            projection_plot = projection.where(projection != 0)
            self.spatial_plot(projection=projection_plot, model_label=label, info_vals=info_vals, vmin=_shared_vmin, vmax=_shared_vmax)

        # ── All-models panel ─────────────────────────────────────────────────
        all_models_panel = {
            label: (projections[label].where(projections[label] != 0),
                    {f"{explained_variances[label]: .1%}"})
            for label in projections
        }
        self.spatial_plot_all_models(
            all_models_panel, _shared_vmin, _shared_vmax,
            fname=f"{self.method}_all_models.pdf",
        )

        if taylor_stats and ref_name is not None:
            self.taylor_plotter.plot(
                model_stats=taylor_stats,
                colors=colors,
                ref_label=ref_name,
                title=f"{self.name}",
                fname=f"{self.method}_taylor_diagram.pdf",
            )

        self.timeseries_plotter.plot(
            model_data=eof_modes,
            colors=colors,
            title=f"{self.name} – EOF Mode 1 Principal Component",
            variable_name="",
            xlabel="Time",
            ylabel="EOF Mode 1 Amplitude",
            xticks=None,
            fname=f"{self.method}_eof_timeseries.pdf",
        )

    def _evaluate_per_member(self, data_containers) -> dict:
        """Per-member evaluation: compute EOF for each individual member and for
        the ensemble mean.  All member results and the member-mean are collected
        and visualised together.

        For each data container that carries a ``member`` dimension:

        * One spatial EOF pattern per member (file: ``{method}_{label}_m{m}.pdf``).
        * One spatial EOF pattern for the member mean
          (file: ``{method}_{label}_member_mean.pdf``).
        * All per-member principal-component time series plus the member-mean
          series are drawn on a single timeseries plot per container.
        * The Taylor diagram uses the member-mean projection as the
          representative pattern for each non-reference container.

        Containers without a ``member`` dimension are handled identically to
        the standard path.
        """
        import glob as _glob

        colors = self._colors(data_containers)
        ref_name, _ = self._reference(data_containers)
        output_path = self.plotter_kwargs.get("output_path", ".")
        os.makedirs(output_path, exist_ok=True)

        # ── 1. Compute / load EOF results ─────────────────────────────────────
        # Outer key: model_label  →  dict of {series_key: (time_da, projection, expl_var)}
        # ``series_key`` is e.g. "m0", "m1", "member_mean" or "" (no member dim).
        all_results: dict = {}   # {model_label: {series_key: (time_da, projection, expl_var)}}

        # Helper: canonical nc path for a (label, series_key) pair.
        def _model_dir(label: str) -> str:
            d = os.path.join(output_path, label)
            os.makedirs(d, exist_ok=True)
            return d

        def _proj_nc(label, key):
            return self._nc_path(_model_dir(label), f"{self.method}_{label}_{key}_projection")

        def _ts_nc(label, key):
            return self._nc_path(_model_dir(label), f"{self.method}_{label}_{key}_timeseries")

        # Helper: load a single cached (time_da, projection, expl_var) tuple.
        def _load_series(label, key):
            ds_proj = self._load_nc(_proj_nc(label, key))
            ds_ts   = self._load_nc(_ts_nc(label, key))
            if ds_proj is None or ds_ts is None:
                return None
            projection = ds_proj["data"].transpose("lat", "lon")
            expl_var   = float(ds_proj.attrs.get("explained_variance", 0.0))
            ds_proj.close()
            t_labels = ds_ts["time_labels"].values.astype(str)
            time_da = xr.DataArray(
                ds_ts["data"].values, coords={"time": t_labels}, dims=["time"]
            )
            ds_ts.close()
            return time_da, projection, expl_var

        # Helper: save a single (time_da, projection, expl_var) tuple.
        def _save_series(label, key, time_da, projection, expl_var):
            self._save_nc(projection, _proj_nc(label, key),
                          attrs={"explained_variance": expl_var})
            t_arr = np.array(time_da.time.values, dtype=str)
            self._save_nc(
                xr.Dataset({
                    "data": xr.DataArray(time_da.values, dims=["t_idx"]),
                    "time_labels": xr.DataArray(t_arr, dims=["t_idx"]),
                }),
                _ts_nc(label, key),
            )

        # Helper: check if a model has all its series cached.
        # We use the "sentinel" files: member_mean (per-member) or "" (no-member).
        def _model_is_cached(label):
            # Check for member_mean sentinel
            if os.path.exists(_proj_nc(label, "member_mean")) and \
               os.path.exists(_ts_nc(label, "member_mean")):
                return True
            # Check for single (no-member) sentinel – empty key encoded as ""
            if os.path.exists(_proj_nc(label, "")) and \
               os.path.exists(_ts_nc(label, "")):
                return True
            return False

        # Helper: discover all cached series keys for a model via glob.
        def _load_all_cached_series(label):
            pattern = os.path.join(output_path, label, f"{self.method}_{label}_*_projection.nc")
            proj_files = sorted(_glob.glob(pattern))
            series = {}
            prefix = f"{self.method}_{label}_"
            suffix = "_projection.nc"
            for pf in proj_files:
                key = os.path.basename(pf)[len(prefix):-len(suffix)]
                result = _load_series(label, key)
                if result is not None:
                    series[key] = result
            return series

        # Process the reference container first so its projection is available
        # for sign-correction of all subsequent containers.
        ref_projection = None
        ref_dc = next((dc for dc in data_containers if dc.model_label == ref_name), None)
        ordered_containers = (
            [ref_dc] + [dc for dc in data_containers if dc.model_label != ref_name]
            if ref_dc is not None else list(data_containers)
        )

        metrics = []
        for dc in ordered_containers:
            is_reference = dc.model_label == ref_name

            if _model_is_cached(dc.model_label):
                logger.info(
                    f"    Loading cached per-member EOF results for '{dc.model_label}'"
                )
                series = _load_all_cached_series(dc.model_label)
            else:
                logger.info(
                    f"--> {self.name}: per_member evaluation for '{dc.model_label}'"
                )
                data = self._extract_data(dc, all_members=True)   # (time, lat, lon[, member])
                series: dict = {}

                if "member" in data.dims:
                    members = data.member.values
                    print(f"Members found in '{dc.model_label}': {members}")
                    logger.info(f"    member dimension: {len(members)} members")

                    for m in members:
                        member_data = data.sel(member=m, drop=True)
                        t, proj, eig, exv = self._compute_single(member_data)
                        if not is_reference and ref_projection is not None:
                            r, _, _ = self.regress_against_reference(proj, ref_projection)
                            if r < 0:
                                logger.debug(
                                    f"    {dc.model_label} m{m}: sign flipped (r={r:.4f})"
                                )
                                proj = -proj
                                eig  = -eig
                        time_da = xr.DataArray(
                            eig, coords={"time": np.array(t, dtype=str)}, dims=["time"]
                        )
                        series[f"m{m}"] = (time_da, proj, exv)
                        _save_series(dc.model_label, f"m{m}", time_da, proj, exv)

                    # Member mean
                    all_member_keys = [f"m{m}" for m in members]
                    mean_eig = np.stack(
                        [series[k][0].values for k in all_member_keys], axis=0
                    ).mean(axis=0)
                    mean_proj = xr.concat(
                        [series[k][1] for k in all_member_keys], dim="_member"
                    ).mean(dim="_member")
                    mean_exv = float(np.mean([series[k][2] for k in all_member_keys]))
                    t_coords = series[all_member_keys[0]][0].time.values
                    time_da_mean = xr.DataArray(mean_eig, coords={"time": t_coords}, dims=["time"])
                    series["member_mean"] = (time_da_mean, mean_proj, mean_exv)
                    _save_series(dc.model_label, "member_mean", time_da_mean, mean_proj, mean_exv)
                else:
                    t, proj, eig, exv = self._compute_single(data)
                    if not is_reference and ref_projection is not None:
                        r, _, _ = self.regress_against_reference(proj, ref_projection)
                        if r < 0:
                            logger.debug(f"    {dc.model_label}: sign flipped (r={r:.4f})")
                            proj = -proj
                            eig  = -eig
                    time_da = xr.DataArray(
                        eig, coords={"time": np.array(t, dtype=str)}, dims=["time"]
                    )
                    series[""] = (time_da, proj, exv)
                    _save_series(dc.model_label, "", time_da, proj, exv)

            all_results[dc.model_label] = series

            if is_reference:
                if "member_mean" in series:
                    ref_projection = series["member_mean"][1]
                elif "" in series:
                    ref_projection = series[""][1]

            # --- Collect metrics for CSV ---
            for series_key, (time_da, projection, expl_var) in series.items():
                row = {"model": dc.model_label, "member": series_key, "explained_variance": expl_var}
                # Compare all non-reference series (including member_mean) to reference.
                if (
                    dc.model_label != ref_name and ref_projection is not None
                ):
                    r, p, std_err = self.regress_against_reference(projection, ref_projection)
                    proj_vals = np.asarray(projection)
                    ref_vals  = np.asarray(ref_projection)
                    std_ratio = np.nanstd(proj_vals) / np.nanstd(ref_vals)
                    pm = np.nanmean(proj_vals)
                    rm = np.nanmean(ref_vals)
                    centred_rmse = float(
                        np.sqrt(np.nanmean(((proj_vals - pm) - (ref_vals - rm)) ** 2))
                    )
                    row.update({
                        "centered_rmse": centred_rmse,
                        "r_value": r,
                        "std_ratio": std_ratio,
                    })
                else:
                    row.update({"centered_rmse": None, "r_value": None, "std_ratio": None})
                metrics.append(row)

        # Compute averages for per-member (excluding member_mean and reference)
        member_rows = [r for r in metrics if r["member"].startswith("m") and r["model"] != ref_name]
        if member_rows:
            avg_row = {"model": "AVG", "member": "", "explained_variance": None}
            for k in ["centered_rmse", "r_value", "std_ratio", "explained_variance"]:
                vals = [r[k] for r in member_rows if r[k] is not None]
                avg_row[k] = float(np.mean(vals)) if vals else None
            metrics.append(avg_row)

        # ── 3. Spatial plots + Taylor stats ───────────────────────────────────
        # model_label -> (best_r, best_std_ratio, best_series_key)
        _model_best: dict = {}
        projections = {}
        for model_label, series in all_results.items():
            for series_key, (time_da, projection, expl_var) in series.items():
                projections[f"{model_label}_{series_key}"] = projection

        vmin, vmax = self.get_shared_vmin_vmax(projections)
        for model_label, series in all_results.items():
            model_panel: dict = {}
            for series_key, (time_da, projection, expl_var) in series.items():
                member_suffix = f"_{series_key}" if series_key else ""
                combined_label = f"{model_label}{member_suffix}"

                info_vals = {"explained_variance": f"{float(expl_var * 100)} %"}

                if (
                    combined_label != ref_name
                    and ref_projection is not None
                    and not (model_label == ref_name)
                    and series_key != "member_mean"
                ):
                    r, p, std_err = self.regress_against_reference(
                        projection, ref_projection
                    )
                    proj_vals = np.asarray(projection)
                    ref_vals  = np.asarray(ref_projection)
                    std_ratio = np.nanstd(proj_vals) / np.nanstd(ref_vals)
                    logger.debug(
                        f"    {combined_label}: r={r:.4f}, p={p:.4e}, "
                        f"std_ratio={std_ratio:.4f}"
                    )
                    # Track the best member (highest r) per model.
                    if model_label not in _model_best or r > _model_best[model_label][0]:
                        _model_best[model_label] = (float(r), float(std_ratio), series_key)

                panel_label = f"{model_label} ({series_key})" if series_key else model_label
                model_panel[panel_label] = (projection.where(projection != 0), info_vals)

            # Plot all members for this model in a single subplot figure.
            self.spatial_plot_all_models(
                model_panel, vmin, vmax,
                fname=f"{self.method}_{model_label}_all_members.pdf",
            )

        # ── All-models best-member panel ──────────────────────────────────────
        pm_panel: dict = {}
        for model_label, series in all_results.items():
            if model_label == ref_name:
                key = "member_mean" if "member_mean" in series else ""
                if key in series:
                    _, proj, exv = series[key]
                    pm_panel[model_label] = (proj.where(proj != 0),
                                            {"explained_variance": f"{float(exv):.1%}"})
            elif model_label in _model_best:
                best_r, best_std_ratio, best_key = _model_best[model_label]
                _, proj, exv = series[best_key]
                if best_key.startswith("m") and best_key[1:].isdigit():
                    disp = f"{model_label}" # (m={int(best_key[1:]) + 1})
                else:
                    disp = model_label
                pm_panel[disp] = (proj.where(proj != 0),
                                  {"explained_variance": f"{float(exv):.1%}"})
        self.spatial_plot_all_models(
            pm_panel, vmin, vmax,
            fname=f"{self.method}_all_models_best_member.pdf",
        )

        # Build taylor_stats keyed by "ModelName (m=N)" using the best member.
        taylor_stats: dict = {}
        taylor_colors: dict = {}
        for model_label, (best_r, best_std_ratio, best_key) in _model_best.items():
            if best_key.startswith("m") and best_key[1:].isdigit():
                int(best_key[1:]) + 1
                legend_label = f"{model_label}" # (m={member_idx})
            else:
                legend_label = model_label
            taylor_stats[legend_label] = (best_r, best_std_ratio)
            taylor_colors[legend_label] = colors.get(model_label, "black")

        if taylor_stats and ref_name is not None:
            self.taylor_plotter.plot(
                model_stats=taylor_stats,
                colors=taylor_colors,
                ref_label=ref_name,
                title=f"{self.name}",
                fname=f"{self.method}_taylor_diagram_per_member.pdf",
            )

        # Combined timeseries plot per model (all members + member mean)
        for model_label, series in all_results.items():
            member_eof_modes: dict = {}
            member_colors: dict = {}
            base_color = colors.get(model_label, "black")
            n_series = len(series)

            for i, (series_key, (time_da, _, _)) in enumerate(series.items()):
                combined_label = f"{model_label} ({series_key})" if series_key else model_label
                member_eof_modes[combined_label] = time_da

                if series_key in ("member_mean", ""):
                    member_colors[combined_label] = base_color
                else:
                    try:
                        import matplotlib.colors as mcolors

                        rgba = list(mcolors.to_rgba(base_color))
                        rgba[3] = max(0.15, 1.0 - 0.7 * (i / max(n_series - 1, 1)))
                        member_colors[combined_label] = tuple(rgba)
                    except Exception:
                        member_colors[combined_label] = base_color

            self.timeseries_plotter.plot(
                model_data=member_eof_modes,
                colors=member_colors,
                title=f"{self.name} – EOF Mode 1 PC – {model_label} (per member)",
                variable_name="",
                xlabel="Time",
                ylabel="EOF Mode 1 Amplitude",
                xticks=None,
                fname=f"{self.method}_eof_timeseries_{model_label}_per_member.pdf",
            )

        return metrics

    def _write_metrics_csv(self, metrics):
        """Write Annular Mode metrics to CSV for all models/members."""
        import pandas as pd
        output_path = self.plotter_kwargs.get("output_path", ".")
        os.makedirs(output_path, exist_ok=True)
        csv_path = os.path.join(output_path, f"{self.method}_metrics.csv")
        df = pd.DataFrame(metrics)
        df.to_csv(csv_path, index=False)
        logger.info(f"AnnularModes: metrics written to {csv_path}")
        


class NorthernAnnularMode(AnnularModes):
    """
    Northern Annular Mode (NAM / Arctic Oscillation).

    Computed from surface pressure (``psl``) or geopotential (``zg``) on a
    user-specified pressure level.  The spatial EOF pattern is rendered using
    an azimuthal-equidistant polar projection.
    """

    def __init__(
        self,
        method: str = "EOF",
        var: str = "psl",
        time=["November", "December", "January", "February"],
        plotter_kwargs: dict = None,
        baseline_period: tuple = ("1981-01-01", "2010-12-31"),
        frequency: str = "monthly",
        latitude_bands: tuple = (20, 90),
        longitude_bands: tuple = (-180, 180),
        per_member: bool = False,
        weight_by_latitude: bool = False,
    ) -> None:
        super().__init__(
            method=method,
            time=time,
            baseline_period=baseline_period,
            plotter_kwargs=plotter_kwargs,
            frequency=frequency,
            latitude_bands=latitude_bands,
            longitude_bands=longitude_bands,
            per_member=per_member,
            weight_by_latitude=weight_by_latitude
        )
        self.name = "NAM"
        self.var = var
        self._cartopy = CartopyProjectionPlotter(
            output_path=self.plotter_kwargs.get("output_path", ".")
        )

    def _projection_params(self) -> dict:
        cbar_label = "Pa / \u03c3(PC\u2081)" if self.var == "psl" else "m / \u03c3(PC\u2081)"
        return {
            "central_latitude": 90,
            "extent": [-180, 180, 20, 90],
            "cbar_label": cbar_label,
        }

    def spatial_plot(self, projection, model_label: str, info_vals=None, member_suffix: str = "", vmin=None, vmax=None) -> None:
        if member_suffix:
            _core = member_suffix.lstrip("_")
            if _core == "member_mean":
                display_label = f"{model_label} (Ensemble mean)"
            elif _core.startswith("m") and _core[1:].isdigit():
                display_label = f"{model_label} (Member {int(_core[1:]) + 1})"
            else:
                display_label = f"{model_label} ({_core})"
        else:
            display_label = model_label
        cbar_label = "Pa / σ(PC₁)" if self.var == "psl" else "m / σ(PC₁)"
        model_dir = os.path.join(self._cartopy.output_path, model_label)
        os.makedirs(model_dir, exist_ok=True)
        self._cartopy.azimuthal_equidistant(
            data=projection,
            central_latitude=90,
            extent=[-180, 180, 20, 90],
            fname=f"{model_label}/{self.method}_{model_label}{member_suffix}.pdf",
            infotext_topleft=display_label,
            info_vals=info_vals,
            vmin=vmin,
            vmax=vmax,
            cbar_label=cbar_label,
        )

    def _extract_data(self, data_container, all_members=False) -> xr.DataArray:
        """Return the raw DataArray with the member dimension preserved (if present)."""
        if self.var == "psl":
            data = data_container.get_variable_data(
                name="psl", frequency=self.frequency, pressure_level=None, all_members=all_members)
        elif self.var == "zg":
            data = data_container.get_variable_data(
                name="zg", frequency=self.frequency, pressure_level=100000, all_members=all_members)
        else:
            raise ValueError(f"Unknown variable for Northern Annular Mode: {self.var}")
        
        if "stat" in data.dims:
            data = data.sel(stat="mean", drop=True)
        return data

    def compute(self, data_container):
        data = self._extract_data(data_container)

        return super().compute(data)

class SouthernAnnularMode(AnnularModes):
    def __init__(
        self,
        method: str = "EOF",
        plotter_kwargs: dict = None,
        baseline_period: tuple = ("1981-01-01", "2010-12-31"),
        frequency: str = "monthly",
        time=["November", "December", "January", "February"],
        latitude_bands: tuple = (-20, -90),
        longitude_bands: tuple = (-180, 180),
        per_member: bool = False,
    ) -> None:
        super().__init__(
            method=method,
            time=time,
            baseline_period=baseline_period,
            plotter_kwargs=plotter_kwargs,
            frequency=frequency,
            latitude_bands=latitude_bands,
            longitude_bands=longitude_bands,
            per_member=per_member,
        )
        self.name = "SAM"
        self._shared_orography_mask: xr.DataArray | None = None
        self._shared_orography_mask_source: str | None = None
        self._reference_label: str | None = None
        self._cartopy = CartopyProjectionPlotter(
            output_path=self.plotter_kwargs.get("output_path", ".")
        )

    def prepare_shared_orography_mask(self, data_containers, all_members: bool = False) -> None:
        """Build a common NaN mask from the non-reference model with most NaN grid cells.

        The selected mask is later applied to all SAM inputs (including the
        reference) so every model shares the same orographic coverage.
        """
        ref_name, _ = self._reference(data_containers)
        self._reference_label = ref_name
        self._shared_orography_mask = None
        self._shared_orography_mask_source = None

        max_nan_cells = -1

        for dc in data_containers:
            if dc.model_label == ref_name:
                continue

            var = self._extract_data(dc, all_members=all_members, apply_shared_mask=False)
            collapse_dims = [d for d in var.dims if d not in ("lat", "lon")]
            spatial_nan_mask = var.isnull()
            if collapse_dims:
                # Orography masks are expected to be static over time/member.
                spatial_nan_mask = spatial_nan_mask.all(dim=collapse_dims)

            nan_cells = int(spatial_nan_mask.sum().compute().item())
            if nan_cells > max_nan_cells:
                max_nan_cells = nan_cells
                self._shared_orography_mask = spatial_nan_mask
                self._shared_orography_mask_source = dc.model_label

        if self._shared_orography_mask is not None:
            logger.info(
                "    SAM shared orography mask: using '%s' (%d NaN grid cells) for all models.",
                self._shared_orography_mask_source,
                max_nan_cells,
            )
        else:
            logger.info("    SAM shared orography mask: no non-reference mask found.")

    def _projection_params(self) -> dict:
        return {
            "central_latitude": -90,
            "extent": [-180, 180, -90, -20],
            "cbar_label": "m / \u03c3(PC\u2081)",
        }

    def spatial_plot(
            self, projection, model_label: str, info_vals=None,
            member_suffix: str = "", vmin=None, vmax=None) -> None:
        if member_suffix:
            _core = member_suffix.lstrip("_")
            if _core == "member_mean":
                display_label = f"{model_label} (Ensemble mean)"
            elif _core.startswith("m") and _core[1:].isdigit():
                display_label = f"{model_label} (Member {int(_core[1:]) + 1})"
            else:
                display_label = f"{model_label} ({_core})"
        else:
            display_label = model_label
        model_dir = os.path.join(self._cartopy.output_path, model_label)
        os.makedirs(model_dir, exist_ok=True)
        self._cartopy.azimuthal_equidistant(
            data=projection,
            central_latitude=-90,
            extent=[-180, 180, -90, -20],
            fname=f"{model_label}/{self.method}_{model_label}{member_suffix}.pdf",
            infotext_topleft=display_label,
            info_vals=info_vals,
            vmin=vmin,
            vmax=vmax,
            cbar_label="m / σ(PC₁)",
        )

    def _extract_data(
        self,
        data_container,
        all_members=False,
        apply_shared_mask: bool = True,
    ) -> xr.DataArray:
        """Return the raw DataArray with the member dimension preserved (if present)."""
        var = data_container.get_variable_data(
            name="zg", frequency=self.frequency,
            pressure_level=70000, all_members=all_members)

        if "stat" in var.dims:
            var = var.sel(stat="mean", drop=True)

        if (
            apply_shared_mask
            and self._shared_orography_mask is not None
        ):
            var = var.where(~self._shared_orography_mask)

        return var

    def evaluate(self, data_containers) -> None:
        self.prepare_shared_orography_mask(
            data_containers,
            all_members=self.per_member,
        )
        super().evaluate(data_containers)

    def compute(self, data_container):
        var = self._extract_data(data_container)

        return super().compute(var)


class NorthernAtlanticOscillationIndex(AnnularModes):
    """
    North Atlantic Oscillation Index (NAOI).

    Spatial EOF pattern is rendered using an azimuthal-equidistant projection
    with an NAO wedge overlay showing the North Atlantic domain.
    """

    def __init__(
        self,
        method: str = "EOF",
        plotter_kwargs: dict = None,
        baseline_period: tuple = ("1981-01-01", "2010-12-31"),
        frequency: str = "monthly",
        latitude_bands: tuple = (20, 80),
        longitude_bands: tuple = (90, 220),
        per_member: bool = False,
    ) -> None:
        super().__init__(
            method=method,
            baseline_period=baseline_period,
            plotter_kwargs=plotter_kwargs,
            frequency=frequency,
            latitude_bands=latitude_bands,
            longitude_bands=longitude_bands,
            per_member=per_member,
        )
        self.name = "NAOI"
        self._cartopy = CartopyProjectionPlotter(
            output_path=self.plotter_kwargs.get("output_path", ".")
        )

    def _projection_params(self) -> dict:
        return {
            "central_latitude": 90,
            "extent": [-180, 180, 20, 90],
            "cbar_label": "Pa / \u03c3(PC\u2081)",
            "wedge": "noa",
        }

    def spatial_plot(self, projection, model_label: str, info_vals=None, member_suffix: str = "", vmin=None, vmax=None) -> None:
        if member_suffix:
            _core = member_suffix.lstrip("_")
            if _core == "member_mean":
                display_label = f"{model_label} (Ensemble mean)"
            elif _core.startswith("m") and _core[1:].isdigit():
                display_label = f"{model_label} (Member {_core[1:]})"
            else:
                display_label = f"{model_label} ({_core})"
        else:
            display_label = model_label
        model_dir = os.path.join(self._cartopy.output_path, model_label)
        os.makedirs(model_dir, exist_ok=True)
        self._cartopy.azimuthal_equidistant(
            data=projection,
            central_latitude=90,
            extent=[-180, 180, 20, 90],
            fname=f"{model_label}/{self.method}_{model_label}{member_suffix}.pdf",
            wedge="noa",
            infotext_topleft=display_label,
            info_vals=info_vals,
            vmin=vmin,
            vmax=vmax,
            cbar_label="Pa / σ(PC₁)",
        )

    def _extract_data(self, data_container) -> xr.DataArray:
        """Return the raw DataArray with the member dimension preserved (if present)."""
        psl = data_container.get_variable_data(name="psl", frequency=self.frequency, pressure_level=None)
        if "stat" in psl.dims:
            psl = psl.sel(stat="mean", drop=True)
        return psl

    def compute(self, data_container):
        psl = self._extract_data(data_container)
        if "member" in psl.dims:
            logger.info(f"    NorthernAtlanticOscillationIndex: member dimension detected (size {psl.sizes['member']}), averaging over members")
            psl = psl.mean(dim="member")
        return super().compute(psl)


# ============================================================================
# Spectral metrics
# ============================================================================

class RadialSpectrum(BaseMetric):
    """
    Radial (isotropic) power spectrum on a spherical grid.

    Inherits :class:`BaseMetric` for variable handling and ``plotter_kwargs``
    storage.  Uses :class:`~plot.modules.FrequencyPlotter` for all rendering.
    """

    def __init__(
        self,
        variables: list,
        time_instances: list,
        frequency: str = "monthly",
        plotter_kwargs: dict = None,
        per_member: bool = False,
    ) -> None:
        super().__init__(variables=variables, frequency=frequency, plotter_kwargs=plotter_kwargs, per_member=per_member)
        self.time_instances = time_instances
        self._freq_plotter = FrequencyPlotter(**self.plotter_kwargs)

    def compute_spectrum(self, data: xr.DataArray) -> dict:
        """Return ``{instance: radial_spectrum}`` averaged over all matching time steps."""
        radial_spectra = {}
        for instance in self.time_instances:
            logger.info(f"--> RadialSpectrum: computing spectrum for instance '{instance}'")
            if isinstance(instance, str) and instance.upper() in ("DJF", "MAM", "JJA", "SON"):
                _data = data.sel(time=data.time.dt.season == instance.upper())
            else:
                # [start_mmdd, end_mmdd] window within each year
                _data = data.sel(
                    time=(
                        (data.time.dt.strftime("%Y-%m-%d") >= instance[0])
                        & (data.time.dt.strftime("%Y-%m-%d") <= instance[1])
                    )
                )

            if _data.shape[0] == 0:
                logger.warning(f"    No data found for instance '{instance}', skipping.")
                continue

            radial_spectra[instance] = (
                sum(spectral.compute_radial_spectrum(x) for x in _data.values)
                / _data.shape[0]
            )
        return radial_spectra

    def compute(self, data_containers, variable_name: str, lvl) -> dict:
        """Return ``{model_label: {instance: spectrum}}`` for all containers.

        When the data container exposes a ``member`` dimension (ensemble),
        the radial spectrum is computed independently for each member and the
        results are averaged, preserving the full distribution of energy at
        each scale without the smoothing caused by averaging the fields first.
        """
        radial_spectra = {}
        for dc in data_containers:
            data = dc.get_variable_data(
                name=variable_name, frequency=self.frequency, pressure_level=lvl,
                all_members=True,
            )
            if "member" in data.dims:
                # Compute spectrum independently per member, then average.
                per_member = [
                    self.compute_spectrum(data.sel(member=m, drop=True))
                    for m in data.member.values
                ]
                avg: dict = {}
                for instance in per_member[0]:
                    arrays = [ms[instance] for ms in per_member if instance in ms]
                    avg[instance] = np.mean(arrays, axis=0)
                radial_spectra[dc.model_label] = avg
            else:
                radial_spectra[dc.model_label] = self.compute_spectrum(data)
        return radial_spectra

    def evaluate(self, data_containers) -> None:
        colors = self._colors(data_containers)
        base_output_path = self.plotter_kwargs.get("output_path", ".")

        all_var_data = []  # list of (title, ylabel, model_spectra)

        for var in self.variables:
            name, pressure_level = var
            self._log(name, pressure_level)

            display_name = _long_var_label(name, pressure_level)
            if pressure_level is None:
                fname_var = name
            else:
                fname_var = f"{name}_{int(round(pressure_level / 100))}hPa"

            output_path = os.path.join(base_output_path, fname_var)
            os.makedirs(output_path, exist_ok=True)

            # ── cache paths per model ─────────────────────────────────────────
            nc_paths = {
                dc.model_label: self._nc_path(
                    output_path, f"radial_spectrum_{fname_var}_{dc.model_label}"
                )
                for dc in data_containers
            }

            radial_spectra: dict = {}
            if self._all_cached(list(nc_paths.values())):
                logger.info(
                    f"    Loading cached radial spectrum results from '{output_path}'"
                )
                for dc in data_containers:
                    ds = self._load_nc(nc_paths[dc.model_label])
                    import json as _json
                    raw_instances = _json.loads(ds.attrs.get("instances", "[]"))
                    instance_spectra: dict = {}
                    for i, inst_str in enumerate(raw_instances):
                        # Raw instances is a string like '["yyyy-mm-dd", "yyyy-mm-dd"]'
                        # so we have to turn it back into a list +
                        inst_str = inst_str.replace("[", "").replace("]", "").replace('"', "").replace("'", "")
                        inst_str = inst_str.replace(" ", "")
                        print(f"Parsed instance string: {inst_str}")
                        var_name = f"inst_{i}"
                        if var_name in ds:
                            instance_spectra[inst_str] = ds[var_name].values
                    radial_spectra[dc.model_label] = instance_spectra
                    ds.close()
            else:
                radial_spectra = self.compute(data_containers, name, pressure_level)
                import json as _json
                for dc in data_containers:
                    instance_spectra = radial_spectra.get(dc.model_label, {})
                    inst_list = list(instance_spectra.keys())
                    ds_vars = {
                        f"inst_{i}": xr.DataArray(instance_spectra[inst], dims=["wavenumber"])
                        for i, inst in enumerate(inst_list)
                    }
                    ds = xr.Dataset(ds_vars, attrs={"instances": _json.dumps([str(k) for k in inst_list])})
                    self._save_nc(ds, nc_paths[dc.model_label])

            all_var_data.append((display_name, "Power Spectral Density", radial_spectra))

        self._freq_plotter.plot_radial_multi(
            var_data=all_var_data,
            colors=colors,
            fname="radial_spectra.pdf",
        )


# ============================================================================
# Distribution metrics
# ============================================================================

class Distribution(BaseMetric):
    """
    Abstract base for distribution-based diagnostics.

    Subclasses implement :py:meth:`compute` (extract a 1-D sample array) and
    :py:meth:`visualize` (render the distribution).
    """

    def __init__(
        self,
        variables: list,
        time: list = None,
        frequency: str = "monthly",
        lat_band: tuple = None,
        lon_band: tuple = None,
        plotter_kwargs: dict = None,
        per_member: bool = False,
    ) -> None:
        super().__init__(variables=variables, plotter_kwargs=plotter_kwargs, per_member=per_member)
        # ``time`` is expected to be a list of [start, end] pairs, e.g.
        # [["2020-01-01", "2020-12-31"], ["2021-01-01", "2021-12-31"]].
        # Pass ``None`` to skip time filtering entirely.
        self.time = time
        self.frequency = frequency
        # Optional spatial restriction applied before sampling values.
        # lat_band: (south_lat, north_lat), e.g. (-30, 30) for the tropics.
        # lon_band: (west_lon, east_lon), e.g. (0, 180).
        self.lat_band = lat_band
        self.lon_band = lon_band

    def _apply_spatial_selection(self, data: xr.DataArray) -> xr.DataArray:
        """Restrict *data* to ``lat_band`` / ``lon_band`` when set."""
        if self.lat_band is not None:
            lat_min, lat_max = min(self.lat_band), max(self.lat_band)
            if data.lat.values[0] > data.lat.values[-1]:
                data = data.sel(lat=slice(lat_max, lat_min))
            else:
                data = data.sel(lat=slice(lat_min, lat_max))
        if self.lon_band is not None:
            lon_min, lon_max = min(self.lon_band), max(self.lon_band)
            data = data.sel(lon=slice(lon_min, lon_max))
        return data

    def compute(self, data: xr.DataArray) -> np.ndarray:
        raise NotImplementedError

    def visualize(
        self,
        distributions: dict,
        variable_name: str = "Variable",
        time_range=None,
        lat_band=None,
        lon_band=None,
    ) -> None:
        raise NotImplementedError

    def evaluate(self, data_containers) -> None:
        # Iterate over every [start, end] time range supplied; fall back to a
        # single pass with no time filtering when ``self.time`` is None.
        time_ranges = self.time if self.time is not None else [None]
        # Normalize lat/lon band parameters to lists so they can be iterated
        # independently, mirroring how ``time`` iterates over multiple ranges.
        self.lat_band = omegaconf.OmegaConf.to_container(self.lat_band) if self.lat_band is not None else None
        self.lon_band = omegaconf.OmegaConf.to_container(self.lon_band) if self.lon_band is not None else None

        lat_bands = _normalize_band_list(self.lat_band)
        lon_bands = _normalize_band_list(self.lon_band)
        output_path = self.plotter_kwargs.get("output_path", ".")

        for var in self.variables:
            name, pressure_level = var
            if pressure_level is None:
                var_fname = name
            else:
                var_fname = f"{name}_{int(round(pressure_level / 100))}hPa"

            for time_range in time_ranges:
                time_dir = (
                    f"{time_range[0]}_{time_range[1]}"
                    if time_range is not None else "all_times"
                )

                for lat_b in lat_bands:
                    for lon_b in lon_bands:
                        # Make the active selection available to helpers that
                        # call ``self._apply_spatial_selection``.
                        self.lat_band = lat_b
                        self.lon_band = lon_b
                        latlon_d = _latlon_dir(lat_b, lon_b)
                        cache_dir = os.path.join(output_path, time_dir, latlon_d, var_fname)

                        # ── cache paths ───────────────────────────────────────
                        nc_paths = {
                            dc.model_label: self._nc_path(
                                cache_dir, f"dist_samples_{var_fname}_{dc.model_label}"
                            )
                            for dc in data_containers
                        }

                        distributions: dict = {}
                        if self._all_cached(list(nc_paths.values())):
                            logger.info(
                                f"    Loading cached distribution samples from '{cache_dir}'"
                            )
                            for dc in data_containers:
                                ds = self._load_nc(nc_paths[dc.model_label])
                                if ds is not None:
                                    distributions[dc.model_label] = (
                                        ds["samples"].values, dc.model_color
                                    )
                                    ds.close()
                        else:
                            for dc in data_containers:
                                # Request all members so that each physical realisation is
                                # treated independently – averaging fields first would smooth
                                # extremes and distort the distribution.
                                logger.info(
                                    f"--> {dc.model_label}: extracting data for '{name}' "
                                    f"at {pressure_level} Pa ")
                                data = dc.get_variable_data(
                                    name=name,
                                    pressure_level=pressure_level,
                                    frequency=self.frequency,
                                    all_members=True,
                                )

                                if time_range is not None:
                                    # Check if the requested time range overlaps with the data's time span.
                                    data_start = data.time.min().values
                                    data_end = data.time.max().values
                                    bool1 = np.datetime64(time_range[1], "ns") < data_start
                                    bool2 = np.datetime64(time_range[0], "ns") > data_end
                                    if bool1 or bool2:
                                        logger.warning(
                                            f"    Time range {time_range} is outside "
                                            f"the data time span for '{dc.model_label}'. "
                                            f"Skipping this range for this model."
                                        )
                                        continue

                                    data = data.sel(time=slice(
                                        np.datetime64(time_range[0], "ns"),
                                        np.datetime64(time_range[1], "ns"),
                                    ))

                                data = self._apply_spatial_selection(data)

                                if "member" in data.dims:
                                    # Compute samples per member and concatenate.  Using
                                    # density=True in the histogram then naturally normalises
                                    # the combined sample, which is equivalent to averaging
                                    # the per-member normalised histograms.
                                    samples = np.concatenate([
                                        self.compute(data.sel(member=m, drop=True))
                                        for m in data.member.values
                                    ])
                                else:
                                    samples = self.compute(data)

                                distributions[dc.model_label] = (samples, dc.model_color)
                                # ── save samples to cache ─────────────────────
                                os.makedirs(cache_dir, exist_ok=True)
                                #self._save_nc(
                                #    xr.Dataset({"samples": xr.DataArray(samples, dims=["sample"])}),
                                #    nc_paths[dc.model_label],
                                #)

                        self.visualize(
                            distributions,
                            variable_name=(name, pressure_level),
                            time_range=time_range,
                            lat_band=lat_b,
                            lon_band=lon_b,
                        )


class Histogram(Distribution):
    """Value-distribution histogram with per-model statistics annotation."""

    def __init__(
        self,
        variables: list,
        time=None,
        frequency: str = "monthly",
        lat_band: tuple = None,
        lon_band: tuple = None,
        plotter_kwargs: dict = None,
        per_member: bool = False,
    ) -> None:
        super().__init__(variables, time=time, frequency=frequency,
                         lat_band=lat_band, lon_band=lon_band,
                         plotter_kwargs=plotter_kwargs, per_member=per_member)
        self.figsize = self.plotter_kwargs.get("figsize", (6.7, 5.0))
        self.dpi = self.plotter_kwargs.get("dpi", 150)
        self.output_path = self.plotter_kwargs.get("output_path", ".")
        os.makedirs(self.output_path, exist_ok=True)
        self.cmor_units = TimeseriesPlotter(**self.plotter_kwargs).cmor_units

    def compute(self, data: xr.DataArray) -> np.ndarray:
        return data.values.flatten()

    def visualize(
        self,
        data: dict,
        variable_name: str = "Variable",
        time_range=None,
        lat_band=None,
        lon_band=None,
    ) -> None:
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        for label, (density, bin_edges, color) in data.items():
            legend_label = f"{label}"
            ax.stairs(
                density, bin_edges, alpha=1.0, label=legend_label, color=color,
                linewidth=1.5,
            )
        ax.set_yscale("log")

        if len(data) > 2:
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.1),
                ncol=2, fontsize=6, framealpha=0.8,
            )
        else:
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.1),
                ncol=1, fontsize=6, framealpha=0.8,
            )

        plt.grid(True, which="both", linestyle="-.", linewidth=0.5, alpha=0.5)
        plt.xlabel(self.cmor_units.get(variable_name[0]), fontsize=8)
        plt.ylabel("Density", fontsize=8)

        var_display = _long_var_label(variable_name[0], variable_name[1])
        # Keep filenames short and filesystem-safe.
        if variable_name[1] is None:
            var_fname = variable_name[0]
        else:
            var_fname = f"{variable_name[0]}_{int(round(variable_name[1] / 100))}hPa"

        plt.text(
            0.0, 1.01, var_display, fontsize=10, ha="left", va="bottom", 
            transform=ax.transAxes)

        annot = _build_selection_annotation(
            lat_band=lat_band,
            lon_band=lon_band,
            time_range=time_range,
        )
        
        if annot:
            ax.text(
                1.0, 1.01, annot,
                transform=ax.transAxes,
                fontsize=10,
                va="bottom", ha="right",
            )

        # Build sub-directory tree: time-range / lat-lon / var.
        if time_range is not None:
            time_dir = f"{time_range[0]}_{time_range[1]}"
        else:
            time_dir = "all_times"
        latlon_d = _latlon_dir(lat_band, lon_band)
        # set yticks and x ticks labels size
        ax.tick_params(axis='both', which='major', labelsize=8)
        out_dir = os.path.join(self.output_path, time_dir, latlon_d, var_fname)
        os.makedirs(out_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"histogram_{var_fname}.pdf"), bbox_inches="tight",
            dpi=300)
        plt.close()

    def visualize_multi(
        self,
        var_data: list,
        time_range=None,
        lat_band=None,
        lon_band=None,
        out_dir: str = ".",
        fname: str = "histograms.pdf",
    ) -> None:
        """Plot up to 3 variables per row in one figure with a single shared legend.

        Parameters
        ----------
        var_data :
            List of ``((name, pressure_level), distributions)`` tuples where
            ``distributions`` is ``{model_label: (samples_array, color)}``.
        """
        import math
        n = len(var_data)
        n_cols = min(n, 2)
        n_rows = math.ceil(n / 2)
        fw, fh = self.figsize
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fw, fh),
            dpi=self.dpi, squeeze=False,
        )

        for idx, ((name, pressure_level), distributions) in enumerate(var_data):
            row, col = divmod(idx, 2)
            ax = axes[row][col]

            for label, (density, bin_edges, color) in distributions.items():
                ax.stairs(
                    density, bin_edges, alpha=1.0, label=label, color=color,
                    linewidth=1.,
                )
            ax.set_yscale("log")

            ax.grid(True, which="both", linestyle="-.", linewidth=0.3, alpha=0.5)
            unit = self.cmor_units.get(name, "")
            ax.set_xlabel(unit, fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=8)
            var_display = _long_var_label(name, pressure_level)
            ax.text(
                0.0, 1.01, var_display, fontsize=10, ha="left", va="bottom",
                transform=ax.transAxes,
            )

        # Annotation on top-right of the last used panel
        annot = _build_selection_annotation(lat_band=lat_band, lon_band=lon_band, time_range=time_range)
        if annot:
            last_row, last_col = divmod(n - 1, 2)
            axes[last_row][last_col].text(
                1.0, 1.01, annot, fontsize=10,
                va="bottom", ha="right",
                transform=axes[last_row][last_col].transAxes,
            )

        # Hide unused axes in the last row
        for idx in range(n, n_rows * n_cols):
            row, col = divmod(idx, 2)
            axes[row][col].set_visible(False)

        # One shared legend collected from the first panel
        handles, labels = axes[0][0].get_legend_handles_labels()
        ncols_legend = 2 if len(handles) > 2 else 1
        axes[0][0].legend(
            handles, labels,
            loc="best",
            ncol=ncols_legend,
            fontsize=6,
            framealpha=0.8,
        )

        os.makedirs(out_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def evaluate(self, data_containers) -> None:
        """Override: collect all variables per time/lat/lon selection, plot in groups of 3."""
        import math
        time_ranges = self.time if self.time is not None else [None]
        self.lat_band = omegaconf.OmegaConf.to_container(self.lat_band) if self.lat_band is not None else None
        self.lon_band = omegaconf.OmegaConf.to_container(self.lon_band) if self.lon_band is not None else None
        lat_bands = _normalize_band_list(self.lat_band)
        lon_bands = _normalize_band_list(self.lon_band)
        output_path = self.plotter_kwargs.get("output_path", ".")

        for time_range in time_ranges:
            time_dir = (
                f"{time_range[0]}_{time_range[1]}" if time_range is not None else "all_times"
            )
            for lat_b in lat_bands:
                for lon_b in lon_bands:
                    self.lat_band = lat_b
                    self.lon_band = lon_b
                    latlon_d = _latlon_dir(lat_b, lon_b)

                    all_var_data = []  # list of ((name, lvl), distributions)

                    for var in self.variables:
                        name, pressure_level = var
                        if pressure_level is None:
                            var_fname = name
                        else:
                            var_fname = f"{name}_{int(round(pressure_level / 100))}hPa"

                        cache_dir = os.path.join(output_path, time_dir, latlon_d, var_fname)
                        npz_paths = {
                            dc.model_label: os.path.join(
                                cache_dir, f"hist_{var_fname}_{dc.model_label}.npz"
                            )
                            for dc in data_containers
                        }

                        distributions: dict = {}
                        if self._all_cached(list(npz_paths.values())):
                            logger.info(
                                f"    Loading cached histogram data from '{cache_dir}'"
                            )
                            for dc in data_containers:
                                npz_path = npz_paths[dc.model_label]
                                if os.path.exists(npz_path):
                                    arr = np.load(npz_path)
                                    distributions[dc.model_label] = (
                                        arr["density"], arr["bin_edges"], dc.model_color
                                    )
                        else:
                            for dc in data_containers:
                                logger.info(
                                    f"--> {dc.model_label}: extracting data for '{name}' "
                                    f"at {pressure_level} Pa"
                                )
                                data = dc.get_variable_data(
                                    name=name,
                                    pressure_level=pressure_level,
                                    frequency=self.frequency,
                                    all_members=True,
                                )

                                if time_range is not None:
                                    data_start = data.time.min().values
                                    data_end = data.time.max().values
                                    bool1 = np.datetime64(time_range[1], "ns") < data_start
                                    bool2 = np.datetime64(time_range[0], "ns") > data_end
                                    if bool1 or bool2:
                                        logger.warning(
                                            f"    Time range {time_range} is outside "
                                            f"the data time span for '{dc.model_label}'. "
                                            f"Skipping this range for this model."
                                        )
                                        continue
                                    data = data.sel(time=slice(
                                        np.datetime64(time_range[0], "ns"),
                                        np.datetime64(time_range[1], "ns"),
                                    ))

                                data = self._apply_spatial_selection(data)

                                if "member" in data.dims:
                                    samples = np.concatenate([
                                        self.compute(data.sel(member=m, drop=True))
                                        for m in data.member.values
                                    ])
                                else:
                                    samples = self.compute(data)

                                samples = samples[np.isfinite(samples)]
                                if samples.size == 0:
                                    logger.warning(
                                        f"    No finite values for '{name}' in "
                                        f"'{dc.model_label}' – skipping."
                                    )
                                    continue
                                density, bin_edges = np.histogram(
                                    samples, bins=100, density=True
                                )
                                os.makedirs(cache_dir, exist_ok=True)
                                np.savez(
                                    npz_paths[dc.model_label],
                                    density=density,
                                    bin_edges=bin_edges,
                                )
                                distributions[dc.model_label] = (
                                    density, bin_edges, dc.model_color
                                )

                        if distributions:
                            all_var_data.append(((name, pressure_level), distributions))

                    # Plot all variables in chunks of ≤3 per figure
                    n_chunks = math.ceil(len(all_var_data) / 3) if all_var_data else 0
                    out_dir = os.path.join(output_path, time_dir, latlon_d)
                    for chunk_idx in range(n_chunks):
                        chunk = all_var_data[chunk_idx * 3:(chunk_idx + 1) * 3]
                        suffix = f"_part{chunk_idx + 1}" if n_chunks > 1 else ""
                        self.visualize_multi(
                            var_data=chunk,
                            time_range=time_range,
                            lat_band=lat_b,
                            lon_band=lon_b,
                            out_dir=out_dir,
                            fname=f"histograms{suffix}.pdf",
                        )

class ReturnPeriods(Distribution):
    """Block-maxima extreme value analysis (GEV) of domain-wide extremes.

    For each variable the field is first reduced to its spatial maximum at
    every time step (optionally restricted to *lat_band* / *lon_band*), then
    split into annual block maxima.  Ensemble members are pooled together –
    more block maxima give a better-constrained fit – before a GEV
    distribution is fit via :func:`scipy.stats.genextreme.fit`.  A single
    return-period plot (return level vs. return period, with empirical
    points overlaid) is produced per variable, overlaying every model on
    shared axes, alongside a CSV of the fitted parameters and return levels.
    """

    def __init__(
        self,
        variables: list,
        time: list = None,
        frequency: str = "daily",
        lat_band: tuple = None,
        lon_band: tuple = None,
        return_periods: list = None,
        plotter_kwargs: dict = None,
        per_member: bool = False,
    ) -> None:
        super().__init__(
            variables, time=time, frequency=frequency,
            lat_band=lat_band, lon_band=lon_band,
            plotter_kwargs=plotter_kwargs, per_member=per_member,
        )
        self.return_periods = np.asarray(
            return_periods or [2, 5, 10, 20, 50, 100, 200], dtype=float
        )
        self.figsize = self.plotter_kwargs.get("figsize", (8, 5))
        self.dpi = self.plotter_kwargs.get("dpi", 300)
        self.output_path = self.plotter_kwargs.get("output_path", ".")
        os.makedirs(self.output_path, exist_ok=True)
        self.cmor_units = EarthPlotter.cmor_units

    def compute(self, data: xr.DataArray) -> np.ndarray:
        """Return one block maximum per calendar year (domain max reduced first)."""
        return annual_block_maxima(domain_max_series(data))

    def visualize(
        self,
        distributions: dict,
        variable_name=("Variable", None),
        time_range=None,
        lat_band=None,
        lon_band=None,
    ) -> None:
        name, pressure_level = variable_name
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)

        csv_rows = []
        for label, (samples, color) in distributions.items():
            samples = samples[np.isfinite(samples)]
            if samples.size < 3:
                logger.warning(
                    f"    Not enough block maxima for '{label}' ({samples.size}) "
                    f"– skipping GEV fit."
                )
                continue

            shape, loc, scale = fit_gev(samples)
            levels = return_levels(self.return_periods, shape, loc, scale)
            ax.plot(self.return_periods, levels, color=color, linewidth=1.8, label=label)

            emp_vals, emp_periods = empirical_return_periods(samples)
            ax.scatter(emp_periods, emp_vals, color=color, s=12, alpha=0.6, zorder=3)

            for period, level in zip(self.return_periods, levels):
                csv_rows.append({
                    "model": label,
                    "return_period_years": period,
                    "return_level": level,
                    "gev_shape": shape,
                    "gev_loc": loc,
                    "gev_scale": scale,
                })

        ax.set_xscale("log")
        ax.set_xticks(self.return_periods)
        ax.set_xlabel("Return Period (years)", fontsize=10)
        ax.set_ylabel(self.cmor_units.get(name, ""), fontsize=10)
        ax.grid(True, which="both", linestyle="-.", linewidth=0.5, alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=8)

        var_display = _long_var_label(name, pressure_level)
        ax.text(
            0.0, 1.01, var_display, fontsize=10, ha="left", va="bottom",
            transform=ax.transAxes,
        )
        annot = _build_selection_annotation(lat_band=lat_band, lon_band=lon_band, time_range=time_range)
        if annot:
            ax.text(
                1.0, 1.01, annot, transform=ax.transAxes, fontsize=10,
                va="bottom", ha="right",
            )
        if distributions:
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.15),
                ncol=min(len(distributions), 4), fontsize=8, framealpha=0.8,
            )

        if pressure_level is None:
            var_fname = name
        else:
            var_fname = f"{name}_{int(round(pressure_level / 100))}hPa"

        time_dir = f"{time_range[0]}_{time_range[1]}" if time_range is not None else "all_times"
        latlon_d = _latlon_dir(lat_band, lon_band)
        out_dir = os.path.join(self.output_path, time_dir, latlon_d, var_fname)
        os.makedirs(out_dir, exist_ok=True)

        plt.tight_layout()
        fpath = os.path.join(out_dir, f"return_periods_{var_fname}.pdf")
        plt.savefig(fpath, bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)
        logger.info(f"    Saved return-period plot: {fpath}")

        if csv_rows:
            csv_path = os.path.join(out_dir, f"return_levels_{var_fname}.csv")
            pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
            logger.info(f"    Saved return levels CSV: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Quantitative Climate Baseline
# ═══════════════════════════════════════════════════════════════════════════


class QuantitativeBaseline(BaseMetric):
    """Score model maps against a reference to build a quantitative baseline.

    This metric **does not compute its own maps**.  Instead it loads NetCDF
    files produced by upstream metric evaluators (``XYMaps``,
    ``AnnularModes``, ``SouthernOscillationIndex``, ``XYTrendMaps``, …) and
    evaluates every model's map against the reference map using
    latitude-weighted pattern scores:

    * **rmse** – root-mean-square error
    * **pattern_correlation** – centred, area-weighted Pearson *r*
    * **bias** – mean (model − reference)
    * **centered_rmse** – RMSE after removing area-weighted field means
    * **std_ratio** – σ_model / σ_reference

    A user-configurable subset of models (``baseline_models``) is averaged to
    yield a single "baseline" score per map source and score function,
    providing a compact, CMIP-style multi-model performance summary.

    Parameters
    ----------
    map_sources : list[dict]
        Each entry selects one diagnostic to score.  Required keys:

        * ``name`` – human-readable label (file-name and table-row key).
        * ``path_template`` – path *relative to the results root* containing
          a ``{model_label}`` placeholder, e.g.
          ``"SeasonalMeans/annual/tas/{model_label}_tas.nc"``.

        Optional keys:

        * ``variable`` – NetCDF variable to read (default ``"data"``).
        * ``lat_weighted`` – apply cosine-latitude weights (default ``True``).

    baseline_models : list[str]
        Model labels whose per-source scores are averaged into the
        ``BASELINE_AVG`` summary row.
    scores : list[str], optional
        Subset of ``{"rmse", "pattern_correlation", "bias",
        "centered_rmse", "std_ratio"}``.  Default:
        ``["rmse", "pattern_correlation", "bias"]``.
    results_root : str, optional
        Root directory that contains upstream metric outputs.  When omitted
        the parent directory of this metric's own *output_path* is used
        (which matches the default layout created by ``GeoClimate``).
    plotter_kwargs : dict
        Must contain ``output_path``.

    Outputs
    -------
    Per map-source
        ``{source_name}_scores.nc`` — one score variable per score function,
        indexed by ``model_label``.
    Aggregate
        ``all_scores.csv`` — full table (model × source × score).
        ``baseline_scores.csv`` — baseline models + ``BASELINE_AVG`` row.
        ``other_scores.csv`` — non-baseline models + ``OTHER_AVG`` row.
        ``summary_scores.nc`` — 2-D ``(source × model)`` NetCDF plus
        ``{score}_baseline_avg`` 1-D variables.
    """

    SCORE_FUNCTIONS = frozenset({
        "rmse",
        "pattern_correlation",
        "bias",
        "centered_rmse",
        "std_ratio",
    })

    def __init__(
        self,
        map_sources: list,
        baseline_models: list,
        scores: list = None,
        results_root: str = None,
        plotter_kwargs: dict = None,
        frequency: str = "monthly",
    ) -> None:
        super().__init__(frequency=frequency, plotter_kwargs=plotter_kwargs)
        self.map_sources = list(map_sources)
        self.baseline_models = list(baseline_models)
        self.scores = list(scores) if scores else [
            "rmse", "pattern_correlation", "bias",
        ]
        self._results_root_override = results_root

        unknown = set(self.scores) - self.SCORE_FUNCTIONS
        if unknown:
            raise ValueError(
                f"Unknown score(s): {unknown}.  "
                f"Choose from {sorted(self.SCORE_FUNCTIONS)}."
            )

    # ── latitude weights ──────────────────────────────────────────────────

    @staticmethod
    def _lat_weights(lat: np.ndarray, shape: tuple) -> np.ndarray:
        """Cosine-latitude area weights, normalised and broadcast to *shape*."""
        w = np.cos(np.deg2rad(lat.astype(np.float64)))
        w = np.clip(w, 0.0, None)
        total = w.sum()
        if total > 0:
            w = w / total
        return np.broadcast_to(w[:, np.newaxis], shape)

    # ── scoring ───────────────────────────────────────────────────────────

    @classmethod
    def _compute_scores(
        cls,
        model: np.ndarray,
        ref: np.ndarray,
        weights: np.ndarray | None,
    ) -> dict[str, float]:
        """Compute all available pattern scores between two 2-D fields.

        Returns a dict keyed by every member of :pyattr:`SCORE_FUNCTIONS`.
        """
        valid = ~(np.isnan(model) | np.isnan(ref))
        if not np.any(valid):
            return {s: np.nan for s in cls.SCORE_FUNCTIONS}

        m = model[valid].astype(np.float64)
        r = ref[valid].astype(np.float64)
        w = (
            weights[valid].astype(np.float64)
            if weights is not None
            else np.ones_like(m)
        )
        w = w / w.sum()

        bias = float(np.sum(w * (m - r)))
        rmse = float(np.sqrt(np.sum(w * (m - r) ** 2)))

        m_wm = float(np.sum(w * m))
        r_wm = float(np.sum(w * r))
        ma, ra = m - m_wm, r - r_wm

        m_std = float(np.sqrt(np.sum(w * ma**2)))
        r_std = float(np.sqrt(np.sum(w * ra**2)))

        if m_std > 0 and r_std > 0:
            pattern_correlation = float(
                np.sum(w * ma * ra) / (m_std * r_std)
            )
        else:
            pattern_correlation = np.nan

        centered_rmse = float(np.sqrt(np.sum(w * (ma - ra) ** 2)))
        std_ratio = float(m_std / r_std) if r_std > 0 else np.nan

        return {
            "rmse": rmse,
            "pattern_correlation": pattern_correlation,
            "bias": bias,
            "centered_rmse": centered_rmse,
            "std_ratio": std_ratio,
        }

    # ── persistence helpers ───────────────────────────────────────────────

    def _save_source_scores(
        self,
        source_scores: dict[str, dict[str, float]],
        nc_path: str,
    ) -> None:
        """Write per-model scores for **one** map source to NetCDF."""
        labels = list(source_scores.keys())
        if not labels:
            return
        score_names = list(next(iter(source_scores.values())).keys())
        data_vars = {}
        for sn in score_names:
            vals = [source_scores[ml].get(sn, np.nan) for ml in labels]
            data_vars[sn] = xr.DataArray(
                vals,
                dims=["model_label"],
                coords={"model_label": labels},
            )
        ds = xr.Dataset(data_vars)
        os.makedirs(os.path.dirname(nc_path), exist_ok=True)
        ds.to_netcdf(nc_path)
        logger.debug("  [cache] saved source scores → %s", nc_path)

    def _save_aggregated(
        self,
        all_source_scores: dict[str, dict[str, dict[str, float]]],
        model_labels: list[str],
        output_path: str,
    ) -> None:
        """Write summary CSV and NetCDF files for all sources combined."""
        baseline_set = set(self.baseline_models)
        bl_labels = [ml for ml in model_labels if ml in baseline_set]
        oth_labels = [ml for ml in model_labels if ml not in baseline_set]

        # ── flat table ────────────────────────────────────────────────────
        rows: list[dict] = []
        for src_name, src_scores in all_source_scores.items():
            for ml, sc in src_scores.items():
                row = {
                    "source": src_name,
                    "model_label": ml,
                    "group": (
                        "baseline" if ml in baseline_set else "other"
                    ),
                }
                row.update(sc)
                rows.append(row)

            # baseline average
            if bl_labels:
                bl_row = {
                    "source": src_name,
                    "model_label": "BASELINE_AVG",
                    "group": "baseline_avg",
                }
                for sn in self.scores:
                    vals = [
                        src_scores[ml][sn]
                        for ml in bl_labels
                        if ml in src_scores
                        and not np.isnan(src_scores[ml].get(sn, np.nan))
                    ]
                    bl_row[sn] = float(np.mean(vals)) if vals else np.nan
                rows.append(bl_row)

            # non-baseline average
            if oth_labels:
                oth_row = {
                    "source": src_name,
                    "model_label": "OTHER_AVG",
                    "group": "other_avg",
                }
                for sn in self.scores:
                    vals = [
                        src_scores[ml][sn]
                        for ml in oth_labels
                        if ml in src_scores
                        and not np.isnan(src_scores[ml].get(sn, np.nan))
                    ]
                    oth_row[sn] = float(np.mean(vals)) if vals else np.nan
                rows.append(oth_row)

        if not rows:
            logger.warning(
                "  QuantitativeBaseline: no scores computed – nothing to save."
            )
            return

        df = pd.DataFrame(rows)

        csv_all = os.path.join(output_path, "all_scores.csv")
        df.to_csv(csv_all, index=False, float_format="%.6f")
        logger.info("  Saved full table → %s", csv_all)

        df_bl = df[df["group"].isin(["baseline", "baseline_avg"])]
        csv_bl = os.path.join(output_path, "baseline_scores.csv")
        df_bl.to_csv(csv_bl, index=False, float_format="%.6f")

        df_oth = df[df["group"].isin(["other", "other_avg"])]
        csv_oth = os.path.join(output_path, "other_scores.csv")
        df_oth.to_csv(csv_oth, index=False, float_format="%.6f")

        # ── multi-dimensional NetCDF summary ──────────────────────────────
        src_names = list(all_source_scores.keys())
        data_vars: dict[str, xr.DataArray] = {}
        for sn in self.scores:
            vals = np.full((len(src_names), len(model_labels)), np.nan)
            for i, src in enumerate(src_names):
                for j, ml in enumerate(model_labels):
                    if ml in all_source_scores[src]:
                        vals[i, j] = all_source_scores[src][ml].get(
                            sn, np.nan
                        )
            data_vars[sn] = xr.DataArray(
                vals,
                dims=["source", "model_label"],
                coords={"source": src_names, "model_label": model_labels},
            )

            # 1-D baseline average per source
            if bl_labels:
                bl_vals: list[float] = []
                for src in src_names:
                    vs = [
                        all_source_scores[src][ml].get(sn, np.nan)
                        for ml in bl_labels
                        if ml in all_source_scores[src]
                    ]
                    valid = [v for v in vs if not np.isnan(v)]
                    bl_vals.append(
                        float(np.mean(valid)) if valid else np.nan
                    )
                data_vars[f"{sn}_baseline_avg"] = xr.DataArray(
                    bl_vals,
                    dims=["source"],
                    coords={"source": src_names},
                )

        ds = xr.Dataset(
            data_vars,
            attrs={"baseline_models": ", ".join(self.baseline_models)},
        )
        nc_sum = os.path.join(output_path, "summary_scores.nc")
        ds.to_netcdf(nc_sum)
        logger.info("  Saved summary NetCDF → %s", nc_sum)

    # ── main entry point ──────────────────────────────────────────────────

    def evaluate(self, data_containers) -> None:
        ref_label, _ = self._reference(data_containers)
        if ref_label is None:
            logger.warning(
                "QuantitativeBaseline: no reference container – skipping."
            )
            return

        output_path = self.plotter_kwargs.get("output_path", ".")
        os.makedirs(output_path, exist_ok=True)

        results_root = (
            self._results_root_override
            or os.path.dirname(output_path.rstrip("/"))
        )

        model_labels = [
            dc.model_label
            for dc in data_containers
            if not dc.is_reference
        ]

        logger.info(
            "==> QuantitativeBaseline: %d source(s), %d model(s), "
            "baseline = %s",
            len(self.map_sources),
            len(model_labels),
            self.baseline_models,
        )

        all_source_scores: dict[str, dict[str, dict[str, float]]] = {}

        for src in self.map_sources:
            src_name: str = src["name"]
            path_tpl: str = src["path_template"]
            nc_var: str = src.get("variable", "data")
            use_lat_w: bool = src.get("lat_weighted", True)

            logger.info("--> Scoring source '%s'", src_name)

            # ── per-source cache ──────────────────────────────────────────
            score_nc = self._nc_path(output_path, f"{src_name}_scores")
            if os.path.exists(score_nc):
                logger.info("    Cached – loading '%s'", score_nc)
                ds = xr.open_dataset(score_nc)
                source_scores: dict[str, dict[str, float]] = {}
                for ml in ds.model_label.values:
                    ml_str = str(ml)
                    source_scores[ml_str] = {
                        s: float(ds[s].sel(model_label=ml).values)
                        for s in self.scores
                        if s in ds
                    }
                ds.close()
                all_source_scores[src_name] = source_scores
                continue

            # ── load reference map ────────────────────────────────────────
            ref_path = os.path.join(
                results_root,
                path_tpl.format(model_label=ref_label),
            )
            ref_ds = self._load_nc(ref_path)
            if ref_ds is None:
                logger.warning(
                    "    Reference map not found at '%s' – skipping source.",
                    ref_path,
                )
                continue

            ref_map = ref_ds[nc_var].values.astype(np.float64)

            weights = None
            if use_lat_w and "lat" in ref_ds[nc_var].dims:
                weights = self._lat_weights(
                    ref_ds[nc_var].lat.values, ref_map.shape,
                )
            ref_ds.close()

            # ── score every non-reference model ───────────────────────────
            source_scores = {}
            for ml in model_labels:
                m_path = os.path.join(
                    results_root,
                    path_tpl.format(model_label=ml),
                )
                m_ds = self._load_nc(m_path)
                if m_ds is None:
                    logger.warning(
                        "    Map for '%s' not found (%s) – NaN.",
                        ml,
                        m_path,
                    )
                    source_scores[ml] = {s: np.nan for s in self.scores}
                    continue

                m_map = m_ds[nc_var].values.astype(np.float64)
                m_ds.close()

                all_sc = self._compute_scores(m_map, ref_map, weights)
                source_scores[ml] = {s: all_sc[s] for s in self.scores}

                parts = ", ".join(
                    f"{s}={_fmt_scalar(all_sc[s])}" for s in self.scores
                )
                logger.info("    %s : %s", ml, parts)

            all_source_scores[src_name] = source_scores
            self._save_source_scores(source_scores, score_nc)

        # ── aggregate and persist ─────────────────────────────────────────
        self._save_aggregated(all_source_scores, model_labels, output_path)
        logger.info("==> QuantitativeBaseline: complete.")

