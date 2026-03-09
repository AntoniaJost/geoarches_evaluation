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
│   │   ├─ XYBiasMaps
│   │   └─ XYAnomalyMaps
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
│   └─ AnimatedHistogram
└─ TropicalCycloneFrequency

EOF is a standalone computational helper (not a metric evaluator).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import CenteredNorm
from scipy import stats

from metrics import functional
from metrics.functional import spectral
from metrics.functional.extremes import (
    TC_SLP_THRESHOLDS_HPA,
    TC_INTENSITY_LABELS,
    WNP_LON,
    WNP_LAT,
    detect_tc_candidates,
    build_tc_tracks,
    compute_tc_frequency,
    compute_tc_count_per_year,
    compute_tc_count_by_intensity,
    compute_clim_z_dayofyear,
)
from metrics.base import (
    BaseMetric,
    SpatialMetric,
    TimeseriesMetric,
    _format_var_name,
    _get_model_colors,
    _get_reference_container,
    _log_variable_info,
    annual_mean,
    seasonal_mean,
    instantaneous,
    select_by_time,
    compute_latitude_weights,
    compute_anomaly,
    compute_bias,
    compute_eof,
    compute_correlation_matrix,
    compute_soi,
    detrend_data,
)
from plot.modules import FrequencyPlotter, TimeseriesPlotter, TaylorDiagramPlotter
from plot.projections import CartopyProjectionPlotter


# ============================================================================
# Spatial map metrics
# ============================================================================

class XYMaps(SpatialMetric):
    """
    Global or regional lat/lon mean maps.

    Produces one image per variable × temporal-selection × model, optionally
    computing RMSE and bias against a reference dataset.
    """

    def eval_against_reference(
        self, data: xr.DataArray, reference_data: xr.DataArray
    ) -> tuple:
        """Return latitude-weighted (RMSE, bias) against *reference_data*.

        Works directly on NumPy arrays to avoid creating redundant xarray
        copies (which would each trigger a dask graph materialisation).
        """
        latitude = data.lat.values
        longitude = data.lon.values
        w = compute_latitude_weights(latitude, longitude)
        if len(data.values.shape) == 2:
            w = w[0]

        # Materialise once and weight in NumPy – avoids two xarray copies.
        wd = data.values * w
        wr = reference_data.values * w

        mask = ~np.isnan(wr)
        wd, wr = wd[mask], wr[mask]
        rmse = float(np.sqrt(np.mean((wd - wr) ** 2)))
        bias = float((wd - wr).mean())
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
            print(f"Computing {self.xdim}-{self.ydim} map for temporal selection: {ts}")
            print("-" * 72)
            for variable in self.variables:
                name, pressure_level = variable["name"], variable["pressure_level"]
                _log_variable_info(name, pressure_level)

                maps = {}
                for dc in data_containers:
                    data = dc.get_variable_data(**variable, frequency=self.frequency)
                    data = self.compute(data, temporal_dim=ts, var=variable, frequency=self.frequency)
                    data = data.transpose(self.ydim, self.xdim)
                    var_name = _format_var_name(name, pressure_level)
                    output_path = os.path.join(self.plotter.output_path, ts)
                    maps[dc.model_label] = (data, output_path, var_name)

                # Compute shared colorbar limits across all models so the
                # figures are directly comparable.
                all_values = np.concatenate([d.values.ravel() for d, _, _ in maps.values()])
                shared_vmin = float(np.nanmin(all_values))
                shared_vmax = float(np.nanmax(all_values))
                cbar_label = self.plotter.cmor_units.get(name, "")

                for model_label, (data, output_path, var_name) in maps.items():
                    if self.reference_data is not None:
                        rmse, bias = self.eval_against_reference(data, maps[self.reference_label][0])
                        print(f"RMSE against reference for {model_label}: {rmse:.4f}")
                        print(f"Bias against reference for {model_label}: {bias:.4f}")
                    else:
                        rmse, bias = None, None
                        print(f"No reference data available for evaluation of {model_label}.")

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
                        info={
                            "rmse": f"{rmse:.2f}" if rmse is not None else "N/A",
                            "bias": f"{bias:.2f}" if bias is not None else "N/A",
                        },
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
        print(f"Computing {self.xdim}-{self.ydim} Hovmöller diagram")
        print("-" * 72)
        for variable in self.variables:
            name, pressure_level = variable["name"], variable["pressure_level"]
            _log_variable_info(name, pressure_level)

            maps = {}
            for dc in data_containers:
                data_array = dc.get_variable_data(**variable, frequency=self.frequency)
                time_vals, lat_vals, z = self.compute(
                    data_array, temporal_dim=None, var=variable, frequency=self.frequency
                )
                var_name = _format_var_name(name, pressure_level)
                maps[dc.model_label] = (time_vals, lat_vals, z, self.plotter.output_path, var_name)

            # Compute shared colorbar limits across all models so the
            # Hovmöller diagrams are directly comparable.
            all_z_values = np.concatenate([z.ravel() for _, _, z, _, _ in maps.values()])
            shared_vmin = float(np.nanmin(all_z_values))
            shared_vmax = float(np.nanmax(all_z_values))

            for model_label, (time_vals, lat_vals, z, output_path, var_name) in maps.items():
                fig = plt.figure(figsize=(12, 6), dpi=150)
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
                plt.savefig(
                    os.path.join(output_path, f"{model_label}_{var_name}.png"),
                    bbox_inches="tight",
                )
                plt.close(fig)


class XYBiasMaps(XYMaps):
    """Spatial bias maps (model minus reference)."""

    def evaluate(self, data_containers) -> None:
        self.get_reference_data(data_containers)
        for ts in self.temporal_selection:
            for variable in self.variables:
                name, pressure_level = variable["name"], variable["pressure_level"]
                _log_variable_info(name, pressure_level)

                maps = {}
                for dc in data_containers:
                    print(f"... for model: {dc.model_label}: ", end="")
                    if dc.model_label == self.reference_label:
                        print("Skipping reference model for bias map.")
                        continue
                    data = dc.get_variable_data(**variable, frequency=self.frequency)
                    ref_data = self.reference_data.get_variable_data(
                        **variable, frequency=self.frequency
                    )
                    # Compute model and reference maps separately so they can be
                    # passed to eval_against_reference for scalar RMSE / bias stats.
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
                    rmse, bias_val = self.eval_against_reference(model_map, ref_map)
                    print(f"RMSE: {rmse:.4f}, Bias: {bias_val:.4f}")
                    var_name = _format_var_name(name, pressure_level)
                    output_path = os.path.join(self.plotter.output_path, ts)
                    maps[dc.model_label] = (bias_map, output_path, var_name, rmse, bias_val)

                # Symmetric shared limits so the bias colorbars match across
                # all models and the colour scale is centred on zero.
                all_values = np.concatenate([bm.values.ravel() for bm, _, _, _, _ in maps.values()])
                abs_max = float(np.nanmax(np.abs(all_values)))
                shared_vmin = -abs_max
                shared_vmax = abs_max
                cbar_label = self.plotter.cmor_units.get(name, "")

                for model_label, (bias_map, output_path, var_name, rmse, bias_val) in maps.items():
                    self.plotter.plot(
                        x=bias_map,
                        variable_name=f"{var_name}_bias",
                        title="",
                        model_label=model_label,
                        style="imshow",
                        output_path=output_path,
                        vmin=shared_vmin,
                        vmax=shared_vmax,
                        cbar_label=cbar_label,
                        info={
                            "RMSE": f"{rmse:.4f}",
                            "Bias": f"{bias_val:.4f}",
                        },
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
    ) -> None:
        super().__init__(variables, xdim, ydim, temporal_selection, frequency, plotter_kwargs)
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


def _compute_linear_trend(x_values, y_values):
    """Return (trend_line, slope_per_decade) from a 1-D time series."""
    coeffs = np.polyfit(range(len(x_values)), y_values, deg=1)
    trend = np.polyval(coeffs, range(len(x_values)))
    slope_per_decade = coeffs[0] * 12 * 10  # monthly → per decade
    return trend, slope_per_decade


class SeasonalCycles(Timeseries):
    """Global-mean seasonal cycle (or anomaly cycle) per variable."""

    def __init__(
        self,
        variables: list,
        mean_groups: list = None,
        linear_trend: bool = False,
        detrend: bool = False,
        compute_anomalies: bool = False,
        baseline_period: tuple = None,
        baseline_mean_groups: list = None,
        plotter_kwargs: dict = None,
    ) -> None:
        super().__init__(
            variables=variables,
            detrend=detrend,
            compute_anomalies=compute_anomalies,
            baseline_period=baseline_period,
            baseline_mean_groups=baseline_mean_groups,
            mean_groups=mean_groups or ["year"],
            plotter_kwargs=plotter_kwargs,
        )
        if linear_trend and detrend:
            raise ValueError("Cannot apply both linear_trend and detrend simultaneously.")
        self.linear_trend = linear_trend

    def compute(self, data: xr.DataArray) -> xr.DataArray:
        data = data.mean(dim=["lat", "lon"])
        print("Data spatially averaged.", end="; ")

        if self.detrend:
            data.values = detrend_data(data).values
            print("Data detrended.", end="; ")

        if self.compute_anomalies:
            seasonal_cycle = compute_anomaly(
                data=data,
                baseline_period=self.baseline_period,
                mean_groups=self.mean_groups,
                baseline_mean_groups=self.baseline_mean_groups,
            )
        else:
            seasonal_cycle = data.groupby(self.mean_groups).mean(dim=["time"])
        print("Computed temporal mean.")

        time = self.xlabels_from_time(seasonal_cycle)
        seasonal_cycle = seasonal_cycle.stack(
            time=[g.split(".")[-1] for g in self.mean_groups]
        ).reset_index("time")
        seasonal_cycle["time"] = time
        return seasonal_cycle.dropna(dim="time", how="any")

    def evaluate(self, data_containers) -> None:
        print("-" * 72)
        for variable in self.variables:
            name, pressure_level = variable["name"], variable["pressure_level"]
            _log_variable_info(name, pressure_level)

            cycles: dict = {}
            standard_deviations: dict = {}
            for dc in data_containers:
                print(f"... for model: {dc.model_label}: ", end="")
                data = dc.get_variable_data(name=name, pressure_level=pressure_level, frequency="monthly")
                if "stat" in data.dims:
                    cycle = self.compute(data.sel(stat="mean", drop=True))
                    std_cycle = self.compute(data.sel(stat="std", drop=True))
                    standard_deviations[dc.model_label] = std_cycle.values
                else:
                    cycle = self.compute(data)
                cycles[dc.model_label] = (cycle.time.values, cycle.values)

            colors = _get_model_colors(data_containers)
            trends: dict = {}
            if self.linear_trend:
                for label, (x, y) in cycles.items():
                    trend_values, m = _compute_linear_trend(x, y)
                    trends[label] = (trend_values, m)

            var_name = _format_var_name(name, pressure_level)
            self.timeseries_plotter.plot(
                model_data=cycles,
                model_stds=standard_deviations,
                linear_trend=trends,
                colors=colors,
                title="",
                variable_name=name,
                xlabel="",
                ylabel=f"{self.timeseries_plotter.cmor_units.get(name, '')}",
                xticks=None,
                fname=f"{var_name}.png",
            )
            print("... Done.")
            print("-" * 72)


class MonsoonIndices(BaseMetric):
    """
    Monsoon strength indices (Webster-Yang and others).

    Inherits :class:`BaseMetric` to obtain shared helpers and consistent
    ``plotter_kwargs`` handling.  Uses :class:`TimeseriesPlotter` for output.
    """

    def __init__(
        self,
        method: str = "webster_yang",
        frequency: str = "monthly",
        target_year: int = 2024,
        baseline_period: tuple = None,
        plotter_kwargs: dict = None,
    ) -> None:
        # Embed method name in output path before passing to base.
        pk = dict(plotter_kwargs or {})
        pk["output_path"] = os.path.join(pk.get("output_path", "."), method)
        super().__init__(variables=[], frequency=frequency, plotter_kwargs=pk)

        self.method = method
        self.target_year = target_year
        self.baseline_period = baseline_period
        self.timeseries_plotter = TimeseriesPlotter(**self.plotter_kwargs)

    # -- Index computation methods ------------------------------------------

    def webster_yang(self, data_container, baseline: bool = False) -> xr.DataArray:
        """Webster-Yang 850-250 hPa wind shear index."""
        u850 = data_container.get_variable_data(name="ua", pressure_level=85000, frequency=self.frequency)
        u250 = data_container.get_variable_data(name="ua", pressure_level=25000, frequency=self.frequency)

        if "stat" in u850.dims:
            u850 = u850.sel(stat="mean", drop=True)
            u250 = u250.sel(stat="mean", drop=True)

        u850 = u850.sel(lat=slice(20, 0), lon=slice(220, 290))
        u250 = u250.sel(lat=slice(20, 0), lon=slice(220, 290))

        if baseline:
            print(f"Selecting baseline period: {self.baseline_period}")
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
        indices: dict = {}
        reference_label, _ = self._reference(data_containers)
        reference: tuple = None

        for dc in data_containers:
            index = self.compute(dc)
            indices[dc.model_label] = (index.time.values, index.values)
            print(f"Computed index for {dc.model_label}.")

            if dc.model_label == reference_label:
                ref_index = self.compute(dc, baseline=True)
                reference = (ref_index.time.values, ref_index.values)
                print(f"Computed baseline index for {reference_label}.")

        colors = self._colors(data_containers)
        linestyles = {dc.model_label: "-" for dc in data_containers}

        if reference_label and reference is not None:
            b_label = (
                f"{reference_label} ({self.baseline_period[0]}-{self.baseline_period[1]})"
                if self.baseline_period
                else reference_label
            )
            indices[b_label] = reference
            colors[b_label] = colors[reference_label]
            linestyles[b_label] = "--"

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
            xlabel="Time",
            ylabel="Index Value",
            xticks=xticks,
            fname=f"{self.method}_monsoon_index.png",
        )


class SouthernOscillationIndex(Timeseries):
    """Southern Oscillation Index (Tahiti minus Darwin standardised pressure)."""

    def __init__(
        self,
        detrend: bool = False,
        compute_anomalies: bool = True,
        baseline_period: tuple = None,
        spectrum: bool = False,
        plotter_kwargs: dict = None,
    ) -> None:
        super().__init__(
            detrend=detrend,
            compute_anomalies=compute_anomalies,
            baseline_period=baseline_period,
            plotter_kwargs=plotter_kwargs,
        )
        self.spectrum_fn = spectral.welch_psd if spectrum else None
        self._freq_plotter = FrequencyPlotter(**self.plotter_kwargs) if spectrum else None

    def compute(self, data_container) -> xr.DataArray:
        data = data_container.get_variable_data(name="psl", frequency="monthly", pressure_level=None)
        if "stat" in data.dims:
            data = data.sel(stat="mean", drop=True)
        soi = compute_soi(data, base_period=self.baseline_period, detrend=self.detrend)
        import itertools
        time = [f"{y}-{m:02d}" for y, m in itertools.product(soi.year.values, soi.month.values)]
        soi = soi.stack(time=("year", "month")).reset_index("time")
        soi["time"] = time
        return soi

    def evaluate(self, data_containers) -> None:
        soi_indices: dict = {}
        psd_spectra: dict = {}
        colors = self._colors(data_containers)
        for dc in data_containers:
            soi = self.compute(dc)
            if self.spectrum_fn:
                psd_spectra[dc.model_label] = self.spectrum_fn(soi.values)
            soi_indices[dc.model_label] = (soi.time.values, soi.values)

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
                fname=f"SOI_{label}.png",
            )

        if self.spectrum_fn and psd_spectra:
            self._freq_plotter.plot_psd(
                model_spectra=psd_spectra,
                colors=colors,
                title="SOI Power Spectrum",
                xlabel="Frequency (1/months)",
                ylabel="Power Spectral Density",
                fname="SOI_Spectrum.png",
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

    def __init__(self, n_modes: int = 1) -> None:
        self.n_modes = n_modes

    def compute_eofs(
        self,
        data: xr.DataArray,
        lat_slicer=None,
        lon_slicer=None,
    ) -> None:
        assert isinstance(data, xr.DataArray), "data must be an xr.DataArray."
        self._data = data
        self._anomaly = compute_anomaly(
            data=self._data,
            baseline_period=None,
            mean_groups=["time.year"],
            baseline_mean_groups=None,
            standardize=True,
        ).rename({"year": "time"}).reset_index("time", drop=True)
        self._anomaly = self._anomaly.transpose("time", "lat", "lon")
        self._anomaly.values = self._anomaly.values * compute_latitude_weights(
            self._data.lat.values, self._data.lon.values
        )
        self._anomaly = self._anomaly.sel(lat=lat_slicer, lon=lon_slicer)

        self.time = self._data.time.values
        eof_modes, A, Lh, E = compute_eof(self._anomaly, n_modes=self.n_modes)
        self.eof_modes = eof_modes
        self.norm_coeff = Lh * Lh / (self.eof_modes.shape[0] - 1)
        # Fraction of total variance explained by the leading mode.
        x_flat = self._anomaly.values.reshape(self._anomaly.values.shape[0], -1)
        total_var = Lh[0].sum() 
        self.explained_variance_ratio = (Lh[0] / total_var) if total_var > 0 else 0.0


    def eigenvals_timeseries(self) -> np.ndarray:
        return self.eof_modes / np.std(self.eof_modes, axis=0, keepdims=True)

    def project_eofs(
        self,
        anomaly: xr.DataArray,
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
        full_lat = anomaly.lat.values
        full_lon = anomaly.lon.values

        # Restrict to the desired band for projection and p-value stats.
        _lat_slicer = lat_slicer if lat_slicer is not None else slice(None)
        _lon_slicer = lon_slicer if lon_slicer is not None else slice(None)
        anomaly_band = anomaly.sel(lat=_lat_slicer, lon=_lon_slicer)

        T = anomaly_band.values.shape[0]
        n_lat_band = len(anomaly_band.lat)
        n_lon_band = len(anomaly_band.lon)
        print(f"Projecting onto band: {n_lat_band} lat x {n_lon_band} lon")

        y = eof_modes[:, 0]                                        # (T,)

        # Flatten spatial dims -> (T, lat_band * lon_band).
        anomaly_flat = anomaly_band.values.reshape(T, -1)          # (T, N)

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

        # Build a DataArray for the band, then reindex onto the full grid
        # (NaN outside the band).
        proj_band = xr.DataArray(
            proj_masked,
            coords={"lat": anomaly_band.lat.values, "lon": anomaly_band.lon.values},
            dims=["lat", "lon"],
        )
        proj_full = proj_band.reindex(lat=full_lat, lon=full_lon, fill_value=np.nan)
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
    ) -> None:
        
        self.name = "Annular Mode"
        super().__init__(variables=[], frequency=frequency, plotter_kwargs=plotter_kwargs)

        self.method = method
        self.time = time
        self.baseline_period = baseline_period
        self.latitudes: np.ndarray = None
        self.longitudes: np.ndarray = None

        # Latitude / longitude slicers
        self.latitude_bands = latitude_bands
        self.latitude_slicer = (
            slice(max(latitude_bands), min(latitude_bands))
            if latitude_bands is not None
            else slice(None)
        )
        self.longitude_bands = longitude_bands
        self.longitude_slicer = (
            slice(min(longitude_bands), max(longitude_bands))
            if longitude_bands is not None
            else slice(None)
        )

        self.eof_op = EOF(n_modes=1)
        self.timeseries_plotter = TimeseriesPlotter(**self.plotter_kwargs)
        self.taylor_plotter = TaylorDiagramPlotter(
            output_path=self.plotter_kwargs.get("output_path", "."),
        )

    # -- Helpers -------------------------------------------------------------

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
        if isinstance(time, list):
            months = [self._month_to_num(m) for m in time]
            return data.sel(time=data.time.dt.month.isin(months))
        if time in ("DJF", "MAM", "JJA", "SON"):
            return data.sel(time=data.time.dt.season == time)
        return data

    def spatial_plot(self, projection, model_label: str, info_vals=None) -> None:
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

        self.eof_op.compute_eofs(data, lat_slicer=self.latitude_slicer, lon_slicer=self.longitude_slicer)
        eigenvals = self.eof_op.eigenvals_timeseries()

        data = compute_anomaly(
            data=data,
            baseline_period=self.baseline_period,
            mean_groups=["time.year"],
            baseline_mean_groups=None,
            standardize=True,
        ).rename({"year": "time"}).reset_index("time", drop=True)
    
        self.latitudes = data.lat.values
        self.longitudes = data.lon.values
        projection = self.eof_op.project_eofs(
            data, eigenvals,
            lat_slicer=self.latitude_slicer,
            lon_slicer=self.longitude_slicer,
        )
        return data.time.values, projection, eigenvals[:, 0], self.eof_op.explained_variance_ratio

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
        print(f"Regressing over {valid.sum()} grid points within the lat/lon band.")
        lr = stats.linregress(model_flat[valid], ref_flat[valid])
        return lr.rvalue, lr.pvalue, lr.stderr

    def evaluate(self, data_containers) -> None:
        eof_modes: dict = {}
        projections: dict = {}
        explained_variances: dict = {}
        for dc in data_containers:
            time, projection, eigenvals, expl_var = self.compute(dc)
            eof_modes[dc.model_label] = (time, eigenvals)
            projections[dc.model_label] = projection
            explained_variances[dc.model_label] = expl_var

        colors = self._colors(data_containers)
        ref_name, _ = self._reference(data_containers)
        reference_projection = projections.get(ref_name)

        # Collect Taylor-diagram statistics for all non-reference models.
        taylor_stats: dict = {}   # {label: (r, normalised_std)}

        for label, projection in projections.items():
            # Explained variance is shown on every projection plot.
            info_vals = {"Expl. var. (%)": explained_variances[label] * 100}
            if label != ref_name and reference_projection is not None:
                r, p, std_err = self.regress_against_reference(projection, reference_projection)
                # Use nan-aware reductions: projection has NaN outside the band.
                proj_vals = np.asarray(projection)
                ref_vals  = np.asarray(reference_projection)
                std_ratio = np.nanstd(proj_vals) / np.nanstd(ref_vals)
                pm = np.nanmean(proj_vals)
                rm = np.nanmean(ref_vals)
                centred_rmse = float(
                    np.sqrt(np.nanmean(((proj_vals - pm) - (ref_vals - rm)) ** 2))
                )
                print({label: {"r": r, "p": p, "std_err": std_err, "std_ratio": std_ratio, "centred_rmse": centred_rmse}})
                taylor_stats[label] = (float(r), float(std_ratio))

            # Replace insignificant (zero) values with NaN for plotting;
            # projection is already an xr.DataArray with NaN outside the band.
            projection = projection.where(projection != 0)
            self.spatial_plot(projection=projection, model_label=label, info_vals=info_vals)

        # Draw Taylor diagram when at least one model can be compared.
        if taylor_stats and ref_name is not None:
            self.taylor_plotter.plot(
                model_stats=taylor_stats,
                colors=colors,
                ref_label=ref_name,
                title=f"{self.name}",
                fname=f"{self.method}_taylor_diagram.png",
            )

        self.timeseries_plotter.plot(
            model_data=eof_modes,
            colors=colors,
            title="",
            variable_name="",
            xlabel="Time",
            ylabel="EOF Mode 1 Amplitude",
            xticks=None,
            fname=f"{self.method}_eof_timeseries.png",
        )


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
        plotter_kwargs: dict = None,
        baseline_period: tuple = ("1981-01-01", "2010-12-31"),
        frequency: str = "monthly",
        latitude_bands: tuple = (20, 90),
        longitude_bands: tuple = (-180, 180),
    ) -> None:
        super().__init__(
            method=method,
            baseline_period=baseline_period,
            plotter_kwargs=plotter_kwargs,
            frequency=frequency,
            latitude_bands=latitude_bands,
            longitude_bands=longitude_bands,
        )
        self.name = "NAM"
        self.var = var
        self._cartopy = CartopyProjectionPlotter(
            output_path=self.plotter_kwargs.get("output_path", ".")
        )

    def spatial_plot(self, projection, model_label: str, info_vals=None) -> None:
        self._cartopy.azimuthal_equidistant(
            data=projection,
            central_latitude=90,
            extent=[-180, 180, 20, 90],
            fname=f"{self.method}_{model_label}.png",
            info_vals=info_vals,
        )

    def compute(self, data_container):
        if self.var == "psl":
            data = data_container.get_variable_data(name="psl", frequency=self.frequency, pressure_level=None)
        elif self.var == "zg":
            data = data_container.get_variable_data(name="zg", frequency=self.frequency, pressure_level=100000)
        else:
            raise ValueError(f"Unknown variable for Northern Annular Mode: {self.var}")
        if "stat" in data.dims:
            data = data.sel(stat="mean", drop=True)

        
        return super().compute(data)

class SouthernAnnularMode(AnnularModes):
    def __init__(
        self,
        method: str = "EOF",
        plotter_kwargs: dict = None,
        baseline_period: tuple = ("1981-01-01", "2010-12-31"),
        frequency: str = "monthly",
        latitude_bands: tuple = (-90, -20),
        longitude_bands: tuple = (-180, 180),
    ) -> None:
        super().__init__(
            method=method,
            baseline_period=baseline_period,
            plotter_kwargs=plotter_kwargs,
            frequency=frequency,
            latitude_bands=latitude_bands,
            longitude_bands=longitude_bands,
        )
        self._cartopy = CartopyProjectionPlotter(
            output_path=self.plotter_kwargs.get("output_path", ".")
        )

    def spatial_plot(self, projection, model_label: str, info_vals=None) -> None:
        self._cartopy.azimuthal_equidistant(
            data=projection,
            central_latitude=-90,
            extent=[-180, 180, -90, -20],
            fname=f"{self.method}_{model_label}.png",
            info_vals=info_vals,
        )
    
    def compute(self, data_container):
        psl = data_container.get_variable_data(name="psl", frequency=self.frequency, pressure_level=None)
        if "stat" in psl.dims:
            psl = psl.sel(stat="mean", drop=True)
        return super().compute(psl)
    



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
    ) -> None:
        super().__init__(
            method=method,
            baseline_period=baseline_period,
            plotter_kwargs=plotter_kwargs,
            frequency=frequency,
            latitude_bands=latitude_bands,
            longitude_bands=longitude_bands,
        )
        self.name = "NAOI"
        self._cartopy = CartopyProjectionPlotter(
            output_path=self.plotter_kwargs.get("output_path", ".")
        )

    def spatial_plot(self, projection, model_label: str, info_vals=None) -> None:
        self._cartopy.azimuthal_equidistant(
            data=projection,
            central_latitude=90,
            extent=[-180, 180, 20, 90],
            fname=f"{self.method}_{model_label}.png",
            info_vals=info_vals,
            wedge="noa",
        )

    def compute(self, data_container):
        psl = data_container.get_variable_data(name="psl", frequency=self.frequency, pressure_level=None)
        if "stat" in psl.dims:
            psl = psl.sel(stat="mean", drop=True)
        
        if "mpi" in data_container.model_label.lower():
            print(psl)
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
    ) -> None:
        super().__init__(variables=variables, frequency=frequency, plotter_kwargs=plotter_kwargs)
        self.time_instances = time_instances
        self._freq_plotter = FrequencyPlotter(**self.plotter_kwargs)

    def compute_spectrum(self, data: xr.DataArray) -> dict:
        """Return ``{instance: radial_spectrum}`` averaged over all matching time steps."""
        radial_spectra = {}
        for instance in self.time_instances:
            print(f"Computing radial spectrum for instance: {instance}")
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
                print(f"  No data found for instance '{instance}', skipping.")
                continue

            radial_spectra[instance] = (
                sum(spectral.compute_radial_spectrum(x) for x in _data.values)
                / _data.shape[0]
            )
        return radial_spectra

    def compute(self, data_containers, variable_name: str, lvl) -> dict:
        """Return ``{model_label: {instance: spectrum}}`` for all containers."""
        radial_spectra = {}
        for dc in data_containers:
            data = dc.get_variable_data(
                name=variable_name, frequency=self.frequency, pressure_level=lvl
            )
            if "stat" in data.dims:
                data = data.sel(stat="mean", drop=True)
            radial_spectra[dc.model_label] = self.compute_spectrum(data)
        return radial_spectra

    def evaluate(self, data_containers) -> None:
        colors = self._colors(data_containers)

        for var in self.variables:
            name, pressure_level = var
            self._log(name, pressure_level)
            radial_spectra = self.compute(data_containers, name, pressure_level)

            short_name = name if pressure_level is None else f"{name} ({pressure_level}hPa)"

            self._freq_plotter.plot_radial(
                model_spectra=radial_spectra,
                colors=colors,
                title=f"Radial power spectrum – {short_name}",
                ylabel="Power Spectral Density",
                fname=f"radial_spectrum_{short_name}.png",
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
        plotter_kwargs: dict = None,
    ) -> None:
        super().__init__(variables=variables, plotter_kwargs=plotter_kwargs)
        self.time = time
        self.frequency = frequency

    def compute(self, data: xr.DataArray) -> np.ndarray:
        raise NotImplementedError

    def visualize(self, distributions: dict, variable_name: str = "Variable") -> None:
        raise NotImplementedError

    def evaluate(self, data_containers) -> None:
        for var in self.variables:
            name, pressure_level = var
            distributions: dict = {}
            for dc in data_containers:
                data = dc.get_variable_data(name=name, pressure_level=pressure_level, frequency=self.frequency)
                if "stat" in data.dims:
                    data = data.sel(stat="mean", drop=True)
                if self.time is not None:
                    data = data.sel(time=slice(
                        np.datetime64(self.time[0], "ns"),
                        np.datetime64(self.time[1], "ns"),
                    ))
                distributions[dc.model_label] = (self.compute(data), dc.model_color)

            self.visualize(distributions, variable_name=(name, pressure_level))


class Histogram(Distribution):
    """Value-distribution histogram with per-model statistics annotation."""

    def __init__(
        self,
        variables: list,
        time=None,
        frequency: str = "monthly",
        plotter_kwargs: dict = None,
    ) -> None:
        super().__init__(variables, time=time, frequency=frequency, plotter_kwargs=plotter_kwargs)
        self.figsize = self.plotter_kwargs.get("figsize", (10, 10))
        self.dpi = self.plotter_kwargs.get("dpi", 150)
        self.output_path = self.plotter_kwargs.get("output_path", ".")
        os.makedirs(self.output_path, exist_ok=True)
        self.cmor_units = TimeseriesPlotter(**self.plotter_kwargs).cmor_units

    def compute(self, data: xr.DataArray) -> np.ndarray:
        return data.values.flatten()

    def visualize(self, data: dict, variable_name: str = "Variable") -> None:
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        for label, (d, color) in data.items():
            #mean = np.mean(d)
            #std = np.std(d)
            #skewness = stats.skew(d)
            #kurtosis = stats.kurtosis(d)

            #legend_label = (
            #    f"{label} (mean={mean:.2f}, std={std:.2f}, "
            #    f"skew={skewness:.2f}, kurtosis={kurtosis:.2f})"
            #)
            legend_label = f"{label}"
            ax.hist(
                d, alpha=0.5, label=legend_label, color=color,
                bins=100, histtype="step", density=True, log=True,
            )
        if len(data) > 2:
            plt.legend(bbox_to_anchor=(0.5, -0.1), loc="center", fontsize=8, ncol=2)
        else:
            plt.legend(bbox_to_anchor=(0.5, -0.1), loc="center", fontsize=8)

        plt.grid(True, which="both", linestyle="-.", linewidth=0.5)
        plt.xlabel(self.cmor_units.get(variable_name[0]))
        plt.ylabel("Density")
        if variable_name[1] is None:
            variable_name = variable_name[0]
        else:
            variable_name = f"{variable_name[0]} ({variable_name[1]}hPa)"
        plt.title(f"Histogram of {variable_name}")
        os.makedirs(os.path.join(self.output_path, f"{self.time[0]}_{self.time[1]}"), exist_ok=True)
        plt.savefig(os.path.join(self.output_path, f"{self.time[0]}_{self.time[1]}", f"histogram_{variable_name}.png"))
        plt.close()


class AnimatedHistogram(Distribution):
    """
    Animated histogram showing how the value distribution evolves over time.

    Frames are computed cumulatively from ``time[0]`` at the following
    milestones:

    * 1 week, 2 weeks, 4 weeks, 3 months, 6 months
    * yearly snapshots (Jan 1 of each year) up to ``time[1]``
    * the exact end date ``time[1]``

    Each frame displays all models' histograms for data in the window
    ``[time[0], milestone]``.  The animation is saved as an MP4 (falling
    back to an animated GIF when FFmpeg is unavailable).

    Parameters
    ----------
    variables : list of (name, pressure_level) tuples
    time : list[str]  ``[start_date, end_date]`` in any format understood by
        :class:`pandas.Timestamp`.
    frequency : str  Temporal resolution of the input data (default ``"daily"``).
    plotter_kwargs : dict
        Forwarded to the parent and used for ``figsize``, ``dpi``,
        ``output_path``, and ``fps`` (frames per second, default 2).
    """

    def __init__(
        self,
        variables: list,
        time=None,
        frequency: str = "daily",
        plotter_kwargs: dict = None,
    ) -> None:
        super().__init__(variables, time=time, frequency=frequency, plotter_kwargs=plotter_kwargs)
        self.figsize = self.plotter_kwargs.get("figsize", (10, 6))
        self.dpi = self.plotter_kwargs.get("dpi", 150)
        self.output_path = self.plotter_kwargs.get("output_path", ".")
        self.fps = self.plotter_kwargs.get("fps", 2)
        os.makedirs(self.output_path, exist_ok=True)
        self.cmor_units = TimeseriesPlotter(**self.plotter_kwargs).cmor_units

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_milestones(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> list:
        """Return ordered cumulative time milestones between *start* and *end*."""
        milestones = []

        # fixed early milestones
        for offset in [
            pd.DateOffset(weeks=1),
            pd.DateOffset(weeks=2),
            pd.DateOffset(weeks=4),
            pd.DateOffset(months=3),
            pd.DateOffset(months=6),
        ]:
            t = start + offset
            if t <= end:
                milestones.append(t)

        # yearly Jan-1 snapshots
        year = start.year + 1
        while True:
            t = pd.Timestamp(year, 1, 1)
            if t > end:
                break
            milestones.append(t)
            year += 1

        # always close with the exact end date
        if not milestones or milestones[-1] < end:
            milestones.append(end)

        return milestones

    @staticmethod
    def _milestone_label(start: pd.Timestamp, milestone: pd.Timestamp) -> str:
        """Human-readable label for a milestone relative to *start*."""
        days = (milestone - start).days
        if days <= 7:
            return "1 week"
        if days <= 14:
            return "2 weeks"
        if days <= 28:
            return "4 weeks"
        if days <= 95:
            return "3 months"
        if days <= 185:
            return "6 months"
        return f"up to {milestone.strftime('%Y-%m-%d')}"

    # ------------------------------------------------------------------
    # Distribution interface
    # ------------------------------------------------------------------

    def compute(self, data: xr.DataArray) -> np.ndarray:
        return data.values.flatten()

    def visualize(self, data: dict, variable_name=None) -> None:
        """Not used directly; animation is driven by :py:meth:`evaluate`."""

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(self, data_containers) -> None:
        from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

        for var in self.variables:
            name, pressure_level = var

            start = pd.Timestamp(self.time[0])
            end = pd.Timestamp(self.time[1])
            milestones = self._build_milestones(start, end)

            # ── pre-load data for every model ────────────────────────
            model_data: dict = {}
            for dc in data_containers:
                da = dc.get_variable_data(
                    name=name, pressure_level=pressure_level, frequency=self.frequency
                )
                if "stat" in da.dims:
                    da = da.sel(stat="mean", drop=True)
                da = da.sel(
                    time=slice(
                        np.datetime64(start, "ns"),
                        np.datetime64(end, "ns"),
                    )
                )
                model_data[dc.model_label] = (da, dc.model_color)

            unit_label = self.cmor_units.get(name, name)
            if pressure_level is not None:
                var_title = f"{name} ({pressure_level} hPa)"
                var_fname = f"{name}_{pressure_level}hPa"
            else:
                var_title = name
                var_fname = name

            # ── figure & update function ─────────────────────────────
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            fig.subplots_adjust(bottom=0.25)

            def _draw_frame(i: int, _start=start, _milestones=milestones,
                            _model_data=model_data, _var_title=var_title,
                            _unit_label=unit_label) -> None:
                ax.cla()
                milestone = _milestones[i]
                t_start_ns = np.datetime64(_start, "ns")
                t_end_ns = np.datetime64(milestone, "ns")

                for label, (da, color) in _model_data.items():
                    subset = (
                        da.sel(time=slice(t_start_ns, t_end_ns))
                        .values.flatten()
                    )
                    finite = subset[np.isfinite(subset)]
                    if finite.size == 0:
                        continue
                    legend_label = (
                        f"{label}  "
                        f"(mean={np.mean(finite):.2f}, "
                        f"std={np.std(finite):.2f}, "
                        f"skew={stats.skew(finite):.2f}, "
                        f"kurt={stats.kurtosis(finite):.2f})"
                    )
                    ax.hist(
                        finite,
                        bins=100,
                        alpha=0.5,
                        color=color,
                        label=legend_label,
                        histtype="step",
                        density=True,
                        log=True,
                    )

                period = self._milestone_label(_start, milestone)
                ax.set_title(f"Histogram of {_var_title} — {period}")
                ax.set_xlabel(_unit_label)
                ax.set_ylabel("Density")
                ax.legend(
                    bbox_to_anchor=(0.5, -0.3),
                    loc="upper center",
                    fontsize=7,
                    ncol=1,
                )
                ax.grid(True, which="both", linestyle="-.", linewidth=0.5)

            anim = FuncAnimation(
                fig,
                _draw_frame,
                frames=len(milestones),
                interval=int(1000 / self.fps),
                repeat=True,
            )

            # ── save ─────────────────────────────────────────────────
            out_mp4 = os.path.join(
                self.output_path, f"histogram_animated_{var_fname}.mp4"
            )
            out_gif = os.path.join(
                self.output_path, f"histogram_animated_{var_fname}.gif"
            )
            try:
                anim.save(out_mp4, writer=FFMpegWriter(fps=self.fps))
            except Exception:
                anim.save(out_gif, writer=PillowWriter(fps=self.fps))

            plt.close(fig)


# ============================================================================
# Tropical Cyclone frequency
# ============================================================================


class TropicalCycloneFrequency(BaseMetric):
    """
    Tropical Cyclone (TC) frequency diagnostics for the western North Pacific.

    Detection follows three criteria applied to daily-resolution data:

    1. **SLP minimum** – a local SLP minimum in the WNP domain
       (100 °E–160 °E, 5 °N–35 °N) falls below at least one of the four
       proxy-percentile thresholds: 1 000, 994, 985, 975 hPa.
    2. **Warm-core** – geopotential thickness between 300 hPa and 700 hPa
       exceeds the seasonally varying climatological value.
    3. **Track continuity** – adjacent centres along the same track are
       separated by ≤ 2 days in time and ≤ 3 ° in distance; tracks with
       fewer than 3 points are rejected.

    The climatological geopotential used in criterion 2 is computed from
    the *reference* data container (``dc.is_reference == True``).  If no
    reference is present the first container is used as climatology.

    Parameters
    ----------
    time : list of str, optional
        ``[start_date, end_date]`` to restrict the analysis window.
    frequency : str
        Temporal resolution of the input data.  ``"daily"`` is strongly
        recommended.
    lat_bins : array-like, optional
        Latitude bin edges for the frequency-density map
        (default: 2 ° grid over the WNP domain).
    lon_bins : array-like, optional
        Longitude bin edges (default: 2 ° grid over the WNP domain).
    time_step_hours : float
        Hours between successive data snapshots (default 24).
    plotter_kwargs : dict, optional
        Forwarded to plotters; recognised keys include ``output_path``,
        ``figsize``, ``dpi``.
    """

    def __init__(
        self,
        time: list = None,
        frequency: str = "daily",
        lat_bins=None,
        lon_bins=None,
        time_step_hours: float = 24.0,
        plotter_kwargs: dict = None,
    ) -> None:
        # No per-variable loop – TC uses fixed physical fields (psl, zg).
        super().__init__(variables=[], frequency=frequency, plotter_kwargs=plotter_kwargs)

        self.time = time
        self.time_step_hours = time_step_hours
        self.figsize = self.plotter_kwargs.get("figsize", (12, 5))
        self.dpi = self.plotter_kwargs.get("dpi", 150)
        self.output_path = self.plotter_kwargs.get("output_path", ".")
        os.makedirs(self.output_path, exist_ok=True)

        # Default 2 ° bins covering the WNP domain
        self.lat_bins = (
            np.arange(WNP_LAT[0], WNP_LAT[1] + 2.0, 2.0)
            if lat_bins is None
            else np.asarray(lat_bins)
        )
        self.lon_bins = (
            np.arange(WNP_LON[0], WNP_LON[1] + 2.0, 2.0)
            if lon_bins is None
            else np.asarray(lon_bins)
        )

    # ── Data retrieval helpers ────────────────────────────────────────────────

    def _get_slp(self, dc) -> xr.DataArray:
        da = dc.get_variable_data(name="psl", pressure_level=None, frequency=self.frequency)
        if "stat" in da.dims:
            da = da.sel(stat="mean", drop=True)
        return self._slice_time(da)

    def _get_zg(self, dc) -> xr.DataArray:
        """Return geopotential stacked over level dimension (time, level, lat, lon)."""
        levels = [300, 700]
        arrays = []
        for lev in levels:
            da = dc.get_variable_data(name="zg", pressure_level=lev, frequency=self.frequency)
            if "stat" in da.dims:
                da = da.sel(stat="mean", drop=True)
            da = self._slice_time(da)
            # Ensure a level coordinate exists
            if "level" not in da.dims:
                da = da.expand_dims({"level": [lev]})
            arrays.append(da)
        return xr.concat(arrays, dim="level")

    def _slice_time(self, da: xr.DataArray) -> xr.DataArray:
        if self.time is None:
            return da
        return da.sel(
            time=slice(
                np.datetime64(self.time[0], "ns"),
                np.datetime64(self.time[1], "ns"),
            )
        )

    # ── Core computation ──────────────────────────────────────────────────────

    def compute(
        self, dc, clim_z_da: xr.DataArray
    ) -> tuple[list[list[dict]], np.ndarray, dict, dict]:
        """Run TC detection and tracking for one data container.

        Parameters
        ----------
        dc :
            A data container providing ``get_variable_data``.
        clim_z_da :
            Climatological day-of-year geopotential (output of
            :func:`~metrics.functional.extremes.compute_clim_z_dayofyear`).

        Returns
        -------
        tracks :
            List of TC trajectories (list of dicts).
        freq_map :
            2-D passage-frequency array.
        count_per_year :
            ``{year: count}`` mapping.
        count_by_intensity :
            ``{intensity_category: count}`` mapping.
        """
        slp_da = self._get_slp(dc)
        zg_da = self._get_zg(dc)

        candidates = detect_tc_candidates(slp_da, zg_da, clim_z_da)
        tracks = build_tc_tracks(candidates, time_step_hours=self.time_step_hours)
        freq_map = compute_tc_frequency(tracks, self.lat_bins, self.lon_bins)
        count_per_year = compute_tc_count_per_year(tracks)
        count_by_intensity = compute_tc_count_by_intensity(tracks)
        return tracks, freq_map, count_per_year, count_by_intensity

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _plot_frequency_map(
        self,
        freq_map: np.ndarray,
        model_label: str,
    ) -> None:
        """Save a filled-contour TC-frequency map for the WNP domain."""
        lat_centres = 0.5 * (self.lat_bins[:-1] + self.lat_bins[1:])
        lon_centres = 0.5 * (self.lon_bins[:-1] + self.lon_bins[1:])

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        cf = ax.contourf(
            lon_centres,
            lat_centres,
            freq_map,
            levels=10,
            cmap="YlOrRd",
        )
        ax.contour(
            lon_centres,
            lat_centres,
            freq_map,
            levels=10,
            colors="k",
            linewidths=0.4,
            alpha=0.5,
        )
        plt.colorbar(cf, ax=ax, label="TC passage count")
        ax.set_xlabel("Longitude [°E]")
        ax.set_ylabel("Latitude [°N]")
        ax.set_title(f"TC passage frequency – {model_label}")
        ax.grid(True, linestyle="-.", linewidth=0.4, alpha=0.6)
        fpath = os.path.join(self.output_path, f"tc_frequency_map_{model_label}.png")
        plt.savefig(fpath, bbox_inches="tight")
        plt.close(fig)

    def _plot_annual_count(
        self,
        counts_per_model: dict[str, tuple[dict, str]],
    ) -> None:
        """Save a bar-chart of annual TC counts for all models."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        all_years = sorted(
            {yr for label, (cnt, _) in counts_per_model.items() for yr in cnt}
        )
        n_models = len(counts_per_model)
        width = 0.8 / max(n_models, 1)

        for k, (label, (cnt, color)) in enumerate(counts_per_model.items()):
            offsets = np.arange(len(all_years)) + (k - n_models / 2.0 + 0.5) * width
            values = [cnt.get(yr, 0) for yr in all_years]
            ax.bar(offsets, values, width=width, label=label, color=color, alpha=0.8)

        ax.set_xticks(range(len(all_years)))
        ax.set_xticklabels([str(y) for y in all_years], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Year")
        ax.set_ylabel("TC count")
        ax.set_title("Annual TC count – western North Pacific")
        ax.legend()
        ax.grid(True, axis="y", linestyle="-.", linewidth=0.4, alpha=0.6)
        fpath = os.path.join(self.output_path, "tc_annual_count.png")
        plt.savefig(fpath, bbox_inches="tight")
        plt.close(fig)

    def _plot_intensity_distribution(
        self,
        counts_per_model: dict[str, tuple[dict, str]],
    ) -> None:
        """Save grouped bar chart of TC counts by intensity category."""
        n_cat = len(TC_SLP_THRESHOLDS_HPA)
        n_models = len(counts_per_model)
        x = np.arange(n_cat)
        width = 0.8 / max(n_models, 1)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        for k, (label, (cnt, color)) in enumerate(counts_per_model.items()):
            offsets = x + (k - n_models / 2.0 + 0.5) * width
            values = [cnt.get(i + 1, 0) for i in range(n_cat)]
            ax.bar(offsets, values, width=width, label=label, color=color, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(TC_INTENSITY_LABELS, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("TC track count")
        ax.set_title("TC count by intensity – western North Pacific")
        ax.legend()
        ax.grid(True, axis="y", linestyle="-.", linewidth=0.4, alpha=0.6)
        fpath = os.path.join(self.output_path, "tc_intensity_distribution.png")
        plt.savefig(fpath, bbox_inches="tight")
        plt.close(fig)

    def _plot_tracks(
        self,
        tracks: list[list[dict]],
        model_label: str,
        color: str,
    ) -> None:
        """Save a simple lat/lon track plot for one model."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_xlim(*WNP_LON)
        ax.set_ylim(*WNP_LAT)

        for track in tracks:
            lats = [pt["lat"] for pt in track]
            lons = [pt["lon"] for pt in track]
            ax.plot(lons, lats, color=color, linewidth=0.6, alpha=0.5)
            # mark genesis point
            ax.scatter(lons[0], lats[0], color=color, s=6, zorder=3)

        ax.set_xlabel("Longitude [°E]")
        ax.set_ylabel("Latitude [°N]")
        ax.set_title(f"TC tracks – {model_label} (n={len(tracks)})")
        ax.grid(True, linestyle="-.", linewidth=0.4, alpha=0.6)
        fpath = os.path.join(self.output_path, f"tc_tracks_{model_label}.png")
        plt.savefig(fpath, bbox_inches="tight")
        plt.close(fig)

    # ── Main entry point ──────────────────────────────────────────────────────

    def evaluate(self, data_containers) -> None:
        """Run TC detection, tracking, and visualisation for all data containers.

        Steps
        -----
        1.  Compute climatological geopotential from the reference container
            (or the first container when no reference is marked).
        2.  For each container: detect candidates, build tracks, compute
            frequency map and count statistics.
        3.  Produce four output figures per model (frequency map, track plot)
            plus two multi-model comparison figures (annual count bar chart,
            intensity distribution bar chart).
        """
        # --- climatology ---------------------------------------------------
        ref_label, ref_dc = self._reference(data_containers)
        clim_dc = ref_dc if ref_dc is not None else data_containers[0]
        zg_clim_raw = self._get_zg(clim_dc)
        clim_z_da = compute_clim_z_dayofyear(zg_clim_raw)
        print(
            f"Using '{clim_dc.model_label}' as geopotential climatology source."
        )

        # --- per-model computation -----------------------------------------
        annual_counts: dict[str, tuple[dict, str]] = {}
        intensity_counts: dict[str, tuple[dict, str]] = {}

        for dc in data_containers:
            print(f"--> TropicalCycloneFrequency: processing '{dc.model_label}'")
            tracks, freq_map, count_yr, count_int = self.compute(dc, clim_z_da)

            print(
                f"    Found {len(tracks)} TC tracks  "
                f"| mean annual count: "
                f"{np.mean(list(count_yr.values())):.1f}" if count_yr else "    No tracks found."
            )

            annual_counts[dc.model_label] = (count_yr, dc.model_color)
            intensity_counts[dc.model_label] = (count_int, dc.model_color)

            self._plot_frequency_map(freq_map, dc.model_label)
            self._plot_tracks(tracks, dc.model_label, dc.model_color)

        # --- multi-model comparison figures --------------------------------
        self._plot_annual_count(annual_counts)
        self._plot_intensity_distribution(intensity_counts)
