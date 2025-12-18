import os
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from sympy import use
from metrics import functional

from metrics.functional import timeseries, frequency_domain, kernel_density_estimation
from plot.timeseries import plot_timeseries, get_xlabel_multiplier
from omegaconf import ListConfig
from plot.spatial import plot_variable

import matplotlib.pyplot as plt

from geoarches.metrics.metric_base import (
    compute_lat_weights,
    compute_lat_weights_weatherbench,
)

import xarray as xr
import numpy as np

import pandas as pd


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
    ):
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
    ):
        super().__init__(
            variables,
            use_lat_weighting,
            latitude,
            figsize,
            dpi,
            base_period,
            output_path,
        )

        self.time = time
        self.x = x
        self.y = y

    def select_time(self, data, time):
        if time == "average_time":
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
        time="all",
        x="latitude",
        y="longitude",
        use_lat_weighting="weatherbench",
        latitude=None,
        figsize=(10, 6),
        dpi=150,
        base_period=None,
        output_path=".",
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
        )
        self.plot_type = plot_type
        self.output_path = output_path + f"/{x}_{y}_plots"
        os.makedirs(self.output_path, exist_ok=True)

    def visualize_on_ax(
        self, fig, ax, data, variable_name, time, model_label, norm=None, infotext=""
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
                title=f"{model_label} - {time}, Variable: {variable_name}",
                ax=None,
                cbar_label=self.units[variable_name]
                if variable_name in self.units
                else "",
                cmap="coolwarm",
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
                levels=10,
            )
            CS2 = ax.contour(CS, levels=CS.levels, colors="k")
            plt.gca().set_aspect("auto")
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
        ax.set_title(
            model_label + " - " + time_title + ", Variable: " + variable_name,
            fontsize=self.fontdict["axes.titlesize"],
        )
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
        if "era5" in [model.label.lower() for model in model_containers]:
            ground_truth_data_index = [
                i
                for i, model in enumerate(model_containers)
                if model.label.lower() == "era5"
            ][0]
            ground_truth_data = model_containers[ground_truth_data_index].data
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

                if model_data.label.lower() != "era5" and ground_truth_data is not None:
                    print(
                        f"Bias correcting model {model_data.label} using ERA5...")
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
                        plot_data = data[variable_name].sel(level=lvl)
                    self.visualize_on_ax(
                        fig, ax, plot_data, variable_name, time, model_label=model_data.label
                    )

                    if diff_data:
                        print(
                            f"Visualizing Bias XY plot for variable: {short_var_name}")
                        fig, ax = self.create_figure()
                        if lvl is None:
                            plot_data = diff_data[variable_name]
                        else:
                            plot_data = diff_data[variable_name].sel(level=lvl)
                        norm = TwoSlopeNorm(vcenter=0)
                        self.visualize_on_ax(
                            fig,
                            ax,
                            plot_data,
                            short_var_name,
                            time,
                            model_label=model_data.label + " Bias",
                            norm=norm,
                            infotext="mean=" +
                            f"{float(plot_data.mean().values):.2f}",
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
                ax.set_title(f"Annual Cycle - {short_var_name} - {lbl}")
                ax.set_xlabel("Time")
            plt.grid()
            plt.legend()
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
                ax0.set_title(f"KDE - {short_var_name}")
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
            ax1.set_title(f"Annual Cycle - {short_var_name}")
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
            ax.set_title(f"{short_var_name}")
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
            ax.legend()
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
                title="Southern Oscillation Index",
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
            ax.set_title("Southern Oscillation Index - Power Spectral Density")
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
