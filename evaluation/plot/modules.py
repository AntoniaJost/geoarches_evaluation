import os
from plot.functional import timeseries, spatial, spectra, stats
from plot.projections import CartopyProjectionPlotter  # re-exported for convenience
from typing import List, Union
import xarray as xr
import numpy as np
from matplotlib.colors import CenteredNorm
import matplotlib.pyplot as plt

class EarthPlotter:

    fontdict = {
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "ytick.labelsize": 12,
            "xtick.labelsize": 12,
            "legend.fontsize": 10,
    }
    
    linestyles = ["-", "--", "-.", ":"]

    markerstyles = ["o", "s", "^", "D", "v", "x", "*"]

    units = {
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

    cmor_units = {
            "tos": "[K]",
            "tas": "[K]",
            "ta": "[K]",
            "msl": "[Pa]",
            "hus": "[kg/kg]",
            "zg": "[m^2/s^2]",
            "ua": "[m/s]",
            "va": "[m/s]",
            "uas": "[m/s]",
            "vas": "[m/s]",
            "siconc": "[fraction]",
            "wap": "[m/s]",
    }

    def __init__(
            self, dpi=150, fontdict=None, output_path=".", 
            figsize=(10, 6), cartopy_projection=None
        ):


        self.dpi = dpi
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.figsize = figsize
        if fontdict is not None:
            self.fontdict = fontdict

        self.cartopy_projection = cartopy_projection


class SpatialPlotter(EarthPlotter):
    def __init__(
            self, xdim, ydim, dpi=150, fontdict=None, output_path=".", 
            figsize=(10, 6), cmap=None, cartopy_projection=None
        ):

        super().__init__(
            dpi=dpi, fontdict=fontdict, output_path=output_path, 
            figsize=figsize, cartopy_projection=cartopy_projection)
        
        # Dimensions to visualize
        self.xdim = xdim
        self.ydim = ydim

        self.cmap = cmap  # Colormap for spatial plots, e.g., 'coolwarm' or 'bwr'

    def contourf(
            self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
            output_path=".", model_label="", variable_name="", 
            add_contourlines=False, **kwargs):
        
        spatial.contourf(
            x=x,
            y=y,
            z=z,
            output_path=os.path.join(output_path, f"{model_label}_{variable_name}.png"),
            fontdict=self.fontdict,
            cartopy_projection=self.cartopy_projection,
            cmap=self.cmap,
            add_contourlines=add_contourlines,
            **kwargs
        )

    def imshow(self, x: np.ndarray, variable_name, model_label, info: dict,
               output_path: str, title: str = "", cbar_label: str = "",
               vmin=None, vmax=None):

        if float(x.min()) < 0 and float(x.max()) > 0:
            norm = CenteredNorm(vcenter=0)
            colormap = self.cmap if self.cmap is not None else 'bwr'
        else:
            norm = None
            colormap = self.cmap if self.cmap is not None else 'coolwarm'

        # Use caller-supplied label; fall back to "<var> <unit>" lookup.
        effective_cbar_label = (
            cbar_label if cbar_label
            else f"{variable_name} {self.cmor_units.get(variable_name, '')}"
        )

        infotext = ", ".join([f"{key}: {value}" for key, value in info.items()])
        spatial.imshow(
            x=x,
            output_path=os.path.join(output_path, f"{model_label}_{variable_name}.png"),
            title=title,
            cbar_label=effective_cbar_label,
            cmap=colormap,
            norm=norm,
            fontdict=self.fontdict,
            infotext=infotext,
            vmin=vmin,
            vmax=vmax,
        )

    def plot(self, x: np.ndarray, model_label,
             title, variable_name, info: dict = {}, style="imshow",
             output_path=None, vmin=None, vmax=None, cbar_label=""
            ):

        if output_path is None:
            output_path = self.output_path
        else:
            os.makedirs(output_path, exist_ok=True)

        getattr(self, style)(
            x=x,
            variable_name=variable_name,
            model_label=model_label,
            title=title,
            output_path=output_path,
            info=info,
            vmin=vmin,
            vmax=vmax,
            cbar_label=cbar_label,
        )

    

class FrequencyPlotter(EarthPlotter):
    """
    Plotter for frequency-domain diagnostics.

    Supports two complementary plot types:

    * :py:meth:`plot_radial` – spherical-harmonic radial power spectrum
      (log-log, inverted wavelength x-axis).  Accepts multiple time instances
      per model.
    * :py:meth:`plot_psd` – Welch / arbitrary 1-D power spectral density
      (semi-log y, frequency x-axis).  Used e.g. for SOI spectrum.

    Both methods share the instance-level style defaults (``figsize``,
    ``dpi``, ``linewidth``, ``linestyles``, ``markerstyles``) and accept
    per-model overrides.
    """

    def __init__(
        self, dpi=150, fontdict=None, output_path=".",
        figsize=(10, 6), linewidth=2.0,
    ):
        super().__init__(dpi=dpi, fontdict=fontdict, output_path=output_path, figsize=figsize)
        self.linewidth = linewidth

    def plot_radial(
        self,
        model_spectra: dict,
        colors: dict = {},
        linestyles: dict = {},
        markers: dict = {},
        linewidths: dict = {},
        title: str = "",
        ylabel: str = "Power Spectral Density",
        fname: str = "radial_spectrum.png",
    ) -> None:
        """Save a radial-spectrum comparison figure.

        Parameters
        ----------
        model_spectra :
            ``{model_label: {instance_key: spectrum_array}}`` – one spectrum
            array per model × time-instance.
        colors, linestyles, markers, linewidths :
            Per-model style overrides keyed by model label.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        for model_label, instance_spectra in model_spectra.items():
            color = colors.get(model_label, "blue")
            lw = linewidths.get(model_label, self.linewidth)
            ls_cycle = self.linestyles
            mk_cycle = self.markerstyles
            for j, (instance, spectrum) in enumerate(instance_spectra.items()):
                start_time, end_time = instance
                if start_time == end_time:
                    instance_str = start_time
                else:
                    if start_time.split("-")[0] == end_time.split("-")[0]:  # same year
                        instance_str = f"{start_time.split('-')[0]}" 
                    else:
                        # year - year
                        instance_str = f"{start_time.split('-')[0]}-{end_time.split('-')[0]}"

                spectra.radial_spectrum_to_ax(
                    ax,
                    spectrum=spectrum,
                    label=f"{model_label} ({instance_str})",
                    color=color,
                    linestyle=linestyles.get(model_label, ls_cycle[j % len(ls_cycle)]),
                    marker=markers.get(model_label, mk_cycle[j % len(mk_cycle)]),
                    linewidth=lw,
                )

        ax.set_title(title, fontsize=self.fontdict["axes.titlesize"])
        ax.set_ylabel(ylabel, fontsize=self.fontdict["axes.labelsize"])
        n_curves = sum(len(v) for v in model_spectra.values())
        ncols = 2 if n_curves > 3 else 1
        ax.legend(
            ncol=ncols, fontsize=self.fontdict["legend.fontsize"],
            framealpha=0.8, loc="upper center", bbox_to_anchor=(0.5, -0.15)
        )
        ax.invert_xaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, fname), dpi=self.dpi)
        plt.close(fig)

    def plot_psd(
        self,
        model_spectra: dict,
        colors: dict = {},
        linestyles: dict = {},
        linewidths: dict = {},
        title: str = "",
        xlabel: str = "Frequency",
        ylabel: str = "Power Spectral Density",
        fname: str = "psd.png",
    ) -> None:
        """Save a PSD comparison figure.

        Parameters
        ----------
        model_spectra :
            ``{model_label: (frequencies_array, psd_array)}`` tuples, as
            returned by :func:`~metrics.functional.spectral.welch_psd`.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ls_cycle = self.linestyles

        for j, (model_label, (freq, psd)) in enumerate(model_spectra.items()):
            spectra.psd_to_ax(
                ax,
                frequencies=freq,
                psd=psd,
                label=model_label,
                color=colors.get(model_label, "blue"),
                linestyle=linestyles.get(model_label, ls_cycle[j % len(ls_cycle)]),
                linewidth=linewidths.get(model_label, self.linewidth),
            )

        ax.set_title(title, fontsize=self.fontdict["axes.titlesize"])
        ax.set_xlabel(xlabel, fontsize=self.fontdict["axes.labelsize"])
        ax.set_ylabel(ylabel, fontsize=self.fontdict["axes.labelsize"])
        ncols = 2 if len(model_spectra) > 2 else 1
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.15),
            ncol=ncols, fontsize=self.fontdict["legend.fontsize"],
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, fname), dpi=self.dpi)
        plt.close(fig)


class TimeseriesPlotter(EarthPlotter):
    def __init__(self, dpi=150, fontdict=None, output_path=".", figsize=(12, 6), linewidth=2.0):
        super().__init__(dpi, fontdict, output_path, figsize)
        self.linewidth = linewidth
        

    def get_xticks_from_timeseries(self, time: np.ndarray):
        mult = timeseries.get_xlabel_multiplier(n_xticks=len(time))
        xticks = [(i, time[i]) for i in range(0, len(time), mult)]
        xticks = zip(*xticks)        
        return xticks

    def plot(
            self, model_data: dict, linear_trend: dict = {}, model_stds: dict = {},  
            fill=None, colors: dict = {}, linewidths: dict = {}, linestyles: dict = {},
            markers: dict = {}, title: str = "", variable_name: str = "", 
            xlabel="", ylabel="", xticks: List = None, fname: str = ""
    ):  
        
        fig, axs = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)

        for model_label, (time, data) in model_data.items():
            std = model_stds.get(model_label, None)
            trend, m = linear_trend.get(model_label, (None, None))
            color = colors.get(model_label, 'blue')
            linewidth = linewidths.get(model_label, self.linewidth) 
            marker = markers.get(model_label, None)
            if m is not None:
                label = f"{model_label} (Trend: {m:.2f}°C per decade)"
            else:
                label = model_label

            timeseries.timerseries_to_ax(
                ax=axs,
                x=range(0, len(time)),
                y=data,
                color=color,
                linear_trend=trend,
                std=std,
                linewidth=linewidth,
                fill=fill,
                label=label,
                marker=marker
            )

        axs.set_title(title, fontsize=self.fontdict['axes.titlesize'])
        axs.set_xlabel(
              xlabel,
              fontsize=self.fontdict['axes.labelsize']
        )
        axs.set_ylabel(
              ylabel,
              fontsize=self.fontdict['axes.labelsize']
        )


        if xticks is not None:
            axs.set_xticks(
                *xticks, rotation=45,
                fontsize=self.fontdict['xtick.labelsize']
            )
        else:
            axs.set_xticks(
                *self.get_xticks_from_timeseries(time), rotation=45,
                fontsize=self.fontdict['xtick.labelsize']
            )
        
        # Create legend with smaller font size
        # place legend above the plot
        ncols = 2 if len(model_data) > 2 else 1
        axs.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), 
            ncol=ncols, fontsize=self.fontdict['legend.fontsize'])
        
        axs.grid(True, linewidth=0.25, linestyle='-.', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{self.output_path}/{fname}", dpi=self.dpi)
        plt.close(fig)


class TaylorDiagramPlotter(EarthPlotter):
    """
    Produces normalised Taylor diagrams comparing multiple model patterns
    against a single reference.

    A Taylor diagram encodes three statistics simultaneously:

    * **Radial axis** – normalised standard deviation (σ_model / σ_ref)
    * **Angular axis** – ``arccos(r)``, where *r* is the Pearson correlation
    * **Centred-RMSE** – readable from curved green iso-contours centred on
      the reference point (θ=0, σ_norm=1).

    Usage
    -----
    .. code-block:: python

        plotter = TaylorDiagramPlotter(output_path="./taylor")
        plotter.plot(
            model_stats={
                "ModelA": (0.95, 1.05),   # (r, σ_norm)
                "ModelB": (0.80, 1.30),
            },
            colors={"ModelA": "steelblue", "ModelB": "tomato"},
            ref_label="ERA5",
            fname="taylor_NAM.png",
        )
    """

    def __init__(
        self,
        dpi: int = 150,
        fontdict: dict = None,
        output_path: str = ".",
        figsize: tuple = (8, 8),
        marker_size: int = 10,
        crmse_levels: int = 5,
    ) -> None:
        super().__init__(
            dpi=dpi, fontdict=fontdict, output_path=output_path, figsize=figsize
        )
        self.marker_size = marker_size
        self.crmse_levels = crmse_levels

    def plot(
        self,
        model_stats: dict,
        colors: dict = None,
        markers: dict = None,
        ref_label: str = "Reference",
        title: str = "",
        fname: str = "taylor_diagram.png",
    ) -> None:
        """
        Draw and save a Taylor diagram.

        Parameters
        ----------
        model_stats:
            ``{model_label: (r, normalised_std)}`` mapping.  *r* is the
            Pearson correlation coefficient; *normalised_std* is
            σ_model / σ_reference.
        colors:
            Optional per-model colour overrides.
        markers:
            Optional per-model marker overrides.
        ref_label:
            Label shown next to the reference point (⭐).
        title:
            Figure title.
        fname:
            Output file name relative to ``self.output_path``.
        """
        colors = colors or {}
        markers = markers or {}

        # Choose r_max slightly beyond the largest normalised std so all
        # points fit comfortably inside the diagram.
        all_stds = [s for _, s in model_stats.values()]
        r_max = max(1.65, max(all_stds) * 1.15) if all_stds else 1.65

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.rcParams.update(self.fontdict)

        ax, aux_ax = stats.taylor_diagram_to_ax(
            fig=fig,
            rect=111,
            model_stats=model_stats,
            colors=colors,
            markers=markers,
            marker_size=self.marker_size,
            ref_label=ref_label,
            title=title,
            r_max=r_max,
            crmse_levels=self.crmse_levels,
        )

        # Legend below the diagram
        ncols = 2 if len(model_stats) > 3 else 1
        aux_ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=ncols,
            fontsize=self.fontdict["legend.fontsize"],
            framealpha=0.8,
            
        )

        plt.tight_layout()
        fpath = os.path.join(self.output_path, fname)
        plt.savefig(fpath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Taylor diagram saved to {fpath}")
