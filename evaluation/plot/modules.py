import colorsys
import logging
import math
import os

from seaborn import colors
from plot.functional import timeseries, spatial, spectra, stats
from plot.projections import CartopyProjectionPlotter  # re-exported for convenience
from typing import List, Union
import xarray as xr
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# A4 single-column text width in inches (used as the default figure width throughout).
A4_WIDTH = 6.7

# Use sans-serif math font throughout all plots.

mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["font.family"] = "DejaVu Sans"  # for non-math text, e.g. axis labels and legends
mpl.rcParams["axes.titlesize"] = 8
mpl.rcParams["axes.labelsize"] = 7
mpl.rcParams["xtick.labelsize"] = 7
mpl.rcParams["ytick.labelsize"] = 7
mpl.rcParams["legend.fontsize"] = 6
logger = logging.getLogger(__name__)

class EarthPlotter:

    fontdict = {
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.labelsize": 7,
            "legend.fontsize": 6,
    }
    
    linestyles = ["-", "--", "-.", ":"]

    markerstyles = ["o", "s", "^", "D", "v", "x", "*"]

    units = {
            "sea_surface_temperature": r"$\mathrm{K}$",
            "2m_temperature":          r"$\mathrm{K}$",
            "mean_sea_level_pressure": r"$\mathrm{Pa}$",
            "specific_humidity":       r"$\mathrm{kg\,kg^{-1}}$",
            "geopotential":            r"$\mathrm{m^{2}\,s^{-2}}$",
            "u_component_of_wind":     r"$\mathrm{m\,s^{-1}}$",
            "v_component_of_wind":     r"$\mathrm{m\,s^{-1}}$",
            "10m_u_component_of_wind": r"$\mathrm{m\,s^{-1}}$",
            "10m_v_component_of_wind": r"$\mathrm{m\,s^{-1}}$",
            "sea_ice_cover":           r"fraction",
            "temperature":             r"$\mathrm{K}$",
            "vertical_velocity":       r"$\mathrm{m\,s^{-1}}$",
    }

    cmor_units = {
            "tos":   r"$\mathrm{K}$",
            "tas":   r"$\mathrm{K}$",
            "ta":    r"$\mathrm{K}$",
            "psl":   r"$\mathrm{Pa}$",
            "msl":   r"$\mathrm{Pa}$",
            "hus":   r"$\mathrm{kg\,kg^{-1}}$",
            "zg":    r"$\mathrm{m^{2}\,s^{-2}}$",
            "ua":    r"$\mathrm{m\,s^{-1}}$",
            "va":    r"$\mathrm{m\,s^{-1}}$",
            "uas":   r"$\mathrm{m\,s^{-1}}$",
            "vas":   r"$\mathrm{m\,s^{-1}}$",
            "siconc": r"fraction",
            "wap":   r"$\mathrm{Pa\,s^{-1}}$",
    }

    def __init__(
            self, dpi=150, fontdict=None, output_path=".", 
            figsize=(A4_WIDTH, 3.5), cartopy_projection=None
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
            figsize=(A4_WIDTH, 3.5), cmap=None, cartopy_projection=None
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
            output_path=os.path.join(output_path, f"{model_label}_{variable_name}.pdf"),
            fontdict=self.fontdict,
            figsize=self.figsize,
            cartopy_projection=self.cartopy_projection,
            cmap=self.cmap,
            add_contourlines=add_contourlines,
            **kwargs
        )

    def map_plot(
            self,
            x: np.ndarray,
            variable_name: str,
            model_label: str,
            output_path: str,
            title: str = "",
            cbar_label: str = "",
            vmin=None,
            vmax=None,
            extent=None,
            **kwargs,
    ):
        """Render a single global map and save it.

        Parameters
        ----------
        variable_name:
            Short variable name; used for the output filename.
        model_label:
            Model identifier; used for the output filename.
        infotext_topleft:
            Label placed in the top-left corner of the map.  Defaults to
            *model_label* when not provided.
        """
        
    

        effective_cbar_label = (
            cbar_label if cbar_label
            else f"{variable_name} {self.cmor_units.get(variable_name, '')}"
        )

        # Discard legacy empty `info={}` dict; pass infotext directly if needed.
        kwargs.pop("info", None)

        # Fix lon coords shifted by roll_coords=False: values were shifted by
        # half the longitude range but labels were left unchanged.
        if "lon" in x.dims:
            n_lon = x.sizes["lon"]
            corrected_lons = np.roll(x.lon.values, -n_lon // 2)
            x = x.assign_coords(lon=corrected_lons).sortby("lon")

        spatial.contourf(
            x=x.lon.values,
            y=x.lat.values,
            z=x.values,
            output_path=os.path.join(output_path, f"{model_label}_{variable_name}.pdf"),
            cbar_label=effective_cbar_label,
            cmap=None,
            norm="bwr" if vmin is not None and vmax is not None and vmin < 0 < vmax else "coolwarm",
            fontdict=self.fontdict,
            figsize=self.figsize,
            infobox_geographic=True,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            **kwargs
        )

    def plot_stacked_maps(
            self,
            maps: list,
            variable_name: str,
            output_path: str,
            fname: str,
            title: str = "",
            cbar_label: str = "",
            vmin=None,
            vmax=None,
            cbar_orientation: str = "horizontal",
            row_info_right: dict = None,
    ) -> None:
        """Render one figure with one map row per model and a shared colorbar."""
        if not maps:
            return

        os.makedirs(output_path, exist_ok=True)
        row_info_right = row_info_right or {}

        import cartopy.crs as ccrs
        from cartopy import feature as cfeature

        nrows = len(maps)
        proj = ccrs.Robinson(central_longitude=180.0)
        # 2 inches per row gives compact, paper-ready stacked maps.
        fig_h = max(2.0 * nrows, 4.0)
        fig = plt.figure(figsize=(A4_WIDTH, fig_h), dpi=self.dpi)
        gs = fig.add_gridspec(nrows=nrows, ncols=1, hspace=0.08)

        use_white_center_bins = vmin is not None and vmax is not None and vmin < 0 < vmax
        if use_white_center_bins:
            halfstep = spatial.round_to_next_nice(max(abs(vmin), abs(vmax)))
            step = halfstep / 10 if halfstep != 0 else 1.0
            levels = np.arange(-halfstep, halfstep + step, step)
            colormap = spatial.get_custom_cmap(levels, base_name="bwr")
            norm = BoundaryNorm(levels, ncolors=colormap.N)
            extend = "both"
        else:
            
            colormap = LinearSegmentedColormap.from_list('cmap_name', ["white", *plt.cm.Reds(np.linspace(0, 1, 256))])
            norm = None

            levels = 11

            # Round vmax up to a nice number for better colorbar ticks.  If vmin is negative, also round it down to a nice number and use symmetric levels around zero.
            vmax = np.round(vmax, -int(math.floor(math.log10(abs(vmax))))) 
            print(f"Rounded vmax to {vmax}")
            vmin = 0.0

            levels = np.linspace(vmin, vmax, levels)
            extend = "max" 
            # Make sure, that white is the first color in the colormap, so that values close to zero are white.
            
        mappable = None
        axes = []
        for i, (model_label, data) in enumerate(maps):
            da = data
            if "lon" in da.dims:
                n_lon = da.sizes["lon"]
                corrected_lons = np.roll(da.lon.values, -n_lon // 2)
                da = da.assign_coords(lon=corrected_lons).sortby("lon")

            ax = fig.add_subplot(gs[i, 0], projection=proj)
            axes.append(ax)
            cnt = ax.contourf(
                da.lon.values,
                da.lat.values,
                da.values,
                transform=ccrs.PlateCarree(),
                cmap=colormap,
                vmin=None if use_white_center_bins else vmin,
                vmax=None if use_white_center_bins else vmax,
                levels=levels,
                norm=norm,
                extend=extend,
            )
            mappable = cnt
            ax.coastlines(linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.2)
            ax.set_global()

            ax.text(
                -0.01, 0.5, model_label,
                transform=ax.transAxes,
                ha="right", va="center",
                fontsize=self.fontdict.get("axes.titlesize", 18),
                
                rotation=90,
            )

            #ax.text(-0.01, 0.5, infotext_left, fontsize=_fs_title,
            #ha="right", va="center", transform=ax.transAxes, rotation=90)

            info_txt = row_info_right.get(model_label, "")
            if info_txt:
                #ax.text(
                #    1.0, 1.01, info_txt,
                #    transform=ax.transAxes,
                #    ha="right", va="bottom",
                #    fontsize=self.fontdict.get("axes.labelsize", 16),
                #)
                pass

            gl = ax.gridlines(draw_labels=False, linewidth=0.2, alpha=0.3)

        if title and axes:
            # Use geographic placement like spatial.contourf(infobox_geographic=True):
            # draw title in lon/lat coordinates so it tracks map projection.
            axes[0].text(
                3.0,
                90.0,
                title,
                transform=ccrs.PlateCarree(),
                ha="left",
                va="bottom",
                fontsize=self.fontdict.get("axes.titlesize", 18),
                
            )

        if mappable is not None:
            # Match contourf colorbar thickness and placement.
            fig.canvas.draw()
            x0 = min(ax.get_position().x0 for ax in axes)
            y0 = min(ax.get_position().y0 for ax in axes)
            x1 = max(ax.get_position().x1 for ax in axes)
            y1 = max(ax.get_position().y1 for ax in axes)

            if cbar_orientation == "vertical":
                cax = fig.add_axes([x1 + 0.015, y0, 0.03, y1 - y0])
                cbar = fig.colorbar(mappable, cax=cax, orientation="vertical", extend="both")
            else:
                cax = fig.add_axes([x0, y0 - 0.06, x1 - x0, 0.03])
                cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend="both")

            # Use scientific notation only when magnitudes warrant it.
            formatter = mticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2, 3))
            axis = cbar.ax.xaxis if cbar_orientation == "horizontal" else cbar.ax.yaxis
            axis.set_major_formatter(formatter)
            cbar.ax.figure.canvas.draw()

            if cbar_label:
                cbar.set_label(cbar_label, fontsize=max(self.fontdict.get("axes.labelsize", 16) - 2, 8))

        fpath = os.path.join(output_path, fname)
        plt.savefig(fpath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_multi_var_bias_grid(
        self,
        var_data: list,
        model_labels: list,
        output_path: str,
        fname: str,
    ) -> None:
        """Grid of bias maps: models in rows, variables in columns.

        Parameters
        ----------
        var_data:
            List of per-variable dicts, each containing:
            ``"title"`` (column header), ``"cbar_label"``, ``"vmin"``,
            ``"vmax"``, ``"maps"`` ({model_label: (bias_map, ...)}).
        model_labels:
            Ordered list of model labels defining the row order.
        output_path:
            Directory in which to write the figure.
        fname:
            Output filename (relative to *output_path*).
        """
        if not var_data or not model_labels:
            return

        import cartopy.crs as ccrs
        from cartopy import feature as cfeature

        os.makedirs(output_path, exist_ok=True)

        n_rows = len(model_labels)
        n_cols = len(var_data)

        proj = ccrs.Robinson(central_longitude=180.0)

        # Total figure width = A4_WIDTH; height = 2 inches per model row.
        cell_w = A4_WIDTH / max(n_cols, 1)
        cell_h = 2.0
        fig_w = A4_WIDTH
        fig_h = cell_h * n_rows

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=self.dpi)
        # Extra bottom row (height_ratio 0.06 of cell_h) for per-column colorbars.
        gs = fig.add_gridspec(
            nrows=n_rows + 1,
            ncols=n_cols,
            hspace=0.05,
            wspace=0.05,
            height_ratios=[1.0] * n_rows + [0.06],
        )

        # Pre-build colormap / level specs once per column so all rows share them.
        col_specs = []
        for vd in var_data:
            vmin, vmax = vd["vmin"], vd["vmax"]
            use_diverging = vmin is not None and vmax is not None and vmin < 0 < vmax
            if use_diverging:
                halfstep = spatial.round_to_next_nice(max(abs(vmin), abs(vmax)))
                step = halfstep / 10 if halfstep != 0 else 1.0
                levels = np.arange(-halfstep, halfstep + step, step)
                # Snap near-zero values to exact 0.0 so the zero level
                # boundary is precise and the two adjacent bins are symmetric.
                levels = np.where(np.isclose(levels, 0.0), 0.0, levels)
                # Build a colormap with TWO white bins: one whose midpoint is
                # just below zero and one just above.  We do this manually
                # rather than via get_custom_cmap (which only whites one bin).
                midpoints = np.array(
                    [(levels[i] + levels[i + 1]) / 2 for i in range(len(levels) - 1)]
                )
                base_cmap = plt.get_cmap("bwr")
                colors_arr = base_cmap(np.linspace(0, 1, len(levels) - 1))
                neg_bins = np.where(midpoints < 0)[0]
                pos_bins = np.where(midpoints > 0)[0]
                if len(neg_bins) > 0:
                    colors_arr[neg_bins[-1]] = [1, 1, 1, 1]   # bin just below zero
                if len(pos_bins) > 0:
                    colors_arr[pos_bins[0]] = [1, 1, 1, 1]    # bin just above zero
                colormap = ListedColormap(colors_arr)
                norm = None
                extend = "both"
            else:
                n_levels = 11
                safe_vmax = vmax if vmax != 0 else 1.0
                vmax_r = np.round(safe_vmax, -int(math.floor(math.log10(abs(safe_vmax)))))
                levels = np.linspace(0.0, vmax_r, n_levels)
                colormap = LinearSegmentedColormap.from_list(
                    "bias_cmap", ["white", *plt.cm.Reds(np.linspace(0, 1, 256))]
                )
                norm = None
                extend = "max"
            col_specs.append({
                "levels": levels, "cmap": colormap, "norm": norm,
                "extend": extend, "use_diverging": use_diverging,
            })

        col_mappables = [None] * n_cols
        for row_i, model_label in enumerate(model_labels):
            for col_j, vd in enumerate(var_data):
                cs = col_specs[col_j]
                maps = vd["maps"]
                if model_label not in maps:
                    continue
                bias_map = maps[model_label][0]

                da = bias_map
                if "lon" in da.dims:
                    n_lon = da.sizes["lon"]
                    corrected_lons = np.roll(da.lon.values, -n_lon // 2)
                    da = da.assign_coords(lon=corrected_lons).sortby("lon")

                ax = fig.add_subplot(gs[row_i, col_j], projection=proj)
                cnt = ax.contourf(
                    da.lon.values, da.lat.values, da.values,
                    transform=ccrs.PlateCarree(),
                    cmap=cs["cmap"],
                    levels=cs["levels"],
                    extend=cs["extend"],
                )
                ax.coastlines(linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.2)
                ax.set_global()
                ax.gridlines(draw_labels=False, linewidth=0.2, alpha=0.3)

                # Row label on the leftmost column only.
                if col_j == 0:
                    ax.text(
                        -0.01, 0.5, model_label,
                        transform=ax.transAxes,
                        ha="right", va="center",
                        fontsize=self.fontdict.get("axes.titlesize", 18),
                        
                        rotation=90,
                    )

                # Column title on the top row only.
                if row_i == 0:
                    ax.set_title(
                        vd["title"],
                        fontsize=self.fontdict.get("axes.titlesize", 18),
                        
                        pad=4,
                    )

                col_mappables[col_j] = cnt

        # One horizontal colorbar per column in the extra bottom row.
        for col_j, vd in enumerate(var_data):
            if col_mappables[col_j] is None:
                continue
            cax = fig.add_subplot(gs[n_rows, col_j])
            cbar = fig.colorbar(
                col_mappables[col_j], cax=cax,
                orientation="horizontal",
                extend=col_specs[col_j]["extend"],
            )
            cbar.set_label(
                vd["cbar_label"],
                fontsize=max(self.fontdict.get("axes.labelsize", 16) - 2, 8),
            )
            formatter = mticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2, 3))
            cbar.ax.xaxis.set_major_formatter(formatter)
            cbar.ax.figure.canvas.draw()

        fpath = os.path.join(output_path, fname)
        plt.savefig(fpath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot(self, x: np.ndarray, model_label,
             title, variable_name, style="imshow",
             output_path=None, vmin=None, vmax=None, cbar_label="",
             extent=None, **kwargs):

        if output_path is None:
            output_path = self.output_path
        else:
            os.makedirs(output_path, exist_ok=True)

        # "imshow" kept as an alias for backwards compatibility.
        method = "map_plot" if style == "imshow" else style
        if "cmap" in kwargs:
            kwargs.pop("cmap")
            
        getattr(self, method)(
            x=x,
            variable_name=variable_name,
            model_label=model_label,
            title=title,
            output_path=output_path,
            vmin=vmin,
            vmax=vmax,
            cbar_label=cbar_label,
            extent=extent,
            **kwargs
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
        figsize=(A4_WIDTH, 3.5), linewidth=2.0,
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
        fname: str = "radial_spectrum.pdf",
        annotation: str = "",
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
            color = colors[model_label]
            lw = linewidths.get(model_label, self.linewidth)
            
            for instance, spectrum in instance_spectra.items():
                print(instance)

                if isinstance(instance, str):
                    instance = instance.split(",")  
                start_time, end_time = instance[0], instance[1]
                if start_time == end_time:
                    instance_str = start_time
                    default_ls = ":"  # single snapshot → dotted
                else:
                    if start_time.split("-")[0] == end_time.split("-")[0]:  # same year
                        instance_str = f"{start_time.split('-')[0]}" 
                    else:
                        # year - year
                        instance_str = f"{start_time.split('-')[0]}-{end_time.split('-')[0]}"
                    default_ls = "-"  # period average → solid

                spectra.radial_spectrum_to_ax(
                    ax,
                    spectrum=spectrum,
                    label=f"{model_label} ({instance_str})",
                    color=color,
                    linestyle=linestyles.get(model_label, default_ls),
                    linewidth=lw,
                )

        ax.text(
            0.0, 1.01, 
            title, 
            fontsize=self.fontdict["axes.labelsize"], 
            ha="left", va="bottom", transform=ax.transAxes,
            
            )
        
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.fontdict["axes.labelsize"])
        ax.set_xlabel("Spatial Frequency", fontsize=self.fontdict["axes.labelsize"])
        n_curves = sum(len(v) for v in model_spectra.values())
        ncols = 2 if n_curves > 3 else 1
        ax.legend(
            ncol=ncols, fontsize=self.fontdict["legend.fontsize"],
            framealpha=0.8, loc="upper center", bbox_to_anchor=(0.5, -0.1)
        )
        if annotation:
            ax.text(
                0.05, 0.1, annotation,
                transform=ax.transAxes,
                fontsize=self.fontdict.get("font.size", 10) - 1,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
                zorder=5,
            )

        # set x and y ticks fontsize to the fontdict size
        ax.tick_params(axis="x", labelsize=self.fontdict["xtick.labelsize"])
        ax.tick_params(axis="y", labelsize=self.fontdict["ytick.labelsize"])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, fname), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_radial_multi(
        self,
        var_data: list,
        colors: dict = {},
        fname: str = "radial_spectra.pdf",
    ) -> None:
        """Multi-variable radial-spectrum grid (up to 3 panels per row, one shared legend).

        Parameters
        ----------
        var_data :
            List of ``(title, ylabel, model_spectra)`` tuples, one per variable.
            ``model_spectra`` follows the same ``{model_label: {instance: spectrum}}``
            convention as :py:meth:`plot_radial`.
        colors :
            Per-model colour overrides keyed by model label.
        fname :
            Output filename relative to ``self.output_path``.
        """
        import math
        n = len(var_data)
        n_cols = 2
        n_rows = n // n_cols + int(n % n_cols > 0)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(A4_WIDTH, 3.0 * n_rows),
            squeeze=False,
        )

        for idx, (title, ylabel, model_spectra) in enumerate(var_data):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]

            for model_label, instance_spectra in model_spectra.items():
                color = colors.get(model_label, "blue")
                for instance, spectrum in instance_spectra.items():
                    if isinstance(instance, str):
                        instance = instance.split(",")
                    start_time, end_time = instance[0], instance[1]
                    if start_time == end_time:
                        instance_str = start_time
                        default_ls = ":"
                    else:
                        if start_time.split("-")[0] == end_time.split("-")[0]:
                            instance_str = f"{start_time.split('-')[0]}"
                        else:
                            instance_str = f"{start_time.split('-')[0]}-{end_time.split('-')[0]}"
                        default_ls = "-"
                    spectra.radial_spectrum_to_ax(
                        ax,
                        spectrum=spectrum,
                        label=f"{model_label}",
                        color=color,
                        linestyle=default_ls,
                        linewidth=self.linewidth,
                    )

            ax.text(
                0.0, 1.01, title,
                fontsize=self.fontdict["axes.titlesize"],
                ha="left", va="bottom", transform=ax.transAxes,
            )
            ax.text(1.0, 1.01, instance_str, fontsize=self.fontdict["axes.labelsize"], ha="right", va="bottom", transform=ax.transAxes)
            ax.set_xlabel("Spatial Frequency", fontsize=self.fontdict["axes.labelsize"])
            ax.tick_params(axis="x", labelsize=self.fontdict["xtick.labelsize"])
            ax.tick_params(axis="y", labelsize=self.fontdict["ytick.labelsize"])

        # Hide unused axes
        for idx in range(n, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].set_visible(False)

        # One shared legend collected from the first panel
        handles, labels = axes[0][0].get_legend_handles_labels()
        n_curves = len(handles)
        ncols_legend = 2 if n_curves > 3 else 1
        axes[0][0].legend(
            handles, labels,
            loc="best",
            ncol=ncols_legend,
            fontsize=self.fontdict["legend.fontsize"],
            framealpha=0.8,
        )

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, fname), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_psd(
        self,
        model_spectra: dict,
        colors: dict = {},
        linestyles: dict = {},
        linewidths: dict = {},
        title: str = "",
        xlabel: str = r"Frequency ($\mathrm{month}^{-1}$)",
        ylabel: str = "Power Spectral Density",
        infotext_topright: str = "",
        fname: str = "psd.pdf",
        semilog: bool = True,
        xmax: float = None,
        highlight_bands: list = None,
        figsize: tuple = None,
        annotation: str = "",
    ) -> None:
        """Save a PSD comparison figure.

        Parameters
        ----------
        model_spectra:
            ``{model_label: (frequencies_array, psd_array)}`` tuples, as
            returned by :func:`~metrics.functional.spectral.welch_psd`.
        xmax:
            Upper limit for the frequency axis.  ``None`` keeps matplotlib's
            default auto-scaling.
        highlight_bands:
            Optional list of band dicts to shade on the plot.  Each dict must
            contain:

            * ``fmin`` / ``fmax`` – band edges in the same frequency units.
            * ``label`` – legend / annotation label.
            * ``color`` – fill colour (default ``"steelblue"``).
            * ``alpha`` – fill transparency (default ``0.12``).
            * ``edge_labels`` – list of ``(freq, text)`` tuples to annotate at
              specific frequencies inside the band (default: none).

            Example (ENSO band, frequency in 1/month)::

                highlight_bands=[
                    dict(
                        fmin=1/84, fmax=1/24,
                        label=r"ENSO band (2\u20137 yr)",
                        color="steelblue",
                        edge_labels=[(1/84, "7 yr"), (1/24, "2 yr")],
                    )
                ]
        """

        fig, ax = plt.subplots(figsize=(6.7, 3.6), dpi=300)
        ls_cycle = self.linestyles

        for j, (model_label, (freq, psd)) in enumerate(model_spectra.items()):
            spectra.psd_to_ax(
                ax,
                frequencies=freq,
                psd=psd,
                semilog=semilog,
                label=model_label,
                color=colors.get(model_label, "blue"),
                linestyle=linestyles.get(model_label, ls_cycle[j % len(ls_cycle)]),
                linewidth=linewidths.get(model_label, self.linewidth),
            )

        if xmax is not None:
            ax.set_xlim(left=0, right=xmax)

        if highlight_bands:
            ymin_ax, ymax_ax = ax.get_ylim()
            for band in highlight_bands:
                fmin  = band["fmin"]
                fmax  = band["fmax"]
                bcolor = band.get("color", "steelblue")
                balpha = band.get("alpha", 0.12)
                ax.axvspan(fmin, fmax, alpha=balpha, color=bcolor,
                           lw=0, zorder=0, label=band.get("label", ""))
                for edge_freq, edge_text in band.get("edge_labels", []):
                    ax.axvline(edge_freq, color=bcolor, lw=0.8,
                               ls="--", alpha=0.6, zorder=1)
                    ax.text(
                        edge_freq, ymin_ax, f" {edge_text}",
                        fontsize=self.fontdict.get("xtick.labelsize", 10),
                        color=bcolor, va="bottom", ha="left", rotation=90,
                    )

        #ax.set_title(title, fontsize=self.fontdict["axes.titlesize"])
        ax.set_xlabel(xlabel, fontsize=self.fontdict["axes.labelsize"])
        if ylabel:
            ax.text(0.0, 1.01, ylabel, fontsize=self.fontdict["axes.titlesize"],
                    ha="left", va="bottom", transform=ax.transAxes)
        if infotext_topright:
            ax.text(1.0, 1.01, infotext_topright, fontsize=self.fontdict["axes.titlesize"],
                    ha="right", va="bottom", transform=ax.transAxes)
        ncols = 2 if len(model_spectra) > 2 else 1
        ax.legend(
            ncol=ncols,
            fontsize=self.fontdict["legend.fontsize"],
            loc="best",
            framealpha=0.8,
        )

        if annotation:
            ax.text(
                0.02, 0.03, annotation,
                transform=ax.transAxes,
                fontsize=self.fontdict.get("font.size", 10) - 1,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
                zorder=5,
            )

        # Set xtick label and ytick label size
        ax.tick_params(axis="x", labelsize=self.fontdict["xtick.labelsize"])
        ax.tick_params(axis="y", labelsize=self.fontdict["ytick.labelsize"])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, fname), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)


class SOIFrequencyPlotter(FrequencyPlotter):
    """Frequency-domain plotter pre-configured for Southern Oscillation Index spectra.

    Inherits :class:`FrequencyPlotter` and adds :py:meth:`plot_soi_psd` which
    automatically applies the ENSO-relevant frequency band shading (2–7 year
    periods) and caps the x-axis at 0.2 month⁻¹.  The generic
    :py:meth:`~FrequencyPlotter.plot_psd` is still available unchanged for any
    non-SOI use-case.
    """

    #: ENSO band definition – periods 2–7 years = 24–84 months.
    ENSO_BAND: dict = dict(
        fmin=1.0 / 84.0,
        fmax=1.0 / 24.0,
        label=r"ENSO band (2$\!-\!$7 yr)",
        color="steelblue",
        alpha=0.12,
        edge_labels=[(1.0 / 84.0, "7 yr"), (1.0 / 24.0, "2 yr")],
    )

    def plot_soi_psd(
        self,
        model_spectra: dict,
        colors: dict = {},
        linestyles: dict = {},
        linewidths: dict = {},
        title: str = "",
        ylabel: str = "Power Spectral Density",
        fname: str = "SOI_Spectrum.pdf",
        semilog: bool = False,
        xmax: float = 0.2,
    ) -> None:
        """PSD plot for the SOI with ENSO band highlighted.

        All parameters mirror :py:meth:`~FrequencyPlotter.plot_psd`.  The
        ENSO band and x-axis cutoff are applied automatically and do not need
        to be supplied by the caller.
        """
        self.plot_psd(
            model_spectra=model_spectra,
            colors=colors,
            linestyles=linestyles,
            linewidths=linewidths,
            title=title,
            ylabel=ylabel,
            fname=fname,
            semilog=semilog,
            xmax=xmax,
            highlight_bands=[self.ENSO_BAND],
        )


class TimeseriesPlotter(EarthPlotter):
    def __init__(self, dpi=150, fontdict=None, output_path=".", figsize=(A4_WIDTH, 3.5), linewidth=2.0):
        super().__init__(dpi, fontdict, output_path, figsize)
        self.linewidth = linewidth
        

    @staticmethod
    def _xtick_step(n: int) -> int:
        """Thinning step for tick labels.

            n <=  8  → step 1
            n <= 32  → step 2
            n  > 32  → step 4
        """
        if n > 32:
            return 4
        if n > 8:
            return 2
        return 1

    def _compute_ticks(self, da: "xr.DataArray", year_ticks_only: bool = False):
        """Return (major_positions, major_labels, minor_positions) from a DataArray.

        The DataArray must have a 1-D ``'time'`` coordinate whose values are
        string labels (e.g. ``'1980'`` or ``'1980-01'``).  The x-axis uses
        ``range(len(da))`` as positions.

        When *year_ticks_only* is True, one major tick is placed at the first
        index for each calendar year (reads the 4-char year prefix from each
        label).  All other positions become unlabelled minor ticks.
        """
        import pandas as pd
        time_vals = da.coords["time"].values
        n = len(time_vals)

        if year_ticks_only:
            # Try pandas DatetimeIndex first; fall back to string prefix parsing.
            try:
                years = pd.DatetimeIndex(time_vals).year
                seen: dict = {}
                for i, y in enumerate(years):
                    if y not in seen:
                        seen[y] = i
                all_pos = list(seen.values())
                all_labels = [str(y) for y in seen.keys()]
            except Exception:
                seen_str: dict = {}
                for i, t in enumerate(time_vals):
                    yr = str(t)[:4]
                    if yr not in seen_str:
                        seen_str[yr] = i
                all_pos = list(seen_str.values())
                all_labels = list(seen_str.keys())
        else:
            all_pos = list(range(n))
            all_labels = [str(t) for t in time_vals]

        step = self._xtick_step(len(all_pos))
        major_pos = all_pos[::step]
        major_labels = all_labels[::step]
        major_set = set(major_pos)
        # When year_ticks_only is True, minor ticks sit only at year boundaries
        # that were skipped by the decimation step — not at every data point.
        # This avoids dense monthly minor ticks on long time-series plots.
        tick_pool = all_pos if year_ticks_only else range(n)
        minor_pos = [p for p in tick_pool if p not in major_set]
        return major_pos, major_labels, minor_pos

    def plot(
            self, model_data: dict, linear_trend: dict = {}, model_stds: dict = {},
            model_members: dict = {},
            fill=None, colors: dict = {}, linewidths: dict = {}, linestyles: dict = {},
            markers: dict = {}, title: str = "", variable_name: str = "", 
            xlabel="", ylabel="", xticks=None, fname: str = "",
            year_ticks_only: bool = False, trend_unit: str = "/ decade", 
            title_fontweight: str = None, legend_loc: str = None, **kwargs
            
    ):
        """Plot timeseries from DataArrays.

        Parameters
        ----------
        model_data:
            ``{model_label: xr.DataArray}`` – each DataArray must have a 1-D
            ``'time'`` coordinate with string tick labels.
        """
        import xarray as _xr
        fig, axs = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)

        longest_da = None
        for model_label, da in model_data.items():
            # Accept plain (time, data) tuples for backward compatibility.
            if isinstance(da, tuple):
                _time, _data = da
                da = _xr.DataArray(_data, coords={"time": np.array(_time, dtype=str)}, dims=["time"])
            if longest_da is None or len(da) > len(longest_da):
                longest_da = da

            std_da = model_stds.get(model_label, None)
            trend_entry = linear_trend.get(model_label, (None, None))
            if isinstance(trend_entry, tuple):
                trend_da, m = trend_entry
            else:
                trend_da, m = trend_entry, None
            trend_vals = trend_da.values if hasattr(trend_da, "values") else trend_da
            std_vals   = std_da.values   if hasattr(std_da,   "values") else std_da

            color     = colors.get(model_label, "blue")
            linewidth = linewidths.get(model_label, self.linewidth)
            linestyle = linestyles.get(model_label, "-")
            marker    = markers.get(model_label, None)
            label     = (f"{model_label} (Trend: {m:.3g} {ylabel} {trend_unit})" if m is not None
                         else model_label)

            # Draw individual member cycles when provided (monthly annual-cycle mode).
            member_series = model_members.get(model_label, None)
            if member_series:
                _base_rgb = mpl.colors.to_rgb(color)
                _h, _l, _s = colorsys.rgb_to_hls(*_base_rgb)
                _n = len(member_series)
                _ls_vals = np.linspace(max(0.2, _l - 0.25), min(0.9, _l + 0.25), _n)
                for _idx, _mc in enumerate(member_series):
                    _mc_vals = _mc.values if hasattr(_mc, "values") else _mc
                    _member_color = colorsys.hls_to_rgb(_h, _ls_vals[_idx], _s)
                    axs.plot(
                        range(0, len(_mc_vals)),
                        _mc_vals,
                        color=_member_color,
                        linewidth=linewidth,
                        linestyle="-",
                        alpha=0.6,
                    )

            timeseries.timeseries_to_ax(
                ax=axs,
                x=range(0, len(da)),
                y=da.values,
                color=color,
                linear_trend=trend_vals,
                std=std_vals,
                linewidth=linewidth,
                fill=fill,
                label=label,
                marker=marker,
                linestyle=linestyle,
            )

        if title:
            axs.set_title(
                title, 
                fontsize=self.fontdict["axes.titlesize"],
                fontweight=title_fontweight            
            )
        axs.set_xlabel(xlabel, fontsize=self.fontdict["axes.labelsize"])
        
        if kwargs.get("infobox_topleft", None):
            axs.text(
                0.0, 1.01, kwargs["infobox_topleft"],
                transform=axs.transAxes,
                fontsize=self.fontdict["axes.titlesize"],
                va="bottom", ha="left",
            
            )
        if kwargs.get("infobox_topright", None):
            axs.text(
                1.0, 1.01, kwargs["infobox_topright"],
                transform=axs.transAxes,
                fontsize=self.fontdict["axes.labelsize"],
                va="bottom", ha="right",
            )
        if kwargs.get("infobox_bottomleft", None):
            axs.text(
                0.02, 0.03, kwargs["infobox_bottomleft"],
                transform=axs.transAxes,
                fontsize=self.fontdict["axes.labelsize"],
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
                zorder=5,
            )
        if kwargs.get("infobox_bottomright", None):
            axs.text(
                0.98, 0.03, kwargs["infobox_bottomright"],
                transform=axs.transAxes,
                fontsize=self.fontdict.get("font.size", 14) - 1,
                va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
                zorder=5,
            )
        if ylabel:
            axs.set_ylabel(ylabel, fontsize=self.fontdict["axes.labelsize"], labelpad=10)

        # --- Ticks -----------------------------------------------------------
        if xticks is not None:
            # Caller-supplied explicit ticks (e.g. month names).
            pos_seq, lbl_seq = xticks
            axs.set_xticks(list(pos_seq))
            axs.set_xticklabels(list(lbl_seq), rotation=45,
                                fontsize=self.fontdict["xtick.labelsize"])
        elif longest_da is not None:
            major_pos, major_labels, minor_pos = self._compute_ticks(
                longest_da, year_ticks_only=year_ticks_only)
            axs.set_xticks(major_pos)
            axs.set_xticklabels(major_labels, rotation=45,
                                fontsize=self.fontdict["xtick.labelsize"])
            if minor_pos:
                axs.set_xticks(minor_pos, minor=True)
                axs.tick_params(axis="x", which="minor", labelbottom=False)

        # Set ytick fontsize after setting x-ticks to avoid resetting them to default.
        axs.tick_params(axis="y", labelsize=self.fontdict["ytick.labelsize"])

        # Legend
        ncols = 2 if len(model_data) > 2 else 1
        if legend_loc is not None:
            axs.legend(loc=legend_loc, framealpha=0.8,
                       ncol=ncols, fontsize=self.fontdict["legend.fontsize"])
        else:
            axs.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), framealpha=0.8,
                       ncol=ncols, fontsize=self.fontdict["legend.fontsize"])

        axs.grid(True, linewidth=0.25, linestyle="-.", alpha=0.7)

        if kwargs.get("despine", False):
            axs.spines["top"].set_visible(False)
            axs.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{self.output_path}/{fname}", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_stem(
        self,
        model_data: dict,
        colors: dict = {},
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        xticks=None,
        fname: str = "stem.pdf",
        year_ticks_only: bool = False,
    ) -> None:
        """Stem plot with blue positive and red negative stems.

        Parameters
        ----------
        model_data:
            ``{model_label: xr.DataArray}`` – each DataArray must have a 1-D
            ``'time'`` coordinate.  Plain ``(time, data)`` tuples are also
            accepted for backward compatibility.
        """
        n = len(model_data)
        fig, axes = plt.subplots(1, 1, figsize=(A4_WIDTH, 3.6),
                                 dpi=300, squeeze=False)

        infoboxes = []
        for ax, (model_label, da) in zip(axes[:, 0], model_data.items()):
            # Accept plain (time, data) tuples.
            if isinstance(da, tuple):
                _time, _data = da
                da = xr.DataArray(_data, coords={"time": np.array(_time, dtype=str)}, dims=["time"])

            timeseries.stem_plot(ax, da.values, label=model_label)

            if title:
                ax.set_title(title, fontsize=self.fontdict["axes.titlesize"])
            ax.set_xlabel(xlabel, fontsize=self.fontdict["axes.labelsize"])
            if ylabel:
                ylabel_txt = ax.text(
                    0.0, 1.01, ylabel,
                    transform=ax.transAxes,
                    fontsize=self.fontdict["axes.titlesize"],
                    va="bottom", ha="left",
                    
                )
                infoboxes.append(ylabel_txt)

            # Ticks from DataArray.
            if xticks is not None:
                pos_seq, lbl_seq = xticks
                ax.set_xticks(list(pos_seq))
                ax.set_xticklabels(list(lbl_seq), rotation=45,
                                   fontsize=self.fontdict["xtick.labelsize"])
            else:
                major_pos, major_labels, minor_pos = self._compute_ticks(
                    da, year_ticks_only=year_ticks_only)
                ax.set_xticks(major_pos)
                ax.set_xticklabels(major_labels, rotation=45,
                                   fontsize=self.fontdict["xtick.labelsize"])
                if minor_pos:
                    ax.set_xticks(minor_pos, minor=True)
                    ax.tick_params(axis="x", which="minor", labelbottom=False)

            # Change y ticks fontsize after setting x-ticks to avoid resetting them to default.
            ax.tick_params(axis="y", labelsize=self.fontdict["ytick.labelsize"])
            # Model name annotation outside top-right of axes.
            txt = ax.text(
                1.0, 1.01, model_label,
                transform=ax.transAxes,
                fontsize=self.fontdict["axes.titlesize"],
                va="bottom", ha="right",
                
            )
            infoboxes.append(txt)
            ax.grid(True, linewidth=0.25, linestyle="-.", alpha=0.7)


        plt.savefig(f"{self.output_path}/{fname}", dpi=self.dpi,
                    bbox_inches="tight", bbox_extra_artists=infoboxes)
        plt.close(fig)


class LatitudinalProfilePlotter(EarthPlotter):
    """Line-plot comparison of latitude profiles across models."""

    def __init__(
        self,
        dpi: int = 150,
        fontdict: dict = None,
        output_path: str = ".",
        figsize: tuple = (A4_WIDTH, 3.5),
        linewidth: float = 2.0,
    ) -> None:
        super().__init__(dpi=dpi, fontdict=fontdict, output_path=output_path, figsize=figsize)
        self.linewidth = linewidth

    def plot(
        self,
        model_profiles: dict,
        colors: dict = None,
        linestyles: dict = None,
        linewidths: dict = None,
        markers: dict = None,
        title: str = "",
        xlabel: str = "Latitude",
        ylabel: str = "",
        fname: str = "latitudinal_profile.pdf",
        output_path: str = None,
        **kwargs,
    ) -> None:
        """Plot latitude (x) against variable magnitude (y) for all models."""
        import xarray as _xr

        colors = colors or {}
        linestyles = linestyles or {}
        linewidths = linewidths or {}
        markers = markers or {}

        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)

        for i, (model_label, profile) in enumerate(model_profiles.items()):
            if isinstance(profile, tuple):
                lat_vals, data_vals = profile
            elif isinstance(profile, _xr.DataArray):
                lat_vals = profile["lat"].values
                data_vals = profile.values
            else:
                raise TypeError(
                    "Each profile must be an xr.DataArray with 'lat' coord or a (lat, values) tuple."
                )

            
            #ls = linestyles.get(model_label, self.linestyles[i % len(self.linestyles)])
            ax.plot(
                lat_vals,
                data_vals,
                label=model_label,
                color=colors.get(model_label, "blue"),
                linestyle="-",
                linewidth=linewidths.get(model_label, self.linewidth),
                marker=markers.get(model_label, None),
                markersize=4,
            )

        if title:
            ax.set_title(title, fontsize=self.fontdict["axes.titlesize"])
        ax.set_xlabel(xlabel, fontsize=self.fontdict["axes.labelsize"])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.fontdict["axes.labelsize"])

        ax.tick_params(axis="x", labelsize=self.fontdict["xtick.labelsize"])
        ax.tick_params(axis="y", labelsize=self.fontdict["ytick.labelsize"])
        ax.grid(True, linewidth=0.25, linestyle="-.", alpha=0.7)

        if kwargs.get("infobox_topleft", None):
            ax.text(
                0.0,
                1.01,
                kwargs["infobox_topleft"],
                transform=ax.transAxes,
                fontsize=self.fontdict["axes.titlesize"],
                va="bottom",
                ha="left",
                
            )
        if kwargs.get("infobox_topright", None):
            ax.text(
                1.0,
                1.01,
                kwargs["infobox_topright"],
                transform=ax.transAxes,
                fontsize=self.fontdict["axes.titlesize"],
                va="bottom",
                ha="right",
                
            )
        if kwargs.get("infobox_bottomleft", None):
            ax.text(
                0.02,
                0.03,
                kwargs["infobox_bottomleft"],
                transform=ax.transAxes,
                fontsize=self.fontdict["axes.labelsize"],
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
                zorder=5,
            )
        if kwargs.get("infobox_bottomright", None):
            ax.text(
                0.98,
                0.03,
                kwargs["infobox_bottomright"],
                transform=ax.transAxes,
                fontsize=self.fontdict.get("font.size", 14) - 1,
                va="bottom",
                ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
                zorder=5,
            )

        ncols = 2 if len(model_profiles) > 2 else 1
        ax.legend(
            ncol=ncols,
            fontsize=self.fontdict["legend.fontsize"],
            bbox_to_anchor=(0.5, -0.12),
            loc="upper center",
            framealpha=0.8,
        )

        plt.tight_layout()
        target_dir = output_path if output_path is not None else self.output_path
        os.makedirs(target_dir, exist_ok=True)
        plt.savefig(os.path.join(target_dir, fname), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_multi(
        self,
        var_data: list,
        colors: dict = None,
        output_path: str = None,
        fname: str = "latprofile_multi.pdf",
    ) -> None:
        """Multi-variable latitude profile grid (up to 3 panels per row, one shared legend).

        Parameters
        ----------
        var_data :
            List of dicts, each with keys: ``model_profiles``
            ({model_label: xr.DataArray or (lat, values) tuple}),
            ``ylabel``, ``infobox_topleft``, ``infobox_topright``,
            ``infobox_bottomleft``, ``infobox_bottomright``.
        colors :
            Per-model colour overrides keyed by model label.
        output_path :
            Directory in which to save the figure.  Defaults to
            ``self.output_path``.
        fname :
            Output filename.
        """
        import math
        import xarray as _xr

        colors = colors or {}
        target_dir = output_path if output_path is not None else self.output_path
        os.makedirs(target_dir, exist_ok=True)

        n = len(var_data)
        n_cols = min(n, 3)
        n_rows = math.ceil(n / 3)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(A4_WIDTH, 2.5 * n_rows),
            dpi=self.dpi, squeeze=False,
        )

        for idx, vd in enumerate(var_data):
            row, col = divmod(idx, 3)
            ax = axes[row][col]
            model_profiles = vd.get("model_profiles", {})
            ylabel = vd.get("ylabel", "")

            for model_label, profile in model_profiles.items():
                if isinstance(profile, tuple):
                    lat_vals, data_vals = profile
                elif isinstance(profile, _xr.DataArray):
                    lat_vals = profile["lat"].values
                    data_vals = profile.values
                else:
                    continue
                ax.plot(
                    lat_vals, data_vals,
                    label=model_label,
                    color=colors.get(model_label, "blue"),
                    linestyle="-",
                    linewidth=self.linewidth,
                )

            if ylabel:
                ax.set_ylabel(ylabel, fontsize=self.fontdict["axes.labelsize"])
            ax.set_xlabel("Latitude", fontsize=self.fontdict["axes.labelsize"])
            ax.tick_params(axis="x", labelsize=self.fontdict["xtick.labelsize"])
            ax.tick_params(axis="y", labelsize=self.fontdict["ytick.labelsize"])
            ax.grid(True, linewidth=0.25, linestyle="-.", alpha=0.7)

            if vd.get("infobox_topleft"):
                ax.text(
                    0.0, 1.01, vd["infobox_topleft"], transform=ax.transAxes,
                    fontsize=self.fontdict["axes.titlesize"],
                    va="bottom", ha="left",
                )
            if vd.get("infobox_topright"):
                ax.text(
                    1.0, 1.01, vd["infobox_topright"], transform=ax.transAxes,
                    fontsize=self.fontdict["axes.labelsize"],
                    va="bottom", ha="right",
                )
            if vd.get("infobox_bottomleft"):
                ax.text(
                    0.02, 0.03, vd["infobox_bottomleft"], transform=ax.transAxes,
                    fontsize=self.fontdict["axes.labelsize"],
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
                    zorder=5,
                )
            if vd.get("infobox_bottomright"):
                ax.text(
                    0.98, 0.03, vd["infobox_bottomright"], transform=ax.transAxes,
                    fontsize=self.fontdict.get("font.size", 14) - 1,
                    va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
                    zorder=5,
                )

        # Hide unused axes
        for idx in range(n, n_rows * n_cols):
            row, col = divmod(idx, 3)
            axes[row][col].set_visible(False)

        # One shared legend below all subplots
        handles, labels = axes[0][0].get_legend_handles_labels()
        ncols_legend = min(len(handles), 5)

        plt.tight_layout()
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=ncols_legend,
            fontsize=self.fontdict["legend.fontsize"],
            framealpha=0.8,
        )
        plt.savefig(os.path.join(target_dir, fname), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

class SOIDashboard:
    # This class creates a dashboard of multiple SOI-related plots (timeseries, PSD, regression plots)
    def __init__(self, dpi=300, width=6.7):
        self.dpi = dpi
        self.width = width

    def create_dashboard(self, num_regression_plots):
        height_figsize = num_regression_plots * 1.5
        mosaic = [["a)", "c)"]["a)", "d)"]["b)", "e)"]["b)", "f)"]]
        fig, axes = plt.subplot_mosaic(
            mosaic, 
            figsize=(self.width, height_figsize), 
            dpi=self.dpi
        )
        

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
            fname="taylor_NAM.pdf",
        )
    """

    def __init__(
        self,
        dpi: int = 150,
        fontdict: dict = None,
        output_path: str = ".",
        figsize: tuple = (A4_WIDTH, A4_WIDTH),  # square Taylor diagram
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
        fname: str = "taylor_diagram.pdf",
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

        fig = plt.figure(figsize=(A4_WIDTH, A4_WIDTH), dpi=self.dpi)

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
            fontsize=mpl.rcParams["legend.fontsize"],
            framealpha=0.8,
            labelspacing=1.5,
            columnspacing=2.5,
            handletextpad=2.5,
            
        )

        plt.tight_layout()
        fpath = os.path.join(self.output_path, fname)
        plt.savefig(fpath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Taylor diagram saved to %s", fpath)
