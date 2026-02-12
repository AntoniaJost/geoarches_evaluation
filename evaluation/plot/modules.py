import os
from plot.functional import timeseries, spatial, spectra, stats
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

    def imshow(self, x: np.ndarray, variable_name, model_label, 
               output_path: str, title: str = "", cbar_label: str = ""):
        if x.min() < 0 and x.max()  > 0:
            vmin = x.min()
            vmax = x.max()
            norm = CenteredNorm(vmin=vmin, vmax=vmax, vcenter=0)
            colormap = self.cmap if self.cmap is not None else 'bwr'
        else:
            norm = None
            colormap = self.cmap if self.cmap is not None else 'Blues'

        spatial.imshow(
            x=x,
            output_path=os.path.join(output_path, f"{model_label}_{variable_name}.png"),
            title=title,
            cbar_label=f"{variable_name} {self.cmor_units.get(variable_name, '')}",
            cmap=colormap,
            vmin=vmin if 'vmin' in locals() else None,
            vmax=vmax if 'vmax' in locals() else None,
            norm=norm,
            fontdict=self.fontdict,
        )

    def plot(self, x: np.ndarray, model_label, 
                title, variable_name, style="imshow", output_path=None
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
            output_path=output_path
        )

    

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
            fill=None, colors: dict = {}, linewidths: dict = {}, 
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
        axs.set_xticks(
            *self.get_xticks_from_timeseries(time), rotation=45,
            fontsize=self.fontdict['xtick.labelsize']
        )
        
        axs.legend(fontsize=self.fontdict['axes.labelsize'])
        axs.grid(True, linewidth=0.25, linestyle='-', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{self.output_path}/{fname}", dpi=self.dpi)
        plt.close(fig)
