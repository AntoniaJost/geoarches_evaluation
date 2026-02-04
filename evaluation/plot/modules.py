from plot.functional import timeseries, spatial, spectra, stats
from typing import List
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
            "tas": "[K]",
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

    def __init__(self, dpi=150, fontdict=None, output_path=".", figsize=(10, 6)):
        self.dpi = dpi
        self.output_path = output_path
        self.figsize = figsize
        if fontdict is not None:
            self.fontdict = fontdict

class SpatialPlotter(EarthPlotter):

    def __init__(self, xdim, ydim, fontdict=None, cmap=None):
        self.xdim = xdim
        self.ydim = ydim
        self.cmap = cmap

        self.fontdict = fontdict if fontdict is not None else {
            'rc.fontsize': 12,
            'rc.fontweight': 'normal',
            'rc.fontfamily': 'serif'
        }


    def plot(self, data: List[xr.Dataset, xr.DataArray], model_label, 
                title, variable_name, output_path
            ):
        
        if data.min().values < 0 and data.max().values > 0:
            vmin = data.min().values
            vmax = data.max().values
            norm = CenteredNorm(vmin=vmin, vmax=vmax, vcenter=0)
            colormap = self.cmap if self.cmap is not None else 'bwr'
        else:
            norm = None
            colormap = self.cmap if self.cmap is not None else 'viridis'


        spatial.plot_variable(
            x=data.transpose(self.ydim, self.xdim).values,
            fname=f"{variable_name}_{model_label}.png",
            output_path=output_path,
            title=title,
            cbar_label=variable_name,
            cmap=colormap,
            vmin=vmin if 'vmin' in locals() else None,
            vmax=vmax if 'vmax' in locals() else None,
            norm=norm,
            fontdict=self.fontdict,
        )

class TimeseriesPlotter(EarthPlotter):
    def __init__(self, dpi=150, fontdict=None, output_path=".", figsize=(10, 6)):
        super().__init__(dpi, fontdict, output_path, figsize)

    def get_xticks_from_timeseries(self, time: np.ndarray):
        return timeseries.get_xticks_from_timeseries(time=time)

    def plot(self, data: dict, title: str, variable_name: str, output_path: str, fname: str):
        
        fig, axs = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)

        for model_label, (time, data) in data.items():
            timeseries.timerseries_to_ax(
                ax=axs,
                x=time,
                y=data,
                color="black",
                xticks=None,
                title=title,
                linear_trend=None,
                std=None,
                linewidth=2.0,
                fill=None,
                label=model_label,
                marker=None,
            )

            

        axs.set_title(title, fontdict=self.fontdict)
        axs.set_ylabel(f"{variable_name} {self.units.get(variable_name, '')}", fontdict=self.fontdict)
        axs.legend(fontsize=12)
        axs.grid(True, linewidth=0.25, linestyle='-', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{output_path}/{fname}", dpi=self.dpi)
        plt.close(fig)