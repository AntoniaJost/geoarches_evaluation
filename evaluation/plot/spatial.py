import itertools
import numpy as np

import matplotlib.pyplot as plt

from matplotlib import patches as mpatches

import cartopy.crs as ccrs

from cartopy import feature as cfeature

from geoarches.dataloaders.era5 import surface_variables_short, level_variables_short

fontdict = {'font.size': 12}
def plot_variable(x, fname, output_path, title=None, ax=None, cbar_label=None, cmap='viridis', extent=None, central_longitude=0., global_projection="Robinson", patch_kwargs=None, vmin=None, vmax=None, norm=None):
    
    # Plot a xarray DataArray with cartopy projection 
    if ax is None:
        if global_projection == "Robinson":
            projection = ccrs.Robinson(central_longitude=central_longitude)
        elif global_projection == 'PlateCarree':
            projection = ccrs.PlateCarree(central_longitude=central_longitude)
        else: 
            raise ValueError(f"Projection {global_projection} not a valid value")
        fig, ax = plt.subplots(
            subplot_kw={
                'projection': projection
            }
        )
    else:
        fig = ax.figure

    plt.rcParams.update(fontdict)
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_title(title)
    ax.gridlines(
        draw_labels=True, 
        dms=True, 
        x_inline=False, 
        y_inline=False
    )

    print('Data Shape: ', x.shape)
    if x.shape[0] == 1:
        x = np.squeeze(x, axis=0)
    
    img = ax.imshow(
        x, transform=ccrs.PlateCarree(central_longitude=central_longitude), cmap=cmap, vmin=vmin, vmax=vmax,
        norm=norm
    )

    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree(central_longitude=central_longitude))

    if patch_kwargs is not None:
        ax.add_patch(mpatches.Rectangle(**patch_kwargs))

    if vmin is not None and vmax is not None:
        # set colorbar limits
        img.set_clim(vmin, vmax)

    # Colorbar with triangular ends and smaller size
    cbar = fig.colorbar(
        img, ax=ax, orientation='horizontal',
        pad=0.1, extend='both', shrink=0.7, aspect=30
    )




    cbar.set_label(cbar_label if cbar_label else "")
    plt.tight_layout()

    if output_path:
        plt.savefig(f"{output_path}/{fname}.png", dpi=300, bbox_inches='tight')

    plt.close(fig)  # Close the figure to free memory

def plot_anomalies(data, variables, levels, output_path, anomaly_type="Monthly", mask=None):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Plot surface variables first
    level_vars = itertools.product(variables['level'], levels)
    surface_vars = [(v, None) for v in variables['surface']]
    #variables = [*surface_vars, *level_vars]

    for var in variables["surface"]:
        anomaly = data[var].to_numpy()


        assert anomaly.shape[0] == 12, "Surface variable should have 12 months"
        for i in range(anomaly.shape[0]):
            if mask is not None and var in ['sea_surface_temperature', 'sea_ice_cover']:

                print(f'Masking {var}')
                x = anomaly[i]
                x[mask] = 0.

            title = f"{anomaly_type} Anomalies {months[i]} {surface_variables_short[var]} "
            plot_variable(x, fname=f"{anomaly_type}_anomaly_{months[i]}_{surface_variables_short[var]}", output_path=output_path, title=title)

    # Plot level variables
    for var in variables["level"]:
        for lvl in levels:
            anomaly = data[var].sel(level=lvl).to_numpy()
            
            for i in range(anomaly.shape[0]):
                title = f"{anomaly_type} Anomalies {months[i]} {level_variables_short[var]}{lvl}"
                plot_variable(
                    anomaly[i], 
                    fname=f"{anomaly_type}_anomaly_{months[i]}_{level_variables_short[var]}{lvl}", 
                    output_path=output_path, 
                    title=title
                )

def plot_temperature_with_geopotential_contours(temp, geopotential, level, output_path, title):

    # Use cartopy to plot temperature with geopotential contours
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()})
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')

    # Plot temperature and geopotential
    temp.plot(ax=ax, transform=ccrs.Robinson(), cmap='coolwarm', vmin=temp.min(), vmax=temp.max())
    geopotential.plot.contour(ax=ax, transform=ccrs.PlateCarree(), levels=4, cmap='gray', linewidths=1.)
    ax.set_title(title)

    plt.savefig(f"{output_path}/2m_temp_geopotential_{level}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)