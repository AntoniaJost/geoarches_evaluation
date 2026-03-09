import itertools
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker

from matplotlib import patches as mpatches
from matplotlib.colors import CenteredNorm
from matplotlib import path as mpath

import cartopy.crs as ccrs

from cartopy import feature as cfeature

from geoarches.dataloaders.era5 import surface_variables_short, level_variables_short

fontdict = {"font.size": 12}

def define_wedge(wedge):
    # Define a wedge shape for the colorbar ends
    if wedge.lower() == "noa":
        # -90 == 180 in the plot, 40 == 310 in the plot

        wedge = mpatches.Wedge(
            (0.5, 0.5), 0.5, 180, 310, fill=False, facecolor="k",
            edgecolor="k", linewidth=1.0,
            transform=ccrs.PlateCarree()
        )

        return wedge
    elif wedge.lower() == "europe":
        wedge = mpatches.Wedge(
            (0.5, 0.5), 0.5, 180, 360, fill=False, facecolor="k",
            edgecolor="k", linewidth=1.0,
            transform=ccrs.PlateCarree()
        )

        return wedge

    else:
        raise ValueError(f"Wedge {wedge} not a valid value, choose between 'noa' and 'europe'")



def set_boundary_to_lambert_conformal_projection(
    ax, central_longitude=-25, central_latitude=55, lowest_lat_cut=20, 
    highest_lat_cut=80, lon_extent=(-90, 40)
):
    """
    This function cuts the axes to the shape of the lambert conformal
    projection by creating a custom path and setting it as the boundary of the axes.

    Args:
        ax (_type_): The axes object to cut
        central_longitude (int, optional): The central longitude of the projection. Defaults to -25.
        central_latitude (int, optional): The central latitude of the projection. Defaults to 55.
        lat_extent (tuple, optional): The latitude extent of the projection. Defaults to (20, 90).
    """
    from matplotlib import path as mpath

    # Get 
    lon1 = np.linspace(
        central_longitude - 90, # in degree  
        central_longitude + 90, # in degree
        num=180 # longitude resolution
    )

    # Get theta for the conic shape of the lambert conformal projection, we only need the lower part of the conic which is between 0 and 180 degree in longitude
    theta = np.linspace(np.pi, 2 * np.pi, 180)  # lower part of conic
    mask = (lon1 >= lon_extent[0]) & (lon1 <= lon_extent[1]) # Get mask where theta is between the longitudes of the extent
    theta = theta[mask] # Restrict theta to the longitude extent

    max_to_np = 90 - lowest_lat_cut # Convert to distance from north pole
    min_to_np= 90 - highest_lat_cut # Convert to distance from north pole
    # distance to np equals radius of 1 on the axes object
    # Given this, calculate the radius for the latitudes of the segments we want to cut at
    r2 = 1 / 90 * max_to_np
    r1 = 1 / 90 * 1.3 * min_to_np # These have to be calculated based on the latitudes of the segments of the lambert conformal projection and the extent of the latitudes we want to cut

    
    x1 = r1 * np.cos(theta)  # We cut the last 10 and first 10 points to avoid the sharp edges of the conic segments which are not well represented in the lambert conformal projection
    y1 = r1 * np.sin(theta) + 0.5  ## rather arbitrary choice to shift the inner circle up a bit to get a better fit to the lambert conformal projection, this can be adjusted based on the specific projection parameters and desired fit

    x2 = r2 * np.cos(theta) # We cut the last 10 and first 10 points to avoid the sharp edges of the conic segments which are not well represented in the lambert conformal projection
    y2 = r2 * np.sin(theta)

    x1, y1, x2, y2 = (x1 + 1.) / 2, (y1 + 1.) / 2, (x2 + 1.) / 2, (y2 + 1.) / 2
    min_1, max_1, min_2, max_2 = np.argmin(x1), np.argmax(x1), np.argmin(x2), np.argmax(x2)
    
    # Create list of verts to respect circular boundaries 
    verts = []
    verts.append([x1[min_1], y1[min_1]])
    verts.append([x2[min_2], y2[min_2]])
    for i in range(min_2, max_2 + 1):
        verts.append([x2[i], y2[i]])
    verts.append([x1[max_1], y1[max_1]])
    for i in range(max_1, min_1 - 1, -1):
        verts.append([x1[i], y1[i]])

    verts = np.array(verts)

    # Verts are between min -1 and max 1, we need to shift them to be between 0 and 1
    path = mpath.Path(verts)
    ax.set_boundary(path, transform=ax.transAxes)

    return ax

def set_boundary_to_azimuthal_equidistant_projection(
    ax, central_longitude=-25, central_latitude=55, lowest_lat_cut=20, 
    highest_lat_cut=80, lon_extent=(-90, 40)
):
    # For the azimuthal equidistant projection, we can simply set the extent 
    # of the axes to the desired extent, as the projection is already circular 
    # and does not have the same issues
    ax.set_boundary(lon_extent + (lowest_lat_cut, highest_lat_cut), crs=ccrs.PlateCarree())
    return ax

def azimuthal_equidistant_projection_plot(
        data, central_latitude=0, central_longitude=0, extent=None, fpath=None, info_vals=None, wedge=None):
    

    proj = ccrs.AzimuthalEquidistant(
        central_longitude=central_longitude,
        central_latitude=central_latitude
    )

    fig = plt.figure(figsize=(8, 8), dpi=150)

    ax = plt.axes(projection=proj)
    if wedge is not None:
        print(f"Adding wedge for {wedge} to the plot")
        wedge = define_wedge(wedge)
        ax.add_patch(wedge)

    if np.max(data.lon) < 360:
        lon = np.linspace(0, 360, num=data.sizes['lon'])
        data = data.assign_coords(lon=lon)

    cnt = ax.contourf(
        data.lon - 180., data.lat, data.values,
        transform=ccrs.PlateCarree(), cmap="bwr", 
        norm=CenteredNorm(vcenter=0), extend='both'
    )

    # add contour lines, get levels from the contourf object to ensure they are the same as the filled contours
    levels = cnt.levels
    ax.contour(
        data.lon - 180., data.lat, data.values,
        transform=ccrs.PlateCarree(), colors='k', linewidths=0.5, levels=levels
    )

    # Add map features
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree()) # Set extent in lon/lat

        r = (extent[-1] - extent[-2]) / 180 # Get radius of the circular boundary based on the latitude extent
        r = 0.5
        circ_x = 0.5 + r * np.cos(np.linspace(0, 2 * np.pi, 100))
        circ_y = 0.5 + r * np.sin(np.linspace(0, 2 * np.pi, 100))
        vertices = np.column_stack((circ_x, circ_y))
        path = mpath.Path(vertices)
        ax.set_boundary(path, transform=ax.transAxes)


    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES, linestyle=':')
    ax.coastlines()
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    if info_vals is not None:
        ax.text(
            1.01, 1.01, "\n".join([f"{key}: {value:.2f}" for key, value in info_vals.items()]),
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
        )
    plt.colorbar(cnt, orientation='vertical', pad=0.05, shrink=0.45)
    plt.savefig(fpath, bbox_inches='tight')
    plt.close(fig)

def lambert_conformal_projection_plot(
        data, central_latitude, central_longitude, extent, fpath, levels=None, lat_cutoff=-30, info_vals: dict = None, cut_boundary=True):
    

        proj = ccrs.LambertConformal(
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            cutoff=lat_cutoff
        )

        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = plt.axes(projection=proj)
        cnt = ax.contourf(
            data.lon - 180., data.lat, data.values,
            transform=ccrs.PlateCarree(), cmap="bwr", 
            norm=CenteredNorm(vcenter=0), extend='both',
            levels=levels
        )

        # add contour lines
        levels = cnt.levels if levels is None else levels
        ax.contour(
            data.lon - 180., data.lat, data.values,
            transform=ccrs.PlateCarree(), colors='k', linewidths=0.5, levels=levels
        )
        # Add map features
        # set boundary of ax to be of conic shape


        if extent is not None:
            lon_extent = [extent[0], extent[1]]
            lat_extent = [extent[2], extent[3]]
            ax.set_extent([*lon_extent, *lat_extent], crs=ccrs.PlateCarree()) # Set extent in lon/lat

            if cut_boundary:
                ax = set_boundary_to_lambert_conformal_projection(
                    ax, central_longitude=central_longitude, central_latitude=central_latitude, 
                    lowest_lat_cut=lat_extent[0], highest_lat_cut=lat_extent[1], lon_extent=lon_extent
                )

        ax.coastlines()
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        if info_vals is not None:
            info_text = "\n".join([f"{key}: {value:.2f}" for key, value in info_vals.items()])
            ax.text(
                0.5, -0.1, info_text, transform=ax.transAxes,
                fontsize=10, ha='center', va='top'
            )


        plt.colorbar(cnt, orientation='vertical', pad=0.05, shrink=0.45)
        plt.savefig(fpath, bbox_inches='tight')
        plt.close(fig)

def get_projection(ax=None, projection=None, central_longitude=0.0):
    # Plot a xarray DataArray with cartopy projection
    if ax is None:
        if projection == "Robinson":
            projection = ccrs.Robinson(central_longitude=central_longitude)
        elif projection == "PlateCarree":
            projection = ccrs.PlateCarree(central_longitude=central_longitude)
        else:
            raise ValueError(f"Projection {projection} not a valid value")
        fig, ax = plt.subplots(subplot_kw={"projection": projection})
    else:
        fig = ax.figure

    return fig, ax
    

def imshow(
    x,
    output_path,
    title=None,
    ax=None,
    cbar_label=None,
    cmap="viridis",
    extent=None,
    central_longitude=0.0,
    projection="Robinson",
    patch_kwargs=None,
    vmin=None,
    vmax=None,
    norm=None,
    fontdict=fontdict,
    infotext="",
):
    # Plot a xarray DataArray with cartopy projection
    fig, ax = get_projection(ax, projection, central_longitude)
    
    plt.rcParams.update(fontdict)
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=0.5, alpha=0.9)


    ax.set_title(title)

    if x.shape[0] == 1:
        x = np.squeeze(x, axis=0)

    if vmin and vmax is not None and norm is not None:
        # Reset norm 
        norm = None
        
    img = ax.imshow(
        x,
        transform=ccrs.PlateCarree(central_longitude=central_longitude),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )

    ax.text(
        0.5,
        1.02,
        infotext,
        fontsize=10,
        ha="center",
        va="bottom",
        transform=ax.transAxes,
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
        img,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        extend="both",
        shrink=0.5,
        aspect=30,
        norm=norm,
    )

    cbar.set_label(cbar_label if cbar_label else "", fontsize=10)

    # Use scientific notation for very small or very large tick values.
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)  # Close the figure to free memory

def contourf(x, y, z, output_path, add_contourlines=False, **kwargs):
    fig, ax = get_projection(
        kwargs.get("cartopy_projection", "Robinson"), central_longitude=0.0)
    plt.rcParams.update(kwargs.get("fontdict", fontdict))
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, edgecolor="black")
    gl = ax.gridlines(
        draw_labels=True, dms=True, x_inline=False,
        y_inline=False, linewidth=0.5, alpha=0.9
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    cnt = ax.contourf(
        x, y, z,
        transform=ccrs.PlateCarree(),
        cmap=kwargs.get("cmap", "viridis"),
        vmin=kwargs.get("vmin", None),
        vmax=kwargs.get("vmax", None),
        norm=kwargs.get("norm", None),
        extend='both'
    )

    if add_contourlines:
        ax.contour(
            x, y, z,
            transform=ccrs.PlateCarree(),
            colors='k',
            linewidths=0.5,
            levels=10
        )


    # Colorbar with triangular ends and smaller size
    cbar = fig.colorbar(
        cnt,
        ax=ax,
        orientation="vertical",
        pad=0.1,
        shrink=0.5,
        aspect=30,
        norm=kwargs.get("norm", None),
    )

    cbar.set_label(kwargs.get("cbar_label", ""), fontsize=10)

    # Use scientific notation for very small or very large tick values.
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)  # Close the figure to free memory
    

def plot_temperature_with_geopotential_contours(
    temp, geopotential, level, output_path, title
):
    # Use cartopy to plot temperature with geopotential contours
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, edgecolor="black")

    # Plot temperature and geopotential
    temp.plot(
        ax=ax,
        transform=ccrs.Robinson(),
        cmap="coolwarm",
        vmin=temp.min(),
        vmax=temp.max(),
    )
    geopotential.plot.contour(
        ax=ax, transform=ccrs.PlateCarree(), levels=4, cmap="gray", linewidths=1.0
    )
    ax.set_title(title)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
