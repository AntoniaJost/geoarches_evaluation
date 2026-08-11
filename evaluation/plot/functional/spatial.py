import itertools
import logging
import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker

from matplotlib import patches as mpatches
from matplotlib.colors import CenteredNorm, LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib import path as mpath

import cartopy.crs as ccrs

from cartopy import feature as cfeature

from geoarches.dataloaders.era5 import surface_variables_short, level_variables_short
from seaborn import colors

# A4 single-column text width in inches.
A4_WIDTH = 6.7

logger = logging.getLogger(__name__)

mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["font.family"] = "DejaVu Sans"  # for non-math text, e.g. axis labels and legends
mpl.rcParams["axes.titlesize"] = 8
mpl.rcParams["axes.labelsize"] = 7
mpl.rcParams["xtick.labelsize"] = 7
mpl.rcParams["ytick.labelsize"] = 7
mpl.rcParams["legend.fontsize"] = 6


def _apply_cbar_formatter(cbar):
    """Apply a ScalarFormatter to a colorbar so that the power-of-ten multiplier
    is shown once at the end of the axis rather than repeated at every tick.
    Works for both horizontal (xaxis) and vertical (yaxis) colorbars.
    """
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 3))  # offset notation outside [0.01, 99]
    axis = cbar.ax.xaxis if cbar.orientation == "horizontal" else cbar.ax.yaxis
    axis.set_major_formatter(formatter)
    cbar.ax.figure.canvas.draw()

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

def round_to_next_nice(x):
    if x == 0: return 0
    # Magnitude of exponent (e.g., 100 for 430, 0.001 for 0.0034)
    exponent = math.floor(math.log10(abs(x)))
    fraction = abs(x) / (10**exponent)
    
    # "Nice" steps within a decade
    nice_steps = [1, 2, 5, 10]
    # Find the smallest step that is >= fraction
    nice_fraction = min(s for s in nice_steps if s >= fraction)
    
    return (1 if x > 0 else -1) * nice_fraction * (10**exponent)

def get_custom_cmap(levels, base_name='bwr'):


    midpoints = [(levels[i] + levels[i+1]) / 2 for i in range(len(levels)-1)]
    
    # 2. Index finden, der am nächsten an 0 liegt
    white_idx = np.argmin(np.abs(midpoints))
    
    # 3. Farben extrahieren und Weiß injizieren
    base_cmap = plt.get_cmap(base_name)
    colors = base_cmap(np.linspace(0, 1, len(levels) - 1))
    colors[white_idx] = [1, 1, 1, 1] # RGBA für Weiß
   
    return ListedColormap(colors)

def azimuthal_equidistant_projection_plot(
        data, central_latitude=0, central_longitude=0,
        extent=None, fpath=None, info_vals=None, wedge=None,
        title=None, vmin=None, vmax=None, infotext_topleft=None,
        cbar_label=None):
    

    proj = ccrs.AzimuthalEquidistant(
        central_longitude=central_longitude,
        central_latitude=central_latitude
    )

    fig = plt.figure(figsize=(8, 8), dpi=150)

    ax = plt.axes(projection=proj)
    if wedge is not None:
        logger.info("Adding wedge for %s to the plot", wedge)
        wedge = define_wedge(wedge)
        ax.add_patch(wedge)

    if np.max(data.lon) < 360:
        lon = np.linspace(0, 360, num=data.sizes['lon'])
        data = data.assign_coords(lon=lon)

    # Pre-compute levels from shared vmin/vmax (or data range).
    # When the range straddles zero, use *half-integer* multiples of the bin
    # width so that zero always falls at the exact CENTRE of a bin (and never
    # on a level boundary).  This guarantees a single symmetric white bin.
    #if vmin is not None and vmax is not None:
    # Round vmin to the nearest smaller integer and vmax to the nearest larger integer
    _vmin = np.floor(float(vmin))
    _vmax = np.ceil(float(vmax))
    print(f"vmin vs / Rounded vmin: {vmin} / {_vmin}")
    print(f"vmax vs / Rounded vmax: {vmax} / {_vmax}")
    _halfrange = max(abs(_vmin), abs(_vmax))
    
    _vmin_use, _vmax_use = -_halfrange, _halfrange
    #else:
    #    _vmin_use = float(np.nanmin(data.values))
    #    _vmax_use = float(np.nanmax(data.values))
     #   _halfrange = max(abs(_vmin_use), abs(_vmax_use))

    #if _vmin_use < 0 < _vmax_use:
        # Build (2*n_side + 1) bins with uniform width; zero is the centre bin.

        # Round levels to full integers
        # (1) Round to nearest integer, then convert back to float (e.g. 0.5 → 1.0, -1.5 → -2.0).
        # (2) Add a small epsilon to the positive levels to ensure they round up (e.g. 1.5 → 2.0, not 1.0).
    #    dist = np.ceil(_vmax_use) - np.floor(_vmin_use) #

    #    if dist < 4:
    #        stepping = 0.2
    #    elif dist <= 10:
    #        stepping = 0.5
    #    else:
    #        stepping = 1
    #    _levels = np.arange(np.floor(_vmin_use), np.ceil(_vmax_use) + stepping, step=stepping)  
    #    _levels = np.round(_levels).astype(float)
    #    _levels = _levels[np.abs(_levels) != 0.]  

    #    print(f"Rounded levels: {_levels}")
    #    n_side = len(_levels) // 2 - 1
    #    _n_bins = len(_levels) - 1  # = 2*n_side + 1 = 9
    #    _colors = [_cmap_base(_i / max(_n_bins - 1, 1)) for _i in range(_n_bins)]
    #    _colors[n_side] = (1.0, 1.0, 1.0, 1.0)  # centre bin → pure white
    #    _cmap = ListedColormap(_colors)
    #    _norm = BoundaryNorm(_levels, _n_bins)
    #else:
    #    _loc = mticker.MaxNLocator(nbins=8)
    #    _levels = _loc.tick_values(_vmin_use, _vmax_use)
    #    if len(_levels) < 2:
    #        _levels = _loc.tick_values(_vmin_use * 1.1, _vmax_use * 1.1)
    #    _cmap = _cmap_base
    #    _norm = CenteredNorm(vcenter=(_vmin_use + _vmax_use) / 2, halfrange=_halfrange)

    levels = np.arange(_vmin_use, _vmax_use + 1, 1)
    # Throw 0 away
    levels = levels[levels != 0.]
    print(f"Final levels for contourf: {levels}")
    _cmap = get_custom_cmap(levels, base_name="bwr")

    cnt = ax.contourf(
        data.lon - 180., data.lat, data.values,
        transform=ccrs.PlateCarree(), cmap=_cmap, 
        extend='both', levels=levels
    )

    #print(f"Levels: {cnt.levels}")
    #print(f"Colors: {_cmap.colors}")
    #cnt.set_cmap(_cmap)

    # add contour lines reusing the same levels; use a masked array so that
    # NaN values (outside the lat/lon band) do not break contour line paths.
    levels = cnt.levels
    ax.contour(
        data.lon - 180., data.lat, data.values,
        transform=ccrs.PlateCarree(), colors='k', linewidths=1.0, 
        levels=levels,
        linestyles='-'
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
    ax.coastlines()
    #gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    # Set font size of gridline labels to 14
    #gl.xlabel_style = {'size': 14}
    #gl.ylabel_style = {'size': 14}
    if info_vals is not None:
        ax.text(
            1.01, 1.1, "\n".join([f"{key}: {value:.2f}" for key, value in info_vals.items()]),
            transform=ax.transAxes, fontsize=16, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'),
        )
    if infotext_topleft is not None:
        ax.text(
            0.0, 1.1, infotext_topleft, fontsize=16,
            ha='left', va='bottom', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'),
        )

    if title:
        ax.set_title(title, pad=10, fontsize=18)

    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.08, pos.width, 0.04])
    cbar = fig.colorbar(cnt, cax=cax, orientation='horizontal', extend='both')
    # insert 0 into level boundaries if not already present to ensure a clear zero reference point on the colorbar
    ticks = cbar.get_ticks().tolist()
    print(ticks)
    # New ticks equals levels
    
    #ticks = levels.tolist()
    
    ticks = np.arange(_vmin_use, _vmax_use + 2, 2).tolist()
    ticks_ids = np.arange(0, 2 * _halfrange + 2, 2).tolist()
    # Set colorbar ticklabels fontsize to 16
    #if 0.0 not in ticks:
    #    ticks.append(0.0)
    #    ticks.sort()
    #    ticks_ids.append(ticks_ids[-1] + 1 // 2)
    #    ticks_ids.sort()
    cbar.set_ticks(ticks=ticks, labels=[f"{tick}" for tick in ticks])
    cbar.ax.tick_params(labelsize=16)

    _apply_cbar_formatter(cbar)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=16)
    plt.savefig(fpath, bbox_inches='tight')
    plt.close(fig)


def azimuthal_equidistant_multi_plot(
        data_dict,
        central_latitude=0,
        central_longitude=0,
        extent=None,
        fpath=None,
        vmin=None,
        vmax=None,
        cbar_label=None,
        wedge=None,
        info_vals_dict=None,
        ncols=4,
):
    """Multi-panel azimuthal-equidistant plot with one subplot per model.

    All panels share identical contour levels and a single colorbar at the
    bottom.  Rendering uses the same logic as
    :func:`azimuthal_equidistant_projection_plot`.

    Parameters
    ----------
    data_dict : dict
        ``{display_label: xr.DataArray}`` – one entry per panel.
    central_latitude / central_longitude : float
        Centre of the projection (e.g. 90 / -90 for NH / SH).
    extent : list or None
        ``[lon_min, lon_max, lat_min, lat_max]`` in PlateCarree degrees.
    fpath : str
        Full output path for the saved figure.
    vmin / vmax : float
        Shared colour-scale bounds (symmetric expected).
    cbar_label : str or None
        Label placed on the shared horizontal colorbar.
    wedge : str or None
        Optional overlay (``"noa"``, ``"europe"``).
    info_vals_dict : dict or None
        ``{display_label: {key: value}}`` annotation boxes (top-right).
    ncols : int
        Maximum number of columns in the subplot grid.
    """
    import os

    n = len(data_dict)
    if n == 0:
        return

    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    proj = ccrs.AzimuthalEquidistant(
        central_longitude=central_longitude,
        central_latitude=central_latitude,
    )

    # Shared levels – same logic as the single-plot function.
    _vmin = np.floor(float(vmin))
    _vmax = np.ceil(float(vmax))
    _halfrange = max(abs(_vmin), abs(_vmax))
    _vmin_use, _vmax_use = -_halfrange, _halfrange
    levels = np.arange(_vmin_use, _vmax_use + 1, 1)
    levels = levels[levels != 0.]
    _cmap = get_custom_cmap(levels, base_name="bwr")

    subplot_size = A4_WIDTH
    row_size = A4_WIDTH / ncols
    figsize = (A4_WIDTH, row_size * nrows + 0.8)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        subplot_kw={"projection": proj},
        squeeze=False,
    )

    last_cnt = None
    for idx, (label, data) in enumerate(data_dict.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        if wedge is not None:
            ax.add_patch(define_wedge(wedge))

        da = data
        if np.max(da.lon) < 360:
            lon = np.linspace(0, 360, num=da.sizes["lon"])
            da = da.assign_coords(lon=lon)

        cnt = ax.contourf(
            da.lon - 180., da.lat, da.values,
            transform=ccrs.PlateCarree(),
            cmap=_cmap,
            extend="both",
            levels=levels,
        )
        last_cnt = cnt

        ax.contour(
            da.lon - 180., da.lat, da.values,
            transform=ccrs.PlateCarree(),
            colors="k",
            linewidths=0.8,
            levels=levels,
            linestyles="-",
        )

        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            r = 0.5
            circ_x = 0.5 + r * np.cos(np.linspace(0, 2 * np.pi, 100))
            circ_y = 0.5 + r * np.sin(np.linspace(0, 2 * np.pi, 100))
            path = mpath.Path(np.column_stack((circ_x, circ_y)))
            ax.set_boundary(path, transform=ax.transAxes)

        ax.add_feature(cfeature.COASTLINE)
        ax.coastlines()

        # Top-left: model / member label.
        ax.text(
            0.0, 1.05, label, fontsize=mpl.rcParams["legend.fontsize"],
            ha="left", va="bottom", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor="black"),
        )

        # Top-right: info_vals annotation.
        iv = (info_vals_dict or {}).get(label)
        if iv:
            ax.text(
                1.0, 1.05,
                "\n".join(f"{v}" for k, v in iv.items()),
                fontsize=mpl.rcParams["legend.fontsize"], ha="right", va="bottom", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8, edgecolor="black"),
            )

    # Hide unused axes.
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Shared horizontal colorbar at the bottom.
    if last_cnt is not None:
        fig.subplots_adjust(bottom=0.12, hspace=0.3, wspace=0.1)
        cax = fig.add_axes([0.15, 0.04, 0.7, 0.025])
        cbar = fig.colorbar(last_cnt, cax=cax, orientation="horizontal", extend="both")
        ticks = np.arange(_vmin_use, _vmax_use + 2, 2).tolist()
        cbar.set_ticks(ticks=ticks, labels=[f"{int(t)}" for t in ticks])
        cbar.ax.tick_params(labelsize=mpl.rcParams["ytick.labelsize"])
        _apply_cbar_formatter(cbar)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=mpl.rcParams["axes.labelsize"])

    os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
    plt.savefig(fpath, bbox_inches="tight")
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

def get_projection(ax=None, projection=None, central_longitude=0.0, figsize=None):
    # Plot a xarray DataArray with cartopy projection
    if ax is None:
        if projection == "Robinson":
            projection = ccrs.Robinson(central_longitude=central_longitude)
        elif projection == "PlateCarree":
            projection = ccrs.PlateCarree(central_longitude=central_longitude)
        else:
            raise ValueError(f"Projection {projection} not a valid value")
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})
    else:
        fig = ax.figure

    return fig, ax


def imshow(x, output_path, **kwargs):
    """Render a global raster on a cartopy projection and save it."""
    _figsize = kwargs.get("figsize", None)
    fig, ax = get_projection(
        None,
        kwargs.get("projection", "Robinson"),
        central_longitude=180.0,
        figsize=_figsize,
    )

    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.LAND, edgecolor="black")

    cmap = kwargs.get("cmap", "viridis")
    norm = kwargs.get("norm", None)
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)

    if x is None:
        raise ValueError("imshow expects a 2-D array in argument 'x'.")

    img = ax.imshow(
        x,
        transform=ccrs.PlateCarree(),
        extent=[0, 360, -90, 90],
        origin="lower",
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )

    title = kwargs.get("title", None)
    if title:
        ax.set_title(title, fontsize=mpl.rcParams["axes.titlesize"], pad=10)

    infotext = kwargs.get("infotext", "")
    if infotext:
        ax.text(
            0.5,
            1.02,
            infotext,
            fontsize=mpl.rcParams["axes.titlesize"],
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )

    cbar = plt.colorbar(img, ax=ax, orientation="horizontal", pad=0.03, shrink=0.9)
    cbar_label = kwargs.get("cbar_label", None)
    if cbar_label:
        cbar.set_label(cbar_label)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    


def contourf(x, y, z, output_path, add_contourlines=False, **kwargs):
    # central_longitude=180 places Greenwich (0°) at the leftmost edge of the map.
    _figsize = kwargs.get("figsize", None)
    fig, ax = get_projection(
        None, kwargs.get("cartopy_projection", "Robinson"), central_longitude=180.0,
        figsize=_figsize)

    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.LAND, edgecolor="black")

    gl = ax.gridlines(
        draw_labels=False, dms=True, x_inline=False,
        y_inline=False, linewidth=0.5, alpha=0.9
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    norm = kwargs.get("norm", None)
    vmin_kw = kwargs["vmin"]
    vmax_kw = kwargs["vmax"]
    print("Arguments: ", vmin_kw, vmax_kw)
    
    if vmin_kw < 0 < vmax_kw:
        _halfstep = max(abs(vmin_kw), abs(vmax_kw))
        _halfstep = round_to_next_nice(_halfstep)

        levels = np.arange(-_halfstep, _halfstep+_halfstep / 10, _halfstep / 10)

        n_colors = len(levels) - 1
        base_cmap = plt.cm.get_cmap('bwr', n_colors)
        new_colors = base_cmap(np.arange(n_colors))

        # 3. Die Bins um die 0 finden und weiß färben
        # In diesem Fall sind die Bins bei Index 3 (-1 bis 0) und Index 4 (0 bis 1)
        mid_idx = n_colors // 2
        new_colors[mid_idx-1 : mid_idx+1] = [1, 1, 1, 1] # RGBA für Weiß

        # 4. Neue Map und Norm erstellen
        custom_cmap = ListedColormap(new_colors)
        print(custom_cmap.colors)
        norm = BoundaryNorm(levels, ncolors=custom_cmap.N)

        cnt = ax.contourf(
            x, y, z,
            transform=ccrs.PlateCarree(),
            cmap=custom_cmap,
            extend='both',
            levels=levels,
            norm=norm,

        )
    else:   
        # Use proper sequential map for climate variables
        # that do not straddle zero, e.g. temperature or precipitation
        colors = ["#ffffff", "#8b0000"]
        cmap_name = "white_to_dark_red"

        # 2. Linearen Farbverlauf erstellen
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
        round_vmin = round_to_next_nice(vmin_kw)
        round_vmax = round_to_next_nice(vmax_kw)
        print(f"Original vmin: {vmin_kw}, Original vmax: {vmax_kw}")
        print(f"Rounded vmin: {round_vmin}, Rounded vmax: {round_vmax}")
        if round_vmin == round_vmax:
            round_vmin = vmin_kw
            round_vmax = vmax_kw
        halfrange = (round_vmin + round_vmax) / 2
        print(f"Halfrange: {halfrange}")
        dist = halfrange - round_vmin
        print(f"Distance between rounded vmin and halfrange: {dist}")
        levels=np.arange(round_vmin, round_vmax + dist / 20, dist / 20)
        print("Levels for sequential red cmap: ", levels)
        cnt = ax.contourf(
            x, y, z,
            transform=ccrs.PlateCarree(),
            cmap=custom_cmap,
            extend='max',
            levels=levels,
            norm=norm,

        )

    # replace the colormap with a custom one that has a white bin around zero 
    #cnt.set_cmap(custom_bwr)


    #if vmin_kw < 0 < vmax_kw:        
    #   #print(get_custom_cmap(levels, base_name=cmap).colors)
    #   colormap = get_custom_cmap(levels, base_name=cmap)
    #   cnt.set_cmap(colormap)
    
    if add_contourlines:
        ax.contour(
            x, y, z,
            transform=ccrs.PlateCarree(),
            colors='k',
            linewidths=0.5,
            levels=cnt.levels,  # reuse same levels as the filled contours
        )

    # Optional stippling: scatter dots at grid points significant at the
    # requested alpha level.  Caller must supply all three kwargs together:
    #   stipple_pvalues : 2-D array (lat, lon) of p-values
    #   stipple_lat     : 1-D latitude coordinate array
    #   stipple_lon     : 1-D longitude coordinate array
    #   stipple_alpha_level : significance threshold (default 0.05)
    stipple_pvalues = kwargs.get("stipple_pvalues", None)
    if stipple_pvalues is not None:
        stipple_lat = kwargs["stipple_lat"]
        stipple_lon = kwargs["stipple_lon"]
        alpha_level = kwargs.get("stipple_alpha_level", 0.05)
        lon2d, lat2d = np.meshgrid(stipple_lon, stipple_lat)
        sig_mask = stipple_pvalues < alpha_level
        ax.scatter(
            lon2d[sig_mask], lat2d[sig_mask],
            s=0.3, color="k", alpha=0.2,
            transform=ccrs.PlateCarree(),
            linewidths=0,
            zorder=5,
        )

    if kwargs.get("nino34_box", False):
        # Niño 3.4 region: 5°S–5°N, 170°W–120°W
        nino34_rect = mpatches.Rectangle(
            xy=(-170, -5),  # (lon_min, lat_min)
            width=50,       # 170°W to 120°W
            height=10,      # 5°S to 5°N
            linewidth=1.5,
            edgecolor="black",
            facecolor="none",
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        ax.add_patch(nino34_rect)

    title = kwargs.get("title", None)
    if title:
        ax.set_title(title, fontsize=mpl.rcParams["axes.titlesize"], pad=10)

    infotext = kwargs.get("infotext", "")
    if infotext is not None and infotext != "":
        ax.text(
            0.5, 1.02, infotext, fontsize=mpl.rcParams["axes.titlesize"],
            ha="center", va="bottom", transform=ax.transAxes,
        )

    infotext_topright = kwargs.get("infotext_topright", "")
    if infotext_topright:
        if kwargs.get("infobox_geographic", False):
            # Place the label in geographic (data) coordinates so it sits just
            # inside the curved left boundary of the Robinson projection.
            # With central_longitude=180 the prime meridian (0° lon) is at the
            # left edge; 3°E puts the text just inside the curved border.
            # The top latitude is derived from the extent when given, or
            # defaults to 78° for a full-globe Robinson view.
            _ext = kwargs.get("extent", None)
            _geo_lat = (_ext[3]) if _ext is not None else 90.0
            _geo_lon = (_ext[0]) if _ext is not None else 357
            ax.text(
                _geo_lon, _geo_lat, infotext_topright, fontsize=mpl.rcParams["axes.titlesize"],
                ha="right", va="bottom", transform=ccrs.PlateCarree(),
            )
        else:
            ax.text(
                1.0, 1.01, infotext_topright, fontsize=mpl.rcParams["axes.titlesize"],
                ha="right", va="bottom", transform=ax.transAxes,
            )

    infotext_topleft = kwargs.get("infotext_topleft", "")
    if infotext_topleft:
        if kwargs.get("infobox_geographic", False):
            # Place the label in geographic (data) coordinates so it sits just
            # inside the curved left boundary of the Robinson projection.
            # With central_longitude=180 the prime meridian (0° lon) is at the
            # left edge; 3°E puts the text just inside the curved border.
            # The top latitude is derived from the extent when given, or
            # defaults to 78° for a full-globe Robinson view.
            _ext = kwargs.get("extent", None)
            _geo_lat = (_ext[3]) if _ext is not None else 90.0
            _geo_lon = (_ext[0] + 3.0) if _ext is not None else 3.0
            ax.text(
                _geo_lon, _geo_lat, infotext_topleft, fontsize=mpl.rcParams["axes.titlesize"],
                ha="left", va="bottom", transform=ccrs.PlateCarree(), fontweight="bold",
            )
        else:
            ax.text(
                0.0, 1.01, infotext_topleft, fontsize=mpl.rcParams["axes.titlesize"],
                ha="left", va="bottom", transform=ax.transAxes,
                fontweight="bold",
            )

    infotext_left = kwargs.get("infotext_left", "")
    infotext_right = kwargs.get("infotext_right", "")
    if infotext_left:
        ax.text(-0.01, 0.5, infotext_left, fontsize=mpl.rcParams["axes.titlesize"], fontweight="bold",
            ha="right", va="center", transform=ax.transAxes, rotation=90)
        
    if infotext_right:
        if kwargs.get("infobox_geographic", False):
            _ext = kwargs.get("extent", None)
            _geo_lat = (_ext[3]) if _ext is not None else 0.0
            _geo_lon = (_ext[0]) if _ext is not None else 360.1
            ax.text(
                _geo_lon, _geo_lat, infotext_right, fontsize=mpl.rcParams["axes.titlesize"],
                ha="left", va="center", transform=ccrs.PlateCarree(), rotation=270,
            )
        else:
            ax.text(0.0, 0.5, infotext_right, fontsize=mpl.rcParams["axes.titlesize"],
                    ha="left", va="center", transform=ax.transAxes, rotation=270)

    extent = kwargs.get("extent", None)
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Force layout so ax.get_position() reflects final axes bounds,
    # then place a colorbar axes whose height exactly matches the map.
    fig.canvas.draw()
    pos = ax.get_position()
    orientation = kwargs.get("cbar_orientation", "vertical")

    if orientation == "vertical":
        cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.03, pos.height])
    else:  # horizontal
        cax = fig.add_axes([pos.x0, pos.y0 - 0.06, pos.width, 0.03])
    cbar = fig.colorbar(
        cnt,
        cax=cax,
        orientation=orientation,
        extend="both",
        norm=norm,
    )

    

    cbar.set_label(kwargs.get("cbar_label", ""), fontsize=mpl.rcParams["axes.labelsize"])

    # Set tick positions at nice round numbers that match the contour levels.
    if levels is None:
        cnt.levels = levels
        if len(cnt.levels) >= 10:
            # set only each second tick to be labeled to avoid overcrowding
            cbar.set_ticks(range(0, len(levels), 2), labels=[f"{tick}" for tick in levels[::2]])
        else:
            cbar.set_ticks(levels)
    else:
        if len(cnt.levels) >= 10:
            # set only each second tick to be labeled to avoid overcrowding
            cbar.set_ticks(levels[::2])
        else:
            cbar.set_ticks(ticks=levels, labels=[f"{tick}" for tick in levels])

    #    # format ticks with scientific notation if they are very small or large, otherwise with 1 decimal place
        
        #cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_cbar_tick))
    _apply_cbar_formatter(cbar)
    if output_path:
        dpi = kwargs.get("dpi", 300)
        print(dpi)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

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
