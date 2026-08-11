"""
plot/projections.py
===================
Cartopy-based map projection plotter.

All map-projection rendering methods are grouped into the single
:class:`CartopyProjectionPlotter` class so callers can share configuration
(output directory, figure size, dpi, colormap) and simply call the
projection they need as a named method.

Supported projections
---------------------
* :py:meth:`azimuthal_equidistant`  – polar / regional views
* :py:meth:`lambert_conformal`      – mid-latitude regional views
* :py:meth:`robinson`               – global imshow (Robinson projection)
* :py:meth:`plate_carree`           – global contourf (Plate Carree)
"""

import os

from plot.functional.spatial import (
    azimuthal_equidistant_projection_plot,
    azimuthal_equidistant_multi_plot,
    lambert_conformal_projection_plot,
    imshow as _imshow,
    contourf as _contourf,
)


class CartopyProjectionPlotter:
    """
    Wraps cartopy projection functions as configurable instance methods.

    All methods share the instance-level defaults (``figsize``, ``dpi``,
    ``cmap``) but individual arguments can be overridden per call.

    Parameters
    ----------
    output_path:
        Directory where figures are saved.  Created automatically if absent.
    figsize:
        Default ``(width, height)`` in inches.
    dpi:
        Dots per inch for saved figures.
    cmap:
        Default colormap name (e.g. ``"bwr"``, ``"coolwarm"``).
    """

    def __init__(
        self,
        output_path: str = ".",
        figsize: tuple = (10, 6),
        dpi: int = 150,
        cmap: str = "bwr",
    ) -> None:
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = cmap

    # -- Internal ------------------------------------------------------------

    def _fpath(self, fname: str) -> str:
        return os.path.join(self.output_path, fname)

    # -- Projection methods --------------------------------------------------

    def azimuthal_equidistant(
        self,
        data,
        *,
        central_latitude: float = 90,
        central_longitude: float = 0,
        extent: list = None,
        fname: str = "azimuthal_equidistant.pdf",
        info_vals: dict = None,
        wedge: str = None,
        title: str = None,
        vmin: float = None,
        vmax: float = None,
        infotext_topleft: str = None,
        cbar_label: str = None,
    ) -> None:
        """
        Azimuthal-equidistant projection plot.

        Suitable for polar views (set ``central_latitude=90`` for the NH,
        ``central_latitude=-90`` for the SH) or any regional circular view.

        Parameters
        ----------
        data:
            ``xr.DataArray`` with ``lat`` and ``lon`` coordinates.
        central_latitude / central_longitude:
            Centre of the projection.
        extent:
            ``[lon_min, lon_max, lat_min, lat_max]`` in degrees (PlateCarree).
            When *None* the full globe is shown.
        fname:
            Output filename (relative to ``self.output_path``).
        info_vals:
            Dict of key/value pairs printed below the plot axes.
        wedge:
            Optional overlay wedge shape (``"noa"`` or ``"europe"``).
        vmin / vmax:
            Symmetric colorbar bounds.  When supplied a :class:`CenteredNorm`
            with ``halfrange=max(|vmin|, |vmax|)`` is used so all plots in a
            comparison share identical colour scales.
        infotext_topleft:
            Text annotation placed at the top-left corner of the axes.
        """
        azimuthal_equidistant_projection_plot(
            data=data,
            central_latitude=central_latitude,
            central_longitude=central_longitude,
            extent=extent,
            fpath=self._fpath(fname),
            info_vals=info_vals,
            wedge=wedge,
            title=title,
            vmin=vmin,
            vmax=vmax,
            infotext_topleft=infotext_topleft,
            cbar_label=cbar_label,
        )

    def azimuthal_equidistant_multi(
        self,
        data_dict: dict,
        *,
        central_latitude: float = 90,
        central_longitude: float = 0,
        extent: list = None,
        fname: str = "azimuthal_equidistant_multi.pdf",
        vmin: float = None,
        vmax: float = None,
        cbar_label: str = None,
        wedge: str = None,
        info_vals_dict: dict = None,
        ncols: int = 4,
    ) -> None:
        """Multi-panel azimuthal-equidistant plot with shared colorbar.

        Parameters
        ----------
        data_dict:
            ``{display_label: xr.DataArray}`` – one panel per entry.
        fname:
            Output filename relative to ``self.output_path``.
        ncols:
            Maximum columns in the subplot grid.
        All other parameters mirror :meth:`azimuthal_equidistant`.
        """
        azimuthal_equidistant_multi_plot(
            data_dict=data_dict,
            central_latitude=central_latitude,
            central_longitude=central_longitude,
            extent=extent,
            fpath=self._fpath(fname),
            vmin=vmin,
            vmax=vmax,
            cbar_label=cbar_label,
            wedge=wedge,
            info_vals_dict=info_vals_dict,
            ncols=ncols,
        )

    def lambert_conformal(
        self,
        data,
        *,
        central_latitude: float = 55,
        central_longitude: float = 0,
        extent: list = None,
        fname: str = "lambert_conformal.pdf",
        info_vals: dict = None,
        levels=None,
        lat_cutoff: float = -30,
        cut_boundary: bool = True,
    ) -> None:
        """
        Lambert-conformal conic projection plot.

        Suited to mid-latitude regional domains (NAO region, Europe, etc.).

        Parameters
        ----------
        data:
            ``xr.DataArray`` with ``lat`` and ``lon`` coordinates.
        central_latitude / central_longitude:
            Standard parallel / central meridian.
        extent:
            ``[lon_min, lon_max, lat_min, lat_max]``.
        fname:
            Output filename.
        info_vals:
            Annotation dict shown below the axes.
        levels:
            Explicit contour levels.  When *None* Matplotlib chooses them.
        lat_cutoff:
            Southern cut-off latitude for the conic (negative = southern
            hemisphere limit).
        cut_boundary:
            When *True* the axes boundary is clipped to the conic shape.
        """
        lambert_conformal_projection_plot(
            data=data,
            central_latitude=central_latitude,
            central_longitude=central_longitude,
            extent=extent,
            fpath=self._fpath(fname),
            info_vals=info_vals,
            levels=levels,
            lat_cutoff=lat_cutoff,
            cut_boundary=cut_boundary,
        )

    def robinson(
        self,
        data,
        *,
        fname: str = "robinson.pdf",
        title: str = None,
        cbar_label: str = None,
        vmin: float = None,
        vmax: float = None,
        norm=None,
        infotext: str = "",
        cmap: str = None,
    ) -> None:
        """
        Global Robinson-projection imshow plot.

        Parameters
        ----------
        data:
            ``xr.DataArray`` or ``np.ndarray`` (2-D).
        fname:
            Output filename.
        title / cbar_label / vmin / vmax / norm / infotext:
            Forwarded to :pyfunc:`plot.functional.spatial.imshow`.
        cmap:
            Override instance colormap for this call.
        """
        _imshow(
            x=data.values if hasattr(data, "values") else data,
            output_path=self._fpath(fname),
            title=title,
            cbar_label=cbar_label,
            cmap=cmap if cmap is not None else self.cmap,
            projection="Robinson",
            figsize=self.figsize,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            infotext=infotext,
        )

    def plate_carree(
        self,
        x,
        y,
        z,
        *,
        fname: str = "plate_carree.pdf",
        cbar_label: str = None,
        add_contourlines: bool = False,
        cmap: str = None,
        extent: list = None,
        **kwargs,
    ) -> None:
        """
        Global Plate-Carree contourf plot.

        Parameters
        ----------
        x / y / z:
            Longitude, latitude, and data arrays.
        fname:
            Output filename.
        cbar_label:
            Colorbar label.
        add_contourlines:
            Overlay thin black contour lines when *True*.
        cmap:
            Override instance colormap for this call.
        **kwargs:
            Additional kwargs forwarded to
            :pyfunc:`plot.functional.spatial.contourf`.
        """
        _contourf(
            x=x,
            y=y,
            z=z,
            output_path=self._fpath(fname),
            add_contourlines=add_contourlines,
            cbar_label=cbar_label,
            cmap=cmap if cmap is not None else self.cmap,
            figsize=self.figsize,
            extent=extent,
            **kwargs,
        )
