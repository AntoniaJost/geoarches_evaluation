"""
plot/functional/stats.py
========================
Low-level drawing helpers for statistical comparison diagrams.

Currently provides:
    taylor_diagram_to_ax  – draw a Taylor diagram on an existing Axes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.projections.polar import PolarAxes
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.grid_finder as grid_finder


# ---------------------------------------------------------------------------
# Taylor diagram
# ---------------------------------------------------------------------------

def _make_taylor_axes(fig, rect=111, r_max=1.65):
    """Create a pair of curved (Floating) axes suited for a Taylor diagram.

    The diagram lives in the first quadrant with:
      * radial axis  = normalised standard deviation  (σ_model / σ_ref)
      * angular axis = arccos(r)  →  r = 1 at θ=0, r = 0 at θ=π/2

    Returns ``(ax, aux_ax)`` where *aux_ax* is the Polar Axes that callers
    use for actual drawing.
    """
    tr = PolarAxes.PolarTransform()

    # Correlation tick locations and their matching angle labels
    r_ticks = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    theta_ticks = [np.arccos(r) for r in r_ticks]
    theta_labels = [str(r) for r in r_ticks]

    gl1 = grid_finder.FixedLocator(theta_ticks)
    tf1 = grid_finder.DictFormatter(dict(zip(theta_ticks, theta_labels)))

    gl2 = grid_finder.MaxNLocator(5)

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr,
        extremes=(0, np.pi / 2, 0, r_max),
        grid_locator1=gl1,
        grid_locator2=gl2,
        tick_formatter1=tf1,
        tick_formatter2=None,
    )

    ax = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax)

    # Orient axes labels
    ax.axis["top"].set_axis_direction("bottom")
    ax.axis["top"].toggle(ticklabels=True, label=True)
    ax.axis["top"].major_ticklabels.set_axis_direction("top")
    ax.axis["top"].label.set_axis_direction("top")
    ax.axis["top"].label.set_text("Correlation")

    ax.axis["left"].set_axis_direction("bottom")
    ax.axis["left"].label.set_text("Normalised std. deviation")

    ax.axis["right"].set_axis_direction("top")
    ax.axis["right"].toggle(ticklabels=True)
    ax.axis["right"].major_ticklabels.set_axis_direction("left")

    ax.axis["bottom"].set_visible(False)

    aux_ax = ax.get_aux_axes(tr)
    aux_ax.patch = ax.patch
    ax.patch.zorder = 0.9

    return ax, aux_ax


def taylor_diagram_to_ax(
    fig,
    rect,
    model_stats: dict,
    colors: dict = None,
    markers: dict = None,
    marker_size: int = 10,
    ref_label: str = "Reference",
    title: str = "",
    r_max: float = 1.65,
    crmse_levels: int = 5,
):
    """Draw a normalised Taylor diagram.

    Parameters
    ----------
    fig:
        Matplotlib figure to draw into.
    rect:
        Subplot specifier (e.g. ``111``).
    model_stats:
        ``{model_label: (r, normalised_std)}`` – Pearson r and
        (σ_model / σ_reference).
    colors:
        Per-model colour overrides.
    markers:
        Per-model marker overrides.
    marker_size:
        Default marker size.
    ref_label:
        Label for the reference point (placed at r=1, σ_norm=1).
    title:
        Axes title.
    r_max:
        Radial extent of the diagram (normalised std).  Should be slightly
        larger than the largest normalised std in *model_stats*.
    crmse_levels:
        Number of centred-RMSE arcs to draw.

    Returns
    -------
    ax, aux_ax
        The FloatingSubplot and its auxiliary Polar Axes.
    """
    colors = colors or {}
    markers = markers or {}

    ax, aux_ax = _make_taylor_axes(fig, rect=rect, r_max=r_max)

    if title:
        ax.set_title(title, pad=12)

    # -- Reference point (r=1, σ_norm=1) ------------------------------------
    aux_ax.plot(
        0.0, 1.0,
        marker="*",
        color="black",
        markersize=marker_size + 2,
        zorder=5,
        label=ref_label,
        clip_on=False,
    )

    # -- Reference std arc (σ_norm = 1 dashed circle) -----------------------
    theta_ref = np.linspace(0, np.pi / 2, 200)
    aux_ax.plot(theta_ref, np.ones_like(theta_ref), "k--", linewidth=0.75, alpha=0.5)

    # -- Centred-RMSE arcs centred on the reference point (theta=0, r=1) -----
    # In polar (theta, r) coords, the reference point in Cartesian is (1, 0).
    # CRMSE iso-contours are circles: x² + (y-0)² = E², i.e. centred at (1,0).
    # We need to convert back to polar for plotting.
    rmse_vals = np.linspace(r_max / (crmse_levels + 1), r_max, crmse_levels, endpoint=False)
    for E in rmse_vals:
        # Parametric circle of radius E centred at Cartesian (1, 0):
        #   xc = 1 + E*cos(phi),  yc = E*sin(phi)
        phi = np.linspace(0, np.pi, 500)
        xc = 1.0 + E * np.cos(phi)
        yc = E * np.sin(phi)
        # Convert to polar: r2=sqrt(xc²+yc²), theta2=atan2(yc,xc)
        r2 = np.sqrt(xc ** 2 + yc ** 2)
        theta2 = np.arctan2(yc, xc)
        # Keep only the part inside the diagram (theta in [0, pi/2], r in [0, r_max])
        mask = (theta2 >= 0) & (theta2 <= np.pi / 2) & (r2 <= r_max)
        if mask.sum() > 1:
            aux_ax.plot(
                theta2[mask], r2[mask],
                color="green", linewidth=0.6, linestyle="--", alpha=0.5,
            )
            # Label the outermost visible point
            idx = np.where(mask)[0][-1]
            aux_ax.text(
                theta2[idx], r2[idx],
                f"{E:.2f}",
                fontsize=7, color="green", alpha=0.7,
                ha="left", va="bottom",
            )

    # -- Radial spokes at round r-values nearest to each data point ---------
    # Candidate "round" r-values that may serve as spoke angles.
    _r_candidates = np.array(
        [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    )
    _spoken_r = set()
    for _, (r_val, _) in model_stats.items():
        r_clipped = float(np.clip(r_val, _r_candidates[0], _r_candidates[-1]))
        nearest = _r_candidates[np.argmin(np.abs(_r_candidates - r_clipped))]
        _spoken_r.add(float(nearest))

    for r_spoke in sorted(_spoken_r):
        theta_spoke = np.arccos(r_spoke)
        aux_ax.plot(
            [theta_spoke, theta_spoke], [0, r_max],
            color="grey", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1,
        )

    # -- Model points -------------------------------------------------------
    _default_markers = ["o", "s", "^", "D", "v", "P", "X", "h"]
    for i, (label, (r, std_norm)) in enumerate(model_stats.items()):
        theta = np.arccos(np.clip(r, -1.0, 1.0))
        color = colors.get(label, f"C{i}")
        marker = markers.get(label, _default_markers[i % len(_default_markers)])
        aux_ax.plot(
            theta, std_norm,
            marker=marker,
            color=color,
            markersize=marker_size,
            linestyle="none",
            label=label,
            clip_on=False,
            zorder=5,
        )
        # Annotate centred RMSE next to each point
        crmse = np.sqrt(1.0 + std_norm ** 2 - 2.0 * std_norm * r)

        # Add some padding to the annotation position to avoid overlap with the marker
        aux_ax.annotate(
            f"  E={crmse:.2f}",
            xy=(theta, std_norm),
            fontsize=7,
            color=color,
            va="bottom",
            ha="left",
            
        )

    return ax, aux_ax
