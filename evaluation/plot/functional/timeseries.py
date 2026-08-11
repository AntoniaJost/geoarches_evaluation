"""
plot/functional/timeseries.py
=============================
Low-level matplotlib helpers for time-series plots.  All functions that draw
onto an axis accept a pre-created ``ax`` and return nothing so they can be
freely composed by higher-level plotters.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Tick-spacing helper
# ---------------------------------------------------------------------------

def get_xlabel_multiplier(n_xticks: int) -> int:
    """Return a step size so that the x-axis has a readable number of ticks."""
    if n_xticks in (365, 366):
        return 30
    if n_xticks <= 12:
        return 1
    if n_xticks <= 24:
        return 2
    if n_xticks <= 36:
        return 3
    if n_xticks <= 48:
        return 4
    if n_xticks <= 120:
        return 6
    if n_xticks <= 240:
        return 12
    if n_xticks <= 360:
        return 30
    return 48


# ---------------------------------------------------------------------------
# Fill helpers
# ---------------------------------------------------------------------------

def _fill_sign(ax, data: np.ndarray) -> None:
    """Fill positive values orange and negative values pale-turquoise."""
    time_ids = np.arange(len(data))
    ax.fill_between(time_ids, 0, np.where(data > 0, data, 0), color="orange")
    ax.fill_between(time_ids, 0, np.where(data < 0, data, 0), color="paleturquoise")
    ax.hlines(0, time_ids[0], time_ids[-1], color="k", linewidth=0.5, linestyle="-")


def fill_timeseries(
    ax,
    x: np.ndarray,
    std: np.ndarray = None,
    fill_type: str = "full",
    alpha: float = 0.2,
    color: str = "gray",
) -> None:
    """Apply an area fill to a timeseries already drawn on *ax*.

    Parameters
    ----------
    fill_type:
        ``"full"`` – symmetric ±std band (requires *std*).
        ``"sign"`` – positive/negative colour fill (no *std* needed).
        Any other value – no-op.
    """
    if fill_type == "full":
        if std is None:
            raise ValueError("std must be provided for fill_type='full'.")
        ax.fill_between(range(len(x)), x - std, x + std, color=color, alpha=alpha)
    elif fill_type == "sign":
        _fill_sign(ax, x)


# ---------------------------------------------------------------------------
# Stem plot
# ---------------------------------------------------------------------------

def stem_plot(ax, x: np.ndarray, label: str = "") -> None:
    """Draw a stem plot with blue positive and red negative stems.

    Parameters
    ----------
    ax:
        Matplotlib axes to draw on.
    x:
        1-D data array.
    label:
        Legend label applied to the positive stems (represents the whole series).
    """
    indices = np.arange(len(x))
    stem_lw = 0.5

    pos_mask = x >= 0
    neg_mask = x < 0

    if pos_mask.any():
        markerline, stemlines, _ = ax.stem(
            indices[pos_mask], x[pos_mask], basefmt=" ", label=label,
        )
        plt.setp(stemlines, color="steelblue", linewidth=stem_lw)
        plt.setp(markerline, color="steelblue", markersize=0.0)

    if neg_mask.any():
        markerline, stemlines, _ = ax.stem(
            indices[neg_mask], x[neg_mask], basefmt=" ",
        )
        plt.setp(stemlines, color="tomato", linewidth=stem_lw)
        plt.setp(markerline, color="tomato", markersize=0.0)

    ax.axhline(0, color="k", linewidth=0.5, linestyle="-")


# ---------------------------------------------------------------------------
# Generic line time-series
# ---------------------------------------------------------------------------

def timeseries_to_ax(
    ax,
    x,
    y: np.ndarray = None,
    color: str = "black",
    linear_trend: np.ndarray = None,
    std: np.ndarray = None,
    linewidth: float = 2.0,
    linestyle: str = "-",
    fill: str = None,
    label: str = "",
    marker=None,
    fill_alpha: float = 0.2,
) -> None:
    """Draw a single time-series line (and optional fill / trend) onto *ax*.

    Parameters
    ----------
    x:
        x-positions (indices or time values).  If *y* is ``None``, *x* is
        treated as the data and plotted against its own integer indices.
    y:
        Data values.  When supplied alongside *x*, *x* provides positions.
    color:
        Line colour.
    linear_trend:
        Pre-computed trend line array (plotted as a dashed overlay when given).
    std:
        Standard-deviation array used for ``fill="full"``.
    linewidth:
        Main line width.
    linestyle:
        Main line style.
    fill:
        ``"full"`` for ±std band, ``"sign"`` for pos/neg colour fill, or
        ``None`` for no fill.
    label:
        Legend label.
    marker:
        Marker style string (e.g. ``"o"``).
    fill_alpha:
        Transparency for the ``"full"`` fill band.
    """
    alpha = 0.7 if linear_trend is not None else 1.0
    data = y if y is not None else x
    positions = x if y is not None else range(len(x))

    ax.plot(
        positions, data,
        color=color, linewidth=linewidth,
        label=label, marker=marker, linestyle=linestyle, alpha=alpha,
    )

    if std is not None and fill is None:
        fill = "full"
    if fill is not None:
        fill_timeseries(ax, data, std, fill_type=fill, alpha=fill_alpha, color=color)

    if linear_trend is not None:
        ax.plot(positions, linear_trend, linestyle="--", color=color, linewidth=linewidth)

