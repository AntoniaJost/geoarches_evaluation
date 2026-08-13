"""
metrics/functional/return_periods.py
=====================================
Block-maxima extreme value analysis (GEV) helpers.

Used by :class:`~metrics.module.ReturnPeriods` to estimate return levels
(e.g. "the daily domain-max temperature expected once every 50 years") from
annual block maxima, pooled across ensemble members.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import stats


def domain_max_series(data: xr.DataArray, spatial_dims=("lat", "lon")) -> xr.DataArray:
    """Reduce *data* to a 1-D time series of the spatial maximum at every time step."""
    dims = [d for d in spatial_dims if d in data.dims]
    return data.max(dim=dims) if dims else data


def annual_block_maxima(data: xr.DataArray, time_dim: str = "time") -> np.ndarray:
    """Return one maximum value per calendar year (block maxima)."""
    return data.groupby(f"{time_dim}.year").max(time_dim).values


def fit_gev(samples: np.ndarray) -> tuple[float, float, float]:
    """Fit a GEV distribution to *samples* via MLE.

    Returns ``(shape, loc, scale)`` using :func:`scipy.stats.genextreme`
    conventions (``shape`` = 0 recovers a Gumbel distribution).
    """
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    shape, loc, scale = stats.genextreme.fit(samples)
    return shape, loc, scale


def return_levels(return_periods: np.ndarray, shape: float, loc: float, scale: float) -> np.ndarray:
    """Return the GEV return level for each value in *return_periods* (in years)."""
    return_periods = np.asarray(return_periods, dtype=float)
    exceedance_prob = 1.0 / return_periods
    return stats.genextreme.ppf(1.0 - exceedance_prob, shape, loc=loc, scale=scale)


def empirical_return_periods(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sorted_values, empirical_return_periods)`` via Weibull plotting positions."""
    sorted_vals = np.sort(np.asarray(samples, dtype=float))
    n = len(sorted_vals)
    ranks = np.arange(1, n + 1)
    exceedance_rank = n - ranks + 1  # largest value -> longest return period
    periods = (n + 1) / exceedance_rank
    return sorted_vals, periods
