"""metrics package – re-exports all public names for convenient access."""

from metrics.base import (
    BaseMetric,
    SpatialMetric,
    TimeseriesMetric,
    annual_mean,
    seasonal_mean,
    instantaneous,
    select_by_time,
    compute_latitude_weights,
    compute_anomaly,
    compute_bias,
    compute_eof,
    compute_correlation_matrix,
    compute_soi,
    detrend_data,
)
from metrics.module import (
    XYMaps,
    LatTimeMap,
    XYBiasMaps,
    XYAnomalyMaps,
    Timeseries,
    SeasonalCycles,
    MonsoonIndices,
    SouthernOscillationIndex,
    EOF,
    AnnularModes,
    NorthernAnnularMode,
    NorthernAtlanticOscillationIndex,
    RadialSpectrum,
    Distribution,
    Histogram,
)

__all__ = [
    # Base classes
    "BaseMetric",
    "SpatialMetric",
    "TimeseriesMetric",
    # Utility functions
    "annual_mean",
    "seasonal_mean",
    "instantaneous",
    "select_by_time",
    "compute_latitude_weights",
    "compute_anomaly",
    "compute_bias",
    "compute_eof",
    "compute_correlation_matrix",
    "compute_soi",
    "detrend_data",
    # Spatial map metrics
    "XYMaps",
    "LatTimeMap",
    "XYBiasMaps",
    "XYAnomalyMaps",
    # Timeseries metrics
    "Timeseries",
    "SeasonalCycles",
    "MonsoonIndices",
    "SouthernOscillationIndex",
    # EOF / circulation modes
    "EOF",
    "AnnularModes",
    "NorthernAnnularMode",
    "NorthernAtlanticOscillationIndex",
    # Spectral
    "RadialSpectrum",
    # Distribution
    "Distribution",
    "Histogram",
]
