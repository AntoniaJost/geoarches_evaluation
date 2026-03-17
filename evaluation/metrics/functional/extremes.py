"""
metrics/functional/extremes.py
===============================
TC (Tropical Cyclone) detection, tracking, and frequency helpers.

Used by :class:`~metrics.module.TropicalCycloneFrequency`.

Detection algorithm
-------------------
1. **SLP minimum** – a local minimum of sea-level pressure in the western North
   Pacific basin (100 °E – 160 °E, 5 °N – 35 °N) drops below at least one of
   the four daily-sampled proxy-percentile thresholds:
   1 000, 994, 985, 975 hPa  (≈ 1st, 0.1th, 0.01th, 0.001th percentile).

2. **Warm-core criterion** – the geopotential-height layer thickness between
   300 hPa and 700 hPa at the candidate grid point exceeds the seasonally
   varying climatological value for the same day-of-year.

3. **Track continuity** – adjacent centres along the same track are separated
   by no more than 2 days in time *and* 3 ° in great-circle distance.
   Tracks with fewer than 3 data points are discarded.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.ndimage import minimum_filter

# ---------------------------------------------------------------------------
# Domain & algorithm constants
# ---------------------------------------------------------------------------

#: Western North Pacific detection domain (lon_min, lon_max, lat_min, lat_max)
WNP_LON: tuple[float, float] = (100.0, 160.0)
WNP_LAT: tuple[float, float] = (5.0, 35.0)

#: SLP proxy-percentile thresholds in hPa (daily data, descending intensity).
TC_SLP_THRESHOLDS_HPA: np.ndarray = np.array([1000.0, 994.0, 985.0, 975.0])

#: Intensity-category labels matching each threshold.
TC_INTENSITY_LABELS: list[str] = [
    "< 1000 hPa (TD)",
    "< 994 hPa (TS)",
    "< 985 hPa (TY)",
    "< 975 hPa (STY)",
]

#: Maximum time gap between consecutive track points (days).
MAX_TIME_GAP_DAYS: int = 2

#: Maximum spatial distance between consecutive track points (degrees).
MAX_SPATIAL_DEG: float = 3.0

#: Minimum number of data points for a valid track.
MIN_TRACK_POINTS: int = 3

#: Size of the neighbourhood (grid cells) used for SLP local-minimum detection.
_LOCAL_MIN_SIZE: int = 5


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def detect_tc_candidates(
    slp_da: xr.DataArray,
    z_da: xr.DataArray,
    clim_z_da: xr.DataArray,
) -> list[dict]:
    """Detect TC candidate centres on every available time step.

    Parameters
    ----------
    slp_da :
        Sea-level pressure in **Pa**, shape ``(time, lat, lon)``.
    z_da :
        Geopotential in m²/s², shape ``(time, level, lat, lon)``.
        Must contain at least the 300 hPa and 700 hPa pressure levels.
    clim_z_da :
        Climatological (day-of-year × level × lat × lon) geopotential in the
        same units as *z_da*.  The ``dayofyear`` dimension must run from 1 to
        365 (or 366).

    Returns
    -------
    list[dict]
        One dict per candidate with keys:
        ``time, dayofyear, year, lat, lon, slp_pa, slp_hpa,
        thick_anom, intensity``.
        *intensity* is an integer 1–4 counting how many thresholds the
        minimum SLP falls below.
    """
    slp_da = _normalise_dims(slp_da)
    z_da = _normalise_dims(z_da)
    clim_z_da = _normalise_dims(clim_z_da)

    # Restrict to WNP domain ------------------------------------------------
    slp_wnp = slp_da.sel(lon=slice(*WNP_LON), lat=slice(*WNP_LAT))
    z_wnp = z_da.sel(lon=slice(*WNP_LON), lat=slice(*WNP_LAT))
    clim_wnp = clim_z_da.sel(lon=slice(*WNP_LON), lat=slice(*WNP_LAT))

    # Pre-compute climatological thickness per day-of-year ------------------
    lev300 = _nearest_level(clim_wnp, 300.0)
    lev700 = _nearest_level(clim_wnp, 700.0)
    clim_thick = lev300 - lev700  # (dayofyear, lat, lon)

    lat_arr = slp_wnp.lat.values
    lon_arr = slp_wnp.lon.values

    candidates: list[dict] = []

    for t in slp_wnp.time.values:
        slp_t = slp_wnp.sel(time=t).values.squeeze()   # (lat, lon)
        slp_hpa_t = slp_t / 100.0

        # 1) SLP local minimum in 5×5 neighbourhood
        min_filt = minimum_filter(slp_hpa_t, size=_LOCAL_MIN_SIZE)
        is_local_min = slp_hpa_t == min_filt

        # Below at least the weakest threshold
        below_thr = slp_hpa_t < TC_SLP_THRESHOLDS_HPA[0]
        slp_mask = is_local_min & below_thr

        if not slp_mask.any():
            continue

        t_da = xr.DataArray(t)
        doy = int(t_da.dt.dayofyear)
        year = int(t_da.dt.year)

        # 2) Warm-core: thickness anomaly > 0
        z300_t = _nearest_level(z_wnp.sel(time=t), 300.0).values.squeeze()   # (lat, lon)
        z700_t = _nearest_level(z_wnp.sel(time=t), 700.0).values.squeeze()
        thick_t = z300_t - z700_t

        clim_t = clim_thick.sel(dayofyear=doy, method="nearest").values.squeeze()
        thick_anom = thick_t - clim_t
        warm_core = thick_anom > 0

        combined = slp_mask & warm_core
        ilats, ilons = np.where(combined)

        for ilat, ilon in zip(ilats, ilons):
            slp_val_hpa = float(slp_hpa_t[ilat, ilon])
            intensity = int(np.sum(slp_val_hpa < TC_SLP_THRESHOLDS_HPA))
            candidates.append(
                dict(
                    time=t,
                    dayofyear=doy,
                    year=year,
                    lat=float(lat_arr[ilat]),
                    lon=float(lon_arr[ilon]),
                    slp_pa=float(slp_t[ilat, ilon]),
                    slp_hpa=slp_val_hpa,
                    thick_anom=float(thick_anom[ilat, ilon]),
                    intensity=intensity,
                )
            )

    return candidates


def build_tc_tracks(
    candidates: list[dict],
    time_step_hours: float = 24.0,
) -> list[list[dict]]:
    """Link TC candidate centres into continuous trajectories.

    Tracking rules
    --------------
    * Consecutive centres must be ≤ :data:`MAX_TIME_GAP_DAYS` apart in time.
    * Consecutive centres must be ≤ :data:`MAX_SPATIAL_DEG` apart
      (great-circle approximation in degrees).
    * Tracks with fewer than :data:`MIN_TRACK_POINTS` points are discarded.

    Uses a greedy nearest-neighbour approach: at each time step every open
    track is extended with the closest unmatched candidate that satisfies both
    constraints.

    Parameters
    ----------
    candidates :
        Output of :func:`detect_tc_candidates`, must be time-sorted.
    time_step_hours :
        Data temporal resolution in hours (default 24 = daily).

    Returns
    -------
    list[list[dict]]
        Each element is a list of candidate dicts forming one track,
        ordered chronologically.
    """
    if not candidates:
        return []

    # Sort by time and group ------------------------------------------------
    candidates = sorted(candidates, key=lambda c: c["time"])
    times = sorted({c["time"] for c in candidates})
    by_time: dict = {t: [c for c in candidates if c["time"] == t] for t in times}

    max_gap_steps = max(1, int(round(MAX_TIME_GAP_DAYS * 24.0 / time_step_hours)))

    tracks: list[list[dict]] = []
    active: list[list[dict]] = []  # open tracks

    for i, t in enumerate(times):
        step_cands = list(by_time[t])
        matched_cand: set[int] = set()
        still_active: list[list[dict]] = []

        for track in active:
            last = track[-1]
            last_idx = times.index(last["time"])
            if i - last_idx > max_gap_steps:
                tracks.append(track)   # too old – close it
                continue

            # Nearest unmatched candidate within spatial threshold
            best_dist = np.inf
            best_j: int | None = None
            for j, cand in enumerate(step_cands):
                if j in matched_cand:
                    continue
                d = _great_circle_deg(
                    last["lat"], last["lon"], cand["lat"], cand["lon"]
                )
                if d < MAX_SPATIAL_DEG and d < best_dist:
                    best_dist = d
                    best_j = j

            if best_j is not None:
                track.append(step_cands[best_j])
                matched_cand.add(best_j)

            still_active.append(track)

        # Unmatched candidates start new tracks
        for j, cand in enumerate(step_cands):
            if j not in matched_cand:
                still_active.append([cand])

        active = still_active

    tracks.extend(active)

    # Filter by minimum length ----------------------------------------------
    return [tr for tr in tracks if len(tr) >= MIN_TRACK_POINTS]


def compute_tc_frequency(
    tracks: list[list[dict]],
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
) -> np.ndarray:
    """2-D TC-passage frequency histogram (track-point counts per grid cell).

    Parameters
    ----------
    tracks :
        Output of :func:`build_tc_tracks`.
    lat_bins, lon_bins :
        Bin edges defining the output grid.

    Returns
    -------
    np.ndarray, shape ``(len(lat_bins)-1, len(lon_bins)-1)``
    """
    freq = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1), dtype=float)
    for track in tracks:
        for pt in track:
            ilat = int(np.searchsorted(lat_bins, pt["lat"])) - 1
            ilon = int(np.searchsorted(lon_bins, pt["lon"])) - 1
            if 0 <= ilat < freq.shape[0] and 0 <= ilon < freq.shape[1]:
                freq[ilat, ilon] += 1.0
    return freq


def compute_tc_count_per_year(tracks: list[list[dict]]) -> dict[int, int]:
    """Count unique TC tracks per calendar year (attributed to first point).

    Returns
    -------
    dict mapping year (int) → count (int).
    """
    counts: dict[int, int] = {}
    for track in tracks:
        year = track[0]["year"]
        counts[year] = counts.get(year, 0) + 1
    return counts


def compute_tc_count_by_intensity(
    tracks: list[list[dict]],
    thresholds: np.ndarray = TC_SLP_THRESHOLDS_HPA,
) -> dict[int, int]:
    """Count tracks by peak intensity (number of SLP thresholds exceeded).

    Parameters
    ----------
    tracks :
        Output of :func:`build_tc_tracks`.
    thresholds :
        SLP thresholds in hPa (descending intensity).

    Returns
    -------
    dict mapping intensity category (1-based int) → count.
    Category *k* means the minimum SLP fell below *thresholds[k-1]*.
    """
    n = len(thresholds)
    counts: dict[int, int] = {i + 1: 0 for i in range(n)}
    for track in tracks:
        min_slp = min(pt["slp_hpa"] for pt in track)
        for i, thr in enumerate(thresholds):
            if min_slp < thr:
                counts[i + 1] += 1
    return counts


def compute_clim_z_dayofyear(z_da: xr.DataArray) -> xr.DataArray:
    """Compute day-of-year mean climatology for geopotential.

    Parameters
    ----------
    z_da :
        Geopotential DataArray with at least a ``time`` dimension.

    Returns
    -------
    xr.DataArray
        Same dimensions as *z_da* minus ``time``, plus a new ``dayofyear``
        dimension (1–365).
    """
    z_da = _normalise_dims(z_da)
    return z_da.groupby("time.dayofyear").mean("time")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalise_dims(da: xr.DataArray) -> xr.DataArray:
    """Rename ``longitude`` → ``lon`` and ``latitude`` → ``lat`` if needed."""
    rename: dict[str, str] = {}
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    return da.rename(rename) if rename else da


def _nearest_level(da: xr.DataArray, level_hpa: float) -> xr.DataArray:
    """Select the level nearest to *level_hpa* regardless of unit convention."""
    return da.sel(level=level_hpa, method="nearest")


def _great_circle_deg(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Spherical-law-of-cosines distance between two points in **degrees**."""
    rlat1, rlon1, rlat2, rlon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    cos_c = np.sin(rlat1) * np.sin(rlat2) + np.cos(rlat1) * np.cos(rlat2) * np.cos(
        rlon2 - rlon1
    )
    return float(np.rad2deg(np.arccos(np.clip(cos_c, -1.0, 1.0))))
