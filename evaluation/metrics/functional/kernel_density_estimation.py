import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import linregress
import matplotlib.gridspec as gridspec
import xarray as xr


def _kde_1d(data: np.ndarray, bandwidth: str | float = "scott"):
    """
    Compute 1D Kernel Density Estimate for `data` array.
    Returns the KDE object.
    """
    kde = gaussian_kde(data, bw_method=bandwidth)

    return kde

def compute_1d_pdf_over_grid(data, ygrid_bounds):
    """
    Compute PDFs over a common y-grid for a list of KDE operators.
    """

    ygrid = np.linspace(*ygrid_bounds, 400)
    pdf = _kde_1d(data.values)

    return pdf

def _get_ygrid_bounds(anom_values):
    """
    Get min and max values over a list of anomaly arrays.
    """

    vmin = float(min(v.min() for v in anom_values))
    vmax = float(max(v.max() for v in anom_values))

    return vmin, vmax


def _global_monthly_anomalies(ds: xr.Dataset, var: str, level, base_period):
    """
    Global-mean monthly anomalies for `var` (optionally at pressure `level`),
    subtracting the month-of-year climatology computed over `base_period`.
    """
    x = ds[var] if level is None else ds[var].sel(level=level)
    # global mean
    x = x.mean(["latitude", "longitude"], skipna=True)
    # monthly mean series (start-of-month)
    xm = x.resample(time="MS").mean()
    # baseline climatology over base_period (per month)
    base = xm.sel(time=slice(f"{base_period[0]}-01-01", f"{base_period[1]}-12-31"))
    # check if base period is empty
    if base.sizes.get("time", 0) == 0:
        print(
            f"Base period {base_period[0]}–{base_period[1]} "
            f"not found in dataset (time range {xm.time.min().values} to {xm.time.max().values}). "
            "Skipping anomaly calculation and returning empty array."
        )
        return xr.full_like(xm, np.nan)
    clim = base.groupby("time.month").mean("time", skipna=True)
    # anomaly relative to that month’s climatology
    anom = xm.groupby("time.month") - clim
    return anom.dropna("time")


# fitting a linear regression to the time series and subtracting it from the fit
def _detrend_series(anom):
    """
    Removes linear trend from 1D time series
    """

    years = anom["time"].dt.year + (anom["time"].dt.dayofyear - 1) / 365.25
    slope, intercept, *_ = linregress(years, anom.values)
    detrended = anom - (slope * years + intercept)

    return detrended


def variability_kde_timeseries(
    data: xr.Dataset,
    era5: xr.Dataset | None,
    variable: str,
    level: int | None,
    base_period: tuple[int, int],
    output_path: str,
    label_model: str = "ArchesWeather",
    include_era5: bool = True,
    bandwidth: str | float = "scott",
    output_fname: str = "variability_kde_timeseries",
    detrend: bool = False,
):
    """
    Make a two-panel figure:
      (a) KDE of monthly global-mean anomalies
      (b) Timeseries of those anomalies

    Saves: <output_path>/variability/plots/<output_fname>.png
    """
    if include_era5 and era5 is None:
        raise RuntimeError("ERA5 is required (set era5_path) when include_era5=True.")

    # anomalies for model (relative to ERA5 climatology period)
    model_anom = _global_monthly_anomalies(data, variable, level, base_period)

    # anomalies for ERA5
    era5_anom = (
        _global_monthly_anomalies(era5, variable, level, base_period)
        if include_era5
        else None
    )

    # remove all-NaN case early
    if model_anom.isnull().all():
        print(f"No valid anomalies for {label_model}, skipping plot.")
        return
    if include_era5 and era5_anom is not None and era5_anom.isnull().all():
        print("No valid anomalies for ERA5, skipping plot.")
        return

    # optionally detrend
    if detrend:
        model_anom = _detrend_series(model_anom)
        if era5_anom is not None:
            era5_anom = _detrend_series(era5_anom)

    # remove NaNs for KDE
    model_vals = model_anom.dropna("time").values
    if model_vals.size == 0:
        print("Model anomalies empty after NaN removal, skipping plot.")
        return

    if era5_anom is not None:
        era5_vals = era5_anom.dropna("time").values
        if era5_vals.size == 0:
            print("ERA5 anomalies empty after NaN removal, skipping plot.")
            return
    else:
        era5_vals = None

    # y-grid for KDE
    vals = [model_anom.values]
    if era5_anom is not None:
        vals.append(era5_anom.values)
    lo = float(min(v.min() for v in vals))
    hi = float(max(v.max() for v in vals))
    ygrid = np.linspace(lo, hi, 400)

    # KDEs
    model_kde = gaussian_kde(model_anom.values, bw_method=bandwidth)
    model_pdf = model_kde(ygrid)
    if era5_anom is not None:
        era5_kde = gaussian_kde(era5_anom.values, bw_method=bandwidth)
        era5_pdf = era5_kde(ygrid)

def plot_variability_kde_timeseries(
        pdfs, anomalies, ygrid, variable, labels, output_path, 
        output_fname, colors
    ):
    
    # make figure
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)

    # (a) KDE plot
    ax0 = fig.add_subplot(gs[0])
    for i, pdf in enumerate(pdfs):
        ax0.plot(
            pdf,
            ygrid,
            label=labels[i],
            color=colors[i],
    )

    ax0.set_ylabel(f"{variable} anomaly")
    ax0.set_xlabel("Density")
    ax0.set_title("(a) Kernel Density Estimate of Monthly Global-Mean Anomalies")
    ax0.legend()
    ax0.grid()

    # (b) Timeseries plot
    ax1 = fig.add_subplot(gs[1])
    for i, anom in enumerate(anomalies):
        ax1.plot(
            anom["time"],
            anom.values,
            label=labels[i],
            color=colors[i],
    )

    ax1.set_ylabel(f"{variable} anomaly")
    ax1.set_xlabel("Time")
    ax1.set_title("(b) Monthly Global-Mean Anomalies Time Series")
    ax1.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax1.legend()
    ax1.grid()

    # save figure
    output_dir = os.path.join(output_path, "variability", "plots")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{output_fname}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
