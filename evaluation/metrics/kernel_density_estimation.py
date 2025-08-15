import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import linregress
import matplotlib.gridspec as gridspec
import xarray as xr

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
    years = anom['time'].dt.year + (anom['time'].dt.dayofyear - 1) / 365.25
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
    detrend:bool = False,
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
    era5_anom = _global_monthly_anomalies(era5, variable, level, base_period) if include_era5 else None

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

    # figure layout
    outdir = os.path.join(output_path, "variability", "plots")
    os.makedirs(outdir, exist_ok=True)
    fig = plt.figure(figsize=(12, 4), dpi=150)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4], wspace=0.25)

    # (a) KDE panel: density (x) vs anomaly (y)
    ax_kde = fig.add_subplot(gs[0, 0])
    if era5_anom is not None:
        ax_kde.plot(era5_pdf, ygrid, color="black", label="ERA5")
    ax_kde.plot(model_pdf, ygrid, color="red", label=label_model)
    lab_var = f"{variable}" + (f" at {level} hPa" if level is not None else "")
    ax_kde.set_xlabel("Density")
    ax_kde.set_ylabel(f"{lab_var} anomaly (K)")
    ax_kde.grid(True, alpha=0.3)
    ax_kde.legend(loc="best", frameon=False)

    # (b) Timeseries panel
    ax_ts = fig.add_subplot(gs[0, 1])
    if era5_anom is not None:
        ax_ts.plot(era5_anom["time"].values, era5_anom.values, color="black", label="ERA5")
    ax_ts.plot(model_anom["time"].values, model_anom.values, color="red", label=label_model)
    ax_ts.axhline(0, lw=0.8, color="k", alpha=0.5)
    ax_ts.set_xlabel("Year")
    ax_ts.set_ylabel("Anomaly (K)")
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(loc="upper right", frameon=False)

    # title
    title_parts = [
        f"{label_model} vs ERA5" if include_era5 else label_model,
        f"{lab_var}",
        f"Base period: {base_period[0]}–{base_period[1]}",
        "Detrended" if detrend else "Raw anomalies",
        f"Kernel: Gaussian",
        f"Bandwidth: {bandwidth}",
    ]
    fig.suptitle(" | ".join(title_parts), fontsize=11)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(outdir, f"{output_fname}.png"))
    plt.close(fig)
