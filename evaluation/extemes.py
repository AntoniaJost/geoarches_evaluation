import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyextremes
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load deja vu sans as maptlotlib font
plt.rcParams["font.family"] = "DejaVu Sans"

# Data T2m
data_paths = {
    #"MPI-ESM1-2-HR": [
    #    "data/mpi-esm-1x1/amip/r1i1p1f1/day",
    #    "data/mpi-esm-1x1/amip/r2i1p1f1/day",
    #    "data/mpi-esm-1x1/amip/r3i1p1f1/day"
    #],
    #"AWM": [
    #    "rollouts/cmorised_aimip/ArchesWeather/full/0k/r1i1p1f1/day",
    #    "rollouts/cmorised_aimip/ArchesWeather/full/0k/r2i1p1f1/day",
    #    "rollouts/cmorised_aimip/ArchesWeather/full/0k/r3i1p1f1/day",
    #    "rollouts/cmorised_aimip/ArchesWeather/full/0k/r4i1p1f1/day",
    #    "rollouts/cmorised_aimip/ArchesWeather/full/0k/r5i1p1f1/day"
    #],
    "AWGen": [
        "/work/bk1450/a270220/cmorised_awm/forced_sst/member1/day",
        "/work/bk1450/a270220/cmorised_awm/forced_sst/member2/day",
        "/work/bk1450/a270220/cmorised_awm/forced_sst/member3/day",
        "/work/bk1450/a270220/cmorised_awm/forced_sst/member4/day",
        "/work/bk1450/a270220/cmorised_awm/forced_sst/member5/day"
     ],
    "AWGen-unforced": [
        "/work/bk1450/a270220/cmorised_awm/free_run_control/member1/day",
        "/work/bk1450/a270220/cmorised_awm/free_run_control/member2/day",
        "/work/bk1450/a270220/cmorised_awm/free_run_control/member3/day",
        "/work/bk1450/a270220/cmorised_awm/free_run_control/member4/day",
        "/work/bk1450/a270220/cmorised_awm/free_run_control/member5/day"
    ],
    "ERA5": ["/home/b/b383170/repositories/geoarches_evaluation/evaluation/data/era5_1x1_averaged_cmor/4_cmorisation/day"],

}

variable = "tas"
level = None
variables = f"{variable}_*.nc"

var_name_mapping = {
    "tas": "Surface Air Temperature",
    "ua": "Eastward Wind",
    "va": "Northward Wind",
    "uv": "Wind Speed"
}

units_mapping = {
    "tas": r"$K$",
    "ua": r"$m/s$",
    "va": r"$m/s$",
    "uv": r"$m/s$"
}

data = {}
for model, paths in data_paths.items():
    data[model] = []
    print("###############################################################")
    logging.info(f"Processing model: {model}")
    for idx, path in enumerate(paths):
        logging.info(f"... Processing member: [{idx+1}/{len(paths)}]")
        def preprocess(ds):
            print(level)
            if level is not None:
                ds = ds.sel(plev=level * 100, method="nearest", drop=True)
            return ds

        open_kwargs = dict(combine="by_coords", parallel=False, preprocess=preprocess, chunks={"time": 365})

        if variable == "uv":
            ua_files = sorted(glob.glob(f"{path}/**/ua_*.nc", recursive=True))
            va_files = sorted(glob.glob(f"{path}/**/va_*.nc", recursive=True))
            ds_ua = xr.open_mfdataset(ua_files, **open_kwargs)
            ds_va = xr.open_mfdataset(va_files, **open_kwargs)
            wind_speed = np.sqrt(ds_ua["ua"] ** 2 + ds_va["va"] ** 2)
            ds = wind_speed.max(("lat", "lon")).compute().dropna("time").to_series()
        else:
            files = sorted(glob.glob(f"{path}/**/{variables}", recursive=True))
            ds = xr.open_mfdataset(files, **open_kwargs)
            ds = ds[variable].max(("lat", "lon")).compute().dropna("time").to_series()
        if len(paths) > 1 and idx > 0:
            # Get number of years from the pandas time series
            n_years = (ds.index[-1] - ds.index[0]).days / 365.2425
            # Shift the time index by the number of years            
            # ds.index = ds.index + pd.Timedelta(days=n_years * 365.2425
            ds.index = ds.index + pd.Timedelta(days=365.2425 * idx * n_years)
        data[model].append(ds)
    if len(data[model]) > 1:
        # Concatenate the pandas series along the time axis
        data[model] = pd.concat(data[model], axis=0)
    else:
        data[model] = data[model][0]

# Create EVA objects

#awgen = pyextremes.EVA(data["AWGen"].stack(series=("time", "lat", "lon")).reindex().to_pandas())

models = {}
for model_name, data in data.items():
    logging.info(f"Performing EVA for model: {model_name}")
    eva = pyextremes.EVA(data)
    eva.get_extremes(method="BM", block_size="365.2425D")
    eva.fit_model()
    models[model_name] = eva

fig, axs = plt.subplots(1, 1, dpi=300, figsize=(8,5))
colors = {
    "ERA5": "black", 
    "AWGen": "#D55E00",
    "AWM": "#0072B2",
    "MPI-ESM1-2-HR": [0.5, 0.5, 0.5],
    "AWGen-unforced": "#CC79A7"
}

alpha = {
    "ERA5": 1.0, 
    "AWGen": 0.95,
    "AWM": 0.95,
    "MPI-ESM1-2-HR": 0.95,
    "AWGen-unforced": 0.95
}
for model_name, eva in models.items():
    
    _, ax = eva.plot_return_values(ax=axs, alpha=alpha[model_name])
    ax.lines[-1].set_label(model_name)
    ax.lines[-1].set_color(colors[model_name])
    ax.lines[-2].set_color(colors[model_name])
    ax.lines[-3].set_color(colors[model_name])
    ax.collections[-1].set_facecolor(colors[model_name])
    ax.collections[-1].set_edgecolor(colors[model_name])
    ax.collections[-2].set_alpha(0.3)  
    ax.collections[-2].set_facecolor(colors[model_name])

if level is not None:
    var_name = f"{var_name_mapping[variable]} ({level} hPa)"
else:
    var_name = var_name_mapping[variable]

axs.text(
    0.0, 1.0, var_name, 
    transform=axs.transAxes, fontsize=18, 
    verticalalignment="bottom", horizontalalignment="left",
    fontweight="bold"
)
axs.set_xlabel("Return Period (years)", fontsize=16)

axs.set_ylabel(f"{units_mapping[variable]}", fontsize=16)
axs.set_xticks([1, 5, 10, 50, 100, 200])
axs.legend(fontsize=16)

# Set xticks and yticks font size tio 16
axs.tick_params(axis="both", which="major", labelsize=16)
# place legend centred below the plot
axs.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=16)
filename = f"return_values_plot_{variable}_unforced.pdf" if level is None else f"return_values_plot_{variable}_{level}.pdf"
plt.savefig(filename, bbox_inches="tight", dpi=300)