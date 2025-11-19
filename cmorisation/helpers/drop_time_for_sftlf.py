import xarray as xr
import os
import sys

# in and output files/paths
INFILE  = "/work/bk1450/a270220/repos/pcmdi_metrics/land_sea_mask_calculations/archesweather-m-seed0-gc-sst_sic-weight_01_01-res/4_cmorisation/Amon/sftlf/gn/sftlf_Amon_ArchesWeather_aimip_r0i1p1f1_gn_201901-201912.nc"
OUTFILE = "/work/bk1450/a270220/repos/pcmdi_metrics/land_sea_mask_calculations/final_land_sea_masks/1.5x1.5/sftlf_fx_ArchesWeatherGen_1-5x1-5_aimip_r0i1p1f1.nc"

def clean_sftlf(infile, outfile):
    ds = xr.open_dataset(infile)

    ds1 = ds.drop_vars("time_bnds") # remove time_bnds data_vars
    ds1['sftlf'] = ds1["sftlf"].isel(time=0,drop=True) # remove time dimension from data_vars for sftlf
    ds2 = ds1.drop_dims("time") # removes time coordinate

    # update global attrs
    ds2.attrs["frequency"]   = "fx" # fixed aka no changes in time / static
    ds2.attrs["table_id"]    = "fx"
    ds2.attrs["variable_id"] = "sftlf"

    # ensure output dir exists
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    ds2.to_netcdf(outfile)
    print(f"✅ Cleaned and saved: {outfile}")

if __name__ == "__main__":
    # use defaults unless args are provided
    infile  = sys.argv[1] if len(sys.argv) > 1 else INFILE
    outfile = sys.argv[2] if len(sys.argv) > 2 else OUTFILE

    clean_sftlf(infile, outfile)
