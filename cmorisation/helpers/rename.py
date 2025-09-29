import os
import glob
import xarray as xr

input_dir = "/work/bk1450/a270220/evalstore/land_sea_mask_dataCollector/"
# "/home/b/b383170/repositories/scripts/evalstore/AW-M-1-aimip-stats-forcings_surface/1980-01-01T12:00/daily/member_00/data"
output_dir = "/work/bk1450/a270220/evalstore/land_sea_mask_dataCollector/"
# "/work/bk1450/a270220/evalstore/AW-M-1-aimip-stats-forcings_surface/1980-01-01T12:00_2018-12-31T12:00/daily/member_0"
os.makedirs(output_dir, exist_ok=True)

files = glob.glob(os.path.join(input_dir, "*.nc"))
print(f"Found {len(files)} files to process.")

for f in files:
    basename = os.path.basename(f)
    out_path = os.path.join(output_dir, basename)
    print(f"Processing {basename} → {out_path}")

    ds = xr.load_dataset(f, decode_times=False)

    # rename only if valid_time exists
    if "valid_time" in ds.dims or "valid_time" in ds.variables:
        ds = ds.rename({"valid_time": "time"})
        ds.to_netcdf(out_path)
        print(f"  ✅ renamed 'valid_time' → 'time' and saved to {out_path}")
    else:
        # copy unchanged
        ds.to_netcdf(out_path)
        print(f" ‼️ no 'valid_time' found, copied unchanged")
