def get_local_min_mslp(dataset):
    long_range=[100, 160]
    lat_range=[35, 5]

    mslp = dataset['mslp'].sel(longitude=slice(*long_range), latitude=slice(*lat_range))
    min_mslp = mslp.min(dim=['longitude', 'latitude'])
    quantiles = [1000, 994, 985, 975]
                 
def find_warm_core(dataset, climatology):
    # warm core is defined as the geopotential height between 300 hpa and 700 hpa
    # that is warmer than the climatology

    z_300 = dataset['z'].sel(level=300)
    z_700 = dataset['z'].sel(level=700)
    z_anomaly = z_300 - z_700
    z_clim_300 = climatology['z'].sel(level=300)
    z_clim_700 = climatology['z'].sel(level=700)
    z_clim_anomaly = z_clim_300 - z_clim_700
    warm_core = z_anomaly - z_clim_anomaly
    return warm_core

def analyze_trajectories(trajectories):
    # Analyze the trajectories to find common patterns
    # For simplicity, we will just compute the mean trajectory
    mean_trajectory = trajectories.mean(dim='trajectory')
    



