import xarray as xr
1

def remove_annual_cycle(data, groupby):
    """
    Remove the annual cycle from the data by subtracting the monthly mean.
    """
    mean_cycle = data.groupby('time.month').mean('time')

    filtered_data = data.groupby(groupby) - mean_cycle

    return filtered_data

def kinetic_energy(data, level, latitude, longitude):
    """
    Compute the kinetic energy at a specific latitude and longitude.
    """
    u = data['u_component_of_wind'].sel(latitude=latitude, longitude=longitude, level=level)
    v = data['v_component_of_wind'].sel(latitude=latitude, longitude=longitude, level=level)

    ke = 0.5 * (u ** 2 + v ** 2)
    return ke


def get_time_slicer(base_period=None):
    """
    Returns a time slicer for the given base period.
    If no base period is provided, it returns a slice for all time.
    """
    if base_period is None:
        return slice(None)
    else:
        return slice(f"{base_period[0]}-01-01", f"{base_period[1]}-12-31")


def preprocess_data(data, groupby=None, base_period=None):
    """
    Preprocess the data by selecting the time period and grouping if necessary.
    This function is a placeholder and should be implemented with actual logic.
    """
    # Placeholder for actual data loading logic

    slicer = get_time_slicer(base_period)
    data = data.sel(time=slicer)

    if groupby:
        data = data.groupby(groupby)
        print(data)

    return data

def compute_mean(data, reduce_dims, groupby=None, base_period=None):
    """
    Compute the mean of the data over the specified dimensions.
    """

    data = preprocess_data(data, groupby=groupby, base_period=base_period)
    data = data.mean(dim=reduce_dims, skipna=True)

    return data     

def compute_std_dev(data, reduce_dims, groupby=None, base_period=None):
    """
    Compute the standard deviation of the data over the specified dimensions.
    """
    data = preprocess_data(data, groupby=groupby, base_period=base_period)
    data = data.std(dim=reduce_dims)

    return data

def detrend_data(data: xr.Dataset, base_period, variables_to_detrend=None) -> xr.Dataset:
    """
    Detrend the dataset over time.
    This function fits a linear trend to the data over the base period and removes it.
    """

    base_data = data.sel(time=data["time"].dt.year.isin(range(base_period[0], base_period[1] + 1)))
    polyfit = base_data.polyfit('time', deg=1)

    if variables_to_detrend is not None:
        for var in variables_to_detrend:
            data[var] = data[var] - xr.polyval(data["time"], polyfit[var + "_polyfit_coefficients"])
    else:
        for var in data.data_vars:
            data[var] = data[var] - xr.polyval(data["time"], polyfit[var + "_polyfit_coefficients"])

    return data


def compute_anomaly(data: xr.Dataset, reduce_dims: list, groupby: list = None, mean_groupby: list = None, base_period=None, detrend=False, running_average: int = 1) -> xr.Dataset:
    """
    Compute anomalies for the given data per year and month.
    This function is a placeholder and should be implemented with actual logic.
    """

    # Detrend variable if required
    if detrend:
        data = detrend_data(data, base_period)  

    # Select base period if given and compute the monthly mean
    mean = compute_mean(data, reduce_dims=reduce_dims, groupby=mean_groupby, base_period=base_period)

    # Compute the anomalies
    anomalies = data.groupby(groupby).mean(reduce_dims) - mean

    return anomalies

def select_tahiti_data(pressure_data: xr.DataArray) -> xr.DataArray:
    """
    Select Tahiti data from pressure data.
    
    Parameters:
    pressure_data (xr.DataArray): Pressure data with dimensions (time, lat, lon).

    Returns:
    xr.DataArray: Tahiti pressure data.
    """
    tahiti_data = pressure_data.sel(latitude=-17.5, longitude=-149.5 + 180, method="nearest")

    return tahiti_data

def select_darwin_data(pressure_data: xr.DataArray) -> xr.DataArray:
    """
    Select Darwin data from pressure data.
    
    Parameters:
    pressure_data (xr.DataArray): Pressure data with dimensions (time, lat, lon).

    Returns:
    xr.DataArray: Darwin pressure data.

    """
    darwin_data = pressure_data.sel(latitude=-12.5, longitude=130.9 + 180., method="nearest")
    return darwin_data

def select_mslp_data(data: xr.DataArray) -> xr.DataArray:
    """
    Select Mean Sea Level Pressure (MSLP) from data.
    
    Parameters:
    data (xr.DataArray): Pressure data with dimensions (time, lat, lon).

    Returns:
    xr.DataArray: MSLP data.
    """
    mslp_data = data['mean_sea_level_pressure'] 

    return mslp_data