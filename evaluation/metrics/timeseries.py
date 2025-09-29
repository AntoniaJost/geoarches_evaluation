import numpy as np
import xarray as xr 

from metrics import utils as eval_utils
import pandas as pd

def compute_annual_cycle(data, detrend, base_period):
        if detrend:
            data = eval_utils.detrend_data(data, base_period=base_period, variables_to_detrend=['sea_surface_temperature', '2m_temperature'])

        data = eval_utils.compute_mean(data, groupby=['time.year', 'time.month'], reduce_dims=['time', 'latitude', 'longitude'])
        data = data.stack(time=('year', 'month')).transpose('time', ...)
        
        data = data.reset_index('time')

        return data

def compute_annual_anomaly(data, detrend, base_period): 

    eval_utils.compute_anomaly(
        data=data, 
        reduce_dims=['time', 'latitude', 'longitude'],
        groupby=['time.year', 'time.month'], 
        mean_groupby='time.month',
        base_period=base_period, 
        detrend=detrend
    )

    return data.stack(time=('year', 'month')).transpose('time', ...)
    
    
def compute_southern_oscillation_index(data, base_period, detrend=False) -> xr.DataArray:
    """
    Calculate the Southern Oscillation Index (SOI) from predictions and reference data.
    
    Parameters:
    predictions (xr.DataArray): Predicted pressure data with dimensions (time, lat, lon).
    reference_data (xr.DataArray): Reference pressure data with dimensions (time, lat, lon).

    Returns:
    xr.DataArray: Southern Oscillation Index.
    """

    # Detrend
    if detrend:
        data = eval_utils.detrend_data(data, base_period=base_period, variables_to_detrend=['mean_sea_level_pressure'])

    # Select Tahiti and Darwin data
    tahiti_data = eval_utils.select_tahiti_data(data)
    darwin_data = eval_utils.select_darwin_data(data)

    # Anomalies
    tahiti_anomaly = eval_utils.compute_anomaly(tahiti_data, reduce_dims=['time', 'latitude', 'longitude'], groupby=['time.year', 'time.month'], mean_groupby=['time.month'], base_period=base_period)
    darwin_anomaly = eval_utils.compute_anomaly(darwin_data, reduce_dims=['time', 'latitude', 'longitude'], groupby=['time.year', 'time.month'], mean_groupby=['time.month'], base_period=base_period)

    # Normalize by dividing through std dev
    tahiti_anomaly = tahiti_anomaly / eval_utils.compute_std_dev(tahiti_anomaly, reduce_dims=['time', 'latitude', 'longitude'], groupby='time.month')
    darwin_anomaly = darwin_anomaly / eval_utils.compute_std_dev(darwin_anomaly, reduce_dims=['time', 'latitude', 'longitude'], groupby='time.month')

    pdiff = tahiti_anomaly - darwin_anomaly
    monthly_diff_deviation = eval_utils.compute_std_dev(pdiff, reduce_dims=['year'])

    soi = pdiff / monthly_diff_deviation

    soi = soi.stack(time=('year', 'month'), create_index=True).transpose('time', ...)
    
    return soi

def compute_oni_index(data, base_period, detrend=False):
    """
    This function computes the ONI index by selecting the
    enso3.4 region and computing the anomalies. The data is
    normalized by further dividing by the standard deviation.
    args:
    data: Data containing SST data
    base_period: Tuple containing the start and end year for the base period
    detrend: Boolean indicating whether to detrend the data
    """


    # Detrend
    if detrend:
        data = eval_utils.detrend_data(data, base_period=base_period, variables_to_detrend=['sea_surface_temperature'])

    data = data['sea_surface_temperature']
    
    # Standard deviation
    std_dev = eval_utils.compute_std_dev(
        data=data, 
        reduce_dims=['time'], 
        groupby='time.month', 
        base_period=base_period
    )

    # Compute anomalies
    anomalies = eval_utils.compute_anomaly(
        data=data,
        reduce_dims=['time'],
        groupby=['time.year', 'time.month'], 
        mean_groupby=['time.month'],
        base_period=base_period, 
        detrend=False,
        running_average=1,
    )

    # Enso34 index calculation
    enso34 = anomalies / std_dev

    # Select the region for the ENSO 3.4 index
    enso34 = enso34.sel(latitude=slice(5, -5), longitude=slice(10, 60))
    enso34 = enso34.mean(dim=['latitude', 'longitude'])
    enso34 = enso34.stack(time=('year', 'month'), create_index=True).transpose('time', ...)
    enso34['time'] = np.unique(data['time'].astype('datetime64[M]').to_numpy())
    
    # Apply running average if specified
    #enso34 = enso34.rolling(time=3, center=True)
    #enso34 = enso34.mean()
    anomalies = anomalies / std_dev
    anomalies = anomalies.stack(time=('year', 'month'), create_index=True).transpose('time', ...)
    anomalies['time'] = np.unique(data['time'].astype('datetime64[M]').to_numpy())
    
    #anomalies = anomalies.rolling(time=3, center=True).mean()

    return enso34, anomalies
