import xarray as xr


def select_tahiti_data(pressure_data: xr.DataArray) -> xr.DataArray:
    """
    Select Tahiti data from pressure data.
    
    Parameters:
    pressure_data (xr.DataArray): Pressure data with dimensions (time, lat, lon).

    Returns:
    xr.DataArray: Tahiti pressure data.
    """
    tahiti_data = pressure_data.sel(latitude=-17.5, longitude=-149.5, method="nearest")
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

def compute_standardized_mslp(prediction, reference):
    """
    Computes standardized MSLP 

    """

    # Calculate the mean and standard deviation of the reference data
    mean_ref = reference.groupby('time.month').mean(dim='time')
    std_ref = reference.groupby('time.month').std(dim='time')

    # Standardize the prediction data
    standardized_mslp = (prediction - mean_ref) / std_ref

    return standardized_mslp

def calculate_southern_oscillation_index(predictions, reference_data) -> xr.DataArray:
    """
    Calculate the Southern Oscillation Index (SOI) from predictions and reference data.
    
    Parameters:
    predictions (xr.DataArray): Predicted pressure data with dimensions (time, lat, lon).
    reference_data (xr.DataArray): Reference pressure data with dimensions (time, lat, lon).

    Returns:
    xr.DataArray: Southern Oscillation Index.
    """

    # Ensure the data is MSLP
    predictions = select_mslp_data(predictions)
    reference_data = select_mslp_data(reference_data)

    # Select Tahiti and Darwin data
    tahiti_predictions = select_tahiti_data(predictions)
    darwin_predictions = select_darwin_data(predictions)
    
    tahiti_reference = select_tahiti_data(reference_data)
    darwin_reference = select_darwin_data(reference_data)

    sSLP_tahiti = compute_standardized_mslp(tahiti_predictions, tahiti_reference)
    sSLP_darwin = compute_standardized_mslp(darwin_predictions, darwin_reference)

    pdiff = sSLP_tahiti - sSLP_darwin
    monthly_deviation = pdiff.std(dim='year')

    soi = pdiff / monthly_deviation
    
    soi = soi.stack(time=('year', 'month'), create_index=True).transpose('time', ...)
    
    return soi