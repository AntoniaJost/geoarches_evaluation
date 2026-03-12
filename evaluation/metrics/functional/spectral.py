from typing import Union
import pyshtools as pysh
import xarray as xr
import numpy as np

from scipy.signal import welch


def _remove_south_pole_lat(x):
    """Remove 90 S lat from data because library requires nlon = nlat*2)

    Assumes: data tensor has shape (..., lat, lon)
    """
    if isinstance(x, xr.DataArray):
        return x.sel(latitude=slice(None, -1))

    elif isinstance(x, np.ndarray):
        return x[:-1, :]


def compute_spectral_coefficients(x):
    """Compute the spectral coefficients of a 2D grid."""
    x = _remove_south_pole_lat(x)

    if isinstance(x, xr.DataArray):
        x = x.values

    grid = pysh.SHGrid.from_array(x)

    return grid.expand()

def compute_radial_spectrum(x):
    """Compute the radial spectrum of a 2D grid."""
    coeffs = compute_spectral_coefficients(x)

    return coeffs.spectrum()

def filter_spectral_coefficients(coeffs, lmin, lmax):
    """Filter the spectral coefficients by degree."""
    coeffs_copy = coeffs.copy()
    coeffs_copy.coeffs[:, :lmin, :] = 0
    coeffs_copy.coeffs[:, lmax + 1 :, :] = 0

    return coeffs_copy

def expand_to_grid(coeffs):
    """Expand the spectral coefficients back to a grid."""
    return pysh.expand().MakeGridDH(coeffs, sampling=2)

def compute_radial_spectra(dataset):
    """Compute the radial spectra for all 2D grids in a dataset."""
    spectra = {}
    for var in dataset.data_vars:
        spectra[var] = compute_radial_spectrum(dataset[var])
    return spectra


def welch_psd(x: np.ndarray | xr.DataArray, fs=1.0, nperseg=128, noverlap=64):
    """Compute the Welch power spectral density estimate."""

    if isinstance(x, xr.DataArray):
        x = x.values

    # drop all nan values
    x = x[~np.isnan(x)]

    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="spectrum")

    return f, Pxx
