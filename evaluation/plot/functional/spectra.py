"""
plot/functional/spectra.py
==========================
Low-level matplotlib helpers for frequency-domain and spherical-spectrum
plots.  All functions draw onto a supplied ``ax`` and return nothing so they
can be composed freely by higher-level plotters.

The legacy ``plot_radial_spectrum`` standalone function is preserved for
backwards compatibility.
"""

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Ax-level helpers
# ---------------------------------------------------------------------------

def radial_spectrum_to_ax(
    ax,
    spectrum: np.ndarray,
    label: str,
    color: str,
    linestyle: str = "-",
    marker=None,
    linewidth: float = 2.0,
    markevery: int = 20,
) -> None:
    """Draw a single radial (spherical-harmonic) spectrum curve onto *ax*.

    The x-axis uses physical wavelength in km derived from spherical-harmonic
    degree *l*: ``wavelength = 2π R_e / sqrt(l(l+1))``.
    """
    l = np.arange(1, spectrum.shape[0] + 1, dtype=float)
    wavelength = 2.0 * np.pi * 6371.0 / np.sqrt(l * (l + 1))

    # use frequency on x axis instead of wavelength, and log-log scale
    frequency = 1.0 / wavelength
    ax.loglog(
        frequency, spectrum,
        linewidth=linewidth, color=color, label=label,
        linestyle=linestyle, marker=marker,
        markevery=markevery if marker is not None else None,
    )
    ax.invert_xaxis()
    ax.set_xlabel("Frequency (1/km)")
    ax.grid(which="both", linestyle="-.", linewidth=0.2)


def psd_to_ax(
    ax,
    frequencies: np.ndarray,
    psd: np.ndarray,
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
) -> None:
    """Draw a single Welch / arbitrary PSD curve onto *ax* (semi-log y)."""
    ax.semilogy(frequencies, psd, linewidth=linewidth, color=color,
                label=label, linestyle=linestyle)
    ax.set_xlabel("Frequency")
    ax.grid(which="both", linestyle="-.", linewidth=0.2)


# ---------------------------------------------------------------------------
# Legacy standalone figure function (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def plot_radial_spectrum(radial_spec, var, output_path, ref_spec=None):
    """Standalone radial-spectrum figure (legacy interface)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    radial_spectrum_to_ax(ax, radial_spec, label="Model", color="black", linewidth=2.0)
    if ref_spec is not None:
        radial_spectrum_to_ax(
            ax, ref_spec, label="ERA5", color="orange", linestyle="--", linewidth=2.0
        )
    ax.set_title(var)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
