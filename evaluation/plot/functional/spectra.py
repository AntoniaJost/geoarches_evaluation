import matplotlib.pyplot as plt
import numpy as np

# set fontsize to 12 for all plots
fontdict = {"font.size": 12}


def plot_radial_spectrum(radial_spec, var, output_path, ref_spec=None):
    """
    Plots the radial spectrum of a variable.
    """

    plt.figure(figsize=(8, 8))
    plt.rcParams.update(fontdict)

    l = np.linspace(1, radial_spec.shape[0], radial_spec.shape[0])
    wavelength = 2 * np.pi * 6371.0 / np.sqrt(l * (l + 1))
    wavelength = list(wavelength)
    # wavelength.reverse()
    plt.loglog(
        wavelength, radial_spec, linewidth=2.0, c="black", label="Model projection"
    )
    if ref_spec is not None:
        plt.loglog(
            wavelength,
            ref_spec,
            linewidth=2.0,
            linestyle="--",
            c="orange",
            label="Era5 Radial Spectrum",
        )
    plt.title(f"{var}")
    plt.gca().invert_xaxis()

    # plt.xticks([0, 1, 2], labels=['{:.2e}'.format(10**i) for i in [4, 3, 2]], minor=True)
    plt.xlabel("Wavelength (km)")
    plt.grid(which="both", linestyle="-", linewidth=0.2)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
