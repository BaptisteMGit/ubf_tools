import numpy as np
import matplotlib.pyplot as plt

from propa.kraken_toolbox.read_modes import readmodes
from propa.kraken_toolbox.read_shd import readshd
from propa.kraken_toolbox.utils import get_component


def plotmode(filename, freq, modes):
    """Plot modes produced by KRAKEN from a '.mod' binary file.
    Usage: plotmode(filename, freq, modes)

    filename don't need to include the extension.

    Adapted from the original Matlab Acoustics Toolbox by Michael B. Porter https://oalib.hlsresearch.com/AcousticsToolbox/

    """
    Modes = readmodes(filename, freq, modes)

    if Modes["M"] == 0:
        raise Exception("No modes in mode file")

    freqdiff = np.abs(Modes["freqVec"] - freq)
    freq_index = np.argmin(freqdiff)
    phi = get_component(Modes, "N")

    nx = phi.shape[1]  # Assuming all modes have the same length

    if nx > 1:
        x = np.arange(1, nx + 1)
        doo = np.real(phi)
        plt.figure()
        plt.pcolor(x, Modes["z"], doo, shading="auto", cmap="jet")
        plt.gca().invert_yaxis()
        plt.colorbar()
        # caxis_lim = max(np.abs(plt.clim()))
        # plt.clim(-caxis_lim, caxis_lim)
        plt.xlabel("Mode index")
        plt.ylabel("Depth (m)")
        plt.title([Modes["title"], f'Freq = {Modes["freqVec"][freq_index]} Hz'])

    Nplots = min(len(modes), 10)
    iskip = len(modes) // Nplots

    fig, ax = plt.subplots(1, Nplots, figsize=(15, 5), sharey=True)
    for iplot in range(Nplots):
        imode = 1 + (iplot) * iskip
        if iplot == 0:
            ax[iplot].plot(np.real(phi[:, imode - 1]), Modes["z"], "k", label="Real")
            ax[iplot].plot(np.imag(phi[:, imode - 1]), Modes["z"], "b--", label="Imag")
            ax[iplot].legend()
        else:
            ax[iplot].plot(np.real(phi[:, imode - 1]), Modes["z"], "k")
            ax[iplot].plot(np.imag(phi[:, imode - 1]), Modes["z"], "b--")
        ax[iplot].set_xlabel(f"Mode {modes[imode - 1]}")

    ax[0].set_ylabel("Depth (m)")
    ax[0].invert_yaxis()
    plt.suptitle([Modes["title"], f'Freq = {Modes["freqVec"][freq_index]} Hz'])
    # plt.show()


def plotshd(filename, freq=None, m=None, n=None, p=None, units="m"):
    """Plot Transmission loss field read from '.shd' binary file produced by FIELD.exe.
    Usage :  plotshd(filename, freq, m, n, p, units)

    Adapted from the original Matlab Acoustics Toolbox by Michael B. Porter https://oalib.hlsresearch.com/AcousticsToolbox/

    """

    # Read data based on the number of input arguments
    filename = filename.lower()  # Convert filename to lowercase
    PlotTitle, _, freqVec, _, read_freq, _, Pos, pressure = readshd(
        filename=filename, freq=freq
    )

    pressure = np.squeeze(pressure, axis=(0, 1))

    if freq is None:
        freq = freqVec[0]

    if m is not None and n is not None and p is not None:
        # Create a subplot
        plt.figure()
        plt.subplot(m, n, p)
    else:
        plt.figure()

    # Calculate caxis limits
    tlt = np.abs(pressure).astype(float)
    # Remove infinities and nan values
    tlt[np.isnan(tlt)] = 1e-6
    tlt[np.isinf(tlt)] = 1e-6

    values_counting = tlt > 1e-37
    tlt[~values_counting] = 1e-37
    tlt = -20.0 * np.log10(tlt)
    tlmed = np.median(tlt[values_counting])
    tlstd = np.std(tlt[values_counting])
    tlmax = tlmed + 0.75 * tlstd
    tlmax = 10 * round(tlmax / 10)
    tlmin = tlmax - 50

    xlab = "Range (m)"
    if units == "km":
        Pos["r"]["r"] = Pos["r"]["r"] / 1000.0
        xlab = "Range (km)"

    # Plot the data
    tej = plt.get_cmap("jet", 256).reversed()
    plt.pcolor(Pos["r"]["r"], Pos["r"]["z"], tlt, shading="auto", cmap=tej)
    plt.clim(tlmin, tlmax)
    plt.gca().invert_yaxis()
    plt.gca().tick_params(direction="out")

    cbar = plt.colorbar()
    cbar.set_label("TL (dB)")
    cbar.ax.invert_yaxis()

    plt.xlabel(xlab)
    plt.ylabel("Depth (m)")
    plt.title(
        PlotTitle.replace("_", " ")
        + f'\nFreq = {read_freq} Hz    z_src = {Pos["s"]["z"][0]} m'
    )
    plt.scatter(0, Pos["s"]["z"][0], marker="<", c="k", s=50)

    # If a subplot is created, return a handle to the figure
    if m is not None and n is not None and p is not None:
        return plt.gcf()
