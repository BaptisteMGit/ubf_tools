#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   plot_utils.py
@Time    :   2024/03/12 08:48:09
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Utility functions to plot kraken outputs
"""

# ======================================================================================================================
# Import
# ======================================================================================================================

import numpy as np
import matplotlib.pyplot as plt

from propa.kraken_toolbox.read_modes import readmodes
from propa.kraken_toolbox.read_shd import readshd
from propa.kraken_toolbox.utils import get_component
from cst import TICKS_FONTSIZE, TITLE_FONTSIZE, LABEL_FONTSIZE


def plotmode(filename, freq=0, modes=None):
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

    Nplots = min(Modes["nb_selected_modes"], 10)
    iskip = Modes["nb_selected_modes"] // Nplots

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
        ax[iplot].set_xlabel(f"Mode {Modes['selected_modes'][imode - 1]}")

    ax[0].set_ylabel("Depth (m)")
    ax[0].invert_yaxis()
    plt.suptitle([Modes["title"], f'Freq = {Modes["freqVec"][freq_index]} Hz'])
    # plt.show()


def plotshd(
    filename,
    freq=None,
    m=None,
    n=None,
    p=None,
    units="m",
    title=None,
    tl_min=None,
    tl_max=None,
    bathy=None,
):
    """Plot Transmission loss field read from '.shd' binary file produced by FIELD.exe.
    Usage :  plotshd(filename, freq, m, n, p, units)

    Adapted from the original Matlab Acoustics Toolbox by Michael B. Porter https://oalib.hlsresearch.com/AcousticsToolbox/

    """

    # Read data based on the number of input arguments
    filename = filename.lower()  # Convert filename to lowercase
    PlotTitle, _, _, _, read_freq, _, Pos, pressure = readshd(
        filename=filename, freq=freq
    )

    pressure = np.squeeze(pressure, axis=(0, 1))

    if m is not None and n is not None and p is not None:
        # Create a subplot
        plt.figure()
        plt.subplot(m, n, p)
    else:
        plt.figure(figsize=(16, 8))

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

    xlab = "Range [m]"
    if units == "km":
        Pos["r"]["r"] = Pos["r"]["r"] / 1000.0
        xlab = "Range [km]"

    if tl_min is not None:
        tlmin_plot = tl_min
    else:
        tlmin_plot = tlmin

    if tl_max is not None:
        tlmax_plot = tl_max
    else:
        tlmax_plot = tlmax
    # tlmin_plot = 30
    # tlmax_plot = 80
    # Plot the data
    tej = plt.get_cmap("jet", 256).reversed()
    plt.pcolor(
        Pos["r"]["r"],
        Pos["r"]["z"],
        tlt,
        shading="auto",
        cmap=tej,
        vmin=tlmin_plot,
        vmax=tlmax_plot,
    )

    # plt.clim(tlmin_plot, tlmax_plot)

    if bathy is not None:
        plt.plot(bathy.bathy_range * 1e3, bathy.bathy_depth, "k")

    plt.gca().invert_yaxis()
    plt.gca().tick_params(direction="out")

    cbar = plt.colorbar()
    cbar.set_label("TL [dB]")
    cbar.ax.invert_yaxis()

    plt.xlabel(xlab, fontsize=LABEL_FONTSIZE)
    plt.ylabel("Depth [m]", fontsize=LABEL_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)

    if title is None:
        title = (
            PlotTitle.replace("_", " ")
            + f'\nFreq = {read_freq} Hz    z_src = {Pos["s"]["z"][0]} m'
        )
    plt.title(title, fontsize=TITLE_FONTSIZE)

    plt.scatter(0, Pos["s"]["z"][0], marker="o", c="k", s=50)

    # If a subplot is created, return a handle to the figure
    if m is not None and n is not None and p is not None:
        return plt.gcf()


def plotshd_from_pressure_field(
    filename,
    pressure_field,
    freq=None,
    m=None,
    n=None,
    p=None,
    units="m",
    title=None,
    tl_min=None,
    tl_max=None,
    bathy=None,
):
    """
    Plot Transmission loss field directly from pressure field array.
    This function is particularly useful when running broadband simulations with range dependent environments.
    Usage :  plotshd_from_pressure_field(filename, pressure_field, freq, m, n, p, units)
    """
    # Dummy read to get freq and position vectors
    filename = filename.lower()  # Convert filename to lowercase
    PlotTitle, _, _, _, read_freq, _, Pos, _ = readshd(filename=filename, freq=freq)

    pressure = np.squeeze(pressure_field, axis=(0, 1))

    if m is not None and n is not None and p is not None:
        # Create a subplot
        plt.figure()
        plt.subplot(m, n, p)
    else:
        plt.figure(figsize=(16, 8))

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

    xlab = "Range [m]"
    if units == "km":
        Pos["r"]["r"] = Pos["r"]["r"] / 1000.0
        xlab = "Range [km]"

    # Plot the data
    tej = plt.get_cmap("jet", 256).reversed()
    plt.pcolor(Pos["r"]["r"], Pos["r"]["z"], tlt, shading="auto", cmap=tej)
    if tl_min is not None:
        tlmin_plot = tl_min
    else:
        tlmin_plot = tlmin

    if tl_max is not None:
        tlmax_plot = tl_max
    else:
        tlmax_plot = tlmax
    plt.clim(tlmin_plot, tlmax_plot)

    if bathy is not None:
        plt.plot(bathy.bathy_range * 1e3, bathy.bathy_depth, "k")

    plt.gca().invert_yaxis()
    plt.gca().tick_params(direction="out")

    cbar = plt.colorbar()
    cbar.set_label("TL [dB]")
    cbar.ax.invert_yaxis()

    plt.xlabel(xlab, fontsize=LABEL_FONTSIZE)
    plt.ylabel("Depth [m]", fontsize=LABEL_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)

    if title is None:
        title = PlotTitle.replace("_", " ")
        +f'\nFreq = {read_freq} Hz    z_src = {Pos["s"]["z"][0]} m'
    plt.title(title, fontsize=TITLE_FONTSIZE)

    plt.scatter(0, Pos["s"]["z"][0], marker="o", c="k", s=50)

    # If a subplot is created, return a handle to the figure
    if m is not None and n is not None and p is not None:
        return plt.gcf()


""" Plot environment profiles """


def plot_ssp(cp_ssp, cs_ssp, z, z_bottom=None, ax=None):
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_ylabel("Depth [m]")

    if np.array(cp_ssp).size == 1:
        cp = np.ones(z.size) * cp_ssp
    else:
        cp = cp_ssp

    if np.array(cs_ssp).size == 1:
        cs = np.ones(z.size) * cs_ssp
    else:
        cs = cs_ssp

    # No need to plot the C-wave celerity if it is 0 and cs is not 0
    if np.all(cp == 0) and not np.all(cs == 0):
        cp = np.ones(z.size) * np.nan
        min_cp = np.nan
        max_cp = np.nan
        plot_cp = False
    else:
        min_cp = np.min(cp)
        max_cp = np.max(cp)
        plot_cp = True

    # No need to plot the S-wave celerity if it is 0 and cp is not 0
    cs = 0
    if np.all(cs == 0) and not np.all(cp == 0):
        cs = np.ones(z.size) * np.nan
        min_cs = np.nan
        max_cs = np.nan
        plot_cs = False
    else:
        min_cs = np.min(cs)
        max_cs = np.max(cs)
        plot_cs = True

    if plot_cp:
        col1 = "red"
        ax.plot(cp, z, color=col1, label="C-wave")
    ax.set_xlabel("Celerity " + r"[$m.s^{-1}$]")

    if plot_cs:
        col2 = "blue"
        ax.plot(cs, z, color=col2, label="S-wave")

    ax.invert_yaxis()
    ax.legend(loc="upper right")

    # Color domains with water and sediment
    min_x = np.nanmin([min_cp, min_cs])
    max_x = np.nanmax([max_cp, max_cs])

    # Set the x-axis limits
    x_offset = max(0.1 * (max_x - min_x), 10)
    min_x -= x_offset
    max_x += x_offset
    ax.set_xlim(min_x, max_x)

    color_domains(ax, min_x=min_x, max_x=max_x, z=z, z_bottom=z_bottom)


def plot_attenuation(ap, ash, z, z_bottom=None, ax=None):
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_ylabel("Depth [m]")

    if np.array(ap).size == 1:
        ap = np.ones(z.size) * ap
    else:
        pass

    if np.array(ash).size == 1:
        ash = np.ones(z.size) * ash
    else:
        pass

    # No need to plot the C-wave attenuation if it is 0 and as is not 0
    if np.all(ap == 0) and not np.all(ash == 0):
        ap = np.ones(z.size) * np.nan
        min_ap = np.nan
        max_ap = np.nan
        plot_ap = False
    else:
        min_ap = np.min(ap)
        max_ap = np.max(ap)
        plot_ap = True

    # No need to plot the S-wave attenuation if it is 0 and ap is not 0
    if np.all(ash == 0) and not np.all(ap == 0):
        ash = np.ones(z.size) * np.nan
        min_as = np.nan
        max_as = np.nan
        plot_as = False
    else:
        min_as = np.min(ash)
        max_as = np.max(ash)
        plot_as = True

    if plot_ap:
        col1 = "red"
        ax.plot(ap, z, color=col1, label="C-wave")
    ax.set_xlabel(r"$\alpha$ " + r"[$dB.\lambda^{-1}$]")

    if plot_as:
        col2 = "blue"
        ax.plot(ash, z, color=col2, label="S-wave")

    # Color domains with water and sediment
    min_x = np.nanmin([min_ap, min_as])
    max_x = np.nanmax([max_ap, max_as])

    # Set the x-axis limits
    x_offset = max(0.1 * (max_x - min_x), 2)
    min_x -= x_offset
    max_x += x_offset
    ax.set_xlim(min_x, max_x)

    color_domains(ax, min_x=min_x, max_x=max_x, z=z, z_bottom=z_bottom)

    ax.invert_yaxis()
    ax.legend(loc="upper right")


def plot_density(rho, z, z_bottom=None, ax=None):
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_ylabel("Depth [m]")

    if np.array(rho).size == 1:
        rho = np.ones(z.size) * rho
    else:
        pass

    ax.plot(rho, z, label=r"$\rho$", color="k")
    ax.invert_yaxis()
    ax.set_xlabel(r"$\rho$ " + r"[$g.cm^{-3}$]")

    # Color domains with water and sediment
    min_x = np.min(rho)
    max_x = np.max(rho)

    # Set the x-axis limits
    x_offset = max(0.1 * (max_x - min_x), 0.1)
    min_x -= x_offset
    max_x += x_offset
    ax.set_xlim(min_x, max_x)

    color_domains(ax, min_x=min_x, max_x=max_x, z=z, z_bottom=z_bottom)


def color_domains(ax, min_x, max_x, z, z_bottom=None):
    if z_bottom is not None:
        # Bottom domain
        ax.fill_between(
            [min_x, max_x],
            [z_bottom, z_bottom],
            np.max(z),
            color="lightgrey",
        )
        # Water domain
        ax.fill_between(
            [min_x, max_x],
            [z_bottom, z_bottom],
            0,
            color="lightblue",
        )
    else:
        pass


if __name__ == "__main__":
    pass
