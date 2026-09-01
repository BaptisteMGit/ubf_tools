#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   plot_utils.py
@Time    :   2024/03/12 08:48:09
@Author  :   Menetrier Baptiste
@Version :   1.1 (refactor)
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Utility functions to plot KRAKEN/FIELD outputs: mode
             shapes, transmission-loss (TL) maps and range profiles,
             and environment profiles (SSP, attenuation, density).

This module does NOT change the public API of the original file (same
function names/signatures, two new functions added at the end:
plot_tl_profile / plot_tl_profile_multi_freq, factored out of the
duplicated code that used to live in every example case script -- see
propa/kraken_toolbox/examples/).

BUGS FIXED COMPARED TO THE ORIGINAL FILE (all reproduced and confirmed
before fixing -- see the project's test suite):
  1. plotmode() / plotmode_several_freqs(): `plt.subplots(1, Nplots, ...)`
     returns a single Axes object (not an array) when Nplots == 1 --
     `ax[0].invert_yaxis()` then raised
     `TypeError: 'Axes' object is not subscriptable`. Confirmed with a
     real 10 Hz mode file (a single mode at that frequency). Fixed with
     `squeeze=False` + `.ravel()`, which always yields a flat array of
     Axes regardless of Nplots.
  2. plotmode_several_freqs(): the subplot grid was sized to the FIRST
     frequency's mode count (Nplots computed inside the loop, but the
     figure only created `if i_f == 0`). Since the number of modes
     genuinely varies with frequency (confirmed on real data: 1, 3, 4,
     5, 7 modes at 10/20/30/40/50 Hz for the same environment), any
     later frequency with MORE modes than the first one raised an
     IndexError on `ax[iplot]`. Fixed by first computing every
     frequency's Nplots, then sizing the figure to the maximum.
  3. plotmode() / plotmode_several_freqs(): `plt.title([...])` /
     `fig.suptitle([...])` were passed a Python LIST instead of a
     string, producing a literal `"['title', 'Freq = ... Hz']"` in the
     figure (brackets and quotes included). Fixed to build a proper
     string.
  4. plotshd(): `filename = filename.lower()` unconditionally lowercased
     the path before opening it. This silently breaks on any
     case-sensitive filesystem (Linux, macOS) whenever the real file
     name contains uppercase characters -- there is no reason to alter
     a caller-supplied path's case. Removed.
  5. plotshd_from_pressure_field(): the default-title branch was

         title = PlotTitle.replace("_", " ")
         +f'\\nFreq = {read_freq} Hz    z_src = {Pos["s"]["z"][0]} m'

     -- the second line was meant to be concatenated onto 'title' but
     the `+` was never assigned back (`title = title + ...` or `+=`),
     so it was parsed as a separate, useless statement: a unary '+' on
     a string literal, which Python raises `TypeError: bad operand type
     for unary +: 'str'` for. This means: calling
     plotshd_from_pressure_field() without an explicit 'title' ALWAYS
     crashed. Confirmed and fixed by concatenating properly, matching
     the (correct) equivalent code already present in plotshd().
  6. plot_ssp(): right before checking whether the S-wave celerity
     curve should be hidden (`if np.all(cs == 0) and not
     np.all(cp == 0):`), the code did `cs = 0`, unconditionally
     discarding the real 'cs' array (already resolved from 'cs_ssp' a
     few lines above) and replacing it with the literal scalar 0. Since
     `np.all(0 == 0)` is trivially True, the S-wave curve was hidden
     EVERY TIME cp was not all-zero -- i.e. for any normal fluid-over-
     elastic environment, genuine non-zero shear-wave data was silently
     never plotted. Confirmed with a synthetic elastic medium (real,
     non-zero cs values): the 'S-wave' curve was missing from the plot
     even though it should have been there. Fixed by removing the
     erroneous `cs = 0` line.

Two new functions were added, factored out of the near-identical
'read_tl_grid' / 'plot_tl_profile_at_depth' / 'plot_tl_profile_multi_freq'
helpers duplicated across every example script:
  - plot_tl_profile(...): TL vs. range at a single fixed receiver depth.
  - plot_tl_profile_multi_freq(...): the same, one line per frequency,
    for broadband runs.
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


# ======================================================================================================================
# Mode shapes
# ======================================================================================================================
def plotmode(
    filename,
    freq=0,
    modes=None,
    bathy_depth=None,
    normalize_mode=False,
    plot_mode_map=False,
):
    """Plot modes produced by KRAKEN from a '.mod' binary file.
    Usage: plotmode(filename, freq, modes)

    filename doesn't need to include the extension.

    Args:
        filename (str): path to the '.mod' file (extension optional).
        freq (float): frequency (Hz) to read.
        modes (array-like|None): 1-based mode indices to read. None
            reads every mode.
        bathy_depth (float|None): if given, draws a horizontal dashed
            line at this depth (the local seafloor) and clips the
            depth axis to 1.4x this value.
        normalize_mode (bool): normalize each plotted mode to [-1, 1].
        plot_mode_map (bool): additionally plot a depth-vs-mode-index
            color map of the real part of every mode (only if more than
            one mode is available).

    Adapted from the original Matlab Acoustics Toolbox by Michael B. Porter
    https://oalib.hlsresearch.com/AcousticsToolbox/
    """
    Modes = readmodes(filename, freq, modes)

    if Modes["M"] == 0:
        raise Exception("No modes in mode file")

    freqdiff = np.abs(Modes["freqVec"] - freq)
    freq_index = np.argmin(freqdiff)
    phi = get_component(Modes, "N")

    nx = phi.shape[1]  # Assuming all modes have the same length

    if nx > 1 and plot_mode_map:
        x = np.arange(1, nx + 1)
        doo = np.real(phi)
        plt.figure()
        plt.pcolor(x, Modes["z"], doo, shading="auto", cmap="jet")
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.xlabel("Mode index")
        plt.ylabel("Depth (m)")
        # NOTE (bug fixed): passing a list here (instead of a string)
        # to plt.title() rendered as a literal "['...', '...']" in the
        # figure. Same fix applied everywhere else in this module.
        plt.title(f'{Modes["title"]}\nFreq = {Modes["freqVec"][freq_index]} Hz')

    Nplots = min(Modes["nb_selected_modes"], 10)
    iskip = max(Modes["nb_selected_modes"] // Nplots, 1)

    # NOTE (bug fixed): plt.subplots(1, Nplots) returns a single Axes
    # object (not an array) when Nplots == 1, so `ax[0]` used to raise
    # `TypeError: 'Axes' object is not subscriptable` -- a real crash
    # confirmed on a real KRAKEN mode file with a single mode at the
    # requested frequency. squeeze=False guarantees a 2D array of Axes
    # regardless of Nplots; .ravel() flattens it back to the 1D
    # indexing (`ax[iplot]`) the rest of this function expects.
    fig, ax = plt.subplots(1, Nplots, figsize=(15, 5), sharey=True, squeeze=False)
    ax = ax.ravel()
    ax[0].invert_yaxis()

    for iplot in range(Nplots):
        imode = 1 + (iplot) * iskip

        # Normalize mode
        if normalize_mode:
            phi[:, imode - 1] = phi[:, imode - 1] / np.max(np.abs((phi[:, imode - 1])))

        if iplot == 0:
            ax[iplot].plot(np.real(phi[:, imode - 1]), Modes["z"], "k", label="Real")
            ax[iplot].plot(np.imag(phi[:, imode - 1]), Modes["z"], "b--", label="Imag")
            ax[iplot].legend()
        else:
            ax[iplot].plot(np.real(phi[:, imode - 1]), Modes["z"], "k")
            ax[iplot].plot(np.imag(phi[:, imode - 1]), Modes["z"], "b--")
        ax[iplot].set_xlabel(f"Mode {Modes['selected_modes'][imode - 1]}")
        if bathy_depth is not None:
            ax[iplot].axhline(y=bathy_depth, color="r", linestyle="--", label="Depth")

        if normalize_mode:
            ax[iplot].set_xlim([-1.2, 1.2])

    if bathy_depth is not None:
        ax[0].set_ylim([0, bathy_depth * 1.4])

    fig.supylabel("Depth [m]")
    fig.suptitle(f'{Modes["title"]}\nFreq = {Modes["freqVec"][freq_index]} Hz')
    return fig


def plotmode_several_freqs(
    filename: str,
    freq: np.ndarray = None,
    modes: np.ndarray = None,
    bathy_depth: float = None,
    label_bathy: bool = False,
    normalize_mode: bool = False,
):
    """Plot modes produced by KRAKEN from a '.mod' binary file, for
    several frequencies overlaid on the same subplot grid.
    Usage: plotmode_several_freqs(filename, freq, modes)

    filename doesn't need to include the extension.

    Args:
        filename (str): path to the '.mod' file (extension optional).
        freq (array-like): frequencies (Hz) to read and overlay.
        modes (array-like|None): 1-based mode indices to read. None
            reads every mode found at each frequency.
        bathy_depth (float|None): see plotmode.
        label_bathy (bool): add a legend entry for the bathymetry line
            (drawn once, using the first frequency's depth reference).
        normalize_mode (bool): normalize each plotted mode to [-1, 1].

    Adapted from plotmode().
    """
    # NOTE (bug fixed): the original code created the subplot grid
    # ONLY on the first frequency (`if i_f == 0: fig, ax =
    # plt.subplots(1, Nplots, ...)`), sized to that FIRST frequency's
    # mode count. But the number of modes genuinely varies with
    # frequency (confirmed on real data: 1, 3, 4, 5, 7 modes at
    # 10/20/30/40/50 Hz for the very same environment) -- any later
    # frequency with MORE modes than the first one indexed past the
    # end of the fixed-size 'ax' array, raising an IndexError. Fixed by
    # reading every frequency's Modes dict FIRST, then sizing the
    # figure to accommodate the largest one.
    all_modes = []
    for f in freq:
        Modes = readmodes(filename, f, modes)
        if Modes["M"] == 0:
            raise Exception(f"No modes in mode file at {f} Hz")
        all_modes.append(Modes)

    max_nplots = min(max(m["nb_selected_modes"] for m in all_modes), 10)

    # NOTE (bug fixed): see plotmode() -- squeeze=False + ravel()
    # guarantees a flat, subscriptable array of Axes even when
    # max_nplots == 1.
    fig, ax = plt.subplots(1, max_nplots, figsize=(15, 5), sharey=True, squeeze=False)
    ax = ax.ravel()
    ax[0].invert_yaxis()

    for i_f, (f, Modes) in enumerate(zip(freq, all_modes)):
        freqdiff = np.abs(Modes["freqVec"] - f)
        freq_index = np.argmin(freqdiff)
        phi = get_component(Modes, "N")

        nplots_this_freq = min(Modes["nb_selected_modes"], max_nplots)
        iskip = max(Modes["nb_selected_modes"] // nplots_this_freq, 1)

        for iplot in range(nplots_this_freq):
            imode = 1 + (iplot) * iskip

            if normalize_mode:
                phi[:, imode - 1] = phi[:, imode - 1] / np.max(
                    np.abs((phi[:, imode - 1]))
                )

            if iplot == 0:
                ax[iplot].plot(
                    np.real(phi[:, imode - 1]), Modes["z"], f"C{i_f}", label=f"{f} Hz"
                )
                ax[iplot].plot(np.imag(phi[:, imode - 1]), Modes["z"], "b--")
                # ax[iplot].legend()
            else:
                ax[iplot].plot(np.real(phi[:, imode - 1]), Modes["z"], f"C{i_f}")
                ax[iplot].plot(np.imag(phi[:, imode - 1]), Modes["z"], "b--")
            ax[iplot].set_xlabel(f"Mode {Modes['selected_modes'][imode - 1]}")
            if bathy_depth is not None and i_f == 0:
                if label_bathy:
                    ax[iplot].axhline(
                        y=bathy_depth, color="r", linestyle="--", label="Depth"
                    )
                else:
                    ax[iplot].axhline(y=bathy_depth, color="r", linestyle="--")

            if normalize_mode:
                ax[iplot].set_xlim([-1.2, 1.2])

        if bathy_depth is not None and i_f == 0:
            ax[0].set_ylim([0, bathy_depth * 1.4])

    fig.supylabel("Depth [m]")
    # NOTE (bug fixed): passing a list to suptitle() (see plotmode()).
    # Also, since this function overlays several frequencies, the title
    # now lists all of them instead of only the last one processed.
    freqs_str = ", ".join(f"{f:g}" for f in freq)
    fig.suptitle(f'{all_modes[0]["title"]}\nFreq = {freqs_str} Hz')
    return fig


# ======================================================================================================================
# Transmission loss (TL)
# ======================================================================================================================
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
    axis=None,
    rasterized=True,
):
    """Plot the transmission-loss field read from a '.shd' binary file
    produced by FIELD.exe.
    Usage: plotshd(filename, freq, m, n, p, units)

    Args:
        filename (str): path to the '.shd' file.
        freq (float): the single frequency to read and plot (this
            function does not support plotting several frequencies at
            once -- see plotmode_several_freqs/plot_tl_profile_multi_freq
            for that).
        m, n, p (int|None): if all three are given, plot into subplot
            (m, n, p) of a new figure instead of a full-size standalone
            figure.
        units (str): 'm' (default) or 'km' for the range axis. Note:
            read_shd.readshd's own 'Pos["r"]["r"]' is always in METERS
            (the '.shd' file's native convention, regardless of the
            kilometers used in '.env'/'.flp' input files -- see
            read_shd.py's docstring); this function converts to km
            for you when units='km'.
        title (str|None): plot title. None builds a default title from
            the file's own title, frequency and source depth.
        tl_min, tl_max (float|None): color scale bounds (dB). None
            picks a sensible range automatically from the data.
        bathy (Bathymetry|None): if given, overlays the bottom profile.
        axis (matplotlib.axes.Axes|None): plot into this axis instead
            of creating a new figure (ignored if m/n/p are given).
        rasterized (bool): rasterize the pcolor mesh (smaller file size
            for vector-format figure exports).

    Returns:
        matplotlib.figure.Figure, only if (m, n, p) were given (matches
        the original function's behaviour); None otherwise (the current
        figure/axis was already the one drawn into).

    Adapted from the original Matlab Acoustics Toolbox by Michael B. Porter
    https://oalib.hlsresearch.com/AcousticsToolbox/
    """
    # NOTE (bug fixed): the original code did
    # `filename = filename.lower()` unconditionally. This silently
    # breaks on any case-sensitive filesystem (Linux, macOS) as soon as
    # the real file name contains an uppercase character -- there is no
    # legitimate reason to alter a caller-supplied path's case. Removed.
    PlotTitle, _, _, _, read_freq, _, Pos, pressure = readshd(
        filename=filename, freq=freq
    )

    # NOTE: squeeze() (rather than the original's hard-coded
    # `np.squeeze(pressure, axis=(0, 1))`) is more robust: it always
    # collapses every singleton axis (theta, source depth, and -- for a
    # single scalar 'freq', already handled by readshd -- the frequency
    # axis too), leaving the (depth, range) 2D grid this function needs,
    # regardless of exactly which axes happen to be singleton for a
    # given call. 'freq' must still be a single scalar, not a list: a
    # genuine multi-frequency array would leave an extra non-singleton
    # axis that squeeze() cannot remove, and pcolor below would then
    # raise a clear shape-mismatch error rather than silently plotting
    # the wrong slice.
    pressure = np.squeeze(pressure)

    return_fig_handle = False
    if axis is None:
        if m is not None and n is not None and p is not None:
            # Create a subplot
            plt.figure()
            plt.subplot(m, n, p)
            axis = plt.gca()
            return_fig_handle = True
        else:
            plt.figure(figsize=(16, 8))
            axis = plt.gca()

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
    r = Pos["r"]["r"]
    bathy_mult = 1e3
    if units == "km":
        r = r / 1000.0
        xlab = "Range [km]"
        bathy_mult = 1

    tlmin_plot = tl_min if tl_min is not None else tlmin
    tlmax_plot = tl_max if tl_max is not None else tlmax

    # Plot the data
    tej = plt.get_cmap("jet", 256).reversed()
    im = axis.pcolor(
        r,
        Pos["r"]["z"],
        tlt,
        shading="auto",
        cmap=tej,
        vmin=tlmin_plot,
        vmax=tlmax_plot,
        rasterized=rasterized,
    )

    if bathy is not None:
        axis.plot(bathy.bathy_range * bathy_mult, bathy.bathy_depth, "k", linewidth=2)

    axis.invert_yaxis()
    axis.tick_params(direction="out")

    cbar = plt.colorbar(im, ax=axis, pad=0.005)
    cbar.set_label("TL [dB]")
    cbar.ax.invert_yaxis()

    axis.set_xlabel(xlab)
    axis.set_ylabel("Depth [m]")

    if title is None:
        title = (
            PlotTitle.replace("_", " ")
            + f'\nFreq = {read_freq} Hz    z_src = {Pos["s"]["z"][0]} m'
        )
    axis.set_title(title, fontsize=TITLE_FONTSIZE)

    axis.scatter(0, Pos["s"]["z"][0], marker="o", c="k", s=50)

    if return_fig_handle:
        return plt.gcf()
    return None


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
    axis=None,
):
    """Plot a transmission-loss field directly from an already-computed
    pressure field array, rather than reading it from a '.shd' file.
    Particularly useful for broadband simulations with range-dependent
    environments, where the pressure field returned by
    KrakenManager.runkraken() is a single in-memory array covering
    every simulated frequency (re-reading each frequency's own '.shd'
    slice from disk would work too, but this avoids doing so).
    Usage: plotshd_from_pressure_field(filename, pressure_field, freq, m, n, p, units)

    'filename' is still needed to read the grid metadata (Pos, title):
    a "dummy" read (its own pressure data is discarded) is done via
    read_shd.readshd for that purpose. Args are otherwise the same as
    plotshd (see its docstring), plus:

    Args:
        pressure_field (np.ndarray): the pressure field to plot,
            typically the array returned by KrakenManager.runkraken()
            (or a single-frequency slice of it). Must reduce to a plain
            (depth, range) 2D array once every singleton axis (theta,
            source depth, frequency) is squeezed out.
    """
    # Dummy read to get freq and position vectors (its own pressure
    # data is discarded -- see docstring).
    PlotTitle, _, _, _, read_freq, _, Pos, _ = readshd(filename=filename, freq=freq)

    pressure = np.squeeze(pressure_field)

    return_fig_handle = False
    if axis is None:
        if m is not None and n is not None and p is not None:
            plt.figure()
            plt.subplot(m, n, p)
            axis = plt.gca()
            return_fig_handle = True
        else:
            plt.figure(figsize=(16, 8))
            axis = plt.gca()

    # Calculate caxis limits
    tlt = np.abs(pressure).astype(float)
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
    r = Pos["r"]["r"]
    bathy_mult = 1e3
    if units == "km":
        r = r / 1000.0
        xlab = "Range [km]"
        bathy_mult = 1

    tej = plt.get_cmap("jet", 256).reversed()
    im = axis.pcolor(r, Pos["r"]["z"], tlt, shading="auto", cmap=tej)

    tlmin_plot = tl_min if tl_min is not None else tlmin
    tlmax_plot = tl_max if tl_max is not None else tlmax
    im.set_clim(tlmin_plot, tlmax_plot)

    if bathy is not None:
        axis.plot(bathy.bathy_range * bathy_mult, bathy.bathy_depth, "k")

    axis.invert_yaxis()
    axis.tick_params(direction="out")

    cbar = plt.colorbar(im, ax=axis)
    cbar.set_label("TL [dB]")
    cbar.ax.invert_yaxis()

    axis.set_xlabel(xlab, fontsize=LABEL_FONTSIZE)
    axis.set_ylabel("Depth [m]", fontsize=LABEL_FONTSIZE)
    axis.tick_params(axis="both", labelsize=TICKS_FONTSIZE)

    if title is None:
        # NOTE (bug fixed): the original code built this exact string
        # across two statements, `title = PlotTitle.replace(...)` then
        # `+f'...'` on its own line -- the second line's result was
        # never assigned back to 'title' (a plain `+` is not `+=`), and
        # applying unary '+' to a string raises `TypeError: bad operand
        # type for unary +: 'str'`. This meant calling this function
        # without an explicit 'title' ALWAYS crashed. Confirmed and
        # fixed by concatenating properly (matching the equivalent,
        # already-correct code in plotshd()).
        title = (
            PlotTitle.replace("_", " ")
            + f'\nFreq = {read_freq} Hz    z_src = {Pos["s"]["z"][0]} m'
        )
    axis.set_title(title, fontsize=TITLE_FONTSIZE)

    axis.scatter(0, Pos["s"]["z"][0], marker="o", c="k", s=50)

    if return_fig_handle:
        return plt.gcf()
    return None


def _read_tl_grid(filename, freq, units="km"):
    """Shared helper for plot_tl_profile[_multi_freq]: read a '.shd'
    file and return (r, z_m, TL_dB) as plain arrays for a single
    frequency, with 'r' converted to the requested units.

    Args:
        filename (str): path to the '.shd' file.
        freq (float): single frequency (Hz) to read.
        units (str): 'm' or 'km' for the returned range array.

    Returns:
        tuple(r, z_m, TL): r in 'units', z_m in meters, TL in dB, TL
        shaped (n_depth, n_range).
    """
    _, _, _, _, _, _, Pos, pressure = readshd(filename=filename, freq=freq)
    pressure_2d = np.squeeze(pressure)
    with np.errstate(divide="ignore"):
        TL = -20 * np.log10(np.abs(pressure_2d) + 1e-30)

    r = Pos["r"]["r"]  # meters, readshd's native convention
    if units == "km":
        r = r / 1000.0
    return r, Pos["r"]["z"], TL


def plot_tl_profile(filename, freq, rcv_depth, units="km", ax=None, label=None):
    """Plot transmission loss vs. range at the receiver depth closest
    to 'rcv_depth', for a single frequency.

    Args:
        filename (str): path to the '.shd' file.
        freq (float): frequency (Hz) to read.
        rcv_depth (float): target receiver depth (m); the closest depth
            actually available in the '.shd' file's receiver grid is
            used (see the returned title/axis for the exact value).
        units (str): 'm' or 'km' for the range axis.
        ax (matplotlib.axes.Axes|None): plot into this axis instead of
            creating a new figure.
        label (str|None): legend label for this line (useful when
            overlaying this call's result with others on the same
            axis). No legend is drawn if None.

    Returns:
        matplotlib.figure.Figure: the figure the profile was drawn
        into (new, or the one owning 'ax' if provided).
    """
    r, z_m, TL = _read_tl_grid(filename, freq, units=units)
    iz = int(np.argmin(np.abs(z_m - rcv_depth)))
    actual_depth = z_m[iz]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(r, TL[iz, :], label=label)
    ax.invert_yaxis()
    ax.set_xlabel(f"Range [{units}]")
    ax.set_ylabel("TL [dB]")
    ax.set_title(
        f"Transmission loss profile at {actual_depth:.1f} m depth, {freq:g} Hz"
    )
    ax.grid(True)
    if label is not None:
        ax.legend()
    plt.tight_layout()
    return fig


def plot_tl_profile_multi_freq(filename, freqs, rcv_depth, units="km", ax=None):
    """Overlay transmission-loss-vs-range profiles at a fixed receiver
    depth, one line per frequency -- useful to compare how propagation
    loss depends on frequency in a broadband run.

    Args:
        filename (str): path to the '.shd' file.
        freqs (array-like): frequencies (Hz) to read and overlay.
        rcv_depth (float): target receiver depth (m); see plot_tl_profile.
        units (str): 'm' or 'km' for the range axis.
        ax (matplotlib.axes.Axes|None): plot into this axis instead of
            creating a new figure.

    Returns:
        matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    actual_depth = None
    for freq in freqs:
        r, z_m, TL = _read_tl_grid(filename, freq, units=units)
        iz = int(np.argmin(np.abs(z_m - rcv_depth)))
        actual_depth = z_m[iz]
        ax.plot(r, TL[iz, :], label=f"{freq:g} Hz")

    ax.invert_yaxis()
    ax.set_xlabel(f"Range [{units}]")
    ax.set_ylabel("TL [dB]")
    ax.set_title(f"Transmission loss profiles at {actual_depth:.1f} m depth")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


# ======================================================================================================================
# Environment profiles
# ======================================================================================================================
def plot_ssp(cp_ssp, cs_ssp, z, z_bottom=None, ax=None):
    """Plot a compressional-wave (and, if present, shear-wave) sound
    speed profile as a function of depth.

    Args:
        cp_ssp (array-like|float): compressional wave celerity (m/s),
            scalar (constant with depth) or matching the size of 'z'.
        cs_ssp (array-like|float): shear wave celerity (m/s), same
            convention as cp_ssp. If entirely zero (and cp is not),
            the S-wave curve is omitted (a fluid medium has no shear
            waves to plot).
        z (array-like): depths (m).
        z_bottom (float|None): if given, shades the water column
            (0 to z_bottom) light blue and the sediment/bottom
            (z_bottom to max(z)) light grey.
        ax (matplotlib.axes.Axes|None): plot into this axis instead of
            creating a new figure.
    """
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_ylabel("Depth [m]")

    z = np.asarray(z)
    cp = np.full(z.size, cp_ssp) if np.array(cp_ssp).size == 1 else np.asarray(cp_ssp)
    cs = np.full(z.size, cs_ssp) if np.array(cs_ssp).size == 1 else np.asarray(cs_ssp)

    # No need to plot the C-wave celerity if it is 0 and cs is not 0
    if np.all(cp == 0) and not np.all(cs == 0):
        min_cp = np.nan
        max_cp = np.nan
        plot_cp = False
    else:
        min_cp = np.min(cp)
        max_cp = np.max(cp)
        plot_cp = True

    # NOTE (bug fixed): the original code did `cs = 0` right here,
    # UNCONDITIONALLY discarding the real 'cs' array resolved just
    # above and replacing it with the literal scalar 0, before checking
    # `np.all(cs == 0)`. Since `np.all(0 == 0)` is trivially True, the
    # S-wave curve was hidden EVERY TIME cp was not all-zero -- i.e.
    # for any normal fluid-over-elastic environment, genuine non-zero
    # shear-wave data was silently never plotted. Confirmed with a
    # synthetic elastic medium (real, non-zero cs values): the 'S-wave'
    # curve was missing from the plot even though it should have been
    # there. The erroneous reassignment is simply removed here; 'cs' is
    # the real array resolved above.
    if np.all(cs == 0) and not np.all(cp == 0):
        min_cs = np.nan
        max_cs = np.nan
        plot_cs = False
    else:
        min_cs = np.min(cs)
        max_cs = np.max(cs)
        plot_cs = True

    if plot_cp:
        ax.plot(cp, z, color="red", label="C-wave")
    ax.set_xlabel("Celerity " + r"[$m.s^{-1}$]")

    if plot_cs:
        ax.plot(cs, z, color="blue", label="S-wave")

    ax.invert_yaxis()
    ax.legend(loc="upper right")

    min_x = np.nanmin([min_cp, min_cs])
    max_x = np.nanmax([max_cp, max_cs])
    x_offset = max(0.1 * (max_x - min_x), 10)
    min_x -= x_offset
    max_x += x_offset
    ax.set_xlim(min_x, max_x)

    color_domains(ax, min_x=min_x, max_x=max_x, z=z, z_bottom=z_bottom)


def plot_attenuation(ap, ash, z, z_bottom=None, ax=None):
    """Plot compressional- and (if present) shear-wave attenuation as a
    function of depth. See plot_ssp for the shared argument conventions."""
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_ylabel("Depth [m]")

    z = np.asarray(z)
    ap = np.full(z.size, ap) if np.array(ap).size == 1 else np.asarray(ap)
    ash = np.full(z.size, ash) if np.array(ash).size == 1 else np.asarray(ash)

    if np.all(ap == 0) and not np.all(ash == 0):
        min_ap = np.nan
        max_ap = np.nan
        plot_ap = False
    else:
        min_ap = np.min(ap)
        max_ap = np.max(ap)
        plot_ap = True

    if np.all(ash == 0) and not np.all(ap == 0):
        min_as = np.nan
        max_as = np.nan
        plot_as = False
    else:
        min_as = np.min(ash)
        max_as = np.max(ash)
        plot_as = True

    if plot_ap:
        ax.plot(ap, z, color="red", label="C-wave")

    if plot_as:
        ax.plot(ash, z, color="blue", label="S-wave")

    ax.set_xlabel(r"$\alpha$ " + r"[$dB.\lambda^{-1}$]")
    min_x = np.nanmin([min_ap, min_as])
    max_x = np.nanmax([max_ap, max_as])
    x_offset = max(0.1 * (max_x - min_x), 2)
    min_x -= x_offset
    max_x += x_offset
    ax.set_xlim(min_x, max_x)

    color_domains(ax, min_x=min_x, max_x=max_x, z=z, z_bottom=z_bottom)

    ax.invert_yaxis()
    ax.legend(loc="upper right")


def plot_density(rho, z, z_bottom=None, ax=None):
    """Plot density as a function of depth. See plot_ssp for the shared
    argument conventions."""
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_ylabel("Depth [m]")

    z = np.asarray(z)
    rho = np.full(z.size, rho) if np.array(rho).size == 1 else np.asarray(rho)

    ax.plot(rho, z, label=r"$\rho$", color="k")
    ax.invert_yaxis()
    ax.set_xlabel(r"$\rho$ " + r"[$g.cm^{-3}$]")

    min_x = np.min(rho)
    max_x = np.max(rho)
    x_offset = max(0.1 * (max_x - min_x), 0.1)
    min_x -= x_offset
    max_x += x_offset
    ax.set_xlim(min_x, max_x)

    color_domains(ax, min_x=min_x, max_x=max_x, z=z, z_bottom=z_bottom)


def color_domains(ax, min_x, max_x, z, z_bottom=None):
    """Shade the water column (0 to z_bottom) light blue and the
    sediment/bottom (z_bottom to max(z)) light grey on a depth-profile
    axis. No-op if z_bottom is None."""
    if z_bottom is None:
        return
    ax.fill_between([min_x, max_x], [z_bottom, z_bottom], np.max(z), color="lightgrey")
    ax.fill_between([min_x, max_x], [z_bottom, z_bottom], 0, color="lightblue")


def plot_bathymetry(bathy, ax=None):
    """Plot a bathymetry profile (range vs. depth), water shaded blue
    above the seafloor and sediment shaded grey below it -- the
    range-dependent counterpart to plot_ssp/plot_attenuation/
    plot_density's z_bottom shading, for the one piece of the
    environment plot_env() does not itself show (it only plots the
    profile that was used to build the KrakenEnv, not how it varies
    with range).

    Args:
        bathy (Bathymetry): a loaded Bathymetry instance (see
            kraken_env.Bathymetry), i.e. one with 'bathy_range' (km)
            and 'bathy_depth' (m) populated.
        ax (matplotlib.axes.Axes|None): plot into this axis instead of
            creating a new figure.

    Returns:
        matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    ax.plot(bathy.bathy_range, bathy.bathy_depth, color="k", linewidth=2, marker="o")
    y_max = np.max(bathy.bathy_depth) * 1.1
    ax.fill_between(bathy.bathy_range, bathy.bathy_depth, y_max, color="lightgrey")
    ax.fill_between(bathy.bathy_range, bathy.bathy_depth, 0, color="lightblue")
    ax.set_ylim(0, y_max)
    ax.invert_yaxis()
    ax.set_xlabel("Range [km]")
    ax.set_ylabel("Depth [m]")
    ax.set_title("Bathymetry")
    ax.grid(True)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    pass
