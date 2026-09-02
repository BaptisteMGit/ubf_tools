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

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from propa.kraken_toolbox.read_modes import readmodes
from propa.kraken_toolbox.read_shd import readshd
from cst import TICKS_FONTSIZE, TITLE_FONTSIZE, LABEL_FONTSIZE


# ======================================================================================================================
# Mode shapes
# ======================================================================================================================
def _grid_shape(n_panels, ncols=None, panel_w=3.2, panel_h=4.2, target_fig_aspect=1.6):
    """Compute a (nrows, ncols) grid for 'n_panels' subplots, chosen so
    the OVERALL figure comes out landscape (wider than tall), given
    that each individual panel is itself taller than wide
    (panel_w x panel_h, per user request).

    A grid that is "roughly square" in terms of CELL COUNT (e.g. 2x3
    for 6 panels) is not the same as a landscape FIGURE once each cell
    is itself a tall rectangle: 2 rows x 3 columns of (3.2, 4.2) panels
    gives a (9.6, 8.4) figure -- already close to square, and easily
    portrait for other panel counts. Instead, every achievable (nrows,
    ncols) pair (nrows from 1 to n_panels, ncols = ceil(n_panels /
    nrows)) is scored by how close its resulting OVERALL aspect ratio
    (ncols*panel_w / (nrows*panel_h)) comes to 'target_fig_aspect',
    restricted to pairs that are actually landscape (aspect >= 1)
    whenever at least one such pair exists.

    Args:
        n_panels (int): number of subplots needed.
        ncols (int|None): force this many columns instead of searching
            for a landscape-optimal layout.
        panel_w, panel_h (float): the size (inches) of a single panel,
            matching plotmode's own figsize-per-panel -- must be kept
            in sync with the figsize passed to plt.subplots() there.
        target_fig_aspect (float): the ideal overall width/height ratio
            to aim for among the landscape-qualifying layouts (1.6 is a
            fairly standard "wide" report-figure ratio; the search does
            not need to hit it exactly, just get close while staying
            landscape).

    Returns:
        tuple(nrows, ncols)
    """
    if ncols is not None:
        nrows = math.ceil(n_panels / ncols)
        return nrows, ncols

    candidates = []
    for nrows in range(1, n_panels + 1):
        ncols_candidate = math.ceil(n_panels / nrows)
        aspect = (ncols_candidate * panel_w) / (nrows * panel_h)
        candidates.append((nrows, ncols_candidate, aspect))

    landscape_candidates = [c for c in candidates if c[2] >= 1.0]
    if landscape_candidates:
        nrows, ncols, _aspect = min(
            landscape_candidates, key=lambda c: abs(c[2] - target_fig_aspect)
        )
    else:
        # No landscape layout is achievable at all (only possible for a
        # single panel, since panel_h > panel_w) -- fall back to
        # whichever grid comes closest to landscape.
        nrows, ncols, _aspect = max(candidates, key=lambda c: c[2])
    return nrows, ncols


def plotmode(
    filename,
    freq=0,
    n_modes=6,
    modes=None,
    bathy_depth=None,
    normalize_mode=False,
    ncols=None,
):
    """Plot mode shapes produced by KRAKEN, read from a '.mod' binary
    file: one subplot per mode, arranged in an automatically-sized
    grid, sharing a single depth axis and a single legend for the whole
    figure.

    Supports one or several frequencies at once: 'freq' can be a scalar
    or an array-like. With several frequencies, each mode's subplot
    overlays that SAME mode number's shape at every frequency (one
    color per frequency, real part solid, imaginary part dashed in the
    same color) -- the natural way to compare how a given mode changes
    with frequency. A frequency that has fewer modes than another
    simply contributes no curve to the subplots beyond its own count
    (grid size is set by whichever frequency has the most).

    This single function replaces the previous 'plotmode' (single
    frequency) and 'plotmode_several_freqs' (several frequencies), and
    fixes two presentation issues found in them along the way:
      - the bathymetry/seafloor line ('bathy_depth') is now always
        drawn on EVERY subplot. In the old 'plotmode_several_freqs',
        it was only added while processing the FIRST frequency
        (`if bathy_depth is not None and i_f == 0:`), inside a loop
        bounded by THAT frequency's own mode count -- so subplots that
        only existed because a LATER frequency had more modes never
        got the line at all.
      - the legend used to be rebuilt on every subplot (or, for
        several frequencies, repeatedly on the first one) with a
        per-frequency label on the "real part" curve only (the
        imaginary part was never labelled at all). It is now built
        ONCE, explicitly, as a single figure-level legend covering
        every frequency (if more than one) and the real/imaginary line
        style -- see the module docstring's third bug-fix-adjacent note
        above for why a single, explicit legend is more robust than
        accumulating one across repeated `ax.legend()` calls.

    Args:
        filename (str): path to the '.mod' file (extension optional).
        freq (float|array-like): frequency/frequencies (Hz) to read and
            plot.
        n_modes (int): number of modes to plot, starting from mode 1,
            when 'modes' is not given. Capped automatically by however
            many modes are actually available (at whichever frequency
            has the most).
        modes (array-like|None): explicit 1-based mode indices to plot
            instead of 1..n_modes (e.g. [1, 5, 10, 20] to inspect a
            sparse selection rather than the lowest-order modes). Give
            them in ascending order.
        bathy_depth (float|None): if given, draws a horizontal dashed
            line at this depth (the local seafloor) on every subplot,
            clips the shared depth axis to 1.4x this value, and adds a
            "Seafloor" entry to the legend.
        normalize_mode (bool): normalize each plotted mode curve to
            [-1, 1] (by its own peak absolute value).
        ncols (int|None): force this many columns in the subplot grid
            instead of the automatic, roughly-square layout.

    Returns:
        matplotlib.figure.Figure

    Raises:
        ValueError: if the mode file contains an ELASTIC medium. Mode
            shapes are read directly from Modes["phi"] (see "NOTE (bug
            fixed)" below), which requires a 1-to-1 correspondence
            between Modes["phi"]'s rows and Modes["z"]'s depths -- true
            for an ACOUSTIC-only mode file (every case in this
            project's examples), but not for an ELASTIC one (where each
            depth occupies 4 rows in Modes["phi"], see
            utils.get_component). Reading an ELASTIC file's mode shapes
            correctly needs that per-point, per-component unpacking;
            this function does not attempt it and fails loudly instead
            of silently plotting misaligned data.
    """
    # NOTE (bug fixed): this function used to extract mode shapes via
    # `utils.get_component(Modes, "N")`, which partitions Modes["phi"]'s
    # rows across media using `Modes["N"]` (the number of points per
    # medium) as the boundary. On real data, this produced visibly
    # wrong mode shapes: confirmed that `Modes["N"]` is actually the
    # number of MESH SUBDIVISIONS requested in the '.env' file's medium
    # block (KrakenMedium's own 'nmesh' parameter), not the number of
    # points in the '.mod' file's actual output z/phi grid, which is
    # typically much finer (KRAKEN interpolates internally). Confirmed
    # on a real single-ACOUSTIC-medium mode file: N=[25] while the
    # actual z/phi grid had 2601 points for that same medium --
    # get_component filled only the first 25 rows and silently left the
    # remaining 2576 at zero. For an ACOUSTIC-only mode file (every
    # example in this project), Modes["phi"]'s rows already correspond
    # 1-to-1 to Modes["z"]'s depths in order, with no need to partition
    # by medium at all -- reading it directly is both simpler and
    # actually correct, confirmed against the same real data.
    freqs = np.atleast_1d(freq).astype(float)
    requested_modes = (
        np.atleast_1d(modes).astype(int)
        if modes is not None
        else np.arange(1, n_modes + 1)
    )

    all_modes = []
    for f in freqs:
        Modes = readmodes(filename, f, requested_modes)
        if Modes["M"] == 0:
            raise Exception(f"No modes in mode file at {f} Hz")
        all_modes.append(Modes)

    return _render_mode_grid(
        all_modes,
        freqs,
        requested_modes,
        bathy_depth=bathy_depth,
        normalize_mode=normalize_mode,
        ncols=ncols,
    )


def plotmode_from_data(
    all_modes,
    freq,
    n_modes=6,
    modes=None,
    bathy_depth=None,
    normalize_mode=False,
    ncols=None,
):
    """Plot mode shapes from an already-loaded list of Modes dicts (one
    per frequency), instead of reading them from a '.mod' file. Same
    rendering and arguments as plotmode() (see its docstring) -- this
    is its counterpart for when there is no single on-disk '.mod' file
    containing every frequency's modes to read from in the first place.

    This is exactly the situation after a broadband + range-dependent
    KRAKEN run (see KrakenManager.runkraken_broadband_range_dependent /
    run_kraken.runkraken_broadband_range_dependent's module docstring):
    KRAKEN is re-run once per frequency, overwriting the SAME '.mod'
    file each time, so only the LAST frequency's modes are ever left on
    disk afterwards. Both of those functions now collect each
    frequency's Modes dict (via read_modes.readmodes) INSIDE their
    per-frequency loop, before the next iteration overwrites the file,
    and return/expose the resulting list (KrakenManager.runkraken()
    sets it on self.last_modes) for exactly this function to plot --
    mirroring how plotshd_from_pressure_field() plots a pressure field
    that was similarly collected in memory rather than read back from a
    single '.shd' file.

    Args:
        all_modes (list[dict]): one Modes dict (as returned by
            read_modes.readmodes) per frequency, in the same order as
            'freq'.
        freq (float|array-like): the frequency/frequencies 'all_modes'
            corresponds to (used for labelling only -- the actual
            frequency each Modes dict was read at is not re-derived
            from it).
        n_modes, modes, bathy_depth, normalize_mode, ncols: see
            plotmode().

    Returns:
        matplotlib.figure.Figure

    Raises:
        ValueError: if 'all_modes' and 'freq' have different lengths,
            or if any entry contains an ELASTIC medium -- see plotmode().
    """
    freqs = np.atleast_1d(freq).astype(float)
    if len(all_modes) != len(freqs):
        raise ValueError(
            f"'all_modes' has {len(all_modes)} entries but 'freq' has "
            f"{len(freqs)} -- they must correspond 1-to-1, in order."
        )
    requested_modes = (
        np.atleast_1d(modes).astype(int)
        if modes is not None
        else np.arange(1, n_modes + 1)
    )

    return _render_mode_grid(
        all_modes,
        freqs,
        requested_modes,
        bathy_depth=bathy_depth,
        normalize_mode=normalize_mode,
        ncols=ncols,
    )


def _render_mode_grid(
    all_modes, freqs, requested_modes, bathy_depth, normalize_mode, ncols
):
    """Shared rendering logic for plotmode() / plotmode_from_data(): one
    subplot per mode, arranged in an automatically-sized landscape
    grid, sharing a single depth axis and a single legend for the whole
    figure. See plotmode()'s docstring for the full behaviour
    description; this function assumes 'all_modes' has already been
    validated (each entry ACOUSTIC-only, i.e. Modes["phi"].shape[0] ==
    len(Modes["z"])) by its caller.
    """
    for f, Modes in zip(freqs, all_modes):
        if Modes["phi"].shape[0] != len(Modes["z"]):
            raise ValueError(
                f"plotmode()/plotmode_from_data() only support ACOUSTIC-only "
                f"mode files (they read Modes['phi'] directly against "
                f"Modes['z'], depth-for-depth). At {f:g} Hz, Modes['phi'] has "
                f"{Modes['phi'].shape[0]} rows but Modes['z'] has "
                f"{len(Modes['z'])} depths -- this mode data includes an "
                f"ELASTIC medium (4 rows per depth in Modes['phi']; see "
                f"utils.get_component to unpack a specific component from it)."
            )

    n_panels = min(len(requested_modes), max(m["nb_selected_modes"] for m in all_modes))
    if n_panels == 0:
        raise Exception("No modes in mode file")
    mode_numbers = requested_modes[:n_panels]

    # NOTE (per user request): panels are now taller than they are wide
    # (a "portrait" aspect ratio suits a depth profile better than the
    # previous, wider-than-tall layout).
    nrows, ncols = _grid_shape(n_panels, ncols=ncols)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(16, 10),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axs_flat = axs.ravel()
    for extra_ax in axs_flat[n_panels:]:
        extra_ax.axis("off")

    multi_freq = len(freqs) > 1
    for panel_idx, mode_number in enumerate(mode_numbers):
        ax = axs_flat[panel_idx]
        max_abs = 0.0
        for i_f, (f, Modes) in enumerate(zip(freqs, all_modes)):
            local_idx = np.flatnonzero(Modes["selected_modes"] == mode_number)
            if local_idx.size == 0:
                continue  # this frequency doesn't have this mode number
            phi_col = Modes["phi"][:, local_idx[0]]

            if normalize_mode:
                peak = np.max(np.abs(phi_col))
                if peak > 0:
                    phi_col = phi_col / peak

            color = f"C{i_f}"
            ax.plot(np.real(phi_col), Modes["z"], color=color, linestyle="-")
            ax.plot(np.imag(phi_col), Modes["z"], color=color, linestyle="--")
            max_abs = max(
                max_abs, np.max(np.abs(phi_col.real)), np.max(np.abs(phi_col.imag))
            )

        # NOTE (per user request): each panel's x-axis is centered on
        # zero (symmetric limits), rather than matplotlib's default
        # autoscale (which need not be centered, making it harder to
        # visually compare the positive/negative excursions of a mode).
        if max_abs > 0:
            ax.set_xlim(-1.05 * max_abs, 1.05 * max_abs)

        # NOTE (bug fixed, see docstring): drawn unconditionally, on
        # EVERY panel, regardless of which frequency's mode count
        # determines this panel's existence.
        if bathy_depth is not None:
            ax.axhline(y=bathy_depth, color="r", linestyle="--")

        # NOTE (per user request): the mode number is now the subplot
        # TITLE, not an x-axis label.
        ax.set_title(f"Mode {mode_number}")

        if normalize_mode:
            ax.set_xlim([-1.2, 1.2])

    if bathy_depth is not None:
        axs_flat[0].set_ylim([0, bathy_depth * 1.4])

    axs_flat[0].invert_yaxis()  # shared y-axis -> propagates to every panel

    # NOTE (per user request): a single legend for the whole figure,
    # built explicitly rather than accumulated from repeated
    # per-subplot ax.legend() calls (see docstring).
    legend_handles = []
    if multi_freq:
        for i_f, f in enumerate(freqs):
            legend_handles.append(
                Line2D([0], [0], color=f"C{i_f}", linestyle="-", label=f"{f:g} Hz")
            )
        legend_handles.append(
            Line2D([0], [0], color="k", linestyle="-", label="Real part")
        )
        legend_handles.append(
            Line2D([0], [0], color="k", linestyle="--", label="Imag part")
        )
    else:
        legend_handles.append(
            Line2D([0], [0], color="C0", linestyle="-", label="Real part")
        )
        legend_handles.append(
            Line2D([0], [0], color="C0", linestyle="--", label="Imag part")
        )
    if bathy_depth is not None:
        legend_handles.append(
            Line2D([0], [0], color="r", linestyle="--", label="Seafloor")
        )
    # NOTE (bug fixed): the legend used to be placed with
    # `loc="center left", bbox_to_anchor=(1.0, 0.5)`, which positions it
    # OUTSIDE the figure's own canvas (past its right edge). Without the
    # caller remembering to pass `bbox_inches="tight"` to `savefig(...)`
    # -- none of this project's example scripts did -- matplotlib's
    # default save behaviour only captures the canvas as sized by
    # figsize, silently CLIPPING the legend out of the saved file
    # entirely. `loc="outside center right"`, combined with
    # constrained_layout=True (already enabled above), tells matplotlib
    # to reserve real space for the legend WITHIN the canvas (shrinking
    # the subplot grid slightly to make room) instead of floating it
    # past the canvas edge, so it is always included on save, with or
    # without `bbox_inches="tight"`.
    fig.legend(handles=legend_handles, loc="outside center right")

    fig.supxlabel("Mode amplitude")
    fig.supylabel("Depth [m]")
    freqs_str = ", ".join(f"{f:g}" for f in freqs)
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
            once -- see plot_tl_profile_multi_freq for TL profiles, or
            plotmode's own multi-frequency support for mode shapes).
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
        matplotlib.figure.Figure: the figure that was drawn into (newly
        created, a new subplot's, or the one owning 'axis' if given).

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

    # NOTE (bug fixed): this function used to only return the figure
    # handle when (m, n, p) were all given (subplot mode), and `None`
    # otherwise -- even though the common, no-subplot case still
    # creates a brand new figure and draws into it. Every caller that
    # naturally expects `plotshd(...)` to hand back the figure it just
    # drew (e.g. to then call `fig.savefig(...)`) got an
    # `AttributeError: 'NoneType' object has no attribute 'savefig'`
    # in that common case -- confirmed to break every example case
    # script in propa/kraken_toolbox/examples/ that calls plotshd()
    # without (m, n, p). Fixed by always resolving and returning the
    # actual Figure that owns 'axis', regardless of how it was
    # obtained (freshly created, a new subplot, or a caller-supplied
    # axis).
    if axis is None:
        if m is not None and n is not None and p is not None:
            # Create a subplot
            plt.figure()
            plt.subplot(m, n, p)
            axis = plt.gca()
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

    return axis.figure


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
    pos=None,
    base_title=None,
):
    """Plot a transmission-loss field directly from an already-computed
    pressure field array, rather than reading it from a '.shd' file.
    Particularly useful for broadband simulations with range-dependent
    environments, where the pressure field returned by
    KrakenManager.runkraken() is a single in-memory array covering
    every simulated frequency (re-reading each frequency's own '.shd'
    slice from disk would work too, but this avoids doing so).
    Usage: plotshd_from_pressure_field(filename, pressure_field, freq, m, n, p, units)

    By default, 'filename' is used to read the grid metadata (Pos,
    title) via a "dummy" read (its own pressure data is discarded) --
    see read_shd.readshd. Pass 'pos' directly instead (see Args) to
    skip that read entirely: needed whenever no single '.shd' file
    holding this information actually exists in the first place -- see
    the NOTE below.

    Args otherwise the same as plotshd (see its docstring), plus:

    Args:
        pressure_field (np.ndarray): the pressure field to plot,
            typically the array returned by KrakenManager.runkraken()
            (or a single-frequency slice of it). Must reduce to a plain
            (depth, range) 2D array once every singleton axis (theta,
            source depth, frequency) is squeezed out.
        pos (dict|None): the receiver/source grid position dict (see
            read_shd.readshd's docstring for the 'Pos' return value) to
            use directly, instead of reading it from 'filename'. When
            given, 'filename' is not accessed at all (it may be None).
        base_title (str|None): the simulation title to use when
            building the default axis title (see 'title' below),
            instead of the 'PlotTitle' a dummy read of 'filename' would
            otherwise provide (typically env.simulation_title). Only
            used when 'pos' is given; ignored otherwise, and ignored
            entirely if 'title' is given directly.

    Returns:
        matplotlib.figure.Figure: the figure that was drawn into (newly
        created, a new subplot's, or the one owning 'axis' if given).

    NOTE: after a broadband + range-dependent KRAKEN run (see
    KrakenManager.runkraken_broadband_range_dependent's module
    docstring), KRAKEN is re-run once per frequency, overwriting the
    SAME '.shd' file each time -- so, same as for mode shapes (see
    plotmode_from_data), there is no single on-disk '.shd' file left
    containing every frequency's grid metadata to read back afterwards
    (the file that does exist afterwards holds only the LAST
    frequency's data, and typically sits in a different,
    'parallel_working_dir' subdirectory than a naive caller would
    expect). KrakenManager.runkraken() already returns 'field_pos'
    (the very same 'Pos' dict) directly, in memory, precisely so it can
    be passed here as 'pos' -- no file access needed at all in that
    case. Use 'base_title=env.simulation_title' alongside it for a
    sensible default axis title.
    """
    if pos is not None:
        # NOTE (bug fixed, see docstring above): skips the dummy
        # readshd() call entirely -- needed because, after a broadband
        # + range-dependent run, no single '.shd' file exists on disk
        # with every frequency's metadata to read it from in the first
        # place (confirmed to raise FileNotFoundError for a caller
        # naively pointing 'filename' at the expected, but never
        # actually written there, top-level path).
        Pos = pos
        PlotTitle = base_title if base_title is not None else ""
        read_freq = freq
    else:
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

    return axis.figure


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
        fig, ax = plt.subplots(figsize=(16, 8))
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
        fig, ax = plt.subplots(figsize=(16, 8))
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


def plot_tl_profile_multi_freq_from_data(
    pressure_field, freqs, field_pos, rcv_depth, units="km", ax=None
):
    """Like plot_tl_profile_multi_freq(), but reading an already-computed
    broadband pressure field (e.g. from KrakenManager.runkraken(),
    aggregated across every frequency) instead of a '.shd' file.

    NOTE: after a broadband + range-dependent KRAKEN run (see
    KrakenManager.runkraken_broadband_range_dependent's module
    docstring), KRAKEN is re-run once per frequency, overwriting the
    SAME '.shd' file each time -- so, same as for mode shapes (see
    plotmode_from_data) and single-frequency-slice TL maps (see
    plotshd_from_pressure_field), there is no single on-disk '.shd'
    file left containing every frequency's pressure field to read back
    afterwards. KrakenManager.runkraken() already returns the
    aggregated 'pressure_field' and 'field_pos' directly, in memory,
    precisely so they can be passed here.

    Args:
        pressure_field (np.ndarray): pressure field covering every
            frequency in 'freqs', typically KrakenManager.runkraken()'s
            first return value. Shape (n_freq, ...), reducing to a
            plain (depth, range) 2D array per frequency once every
            other singleton axis (theta, source depth) is squeezed out.
        freqs (array-like): the frequencies 'pressure_field' covers, in
            the same order as its first axis.
        field_pos (dict): grid position dict (see read_shd.readshd's
            'Pos' return value), typically KrakenManager.runkraken()'s
            second return value.
        rcv_depth (float): target receiver depth (m); see plot_tl_profile.
        units (str): 'm' or 'km' for the range axis.
        ax (matplotlib.axes.Axes|None): plot into this axis instead of
            creating a new figure.

    Returns:
        matplotlib.figure.Figure
    """
    freqs = np.atleast_1d(freqs).astype(float)
    r = field_pos["r"]["r"]
    if units == "km":
        r = r / 1000.0
    z_m = field_pos["r"]["z"]
    iz = int(np.argmin(np.abs(z_m - rcv_depth)))
    actual_depth = z_m[iz]

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 8))
    else:
        fig = ax.figure

    for i, f in enumerate(freqs):
        pressure_2d = np.squeeze(pressure_field[i])
        with np.errstate(divide="ignore"):
            TL = -20 * np.log10(np.abs(pressure_2d) + 1e-30)
        ax.plot(r, TL[iz, :], label=f"{f:g} Hz")

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
