#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/03/11 10:23:35
@Author  :   Menetrier Baptiste
@Version :   1.1 (refactor)
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Kraken toolbox utilities: '.env'/'.flp' line formatting,
             mode-shape component extraction, receiver grid indexing,
             and frequency-batch load balancing for parallel runs.

This module does NOT change the public API of the original file (same
function names/signatures). See the "NOTE (bug ...)" comments below for
the bugs found and fixed.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================

import numpy as np

import source.global_constants as gc
from propa.kraken_toolbox.read_shd import readshd
from scipy.optimize import minimize


# ======================================================================================================================
# Mode-shape component extraction
# ======================================================================================================================

#: Column index of each displacement/stress component within an ELASTIC
#: medium's 4-value-per-point storage block (Horizontal, Vertical,
#: Tangential stress, Normal stress).
_COMPONENT_INDEX = {"H": 0, "V": 1, "T": 2, "N": 3}


def get_component(Modes, comp):
    """Extract a single displacement/stress component of the mode shapes
    from a Modes dict returned by read_modes.readmodes().

    KRAKEN stores mode shape values (Modes["phi"]) as one row per
    "logical" point, but the number of rows a point occupies in that
    storage depends on the material of the medium it belongs to:
      - an ACOUSTIC medium point occupies 1 row (pressure only);
      - an ELASTIC medium point occupies 4 consecutive rows, in the
        order H (horizontal), V (vertical), T (tangential stress), N
        (normal stress) -- see _COMPONENT_INDEX.

    Args:
        Modes (dict): as returned by read_modes.readmodes(). Must
            contain 'Nmedia', 'N' (points per medium), 'Mater' (material
            per medium, 'ACOUSTIC' or 'ELASTIC'), 'z' (depth grid) and
            'phi' (raw mode shape storage).
        comp (str): one of 'H', 'V', 'T', 'N'.

    Returns:
        np.ndarray of shape (len(Modes["z"]), num_modes): the requested
        component of the mode shapes, at every depth of Modes["z"].

    Raises:
        Exception: if 'comp' or a medium's material is not recognized
            (kept as a plain Exception, matching the original code, for
            backward compatibility with any existing `except Exception`
            handling elsewhere in your codebase).

    NOTE (bug found -- and this function's earlier "fix" retracted):
    the previous version of this function (still visible in git
    history) walked Modes["z"] medium by medium, using
    Modes["N"][medium] to decide how many rows of that medium's
    material to consume. That was validated only against a
    self-constructed synthetic example where N happened to add up to
    len(Modes["z"]) by construction. On a REAL KRAKEN mode file, this
    assumption is false: Modes["N"] is the number of MESH SUBDIVISIONS
    requested in the '.env' file's medium block (KrakenMedium's own
    'nmesh' parameter), not the number of points in the '.mod' file's
    actual output z/phi grid, which KRAKEN typically interpolates onto a
    much finer grid internally. Confirmed on a real single-ACOUSTIC-
    medium mode file: Modes["N"] = [25] while the actual z/phi grid had
    2601 points for that same medium -- the medium-by-medium walk filled
    only the first 25 output rows and silently left the remaining 2576
    at zero, visibly corrupting every mode shape read this way (see
    propa/kraken_toolbox/plot_utils.py's plotmode(), which no longer
    uses this function for exactly this reason -- it reads
    Modes["phi"] directly instead, which is both simpler and correct
    for the common ACOUSTIC-only case).

    This version fixes that confirmed failure mode with a fast path:
    when Modes["phi"] already has exactly one row per depth in
    Modes["z"] (true whenever there is no ELASTIC medium at all, i.e.
    NMat == Ntot), it is returned directly -- no medium-by-medium
    accounting needed or possible to get wrong. The original
    medium-by-medium walk (still using Modes["N"] as the per-medium
    point count) is kept ONLY as a fallback for genuinely mixed/ELASTIC
    mode files, where Modes["phi"] has more rows than Modes["z"] has
    depths. This fallback path has NOT been validated against a real
    ELASTIC '.mod' file (none was available); treat its output with the
    same caution the previous "fix" deserved, and verify independently
    before relying on it.
    """
    if comp not in _COMPONENT_INDEX:
        raise Exception("Fatal Error in get_component: Unknown component")
    comp_index = _COMPONENT_INDEX[comp]

    n_points_total = len(Modes["z"])

    # Fast, confirmed-correct path: no ELASTIC medium at all, so
    # Modes["phi"] already has exactly one row per depth, in order.
    if Modes["phi"].shape[0] == n_points_total:
        return Modes["phi"]

    # Fallback for mixed/ELASTIC mode files -- see NOTE above: uses
    # Modes["N"] as the per-medium point count, same as the previous
    # version of this function, NOT validated against real ELASTIC data.
    num_modes = Modes["phi"].shape[1]
    phi = np.zeros((n_points_total, num_modes), dtype=np.complex64)

    k = 0
    point_idx = 0
    for medium in range(Modes["Nmedia"]):
        material = Modes["Mater"][medium]
        n_points_in_medium = Modes["N"][medium]

        for _ in range(n_points_in_medium):
            if point_idx >= n_points_total or k >= Modes["phi"].shape[0]:
                return phi

            if material == "ACOUSTIC":
                phi[point_idx] = Modes["phi"][k]
                k += 1
            elif material == "ELASTIC":
                phi[point_idx] = Modes["phi"][k + comp_index]
                k += 4
            else:
                raise Exception("Fatal Error in get_component: Unknown material type")

            point_idx += 1

    return phi


# ======================================================================================================================
# '.env'/'.flp' line formatting
# ======================================================================================================================
def align_var_description(var_line, desc):
    """Format a KRAKEN input file line: the value(s) in 'var_line',
    padded to a fixed column, followed by a human-readable comment.

    Example: align_var_description("50.0", "Nominal frequency (Hz)")
        -> "50.0                                                    ! Nominal frequency (Hz)\\n"

    Args:
        var_line (str): the value(s) to write, already formatted as
            text (this function does not format numbers itself).
        desc (str): human-readable description, written after '!'.

    Returns:
        str: a single line, newline-terminated, ready to be written to
        a '.env'/'.flp' file.
    """
    n_align = 55
    blank_space = (max(n_align - len(var_line), 3)) * " "
    return var_line + blank_space + f" ! {desc}\n"


# ======================================================================================================================
# Receiver depth grid sizing
# ======================================================================================================================
def default_nb_rcv_z(fmax, max_depth, n_per_l=7):
    """Compute a default number of receiver depths, sized to resolve the
    shortest wavelength (at fmax) with 'n_per_l' points per wavelength
    over the full water depth.

    Jensen et al. (2000), p. 446, recommend between 5 and 10 points per
    wavelength; n_per_l is floored to 5 if a smaller value is given.

    Args:
        fmax (float): highest frequency of the simulation (Hz). Must be
            strictly positive.
        max_depth (float): total depth to resolve (m).
        n_per_l (int): desired points per wavelength (floored to 5).

    Returns:
        int: number of receiver depths.

    Raises:
        ValueError: if fmax <= 0 (division by zero / meaningless
            wavelength otherwise).
    """
    if fmax <= 0:
        raise ValueError(f"fmax must be strictly positive, got {fmax}")

    if n_per_l < 5:
        n_per_l = 5

    lmin = gc.c0 / fmax
    nz = int(np.ceil(max_depth / lmin * n_per_l))
    return nz


def waveguide_cutoff_freq(waveguide_depth, c0=gc.c0):
    """Estimate the low-frequency cutoff of a waveguide of the given
    depth (below which the first propagating mode ceases to exist),
    clipped to the minimum frequency KRAKEN can reliably handle.

    Args:
        waveguide_depth (float): water depth (m). Must be strictly
            positive.
        c0 (float): reference sound speed (m/s). Defaults to
            source.global_constants.c0 (a plain float, safe to use as a
            default argument -- unlike a mutable object, it cannot be
            accidentally shared/mutated across calls).

    Returns:
        float: cutoff frequency (Hz), never below 0.15 Hz.
    """
    if waveguide_depth <= 0:
        raise ValueError(f"waveguide_depth must be strictly positive, got {waveguide_depth}")

    fc = c0 / (4 * waveguide_depth)
    minimum_kraken_freq = 0.15
    return max(minimum_kraken_freq, fc)


# ======================================================================================================================
# Receiver grid indexing
# ======================================================================================================================
def get_rcv_pos_idx(
    kraken_range=None,
    kraken_depth=None,
    shd_fpath=None,
    rcv_depth=None,
    rcv_range=None,
):
    """Build index grids (range, depth) locating specific receiver
    positions within a KRAKEN/FIELD range-depth grid.

    The KRAKEN range/depth grid can be supplied either directly
    (kraken_range + kraken_depth) or indirectly by pointing to a '.shd'
    file (shd_fpath), from which it is read via read_shd.readshd.

    Args:
        kraken_range (np.ndarray|None): full grid of ranges (as used by
            KRAKEN/FIELD). Must be supplied together with kraken_depth,
            or left as None (together with kraken_depth) to read both
            from shd_fpath instead. When read from shd_fpath, this is in
            METERS (read_shd.readshd's own convention -- see its
            docstring); if you supply kraken_range yourself, make sure
            'rcv_range' below uses the SAME units (this function does
            not know or convert units, it only compares like-for-like).
        kraken_depth (np.ndarray|None): full grid of depths (m). See
            kraken_range.
        shd_fpath (str|None): path to a '.shd' file, used to read
            kraken_range/kraken_depth when neither is supplied directly.
        rcv_depth (array-like|None): specific receiver depths (m) to
            locate in kraken_depth. None -> every depth of the grid is
            used.
        rcv_range (array-like|None): specific receiver ranges to locate
            in kraken_range, in the SAME units as kraken_range (meters
            if kraken_range came from shd_fpath). None -> every range of
            the grid is used.

    Returns:
        tuple(rr, zz, field_pos):
            rr, zz (np.ndarray): meshgrid of range/depth INDICES into
                the KRAKEN grid, matching the requested receiver
                positions (or the full grid if rcv_range/rcv_depth were
                not given).
            field_pos (dict|None): the raw field position dict read from
                shd_fpath, or None if kraken_range/kraken_depth were
                supplied directly (nothing was read from disk).

    Raises:
        ValueError: if only one of kraken_range/kraken_depth is
            provided (they must be given together, or both left as
            None), or if both are None and shd_fpath is also None.
    """
    # NOTE (bug fixed): the original code only triggered the
    # "read from shd_fpath" branch when BOTH kraken_range and
    # kraken_depth were None (`if kraken_range is None and
    # kraken_depth is None:`). Supplying only one of the two silently
    # fell through to the 'else' branch, which then crashed with a
    # confusing `AttributeError: 'NoneType' object has no attribute
    # 'size'` on whichever one was missing. Both valid usages (both
    # given, or both omitted + shd_fpath given) are preserved exactly;
    # the only change is that the invalid, previously-crashing
    # in-between case now raises a clear, actionable error.
    if kraken_range is None and kraken_depth is None:
        if shd_fpath is None:
            raise ValueError(
                "shd_fpath must be provided when kraken_range and kraken_depth are not given"
            )
        # Dummy read to get frequencies used by kraken and field grid information
        _, _, _, _, _, _, field_pos, pressure = readshd(filename=shd_fpath, freq=0)
        nr = pressure.shape[-1]
        nz = pressure.shape[-2]
        kraken_range = field_pos["r"]["r"]
        kraken_depth = field_pos["r"]["z"]
    elif kraken_range is None or kraken_depth is None:
        raise ValueError(
            "kraken_range and kraken_depth must be provided together, or both "
            "left as None (in which case shd_fpath is used to read them)."
        )
    else:
        nr = kraken_range.size
        nz = kraken_depth.size
        field_pos = None

    if rcv_range is not None:
        # No need to process the entire grid : extract pressure field at desired positions
        rcv_pos_idx_r = [
            np.nanargmin(np.abs(kraken_range - rcv_r)) for rcv_r in rcv_range
        ]
    else:
        rcv_pos_idx_r = range(nr)

    if rcv_depth is not None:
        rcv_pos_idx_z = [
            np.nanargmin(np.abs(kraken_depth - rcv_z)) for rcv_z in rcv_depth
        ]
    else:
        rcv_pos_idx_z = range(nz)

    rr, zz = np.meshgrid(rcv_pos_idx_r, rcv_pos_idx_z)

    return rr, zz, field_pos


# ======================================================================================================================
# Optimal frequency-interval splitting for parallel processing
# ======================================================================================================================
#
# Overview: when running a broadband simulation in parallel (see
# KrakenManager / run_kraken.runkraken), the CPU cost of KRAKEN+FIELD is
# not uniform across frequencies. The functions below fit a simple
# polynomial cost model (cost per unit frequency ~ a*f^2 + b*f + c, with
# coefficients 'z' = [a, b, c] pre-fitted offline on real timing data)
# and solve for the frequency-band boundaries that give every worker the
# same *expected* total CPU time, rather than the same *number* of
# frequencies.


def find_optimal_intervals(fmin, fmax, nf, n_workers, mean_cpu_time=None, z=None):
    """Find frequency-band boundaries that balance the estimated CPU
    time across 'n_workers' parallel workers processing 'nf' frequencies
    spread between fmin and fmax.

    Args:
        fmin, fmax (float): frequency range bounds (Hz). Assumes
            fmin < fmax and nf > n_workers (this is only ever called
            from that regime -- see KrakenManager.assign_frequency_intervalls
            / run_kraken.assign_frequency_intervalls, which fall back to
            an equal split otherwise).
        nf (int): total number of frequencies being distributed.
        n_workers (int): number of parallel workers.
        mean_cpu_time (float|None): average expected CPU time (s) per
            frequency over [0, 100] Hz, used as the initial guess for
            the total CPU time. Defaults to an empirically measured
            value (5.2 s).
        z (list[float]|None): [a, b, c] coefficients of the quadratic
            CPU-time-per-Hz model, pre-fitted on the 0-100 Hz range.
            Defaults to an empirically fitted set of coefficients.

    Returns:
        tuple(expected_cpu_time, freq_bounds):
            expected_cpu_time (float): estimated CPU time (s) per
                worker at the optimum.
            freq_bounds (list[float]): n_workers + 1 frequency values
                (including fmin and fmax) delimiting the n_workers
                bands.
    """
    if z is None:
        z = [0.00181359, -0.06963699, 1.38239787]  # win = 50

    if mean_cpu_time is None:
        mean_cpu_time = 5.2  # Mean cpu time for the range 0 - 100 Hz (s)

    # Initial guess: evenly spaced interior boundaries, total CPU time
    # scaled from the average per-frequency cost.
    fi = np.linspace(fmin, fmax, n_workers + 1)
    fi = fi[1:-1]
    alpha = mean_cpu_time * nf
    x0 = np.array([alpha, *fi])

    bounds = [(0, alpha), *[(fmin, fmax)] * len(fi)]
    res = minimize(objective_function, x0, args=(z, fmin, fmax), bounds=bounds)
    expected_cpu_time = res.x[0]
    freq_bounds = [fmin, *res.x[1:], fmax]

    return expected_cpu_time, freq_bounds


def objective_function(x, z, fmin, fmax):
    """Sum of squared per-band CPU-time imbalances (see build_y): the
    quantity minimized by find_optimal_intervals to equalize CPU time
    across bands."""
    Y = build_y(x, z=z, fmin=fmin, fmax=fmax)
    return np.sum(Y**2)


def build_y(x, z, fmin, fmax):
    """Compute, for every candidate band, the difference between that
    band's estimated CPU time (via g()) and the target per-worker CPU
    time (x[0]). A perfect solution has every entry equal to zero.

    Args:
        x (array-like): [alpha, *interior_frequency_bounds], where alpha
            is the target CPU time per worker.
        z, fmin, fmax: see find_optimal_intervals.
    """
    alpha = x[0]
    fi = x[1:]
    Y = np.array([g(fi, alpha, k, z, fmin, fmax) for k in range(len(fi) + 1)])
    return Y


def g(fi, alpha, k, z, fmin, fmax):
    """Estimated CPU time of the k-th frequency band [x[k], x[k+1]]
    (with x = [fmin, *fi, fmax]) under the quadratic cost-per-Hz model
    'z' = [a, b, c], minus the target CPU time 'alpha'.

    The model integrates a*f^2 + b*f + c over the band (its analytic
    antiderivative is a/3*f^3 + b/2*f^2 + c*f), which is the standard
    way to turn a "cost density" model into a per-band total cost.
    """
    x = [fmin, *fi, fmax]
    a, b, c = z
    gk = (
        a / 3 * (x[k + 1] ** 3 - x[k] ** 3)
        + b / 2 * (x[k + 1] ** 2 - x[k] ** 2)
        + c * (x[k + 1] - x[k])
        - alpha
    )
    return gk
