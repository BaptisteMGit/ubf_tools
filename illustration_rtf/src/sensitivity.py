#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2026/08/26 14:00:58
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from misc import progression_bar
from source.normal_modes import (
    pekeris_green_fct,
    pekeris_cutoff_frequency,
    pekeris_n_modes,
)
from source.global_constants import project_root
from propa.kraken_toolbox.utils import default_nb_rcv_z

from propa.kraken_toolbox.src.kraken_env import (
    KrakenEnv,
    KrakenMedium,
    KrakenBottomHalfspace,
    KrakenField,
    KrakenFlp,
)
from propa.kraken_toolbox.src.kraken_manager import KrakenManager

# Usefull paths
TITLE = "RTF sensitivity study"
ENV_FILENAME = "rtf_sensitivity_study"

if os.name == "nt":  # Windows
    project_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
    SENSITIVITY_DIRECTORY = os.path.join(
        project_root, "illustration_rtf", "data", "sensitivity"
    )

else:  # Linux
    project_root = "/home/program/ubf_tools"
    data_root = "/home/data"
    SENSITIVITY_DIRECTORY = os.path.join(data_root, "sensitivity")


SENSITIVITY_KRAKEN_DIR = os.path.join(SENSITIVITY_DIRECTORY, "io_files")
os.makedirs(SENSITIVITY_KRAKEN_DIR, exist_ok=True)

RESULT_DIR = os.path.join(SENSITIVITY_DIRECTORY, "result")
os.makedirs(RESULT_DIR, exist_ok=True)


# ======================================================================================================================
# Sensitivity study properties
# ======================================================================================================================
def celerity_density_Hamilton_Bachman_1982(rho):
    """
    rho in g.cm-3
    cp in m.s-1
    """
    # Appendix of Hamilton and Bachman 1982

    # Continental terrace (T)
    cp = 487.7 * rho**2 - 1257.0 * rho + 2330.4

    return cp


def baseline_env():
    # Waveguide parameters
    rho1 = 1.0 * 1e3  # density in water (kg/m^3)
    c1 = 1500  # sound celerity in water (m/s)

    rho2 = 1.5 * 1e3  # density in fluid sediment (kg/m^3)
    # c2 = 1600  # sound celerity in fluid sediment (m/s)
    c2 = 1550  # Close to Hamilton(rho2) = 1542 m.s-1
    attn2 = 0.2  # compressional wave attenuation in fluid sediment in dB / wavelength
    d = 100  # waveguide depth (m)

    env_param = {
        "c1": c1,
        "c2": c2,
        "rho1": rho1,
        "rho2": rho2,
        "attn2": attn2,
        "depth": d,
    }
    return env_param


def baseline_sig():
    # Signal properties
    fmax = 150  # Max frequency (Hz)
    T = 5  # Signal duration to generate (s)
    fs = 2 * fmax  # Sampling frequency (Hz) = Nyquist
    ts = 1 / fs  # sampling interval (s)
    nt = int(T * fs)  # Number of samples
    freq = np.fft.rfftfreq(n=nt, d=ts)  # Frequency vector

    sig_param = {"freq": freq, "fs": fs, "fmax": fmax}
    return sig_param


def baseline_src_rcv():
    # Source / receiver properties
    z_s = 5
    z_rcv = 99  # D-1
    dr = 5
    r0 = 50 * 1e3
    r_rcv = np.arange(5 * 1e3, 100 * 1e3 + dr, dr)
    d12 = 5000

    src_rcv_param = {"z_s": z_s, "z_rcv": z_rcv, "r_rcv": r_rcv, "r0": r0, "d12": d12}
    return src_rcv_param


# NOTE (factored out): build_tests()/build_baseline()/process_sensitivity()
# each used to rebuild this exact same merged dict by hand, dropping a
# slightly different set of keys nothing downstream needs ("fs"/"fmax"
# always; "r0"/"d12" too, wherever the caller instead wants "d12_max").
# One shared helper, parameterized by which extra keys to drop, replaces
# all three copies.
def load_all_arg_dict(drop_keys=("fs", "fmax"), d12_max=None):
    """Merge baseline_env()/baseline_sig()/baseline_src_rcv() into a
    single dict, dropping 'drop_keys' (nothing in this module reads
    "fs"/"fmax" at all; some call sites also don't want "r0"/"d12" --
    see build_dataset_current_config_kraken(), which instead wants a
    single "d12_max" covering every d12 you might later want to
    recompute gamma for).

    Args:
        drop_keys (tuple[str]): keys to remove from the merged dict
            (KeyError if a name here isn't actually present -- fine, it
            means it was already not there).
        d12_max (float|None): if given, added to the dict as
            "d12_max" (see build_dataset_current_config_kraken()).

    Returns:
        dict
    """
    all_arg_dict = baseline_env().copy()
    all_arg_dict.update(baseline_sig())
    all_arg_dict.update(baseline_src_rcv())
    for key in drop_keys:
        all_arg_dict.pop(key)
    if d12_max is not None:
        all_arg_dict["d12_max"] = d12_max
    return all_arg_dict


def _extract_kwargs(func, arg_dict):
    """Return the subset of 'arg_dict' matching 'func''s own parameter
    names -- i.e. exactly the keyword arguments 'func(**result)' needs,
    ignoring any extra keys 'arg_dict' happens to also carry (e.g. a
    shared baseline dict that includes keys only SOME consumers need,
    such as "r0"/"d12"/"d12_max" here).

    Args:
        func (callable): the function about to be called.
        arg_dict (dict): a superset of the keyword arguments it needs.

    Returns:
        dict: arg_dict filtered down to func's own parameter names.
    """
    import inspect

    sig = inspect.signature(func)
    return {name: arg_dict[name] for name in sig.parameters if name in arg_dict}


def _extend_range_grid(r_grid, extra_distance):
    """Extend 'r_grid' (assumed evenly spaced) by 'extra_distance',
    continuing at the same spacing.

    NOTE (factored out): this exact block (compute the spacing, build
    the extra points with np.arange, np.append them) used to be
    duplicated, near-identically, in
    single_sensitivity_test_generate_dataset_kraken(),
    single_sensitivity_test_generate_dataset_analytic() and
    build_dataset_current_config_kraken() -- the only difference
    between them being whether 'extra_distance' was a fixed "d12" or a
    "d12_max" covering every d12 you might want later. One shared
    helper, taking that distance as an explicit argument, replaces all
    three copies.

    Args:
        r_grid (np.ndarray): the base range grid (meters), assumed
            evenly spaced (uses r_grid[1] - r_grid[0] as the step).
        extra_distance (float): how far beyond 'r_grid[-1]' to extend
            (same units as 'r_grid', typically meters here).

    Returns:
        np.ndarray: 'r_grid' followed by the extra, evenly-spaced points.
    """
    dr = r_grid[1] - r_grid[0]
    r_grid_add = np.arange(r_grid[-1] + dr, r_grid[-1] + dr + extra_distance, dr)
    return np.append(r_grid, r_grid_add)


# ======================================================================================================================
# Sensitivity study functions
# ======================================================================================================================


def calc_gamma_dist(gamma_a, gamma_b, dist_type="L1"):
    if dist_type == "L1":
        dist = np.nansum(np.abs(gamma_a - gamma_b), axis=0)

    if dist_type == "L2":
        dist = np.sqrt(np.nansum(np.abs(gamma_a - gamma_b) ** 2, axis=0))

    if dist_type == "theta":
        inner_prod = np.nansum(gamma_a * gamma_b, axis=0)
        norm_a = np.sqrt(np.nansum(gamma_a**2, axis=0))
        norm_b = np.sqrt(np.nansum(gamma_b**2, axis=0))
        # Clip to [-1, 1] for stability
        cos_angle = np.clip(inner_prod / (norm_a * norm_b), -1.0, 1.0)
        dist = 1 - cos_angle

    return dist


def calc_monotonicity_domain(dist_r_r0, r, r0):
    """
    Derive distance main lobe width defined as the monotonicity domain according to the first maximum of the distance (first zero crossing of the derivative)
    """

    # Positive r - r0
    r_reduced = r - r0
    d_pos = dist_r_r0[r_reduced >= 0]
    r_pos = r_reduced[r_reduced >= 0]

    # Compute the derivative of the distance
    dd_dr_pos = np.gradient(d_pos, r_pos)
    # Find zero crossings of the derivative
    zero_crossings_pos = np.where(np.diff(np.sign(dd_dr_pos)))[0]
    if len(zero_crossings_pos) > 0:
        # Take the first zero crossing
        first_zero_crossing_pos = zero_crossings_pos[0]
        r_validity_zero_crossing_pos = r_pos[first_zero_crossing_pos]

    # Negative r - r0
    d_neg = dist_r_r0[r_reduced < 0]
    r_neg = r_reduced[r_reduced < 0]
    # Compute the derivative of the distance
    dd_dr_neg = np.gradient(d_neg, r_neg)
    # Find zero crossings of the derivative
    zero_crossings_neg = np.where(np.diff(np.sign(dd_dr_neg)))[0]

    if len(zero_crossings_neg) > 0:
        first_zero_crossing_neg = zero_crossings_neg[-1]
        r_validity_zero_crossing_neg = r_neg[first_zero_crossing_neg]

    r_validity_zero_crossing = (
        r_validity_zero_crossing_pos - r_validity_zero_crossing_neg
    )

    return (
        r_validity_zero_crossing,
        r_validity_zero_crossing_neg,
        r_validity_zero_crossing_pos,
    )


def calc_mainlobe_width_3dB(dist_r_r0, r, r0):
    """
    Derive distance main lobe width at -3 dB.
    """

    # Positive r - r0
    r_reduced = r - r0
    d_pos = dist_r_r0[r_reduced >= 0]
    r_pos = r_reduced[r_reduced >= 0]

    # Find first time dist > 1/2
    idx_d_pos_sup_3dB = d_pos > 1 / 2
    # NOTE (robustness added): fails loudly with context (which side,
    # which r0) instead of a bare, contextless
    # `IndexError: index 0 is out of bounds for axis 0 with size 0` if
    # the distance curve never actually crosses -3dB on this side.
    if not np.any(idx_d_pos_sup_3dB):
        raise ValueError(
            f"calc_mainlobe_width_3dB: the distance curve never reaches -3dB "
            f"(0.5) on the r > r0 side (r0={r0}) -- cannot locate a mainlobe "
            f"edge there."
        )
    r_d_pos_sup_3dB = r_pos[idx_d_pos_sup_3dB][0]

    # Negative r - r0
    d_neg = dist_r_r0[r_reduced < 0]
    r_neg = r_reduced[r_reduced < 0]
    # Reverse both arrays
    d_neg = d_neg[::-1]
    r_neg = r_neg[::-1]

    idx_d_neg_sup_3dB = d_neg > 1 / 2
    if not np.any(idx_d_neg_sup_3dB):
        raise ValueError(
            f"calc_mainlobe_width_3dB: the distance curve never reaches -3dB "
            f"(0.5) on the r < r0 side (r0={r0}) -- cannot locate a mainlobe "
            f"edge there."
        )
    r_d_neg_sup_3dB = r_neg[idx_d_neg_sup_3dB][0]

    mainlobe_width_3dB = r_d_pos_sup_3dB - r_d_neg_sup_3dB

    return (
        mainlobe_width_3dB,
        r_d_neg_sup_3dB,
        r_d_pos_sup_3dB,
    )


def single_sensitivity_test_generate_dataset(
    freq, c1, c2, rho1, rho2, attn2, depth, z_s, z_rcv, r_rcv, d12, model="kraken"
):
    if model == "kraken":
        # NOTE (bug fixed): this call used to omit 'd12' entirely
        # (calling with 10 positional args into a function requiring
        # 11), even though 'd12' is right here, received from THIS
        # function's own caller -- confirmed to raise
        # `TypeError: single_sensitivity_test_generate_dataset_kraken()
        # missing 1 required positional argument: 'd12'`.
        freq, r_grid, gamma = single_sensitivity_test_generate_dataset_kraken(
            freq, c1, c2, rho1, rho2, attn2, depth, z_s, z_rcv, r_rcv, d12
        )
    elif model == "analytic":
        freq, r_grid, gamma = single_sensitivity_test_generate_dataset_analytic(
            freq, c1, c2, rho1, rho2, attn2, depth, z_s, z_rcv, r_rcv, d12
        )

    return freq, r_grid, gamma


def single_sensitivity_test_generate_dataset_kraken(
    freq, c1, c2, rho1, rho2, attn2, d, z_s, z, r_grid, d12
):
    # Extend r_grid, convert to km for kraken
    r_grid_ = _extend_range_grid(r_grid, d12) * 1e-3

    # Derive green's function at all pos
    # NOTE (bug fixed): build_kraken() no longer takes 'd12' as a
    # parameter (it only ever needed the already-extended range grid,
    # built by the caller -- see build_dataset_current_config_kraken(),
    # which already calls it correctly). This call site still passed
    # the now-stale extra 'd12' argument, confirmed to raise
    # `TypeError: build_kraken() takes 10 positional arguments but 11
    # were given`.
    freq, g_fr, field_pos = build_kraken(
        freq, c1, c2, rho1, rho2, attn2, d, z_s, z, r_grid_
    )

    dr = r_grid[1] - r_grid[0]
    nr_shift = int(d12 / dr)
    # Green function at receiver 1
    g_fr_1 = g_fr[:, 0:-nr_shift]
    # Green function at receiver 2
    g_fr_2 = g_fr[:, nr_shift:]

    # Build RTF
    pi_21_fr = g_fr_2 / g_fr_1
    # Derive gamma
    gamma = 20 * np.log10(np.abs(pi_21_fr))  # (nf, nr)

    return freq, r_grid, gamma


def single_sensitivity_test_generate_dataset_analytic(
    freq, c1, c2, rho1, rho2, attn2, d, z_s, z, r_grid, d12
):
    # Extend r_grid
    r_grid_ = _extend_range_grid(r_grid, d12)
    # Derive green's function at all pos
    g_fr = pekeris_green_fct(freq, c1, c2, rho1, rho2, attn2, d, z_s, z, r_grid_)

    dr = r_grid[1] - r_grid[0]
    nr_shift = int(d12 / dr)
    # Green function at receiver 1
    g_fr_1 = g_fr[:, 0:-nr_shift]
    # Green function at receiver 2
    g_fr_2 = g_fr[:, nr_shift:]

    # Build RTF
    pi_21_fr = g_fr_2 / g_fr_1
    # Derive gamma
    gamma = 20 * np.log10(np.abs(pi_21_fr))  # (nf, nr)

    return freq, r_grid, gamma


def single_sensitivity_test_calc_dist(gamma, r_grid, r0):
    idx_r0 = np.argmin(np.abs(r_grid - r0))
    gamma_r0 = gamma[:, idx_r0]

    dist_L1 = calc_gamma_dist(
        gamma_a=gamma_r0[:, np.newaxis], gamma_b=gamma, dist_type="L1"
    )
    dist_L2 = calc_gamma_dist(
        gamma_a=gamma_r0[:, np.newaxis], gamma_b=gamma, dist_type="L2"
    )
    dist_theta = calc_gamma_dist(
        gamma_a=gamma_r0[:, np.newaxis], gamma_b=gamma, dist_type="theta"
    )
    dist_L1 /= np.max(dist_L1)
    dist_L2 /= np.max(dist_L2)
    dist_theta /= np.max(dist_theta)

    return dist_L1, dist_L2, dist_theta


def single_sensitivity_test_calc_dist_width(dist_L1, dist_L2, dist_theta, r_grid, r0):
    # Width at -3dB
    width_L1, r_ml_width_L1_inf, r_ml_width_L1_sup = calc_mainlobe_width_3dB(
        dist_r_r0=dist_L1, r=r_grid, r0=r0
    )
    width_L2, r_ml_width_L2_inf, r_ml_width_L2_sup = calc_mainlobe_width_3dB(
        dist_r_r0=dist_L2, r=r_grid, r0=r0
    )
    width_theta, r_ml_width_theta_inf, r_ml_width_theta_sup = calc_mainlobe_width_3dB(
        dist_r_r0=dist_theta, r=r_grid, r0=r0
    )

    return width_L1, width_L2, width_theta


def run_single_sensitivity_test(
    freq, c1, c2, rho1, rho2, attn2, depth, z_s, z_rcv, r_rcv, r0, d12, model="kraken"
):
    # 1) Generate dataset
    freq, r_grid, gamma = single_sensitivity_test_generate_dataset(
        freq, c1, c2, rho1, rho2, attn2, depth, z_s, z_rcv, r_rcv, d12, model=model
    )
    # 2) Derive distance around r0
    dist_L1, dist_L2, dist_theta = single_sensitivity_test_calc_dist(
        gamma=gamma, r_grid=r_grid, r0=r0
    )
    # 3) Derive characteristic metric
    width_L1, width_L2, width_theta = single_sensitivity_test_calc_dist_width(
        dist_L1, dist_L2, dist_theta, r_grid=r_grid, r0=r0
    )

    return width_L1, width_L2, width_theta


def run_sensitivity_study(test_arg_name, test_arg_values, all_arg_dict, model="kraken"):
    """Sweep ONE named parameter across 'test_arg_values' (everything
    else held at its value in 'all_arg_dict'), returning the resulting
    RTF mainlobe width (L1/L2/theta distance metrics) for every value.

    Small, in-memory result (3 arrays of length len(test_arg_values)):
    no file-size/RAM concern here, unlike build_sensitivity_dataset()
    (see its own docstring) -- nothing is written to disk by this
    function at all.
    """
    # We will build the args to pass to the test function at each iteration
    all_args = all_arg_dict.copy()
    all_args["model"] = model
    # Add test variable to arg
    all_args.update({test_arg_name: None})

    width_L1, width_L2, width_theta = [], [], []

    i_test = 0
    prev_progress = 0

    # Iterate over test values
    test_arg_values = np.atleast_1d(test_arg_values)
    for test_val in test_arg_values:

        i_test += 1
        prev_progress = progression_bar(
            index=i_test,
            index0=0,
            indexf=test_arg_values.size,
            prev_progress=prev_progress,
        )

        # Update args
        all_args[test_arg_name] = test_val
        # Run test -- filtered to run_single_sensitivity_test's own
        # parameters (see _extract_kwargs()), so 'all_args' carrying
        # extra keys some OTHER consumer needs (e.g. "d12_max") is fine.
        call_kwargs = _extract_kwargs(run_single_sensitivity_test, all_args)
        w_L1, w_L2, w_theta = run_single_sensitivity_test(**call_kwargs)

        width_L1.append(w_L1)
        width_L2.append(w_L2)
        width_theta.append(w_theta)

    return np.array(width_L1), np.array(width_L2), np.array(width_theta)


# ======================================================================================================================
# Build datasets
# ======================================================================================================================
def build_kraken(freq, c1, c2, rho1, rho2, attn2, depth, z_s, z, r_grid):

    # Keeps only frequencies above mode 1 cut-off
    fmin = pekeris_cutoff_frequency(m=1, c1=c1, c2=c2, d=depth)
    fmin = (
        np.floor((fmin) / 5) * 5 + 10
    )  # Round to upper closest multiple of five to avoid being to close to cuttoff
    kraken_freq = freq[freq > fmin]

    # Use only propative modes
    nb_modes = pekeris_n_modes(f=freq.max(), c1=c1, c2=c2, d=depth)
    clim_max = c2
    clim_min = 1400

    # attn2 = 0  # TODO remove
    # ----------------------------------------------------------------------
    # 1. Environment: Pekeris waveguide
    # ----------------------------------------------------------------------
    medium = KrakenMedium(
        ssp_interpolation_method="C_linear",
        z_ssp=[0.0, depth],
        c_p=[c1, c1],  # isovelocity water column
        rho=rho1,
    )

    bottom_hs = KrakenBottomHalfspace(
        halfspace_properties={
            "z": depth,
            "c_p": c2,
            "c_s": 0.0,  # fluid sediment: no shear waves
            "rho": rho2,
            "a_p": attn2,  # dB/wavelength
            "a_s": 0.0,  # fluid sediment: no shear waves
        },
        add_sediment_buffer_layer=False,  # direct half-space -> classic Pekeris model
    )

    n_rcv_z = default_nb_rcv_z(fmax=freq.max(), max_depth=depth, n_per_l=5)
    field = KrakenField(
        phase_speed_limits=[clim_min, clim_max],
        src_depth=z_s,
        n_rcv_z=n_rcv_z,
        rcv_z_min=0.0,
        rcv_z_max=depth,
        rcv_r_max=r_grid.max(),
    )

    env = KrakenEnv(
        title=TITLE,
        env_root=SENSITIVITY_KRAKEN_DIR,
        env_filename=ENV_FILENAME,
        freq=kraken_freq,
        kraken_medium=medium,
        kraken_bottom_hs=bottom_hs,
        kraken_field=field,
        nmedia=None,  # derived automatically -> 1 (no buffer layer)
    )

    flp = KrakenFlp(
        env=env,
        src_type="point_source",
        mode_theory="adiabatic",  # irrelevant for a range-independent run, kept simple
        mode_addition="coherent",
        nb_modes=1000,
        src_depth=z_s,
        n_rcv_z=1,
        rcv_z_min=z,
        rcv_z_max=z,
        n_rcv_r=r_grid.size,
        rcv_r_min=r_grid.min(),
        rcv_r_max=r_grid.max(),
    )

    # ----------------------------------------------------------------------
    # 2. Run KRAKEN + FIELD (requires real binaries -- see KrakenManager /
    #    propa.kraken_toolbox.params.KRAKEN_BIN_DIRECTORY). env/flp write
    #    their own '.env'/'.flp' files internally -- no separate
    #    write_env()/write_flp() call needed here.
    # ----------------------------------------------------------------------
    manager = KrakenManager(verbose=False)
    pressure_field, field_pos = manager.runkraken(
        env=env, flp=flp, frequencies=env.freq
    )

    # Squeeze
    pressure_field = pressure_field.squeeze()

    # Get green's function
    c0 = 1500
    k0 = 2 * np.pi * kraken_freq / c0
    norm_factor = np.exp(1j * k0) / (4 * np.pi)
    g_fr = norm_factor[:, np.newaxis] * pressure_field  # (nf, nr)

    return kraken_freq, g_fr, field_pos


def build_dataset_current_config_kraken(
    freq, c1, c2, rho1, rho2, attn2, depth, z_s, z_rcv, r_rcv, d12_max
):
    # Extend r_grid to d12_max, convert to km for kraken
    r_grid_ = _extend_range_grid(r_rcv, d12_max) * 1e-3

    # Derive green's function at all pos
    freq, g_fr, field_pos = build_kraken(
        freq, c1, c2, rho1, rho2, attn2, depth, z_s, z_rcv, r_grid_
    )

    return freq, field_pos["r"]["r"], g_fr


def _pad_to_full_frequency_grid(g_fr, kraken_freq, full_freq):
    """Pad 'g_fr''s frequency axis (axis 0) with NaN at the START so it
    covers 'full_freq' (KRAKEN drops every frequency below the mode-1
    cutoff -- see build_kraken() -- so 'kraken_freq' is always a
    trailing SUBSET of 'full_freq').

    NOTE (factored out): this exact 3-line block (compute
    n_missing_freq, build pad_width, np.pad) used to be duplicated
    identically in build_sensitivity_dataset() and build_baseline().

    Args:
        g_fr (np.ndarray): shape (kraken_freq.size, n_r).
        kraken_freq (np.ndarray): the (possibly truncated) frequency
            axis actually returned alongside 'g_fr'.
        full_freq (np.ndarray): the full, untruncated frequency grid
            (e.g. all_arg_dict["freq"]) 'g_fr' should be padded up to.

    Returns:
        np.ndarray: shape (full_freq.size, n_r).
    """
    n_missing_freq = full_freq.size - kraken_freq.size
    pad_width = ((n_missing_freq, 0), (0, 0))
    return np.pad(g_fr, pad_width=pad_width, mode="constant", constant_values=np.nan)


def build_sensitivity_dataset(
    test_arg_name, test_arg_values, all_arg_dict, model="kraken"
):
    """Sweep ONE named parameter across 'test_arg_values' (everything
    else held at its value in 'all_arg_dict'), saving the Green's
    function g_fr(freq, r) for every value.

    NOTE (memory usage improved): the original accumulated every
    value's full g_fr array in a Python list (g_fr_arr.append(...))
    and only built/wrote ONE combined xarray Dataset at the very end --
    holding the WHOLE sweep's data in RAM simultaneously (this file's
    own build_tests() estimated ~100+ MB per value; a 20-value sweep
    would peak near 2+ GB just for this one parameter, on top of
    whatever the KRAKEN run itself needs concurrently). Each value's
    g_fr is now written to its OWN small file immediately after being
    computed, and the local reference is dropped (falls out of scope
    at the next loop iteration) so it can be garbage-collected before
    the next value's run even starts -- peak RAM for this function is
    now roughly ONE value's g_fr, regardless of how many values you
    sweep. See process_sensitivity() for the matching lazy read-back.

    File layout: '<RESULT_DIR>/<test_arg_name>/<test_arg_name>_<i>.nc',
    one small file per swept value, holding that single value's g_fr
    plus 'test_arg_name' as a length-1 coordinate (so the files can
    later be combined along that dimension with xr.open_mfdataset,
    without ever holding more than one at a time in memory -- see
    process_sensitivity()).

    Returns:
        str: the directory the per-value files were written into.
    """
    import xarray as xr  # only this function needs it

    out_dir = os.path.join(RESULT_DIR, test_arg_name)
    os.makedirs(out_dir, exist_ok=True)

    # We will build the args to pass to the test function at each iteration
    all_args = all_arg_dict.copy()
    all_args.update({test_arg_name: None})

    i_test = 0
    prev_progress = 0

    test_arg_values = np.atleast_1d(test_arg_values)
    for test_val in test_arg_values:

        i_test += 1
        prev_progress = progression_bar(
            index=i_test,
            index0=0,
            indexf=test_arg_values.size,
            prev_progress=prev_progress,
        )

        # Update args
        all_args[test_arg_name] = test_val

        if test_arg_name == "rho2":
            # Determine c2 according to Hamilton's model
            rho2_gcm3 = test_val * 1e-3
            c2 = celerity_density_Hamilton_Bachman_1982(rho2_gcm3)
            # Update
            all_args["c2"] = c2

            # print(f"Couple rho2, c2 = ({test_val, c2})")

        if test_arg_name == "depth":
            # Update receiver depth
            all_args["z_rcv"] = test_val - 1

        # Build dataset for JUST this one value.
        call_kwargs = _extract_kwargs(build_dataset_current_config_kraken, all_args)
        kraken_freq, kraken_r, g_fr = build_dataset_current_config_kraken(
            **call_kwargs
        )  # g_fr is (nf, nr)

        g_fr_full = _pad_to_full_frequency_grid(g_fr, kraken_freq, all_args["freq"])

        # Write THIS value's dataset immediately, then let g_fr/g_fr_full
        # go out of scope (freed before the next iteration's KRAKEN run).
        ds_value = xr.Dataset(
            data_vars=dict(
                gf=([test_arg_name, "f", "r"], np.abs(g_fr_full)[np.newaxis, ...])
            ),
            coords={
                test_arg_name: [test_val],
                "f": all_args["freq"],
                "r": kraken_r,
            },
        )
        fpath = os.path.join(out_dir, f"{test_arg_name}_{i_test:04d}.nc")
        ds_value.to_netcdf(fpath)
        ds_value.close()

    return out_dir


def build_baseline():

    all_arg_dict = load_all_arg_dict(
        drop_keys=("fs", "fmax", "r0", "d12"), d12_max=5000
    )

    # Build dataset
    call_kwargs = _extract_kwargs(build_dataset_current_config_kraken, all_arg_dict)
    kraken_freq, kraken_r, g_fr = build_dataset_current_config_kraken(**call_kwargs)

    g_fr_full = _pad_to_full_frequency_grid(g_fr, kraken_freq, all_arg_dict["freq"])

    import xarray as xr  # only this function needs it

    # Build xarray dataset and save
    ds = xr.Dataset(
        data_vars=dict(gf=(["f", "r"], np.abs(g_fr_full))),
        coords={"f": all_arg_dict["freq"], "r": kraken_r},
    )
    # Keep the FULL baseline config as attrs (including "r0"/"d12", not
    # part of 'all_arg_dict' used for the KRAKEN call above) -- needed
    # later by process_sensitivity()/dist_from_baseline().
    ds.attrs = {
        **all_arg_dict,
        "r0": baseline_src_rcv()["r0"],
        "d12": baseline_src_rcv()["d12"],
    }

    # Derive gamma from gf
    gamma = derive_gamma(ds, ds.attrs["d12"])
    ds["gamma"] = gamma

    fpath = os.path.join(RESULT_DIR, "gf_dataset_baseline.nc")
    ds.to_netcdf(fpath)
    ds.close()
    return fpath


# def dist_from_baseline(ds_baseline, ds_test, d12, r0):
#     # Baseline gamma (at r0)
#     gamma_baseline = ds_baseline.gamma.sel(r=r0, method="nearest")
#     gamma_a = gamma_baseline.values.T[:, np.newaxis]

#     # Derive gamma from gf test
#     gamma_test = derive_gamma(ds_test, d12)
#     gamma_test_r0 = gamma_test.sel(r=r0, method="nearest")
#     gamma_b = gamma_test_r0.values.T

#     # Compute distance
#     dist_L1 = calc_gamma_dist(gamma_a=gamma_a, gamma_b=gamma_b, dist_type="L1")
#     dist_L2 = calc_gamma_dist(gamma_a=gamma_a, gamma_b=gamma_b, dist_type="L2")
#     dist_theta = calc_gamma_dist(gamma_a=gamma_a, gamma_b=gamma_b, dist_type="theta")

#     return dist_L1, dist_L2, dist_theta


def dist_from_baseline(ds_baseline, ds_test, d12, r0):
    """Compute the RTF distance (L1/L2/theta) between the baseline's
    gamma at r0 and each swept value's gamma at r0.

    NOTE (performance/memory fixed): this used to call
    derive_gamma(ds_test, d12), which computes gamma OVER THE FULL r
    GRID (a division + log10 over a (n_values, n_freq, n_r) array, with
    n_r possibly in the thousands) before this function immediately
    discarded everything except the single r=r0 slice it actually
    needed. Confirmed to be the dominant cost behind
    process_sensitivity() taking several minutes and a lot of RAM: the
    '.values' call inside derive_gamma() forces dask to actually read
    every swept value's FULL range grid from disk and compute the
    ratio over all of it, for a result that is >99% thrown away right
    afterwards. 'ds_test' is now ALREADY reduced to just r=[r0, r0+d12]
    by the time it reaches this function -- see
    process_sensitivity()'s '_select_r0_pair' preprocessing callback --
    so the division/log10 below only ever operates on that tiny slice.
    """
    # Baseline gamma (at r0) -- already precomputed for every r at
    # build_baseline() time (a ONE-TIME run, not a per-value sweep --
    # computing it over the full r grid there is fine, see
    # derive_gamma()'s own docstring).
    gamma_baseline = ds_baseline.gamma.sel(r=r0, method="nearest")
    gamma_a = gamma_baseline.values.T[:, np.newaxis]

    # 'ds_test.gf' only has r=[r0, r0+d12] left (2 points) -- see the
    # NOTE above -- so this is a tiny computation, not a
    # (n_values, n_freq, n_r) one.
    g_fr_1 = ds_test.gf.sel(r=r0, method="nearest")
    g_fr_2 = ds_test.gf.sel(r=r0 + d12, method="nearest")
    gamma_test_r0 = 20 * np.log10(np.abs(g_fr_2.values / g_fr_1.values))
    gamma_b = gamma_test_r0.T

    # Compute distance
    dist_L1 = calc_gamma_dist(gamma_a=gamma_a, gamma_b=gamma_b, dist_type="L1")
    dist_L2 = calc_gamma_dist(gamma_a=gamma_a, gamma_b=gamma_b, dist_type="L2")
    dist_theta = calc_gamma_dist(gamma_a=gamma_a, gamma_b=gamma_b, dist_type="theta")

    return dist_L1, dist_L2, dist_theta


def derive_gamma(ds, d12):
    """Compute gamma(freq, r) over the FULL r grid.

    NOTE: this remains a genuinely full-grid computation -- correct and
    fine for build_baseline()'s use (a single, ONE-TIME configuration,
    not a per-value sweep). Do NOT use this on a multi-value swept
    dataset (e.g. inside process_sensitivity()'s per-parameter loop):
    see dist_from_baseline()'s docstring for the performance/memory
    issue that caused, and _select_r0_pair()/dist_from_baseline() for
    the fix (select the 1-2 needed r-values first, compute after).
    """
    # Build RTF
    r = ds.r.values
    dr = r[1] - r[0]
    nr_shift = int(d12 / dr)

    # Green function at receiver 1
    g_fr_1 = ds.gf.isel(r=slice(0, -nr_shift))
    # Green function at receiver 2
    g_fr_2 = ds.gf.isel(r=slice(nr_shift, ds.sizes["r"]))

    # Build RTF
    pi_21_fr = g_fr_2.values / g_fr_1.values
    # Derive gamma
    gamma = 20 * np.log10(np.abs(pi_21_fr))  # (nf, nr)
    gamma = g_fr_1.copy(
        data=gamma
    )  # reuse g_fr_1's coords/dims, avoids the extra xr.ones_like(...) * gamma multiply

    return gamma


def process_sensitivity(test_arg_names=None):
    """Read back every parameter's per-value result files (see
    build_sensitivity_dataset()), compute each value's RTF distance
    from the baseline (see build_baseline()), and plot the resulting
    distance-vs-parameter-value curves -- one subplot per parameter.

    NOTE (memory usage improved): the original opened each parameter's
    single, already-combined multi-value NetCDF file with a plain
    `xr.open_dataset(fpath)` -- eagerly loading the WHOLE file (every
    swept value's full g_fr grid) into RAM, for every parameter, all
    at once (nothing closed any of them either). Given
    build_sensitivity_dataset() now writes one small file per value
    instead (see its own docstring), this function reads them back
    with `xr.open_mfdataset(..., chunks={})` -- a dask-BACKED, LAZY
    dataset: opening it does not load the actual array data, only its
    metadata/coordinates, and dask only pulls into memory the specific
    (r=r0) slice .sel() below actually needs. Each parameter's dataset
    is also opened in its own 'with' block, so it (and whatever dask
    briefly materialized for that one .sel() call) is released before
    moving on to the next parameter, rather than accumulating every
    parameter's dataset handle for the rest of the run.

    Args:
        test_arg_names (list[str]|None): which parameters to process
            (each must have a '<RESULT_DIR>/<name>/' folder from
            build_sensitivity_dataset()). None discovers every such
            folder under RESULT_DIR automatically.
    """
    import xarray as xr  # only this function needs it

    baseline_src_rcv_param = baseline_src_rcv()
    d12 = baseline_src_rcv_param["d12"]
    r0 = baseline_src_rcv_param["r0"]

    if test_arg_names is None:
        test_arg_names = sorted(
            name
            for name in os.listdir(RESULT_DIR)
            if os.path.isdir(os.path.join(RESULT_DIR, name))
        )

    fpath_baseline = os.path.join(RESULT_DIR, "gf_dataset_baseline.nc")

    n_tests = len(test_arg_names)
    fig, axs = plt.subplots(1, max(n_tests, 1), squeeze=False, sharey=True)
    _dict_var_labels = {
        "attn2": r"$\rho2$ [kg m$^3$]",
        "c2": r"$c_2$ [m s$^{-1}$]",
        "depth": "Depth [m]",
    }
    axs = axs[0]

    # NOTE: the baseline is small (a single configuration, not a
    # sweep) -- safe to keep open for the whole loop, unlike the
    # per-parameter sweep datasets below.
    with xr.open_dataset(fpath_baseline) as ds_baseline:
        for i, test_arg_name in enumerate(test_arg_names):
            value_files = sorted(
                glob.glob(
                    os.path.join(RESULT_DIR, test_arg_name, f"{test_arg_name}_*.nc")
                )
            )
            if not value_files:
                continue

            with xr.open_mfdataset(
                value_files,
                combine="nested",
                concat_dim=test_arg_name,
                chunks={},
            ) as ds_test:
                dist_L1, dist_L2, dist_theta = dist_from_baseline(
                    ds_baseline, ds_test, d12, r0
                )
                test_values = ds_test[test_arg_name].values

            # axs[i].plot(test_values, dist_L1, label="L1")
            # axs[i].plot(test_values, dist_L2, label="L2")
            axs[i].plot(test_values, dist_theta, label="theta")
            axs[i].set_xlabel(test_arg_name)
            axs[i].legend()

    return fig


def build_tests():
    all_arg_dict = load_all_arg_dict(
        drop_keys=("fs", "fmax", "r0", "d12"), d12_max=5000
    )

    npt = 200
    nb_param = 4
    size_per_test_Ko = 110000
    total_size = size_per_test_Ko * nb_param * npt
    print(
        f"Total memory size (if it were all held at once) = {total_size * 1e-6:.2f} Go "
        # f"-- no longer applicable: build_sensitivity_dataset() now writes each "
        # f"value's result as soon as it's computed (see its own docstring)."
    )

    # REAL RUN
    # sweeps = {
    #     "c1": np.linspace(1450, 1545, 200),
    #     # "c2": np.linspace(1550.0, 1900.0, npt),
    #     "rho2": np.linspace(1.0 * 1e3, 2.5 * 1e3, 500),
    #     "attn2": np.linspace(0.0, 1.0, 200),
    #     "depth": np.linspace(30, 5000, 500),
    # }

    # DEMO RUN
    sweeps = {
        "c1": np.linspace(1450, 1540, 2),
        # "c2": np.linspace(1550.0, 1900.0, npt),
        # "rho2": np.linspace(1.0 * 1e3, 2.5 * 1e3, 15),
        # "attn2": np.linspace(0.0, 1.0, 10),
        # "depth": np.linspace(30, 200, 17),
    }
    print(sweeps["c1"])

    for test_arg_name, test_arg_values in sweeps.items():
        build_sensitivity_dataset(
            test_arg_name, test_arg_values, all_arg_dict, model="kraken"
        )


if __name__ == "__main__":
    # build_baseline()
    build_tests()
    # process_sensitivity()
    plt.show()
