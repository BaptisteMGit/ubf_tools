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
import shutil
from datetime import datetime

import numpy as np
import xarray as xr
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
from publication.publication_figure import PubFigure, color

# Usefull paths
TITLE = "RTF sensitivity study"
ENV_FILENAME = "rtf_sensitivity_study"

if os.name == "nt":  # Windows
    project_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
    SENSITIVITY_DIRECTORY = os.path.join(
        project_root, "illustration_rtf", "data", "sensitivity"
    )
    IMG_DIR = os.path.join(project_root, "illustration_rtf", "img")

else:  # Linux
    project_root = "/home/program/ubf_tools"
    data_root = "/home/data"
    SENSITIVITY_DIRECTORY = os.path.join(data_root, "sensitivity")
    IMG_DIR = os.path.join(data_root, "img")


SENSITIVITY_KRAKEN_DIR = os.path.join(SENSITIVITY_DIRECTORY, "io_files")
os.makedirs(SENSITIVITY_KRAKEN_DIR, exist_ok=True)

# NOTE: KRAKEN/FIELD occasionally fail with an intermittent, hard-to-
# reproduce Fortran runtime error (e.g. "I/O past end of record on
# unformatted file") -- see build_kraken()'s own try/except around
# manager.runkraken(), which logs the failed run's '.env'/'.flp'/'.prt'
# (both kraken's own and field's -- see _log_kraken_crash()'s own
# docstring) file contents here, for later diagnosis.
CRASH_LOG_DIR = os.path.join(SENSITIVITY_KRAKEN_DIR, "crash_logs")

RESULT_DIR = os.path.join(SENSITIVITY_DIRECTORY, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# NOTE (harmonized, per user request): "resilience" -- testing
# robustness to REALISTIC environmental variability, as opposed to the
# main sensitivity study above's wide, largely unrealistic parameter
# sweeps -- covers TWO sub-studies that are conceptually the same kind
# of test: "depth" (tidal water-depth elevation -- see
# build_resilience_tests()) and "celerity" (seasonal/EOF-derived sound-
# speed profile shape -- see build_celerity_baseline()). Both are
# further split by ENVIRONMENT TYPE (see CELERITY_ENV_TYPES -- "sw"
# shallow water/100 m, "dw" deep water/2000 m), each with its OWN
# baseline. The two sub-studies used to live under entirely separate
# top-level directory trees (RESILIENCE_* vs a since-removed
# CELERITY_*) -- now unified as
# '<RESILIENCE_RESULT_DIR|RESILIENCE_IMG_DIR>/<env_type>/<kind>/', kind
# being "depth" or "celerity" (see _resilience_study_dirs()).
RESILIENCE_DIRECTORY = os.path.join(
    os.path.dirname(SENSITIVITY_DIRECTORY), "resilience"
)
RESILIENCE_RESULT_DIR = os.path.join(RESILIENCE_DIRECTORY, "result")
os.makedirs(RESILIENCE_RESULT_DIR, exist_ok=True)
RESILIENCE_IMG_DIR = os.path.join(IMG_DIR, "resilience")
os.makedirs(RESILIENCE_IMG_DIR, exist_ok=True)


def _resilience_study_dirs(env_type, kind):
    """Return (result_dir, img_dir) for ONE environment type's
    resilience sub-study -- both nested under
    '<RESILIENCE_RESULT_DIR|RESILIENCE_IMG_DIR>/<env_type>/<kind>/'
    (see this module's own NOTE on RESILIENCE_RESULT_DIR for the
    rationale). Used as the shared default for both sub-studies'
    "build"/"process"/"plot" functions, so the directory layout only
    needs to be decided in ONE place.

    Args:
        env_type (str): "sw" or "dw" -- see CELERITY_ENV_TYPES.
        kind (str): "depth" (see build_resilience_tests()) or
            "celerity" (see build_celerity_baseline()).

    Returns:
        tuple(str, str): result_dir, img_dir.
    """
    return (
        os.path.join(RESILIENCE_RESULT_DIR, env_type, kind),
        os.path.join(RESILIENCE_IMG_DIR, env_type, kind),
    )


if os.name == "nt":  # Windows
    SSP_DATA_DIR = os.path.join(project_root, "illustration_rtf", "data", "ssp")
else:  # Linux
    # SSP_DATA_DIR = os.path.join(data_root, "ssp")
    SSP_DATA_DIR = os.path.join(project_root, "illustration_rtf", "data", "ssp")

# NOTE: "sw"'s depth (100 m) intentionally matches the classic
# sensitivity study's own baseline depth (see baseline_env()) --
# that's what lets its real profile (which load_ssp_data.py's own
# CMEMS extraction only reaches to ~50 m for) be directly compared on
# the same footing, once linearly extended (see
# load_mean_celerity_profile()). "dw"'s depth (2000 m) is new: its own
# real profile reaches to ~2500 m and is truncated down to it instead.
CELERITY_ENV_TYPES = {
    "sw": {"ssp_filename": "ssp_profiles_sw.nc", "depth": 100.0},
    "dw": {"ssp_filename": "ssp_profiles_dw.nc", "depth": 2000.0},
}

# NOTE: matches illustration_rtf/ssp/ssp_process_eof.py's own
# '__main__' defaults -- "n_new_samples" (how many synthetic profiles
# get_ssp_eof()-based sampling generated per file) and the 5 "filenames"
# entries per environment type (1 fit on the WHOLE, multi-decade
# dataset -- "all" here -- + 4 fit on one season only each). See
# _synthetic_ssp_filename(), which reconstructs
# process_ssp_profiles()'s own f"synthetic_{filename_ssp}_
# {n_new_samples}.nc" naming from an (env_type, situation) pair.
CELERITY_N_SYNTHETIC_SAMPLES = 1000
CELERITY_SITUATIONS = ("all", "winter", "spring", "summer", "automn")

# NOTE (factored out): these two dicts used to be redefined identically
# inside plot_sensitivity_curves() and plot_extremal_width_configs()
# (now a third place would need its own copy too) -- one shared,
# module-level version instead.
ARG_LABEL = {
    "attn2": r"$\alpha_2$ [dB $\lambda^{-1}$]",
    "rho2": r"$\rho_2$ [kg m$^{-3}$]",
    "c1": r"$c_1$ [m s$^{-1}$]",
    "c2": r"$c_2$ [m s$^{-1}$]",
    "depth": "D [m]",
}
METRIC_LABEL = {
    "L1": "$L_1$",
    "L2": "$L_2$",
    "theta": r"$\theta$",
    "wasserstein": "Wasserstein",
}


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

    # Water column
    rho1 = 1.0 * 1e3  # density in water (kg/m^3)
    c1 = 1500  # sound celerity in water (m/s)

    # Fluid sediment
    # New (after 09/09/2026): use the sediment properties from TGalan classification (vases)
    # This sediment lies on the continental shelf.
    # Two advantages :
    # First : it fits the Hamilton and Bachman 1982 model (see celerity_density_Hamilton_Bachman_1982()) and thus
    # will produce a distance close to zero for the density sensitivity test
    # Second : here c2 = 1600 and the impedance constrast is higher, this leads to a lower cut off frequency and thus avoids
    # the risk of having a cut off frequency higher than the maximum frequency of the signal (150 Hz) for the celerity sensitivity test.
    from source.global_constants import vase_TG

    rho2 = vase_TG["rho"] * 1e3  # density in fluid sediment (kg/m^3)
    c2 = vase_TG["c_p"]  # sound celerity in fluid sediment (m/s)
    attn2 = vase_TG[
        "a_p"
    ]  # compressional wave attenuation in fluid sediment in dB / wavelength

    # # Fluid sediment
    # # New (after 09/09/2026): use the sediment properties from TGalan classification (sables fins)
    # # This sediment lies on the continental shelf.
    # # Two advantages :
    # # First : it almost perfectly fits the Hamilton and Bachman 1982 model (see celerity_density_Hamilton_Bachman_1982()) and thus
    # # will produce a distance close to zero for the density sensitivity test
    # # Second : here c2 = 1700 and the impedance constrast is higher, this leads to a lower cut off frequency and thus avoids
    # # the risk of having a cut off frequency higher than the maximum frequency of the signal (150 Hz) for the celerity sensitivity test.
    # from source.global_constants import sables_fins_TG

    # rho2 = sables_fins_TG["rho"] * 1e3  # density in fluid sediment (kg/m^3)
    # c2 = sables_fins_TG["c_p"]  # sound celerity in fluid sediment (m/s)
    # attn2 = sables_fins_TG[
    #     "a_p"
    # ]  # compressional wave attenuation in fluid sediment in dB / wavelength

    # # Before 09/09/2026
    # rho2 = 1.5 * 1e3  # density in fluid sediment (kg/m^3)
    # # c2 = 3000  # sound celerity in fluid sediment (m/s)
    # c2 = 1550  # Close to Hamilton(rho2) = 1542 m.s-1
    # attn2 = 0.2  # compressional wave attenuation in fluid sediment in dB / wavelength

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
    # After 07/09/2026
    r0 = 15 * 1e3
    # r_rcv = np.arange(1 * 1e3, 30 * 1e3 + dr, dr)       # Full
    r_rcv = np.arange(10 * 1e3, 20 * 1e3 + dr, dr)  # Restricted

    # Before 07/09/2026
    # r0 = 50 * 1e3
    # r_rcv = np.arange(45 * 1e3, 55 * 1e3 + dr, dr)

    d12 = 1000

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


def _sensitivity_figure_filename(
    kind, test_arg_names, file_prefix=None, mode=None, metric=None
):
    """Build a consistent, collision-free filename for a saved
    sensitivity-study figure.

    NOTE (bug fixed): plot_sensitivity_curves()'s own save used to
    build its filename as f"{file_prefix}_{test_arg_name}.png" -- but
    'file_prefix' (e.g. "dist_"/"mainlobe_width_") ALREADY ends with
    "_", producing double-underscore names like "dist__depth.png".
    Worse, 'test_arg_name' there was the LOOP VARIABLE left over from
    the (already finished) per-parameter plotting loop just above it,
    i.e. whichever parameter happened to be LAST in 'test_arg_names' --
    so a figure showing several parameters' subplots was saved under a
    name mentioning only one of them, and calling the function again
    for a DIFFERENT parameter set (or a different 'distance'/'metric'
    selection) could silently overwrite an unrelated previous figure.
    plot_extremal_width_configs()'s own save had a related issue: its
    filenames ("dist_<name>.png" / "gamma_<name>.png") don't encode
    'mode' or 'metric' at all, so calling it twice with different
    'mode'/'metric' values (e.g. "baseline" vs "intrinsic", or "L1" vs
    "theta") for the SAME parameter silently overwrites the first
    figure with the second. This one shared helper (used by both) now
    encodes every distinguishing piece of information actually
    available -- what kind of figure, which underlying result kind,
    which mode/metric, and which parameter(s) -- into the filename.

    Args:
        kind (str): what the figure shows, e.g. "curves", "dist", "gamma".
        test_arg_names (str|list[str]): the parameter(s) shown.
        file_prefix (str|None): e.g. "dist_"/"mainlobe_width_"/
            "intrinsic_mainlobe_width_" -- which underlying result kind
            was plotted (see save_sensitivity_distance_results()). Any
            trailing "_" is stripped (it's a directory-listing prefix
            there, not meant to appear literally in a filename).
        mode (str|None): "baseline"/"intrinsic" (see
            plot_extremal_width_configs()).
        metric (str|None): "L1"/"L2"/"theta".

    Returns:
        str: filename (no directory), e.g.
        "curves_mainlobe_width_depth.png" or
        "dist_intrinsic_mainlobe_width_theta_depth.png". When more than
        3 parameters are named, only the first is spelled out (plus a
        count of the rest) to keep the filename reasonably short.
    """
    names = (
        [test_arg_names] if isinstance(test_arg_names, str) else list(test_arg_names)
    )
    parts = [kind]
    if file_prefix:
        parts.append(file_prefix.rstrip("_"))
    if mode:
        parts.append(mode)
    if metric:
        parts.append(metric)
    if len(names) <= 3:
        parts.append("_".join(names))
    else:
        parts.append(f"{names[0]}_and_{len(names) - 1}_more")
    return "_".join(parts) + ".png"


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
        dist = (1 - cos_angle) / 2  # In [-1, 1]

    if dist_type == "wasserstein":
        # NOTE (new, per user request): unlike L1/L2/theta above (which
        # compare GAMMA directly, in dB), the Wasserstein distance
        # compares the RTF MAGNITUDE itself (rtf = 10 ** (gamma / 20))
        # -- see calc_rtf_wasserstein_dist()'s own docstring for why.
        # Only supports 'gamma_a'/'gamma_b' shaped (n_freq, 1) /
        # (n_freq, n_values) -- i.e. dist_from_baseline()'s own usage,
        # NOT dist_from_baseline_around_r0()'s 3D (n_freq, n_r,
        # n_values) one (a per-r Wasserstein "mainlobe width" was not
        # requested and would be considerably more expensive to
        # compute -- scipy has no vectorized wasserstein_distance).
        dist = calc_rtf_wasserstein_dist(gamma_a=gamma_a, gamma_b=gamma_b)

    return dist


def calc_rtf_wasserstein_dist(gamma_a, gamma_b):
    """Wasserstein (earth mover's) distance between two RTF magnitude
    spectra, each treated as a 1D distribution over FREQUENCY-BIN
    INDEX (not the physical frequency values -- both curves share the
    same frequency grid, so the bin index alone is enough to compare
    them; see scipy.stats.wasserstein_distance()'s own 'u_values'/
    'v_values', here just np.arange(n_freq) for both curves). Only the
    RTF MAGNITUDE at each bin -- used as that bin's WEIGHT -- differs
    between the two.

    Converts gamma to the RTF magnitude itself first
    (rtf = 10 ** (gamma / 20), matching build_kraken()'s own gamma
    convention: gamma = 20*log10(|RTF|)) rather than comparing gamma
    values directly the way calc_gamma_dist()'s L1/L2/theta do:
    wasserstein_distance()'s weights must be non-negative (they
    represent a "mass" distribution), which gamma itself -- expressed
    in dB, so routinely negative -- cannot satisfy, while the RTF
    magnitude always can.

    NOTE: scipy.stats.wasserstein_distance() only accepts 1D inputs --
    'gamma_b' can carry several swept values at once (each compared
    against the same baseline 'gamma_a'), so this loops over that
    dimension internally rather than vectorizing (there is no
    vectorized wasserstein_distance in scipy as of this writing).

    Args:
        gamma_a (np.ndarray): baseline gamma, shape (n_freq, 1) (a
            single reference curve, broadcast against every swept
            value below).
        gamma_b (np.ndarray): swept-value(s) gamma, shape
            (n_freq, n_values).

    Returns:
        np.ndarray: shape (n_values,) -- one Wasserstein distance per
        swept value.
    """
    from scipy.stats import wasserstein_distance

    # rtf_a = 10 ** (gamma_a / 20.0)
    # rtf_b = 10 ** (gamma_b / 20.0)
    water_level = min(np.min(gamma_a), np.min(gamma_b)) + 1e-5
    rtf_a = gamma_a + water_level
    rtf_b = gamma_b + water_level
    rtf_a = np.broadcast_to(rtf_a, rtf_b.shape)

    n_freq, n_values = rtf_b.shape
    distribution_support = np.arange(n_freq)

    dist = np.empty(n_values)
    for i in range(n_values):
        u = np.nan_to_num(rtf_a[:, i], nan=0.0)
        v = np.nan_to_num(rtf_b[:, i], nan=0.0)
        dist[i] = wasserstein_distance(
            u_values=distribution_support,
            v_values=distribution_support,
            u_weights=u,
            v_weights=v,
        )
    return dist


def calc_monotonicity_domain(dist_r_r0, r, r0):
    """
    Derive distance main lobe width defined as the monotonicity domain according to the first maximum of the distance (first zero crossing of the derivative)
    """

    # # Apply moving average to avoid spurious detections
    # import pandas as pd
    # dist_r_r0_pd = pd.DataFrame(dist_r_r0).rolling(window=3, center=True, min_periods=1).mean()
    # dist_r_r0 = dist_r_r0_pd.to_numpy().flatten()

    # Positive r - r0
    r_reduced = r - r0
    d_pos = dist_r_r0[r_reduced >= 0]
    r_pos = r_reduced[r_reduced >= 0]

    # Compute the derivative of the distance
    dd_dr_pos = np.gradient(d_pos, r_pos)

    # Apply moving average to avoid spurious detections
    import pandas as pd

    dd_dr_pos_pd = (
        pd.DataFrame(dd_dr_pos).rolling(window=5, center=True, min_periods=1).mean()
    )
    dd_dr_pos = dd_dr_pos_pd.to_numpy().flatten()

    # Find zero crossings of the derivative
    zero_crossings_pos = np.where(np.diff(np.sign(dd_dr_pos)))[0]
    if not len(zero_crossings_pos) > 0:
        raise ValueError(
            f"calc_monotonicity_domain: the distance curve's derivative "
            f"never changes sign on the r > r0 side (r0={r0}) -- cannot "
            f"locate a first maximum (monotonicity boundary) there."
        )

    # Take the first zero crossing
    first_zero_crossing_pos = zero_crossings_pos[0]
    r_validity_zero_crossing_pos = r_pos[first_zero_crossing_pos]

    # Negative r - r0
    d_neg = dist_r_r0[r_reduced < 0]
    r_neg = r_reduced[r_reduced < 0]
    # Compute the derivative of the distance
    dd_dr_neg = np.gradient(d_neg, r_neg)

    dd_dr_neg_pd = (
        pd.DataFrame(dd_dr_neg).rolling(window=5, center=True, min_periods=1).mean()
    )
    dd_dr_neg = dd_dr_neg_pd.to_numpy().flatten()

    # Find zero crossings of the derivative
    zero_crossings_neg = np.where(np.diff(np.sign(dd_dr_neg)))[0]

    if not len(zero_crossings_neg) > 0:
        raise ValueError(
            f"calc_monotonicity_domain: the distance curve's derivative "
            f"never changes sign on the r < r0 side (r0={r0}) -- cannot "
            f"locate a first maximum (monotonicity boundary) there."
        )

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
    """Derive the -3dB mainlobe width (see calc_mainlobe_width_3dB())
    from one or several distance-vs-range curves.

    NOTE (fixed to support process_sensitivity_mainlobe_width()): this
    used to only handle a SINGLE distance curve per metric (1D, shape
    (n_r,) -- run_single_sensitivity_test()'s own usage), returning 3
    scalars. dist_from_baseline_around_r0() instead produces one
    distance curve PER SWEPT VALUE, stacked as a 2D array (shape
    (n_r, n_values), one COLUMN per value) -- calling
    calc_mainlobe_width_3dB() directly on that 2D array would boolean-
    index a 1D r_pos/r_neg array with a 2D mask, raising a shape
    mismatch error. This now branches on the input's dimensionality
    instead: 1D input still returns 3 scalars (unchanged behaviour);
    2D input (one column per value) returns 3 arrays of length
    n_values, one width per column, computed independently. A column
    whose curve never reaches -3dB on one side (see
    calc_mainlobe_width_3dB()'s own ValueError) does not abort the rest
    of the batch: that one value is left as NaN, with a printed
    warning naming which value (by index) and metric failed.

    Args:
        dist_L1, dist_L2, dist_theta (array-like): either all 1D
            (shape (n_r,), a single distance curve) or all 2D (shape
            (n_r, n_values), one curve per column).
        r_grid (array-like): the (shared) range grid the distance
            curves are evaluated on, shape (n_r,).
        r0 (float): reference range (m) the mainlobe is centered on.

    Returns:
        tuple(float, float, float) if the input was 1D, or
        tuple(np.ndarray, np.ndarray, np.ndarray) (each length
        n_values) if 2D.
    """
    dist_L1 = np.asarray(dist_L1)
    dist_L2 = np.asarray(dist_L2)
    dist_theta = np.asarray(dist_theta)

    if dist_L1.ndim == 1:
        # Original behaviour: a single distance curve -> 3 scalars.
        width_L1, r_ml_width_L1_inf, r_ml_width_L1_sup = calc_mainlobe_width_3dB(
            dist_r_r0=dist_L1, r=r_grid, r0=r0
        )
        width_L2, r_ml_width_L2_inf, r_ml_width_L2_sup = calc_mainlobe_width_3dB(
            dist_r_r0=dist_L2, r=r_grid, r0=r0
        )
        width_theta, r_ml_width_theta_inf, r_ml_width_theta_sup = (
            calc_mainlobe_width_3dB(dist_r_r0=dist_theta, r=r_grid, r0=r0)
        )
        return width_L1, width_L2, width_theta

    # 2D case: one column per swept value.
    n_values = dist_L1.shape[1]
    dists = {"L1": dist_L1, "L2": dist_L2, "theta": dist_theta}
    widths = {name: np.full(n_values, np.nan) for name in dists}
    for i in range(n_values):
        for name, dist in dists.items():
            try:
                # widths[name][i], _, _ = calc_mainlobe_width_3dB(
                #     dist_r_r0=dist[:, i], r=r_grid, r0=r0
                # )
                widths[name][i], _, _ = calc_monotonicity_domain(
                    dist_r_r0=dist[:, i], r=r_grid, r0=r0
                )
            except ValueError as exc:
                print(
                    f"Warning: could not compute the {name} mainlobe width for "
                    f"swept value index {i}: {exc}"
                )

    return widths["L1"], widths["L2"], widths["theta"]


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
def _log_kraken_crash(env, flp, exc, extra_context=None):
    """On a KRAKEN/FIELD run failure (see build_kraken()'s own
    try/except around manager.runkraken()), write the content of every
    input/diagnostic file that run produced -- '.env', '.flp', and
    BOTH '.prt' files -- to one timestamped log file under
    CRASH_LOG_DIR.

    NOTE: kraken.exe writes its own print output to
    '<filename>.prt' (same base name as the '.env'/'.flp' -- see
    KrakenEnv's own 'env_filename'), but field.exe writes to a FIXED,
    un-derived filename, 'field.prt', in the same working directory
    (env.root) -- NOT '<filename>.prt' (confirmed from an actual
    'field.prt' sample: it has no notion of the run's own filename at
    all). The two are therefore always separate files, neither
    overwriting the other.

    KRAKEN/FIELD occasionally fail with an intermittent, hard-to-
    reproduce Fortran runtime error (e.g. "I/O past end of record on
    unformatted file") that os.system() (see KrakenManager.run_exec())
    does not turn into a Python exception by itself -- the actual
    exception this catches is typically raised downstream, when
    read_shd.readshd() tries to parse a '.shd' file that the crashed
    run never finished writing. By the time that happens, every input/
    print file below is still exactly as KRAKEN/FIELD last left it,
    since build_sensitivity_dataset()'s per-value loop runs strictly
    sequentially, one full KRAKEN/FIELD run at a time, always into the
    SAME '<SENSITIVITY_KRAKEN_DIR>/<ENV_FILENAME>.*' filenames (see
    KrakenEnv's own 'env_root'/'env_filename' in build_kraken()) -- so
    nothing else overwrites them before this can run.

    Args:
        env (KrakenEnv): the environment the failed run was for.
        flp (KrakenFlp): the field parameters the failed run was for.
        exc (Exception): the exception that was actually raised (see
            above -- rarely the Fortran error itself, but whatever
            downstream symptom it caused).
        extra_context (dict|None): extra key/value pairs recorded at
            the top of the log (e.g. the waveguide parameters this
            configuration was run with) -- purely informational, to
            help correlate a crash with what was being swept at the
            time.

    Returns:
        str: path to the written log file.
    """
    os.makedirs(CRASH_LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_fpath = os.path.join(CRASH_LOG_DIR, f"crash_{timestamp}.log")

    env_fpath = env.env_fpath
    flp_fpath = flp.flp_fpath
    kraken_prt_fpath = env_fpath.replace(".env", ".prt")
    field_prt_fpath = os.path.join(os.path.dirname(env_fpath), "field.prt")

    def _read_or_note_missing(fpath):
        if not os.path.exists(fpath):
            return f"[file not found: {fpath}]\n"
        try:
            with open(fpath, "r", errors="replace") as f:
                return f.read()
        except OSError as read_exc:
            return f"[could not read {fpath}: {read_exc}]\n"

    with open(log_fpath, "w", encoding="utf-8") as f:
        f.write(f"KRAKEN/FIELD crash log -- {datetime.now().isoformat()}\n")
        f.write(f"Exception: {type(exc).__name__}: {exc}\n")
        if extra_context:
            f.write("Context:\n")
            for key, value in extra_context.items():
                f.write(f"  {key}: {value}\n")
        f.write("\n")

        for label, fpath in [
            (".env", env_fpath),
            (".flp", flp_fpath),
            (".prt (kraken)", kraken_prt_fpath),
            (".prt (field)", field_prt_fpath),
        ]:
            f.write(f"{'=' * 70}\n{label} -- {fpath}\n{'=' * 70}\n")
            f.write(_read_or_note_missing(fpath))
            f.write("\n\n")

    return log_fpath


def build_kraken(
    freq,
    c1,
    c2,
    rho1,
    rho2,
    attn2,
    depth,
    z_s,
    z,
    r_grid,
    plot_diag=False,
    z_ssp=None,
    c_p_ssp=None,
    img_dir=IMG_DIR,
):
    """Build a Pekeris-like waveguide (water column over a fluid
    half-space), run KRAKEN/FIELD, and return the resulting Green's
    function.

    Args:
        freq, c1, c2, rho1, rho2, attn2, depth, z_s, z, r_grid,
            plot_diag: see the module's other functions -- 'c1' is the
            water sound speed (m/s) for the classic, ISOVELOCITY water
            column (used when 'z_ssp'/'c_p_ssp' below are not given).
        z_ssp, c_p_ssp (array-like|None): if BOTH given, build a
            REALISTIC, depth-varying water column from this profile
            instead of the isovelocity [c1, c1] one (see
            load_mean_celerity_profile(), which prepares one such
            profile per environment type for the celerity-sensitivity
            study). 'z_ssp' must start at 0 and reach exactly 'depth'.
            'c1' is still required either way: the Pekeris cutoff-
            frequency/mode-count formulas below assume an isovelocity
            column (they are a NECESSARY approximation for a realistic
            profile), and use 'z_ssp'/'c_p_ssp''s own SURFACE value in
            that case rather than 'c1' itself.
        img_dir (str): where the 'plot_diag=True' diagnostic figures
            ("environment.png", "group_speed.png", "mode_shapes.png",
            "tl_profile_at_source_depth.png") are saved. Defaults to
            IMG_DIR; build_baseline() forwards its OWN 'img_dir'
            argument here instead (see its own docstring), so e.g. a
            resilience or celerity study's diagnostics land in that
            study's own image folder rather than always IMG_DIR.

    Returns:
        tuple(np.ndarray, np.ndarray, dict): kraken_freq (the requested
        'freq', trimmed to those above the mode-1 cutoff), g_fr (the
        Green's function, shape (kraken_freq.size, r_grid.size)),
        field_pos (see read_shd.readshd's own 'Pos' return value).
    """
    # NOTE: the Pekeris cutoff-frequency/mode-count formulas below are
    # only valid for an isovelocity water column -- for a realistic
    # profile, its own SURFACE sound speed is used as the closest
    # single-value stand-in (a necessary approximation; there is no
    # single "c1" for a depth-varying profile).
    c1_for_estimate = c_p_ssp[0] if (z_ssp is not None and c_p_ssp is not None) else c1

    # Keeps only frequencies above mode 1 cut-off
    fmin = pekeris_cutoff_frequency(m=1, c1=c1_for_estimate, c2=c2, d=depth)
    # fmin = (
    #     np.floor((fmin) / 5) * 5 + 10
    # )  # Round to upper closest multiple of five to avoid being to close to cuttoff
    kraken_freq = freq[freq > fmin]

    # Use only propative modes
    nb_modes = pekeris_n_modes(f=freq.max(), c1=c1_for_estimate, c2=c2, d=depth)
    # clim_max = c2
    clim_max = (
        np.ceil((c2) / 1000) * 1000
    )  # Round to upper closest multiple of five to avoid being to close to cuttoff
    clim_min = 1400

    # attn2 = 0  # TODO remove
    # ----------------------------------------------------------------------
    # 1. Environment: Pekeris waveguide
    # ----------------------------------------------------------------------
    if z_ssp is not None and c_p_ssp is not None:
        medium = KrakenMedium(
            ssp_interpolation_method="C_linear",
            z_ssp=z_ssp,
            c_p=c_p_ssp,  # realistic, depth-varying water column
            rho=rho1 * 1e-3,
        )
    else:
        medium = KrakenMedium(
            ssp_interpolation_method="C_linear",
            z_ssp=[0.0, depth],
            c_p=[c1, c1],  # isovelocity water column
            rho=rho1 * 1e-3,
        )

    bottom_hs = KrakenBottomHalfspace(
        halfspace_properties={
            "z": depth,
            "c_p": c2,
            "c_s": 0.0,  # fluid sediment: no shear waves
            "rho": rho2 * 1e-3,  # g.cm-3
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
    try:
        pressure_field, field_pos = manager.runkraken(
            env=env, flp=flp, frequencies=env.freq
        )
    except Exception as exc:
        # NOTE: see _log_kraken_crash()'s own docstring for why this is
        # needed and what it captures. The ORIGINAL exception is always
        # re-raised unchanged afterward -- logging is purely a side
        # effect, never a substitute for the caller's own error
        # handling (or lack thereof).
        log_fpath = _log_kraken_crash(
            env,
            flp,
            exc,
            extra_context={
                "freq_min_max_Hz": f"{freq.min():.2f}-{freq.max():.2f}",
                "depth_m": depth,
                "c1_m_s": c1,
                "c2_m_s": c2,
                "rho1_kg_m3": rho1,
                "rho2_kg_m3": rho2,
                "attn2_dB_per_wavelength": attn2,
                "realistic_profile": z_ssp is not None and c_p_ssp is not None,
            },
        )
        print(f"build_kraken: run failed -- input/print files logged to {log_fpath}")
        raise

    # Plot
    if plot_diag:
        fig_env = env.plot_env(plot_src=True, src_depth=z_s)
        fig_env.savefig(os.path.join(img_dir, "environment.png"))
        plt.close(fig_env)

        mod_fpath = env.env_fpath.replace(".env", ".mod")
        shd_fpath = env.shd_fpath

        from propa.kraken_toolbox import plot_utils as pu

        fig0 = pu.plot_group_speed(
            mod_fpath,
            n_modes=5,
            freq=kraken_freq,
            modes=None,
        )
        fig0.savefig(os.path.join(img_dir, "group_speed.png"))
        plt.close(fig0)

        fig1 = pu.plotmode(mod_fpath, freq=freq.max())
        fig1.savefig(os.path.join(img_dir, "mode_shapes.png"))
        plt.close(fig1)

        fig3 = pu.plot_tl_profile(
            shd_fpath,
            freq=freq.max(),
            rcv_depth=z_s,
            # rcv_depth=2500,
            units="km",
            show_spherical_loss=True,
            show_cylindrical_loss=True,
        )
        fig3.savefig(os.path.join(img_dir, "tl_profile_at_source_depth.png"))
        plt.close(fig3)

    # Squeeze
    pressure_field = pressure_field.squeeze()

    # Get green's function
    c0 = 1500
    k0 = 2 * np.pi * kraken_freq / c0
    norm_factor = np.exp(1j * k0) / (4 * np.pi)
    g_fr = norm_factor[:, np.newaxis] * pressure_field  # (nf, nr)

    return kraken_freq, g_fr, field_pos


def build_dataset_current_config_kraken(
    freq,
    c1,
    c2,
    rho1,
    rho2,
    attn2,
    depth,
    z_s,
    z_rcv,
    r_rcv,
    d12_max,
    plot_diag=False,
    z_ssp=None,
    c_p_ssp=None,
    img_dir=IMG_DIR,
):
    """Thin wrapper around build_kraken(): extends the receiver range
    grid by 'd12_max' first (see _extend_range_grid()), so the saved
    Green's function covers every d12 you might later want to derive
    an RTF for, without needing to re-run KRAKEN.

    Args: see build_kraken() -- 'z_ssp'/'c_p_ssp'/'img_dir' are
        forwarded as-is (see its own docstring for the realistic-
        profile use case and where diagnostics get saved).

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray): kraken_freq, r
        (meters, the actual receiver range grid KRAKEN used), g_fr.
    """
    # Extend r_grid to d12_max, convert to km for kraken
    r_grid_ = _extend_range_grid(r_rcv, d12_max) * 1e-3

    # Derive green's function at all pos
    freq, g_fr, field_pos = build_kraken(
        freq,
        c1,
        c2,
        rho1,
        rho2,
        attn2,
        depth,
        z_s,
        z_rcv,
        r_grid_,
        plot_diag=plot_diag,
        z_ssp=z_ssp,
        c_p_ssp=c_p_ssp,
        img_dir=img_dir,
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


def _to_float32(arr):
    """Downcast a numpy array to float32.

    NOTE (memory usage improved, per user feedback): build_sensitivity_dataset()/
    build_baseline() used to save 'gf'/'gamma'/coordinates at their
    default float64 precision, and a later, separate fix downcast the
    re-opened dataset to a smaller dtype at READ time instead (inside
    process_sensitivity_mainlobe_width()/
    process_sensitivity_intrinsic_mainlobe_width()) -- but reading
    still has to load the full float64 data first before that
    downcast can happen, so it saves memory only AFTER the expensive
    part is already done, and does nothing at all for the disk space
    the '.nc' files themselves take up. Downcasting at WRITE time
    instead (both the data variables AND the coordinates, e.g. 'f'/'r'
    /the swept parameter's own values, which can be sizeable arrays
    too) means every later read of these files is smaller and faster
    from the start, with no read-side workaround needed at all -- see
    the two call sites below, and process_sensitivity_mainlobe_width()/
    process_sensitivity_intrinsic_mainlobe_width(), which no longer
    do anything special on read.

    Args:
        arr (array-like): real-valued (gf/gamma are magnitudes -- see
            build_dataset_current_config_kraken()'s own np.abs() --
            not complex; do not use this on a genuinely complex array,
            which float32 would silently truncate to its real part).

    Returns:
        np.ndarray: dtype float32.
    """
    return np.asarray(arr, dtype=np.float32)


def _clear_dir(dir_path):
    """Delete every file already in 'dir_path' (then recreate it
    empty) before a fresh sweep writes into it.

    NOTE (added, per user request): a sweep's own per-value output
    folder (e.g. '<RESULT_DIR>/attn2/', see
    build_sensitivity_dataset()) used to only ever get
    os.makedirs(..., exist_ok=True)'d -- never actually emptied. Stale
    files left over from a PREVIOUS run (a different set/count of
    swept values, or one that crashed partway through -- see
    build_kraken()'s own try/except around manager.runkraken()) would
    then sit alongside the new ones, and get silently picked up by
    whatever later reads every '.nc' file in that folder (see
    process_sensitivity()/process_celerity_sensitivity()'s own
    xr.open_mfdataset() calls, matched by a glob() over the folder's
    contents) -- corrupting the results with leftover data that
    doesn't belong to the CURRENT sweep at all.

    Args:
        dir_path (str): directory to empty (created if it doesn't
            exist yet).
    """
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def build_sensitivity_dataset(
    test_arg_name, test_arg_values, all_arg_dict, model="kraken", result_dir=RESULT_DIR
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

    File layout: '<result_dir>/<test_arg_name>/<test_arg_name>_<i>.nc',
    one small file per swept value, holding that single value's g_fr
    plus 'test_arg_name' as a length-1 coordinate (so the files can
    later be combined along that dimension with xr.open_mfdataset,
    without ever holding more than one at a time in memory -- see
    process_sensitivity()).

    Args:
        test_arg_name, test_arg_values, all_arg_dict, model: see the
            module's other sensitivity-study functions. If
            'all_arg_dict' contains 'z_ssp'/'c_p_ssp' (a realistic
            profile -- see build_kraken()'s own docstring) AND
            'test_arg_name' is "depth", that profile is automatically
            re-adapted (extended/truncated -- see
            _adapt_profile_to_depth()) to reach EACH swept depth value
            exactly, rather than kept fixed at its original shape (see
            build_resilience_tests()'s own NOTE, which is what needs
            this).
        result_dir (str): directory the per-value files are written
            under (as '<result_dir>/<test_arg_name>/...'). Defaults to
            RESULT_DIR (the main sensitivity study); pass e.g.
            RESILIENCE_RESULT_DIR to keep a resilience study's own
            result files in their own, separate location -- see
            build_resilience_tests().

    Returns:
        str: the directory the per-value files were written into.
    """

    out_dir = os.path.join(result_dir, test_arg_name)
    # NOTE (per user request): emptied first, not just created -- see
    # _clear_dir()'s own docstring for why.
    _clear_dir(out_dir)

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

            if (
                all_arg_dict.get("z_ssp") is not None
                and all_arg_dict.get("c_p_ssp") is not None
            ):
                # NOTE (added per user request -- see
                # build_resilience_tests()'s own NOTE): when sweeping
                # "depth" with a REALISTIC celerity profile (not the
                # classic isovelocity one), the profile itself must be
                # re-adapted (extended/truncated -- see
                # _adapt_profile_to_depth()) to reach EACH new swept
                # depth exactly, not just held fixed at its original
                # nominal-depth shape. Re-derived from
                # 'all_arg_dict''s OWN, never-mutated 'z_ssp'/'c_p_ssp'
                # every iteration (NOT from 'all_args', which carries
                # over whatever the PREVIOUS iteration left it at) --
                # otherwise repeated extend/truncate/extend... across
                # iterations would compound approximation error instead
                # of freshly adapting the same real profile each time.
                all_args["z_ssp"], all_args["c_p_ssp"] = _adapt_profile_to_depth(
                    all_arg_dict["z_ssp"], all_arg_dict["c_p_ssp"], test_val
                )

        # Build dataset for JUST this one value.
        call_kwargs = _extract_kwargs(build_dataset_current_config_kraken, all_args)
        kraken_freq, kraken_r, g_fr = build_dataset_current_config_kraken(
            **call_kwargs
        )  # g_fr is (nf, nr)

        g_fr_full = _pad_to_full_frequency_grid(g_fr, kraken_freq, all_args["freq"])

        # Write THIS value's dataset immediately, then let g_fr/g_fr_full
        # go out of scope (freed before the next iteration's KRAKEN run).
        # NOTE: variables AND coordinates saved as float32 -- see
        # _to_float32()'s own docstring for why this is done here,
        # at write time, rather than downcasting after reading it back.
        ds_value = xr.Dataset(
            data_vars=dict(
                gf=(
                    [test_arg_name, "f", "r"],
                    _to_float32(np.abs(g_fr_full))[np.newaxis, ...],
                )
            ),
            coords={
                test_arg_name: _to_float32([test_val]),
                "f": _to_float32(all_args["freq"]),
                "r": _to_float32(kraken_r),
            },
        )
        fpath = os.path.join(out_dir, f"{test_arg_name}_{i_test:04d}.nc")
        ds_value.to_netcdf(fpath)
        ds_value.close()

    return out_dir


def build_baseline(result_dir=RESULT_DIR, img_dir=IMG_DIR, env_overrides=None):
    """Build and save the baseline (nominal, unperturbed) configuration's
    Green's function / gamma(f, r) -- the fixed reference every
    sensibility/resilience study's "distance from baseline" family of
    functions (dist_from_baseline(), dist_from_baseline_around_r0())
    compares each swept value against.

    Args:
        result_dir (str): directory 'gf_dataset_baseline.nc' is written
            into. Defaults to RESULT_DIR (the main sensitivity study);
            pass a study-specific directory (e.g.
            RESILIENCE_RESULT_DIR) to keep that study fully self-
            contained, independent of whether the main sensitivity
            study's own baseline has been built.
        img_dir (str): directory the 2 diagnostic figures
            ("gamma_baseline.png"/"gamma_baseline_r0.png") are saved
            into. Defaults to IMG_DIR.
        env_overrides (dict|None): override/extend
            load_all_arg_dict()'s own baseline values before building
            the environment -- e.g. {"depth": 2000.0, "z_ssp": ...,
            "c_p_ssp": ...} for a realistic-profile, deep-water
            baseline (see build_celerity_baseline(), which uses this).
            None (the default): the plain, isovelocity baseline,
            unchanged from before this parameter existed.

    Returns:
        str: path to the written '.nc' file.
    """
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    all_arg_dict = load_all_arg_dict(
        drop_keys=("fs", "fmax", "r0", "d12"),
        d12_max=5000,
    )
    if env_overrides:
        all_arg_dict.update(env_overrides)

    # Build dataset
    call_kwargs = _extract_kwargs(build_dataset_current_config_kraken, all_arg_dict)
    call_kwargs["plot_diag"] = True
    # NOTE (bug fixed): build_kraken()'s diagnostic figures
    # ("environment.png", "mode_shapes.png", ...) used to always save
    # to the hardcoded, module-level IMG_DIR, regardless of which
    # 'img_dir' THIS build_baseline() call was actually given --
    # confirmed to make e.g. build_celerity_baseline()'s "sw"/"dw"
    # diagnostics both silently overwrite IMG_DIR's own files instead
    # of landing in their own study-specific image folder.
    # Forwarded explicitly here since it's not part of 'all_arg_dict'
    # (load_all_arg_dict() has no notion of it at all).
    call_kwargs["img_dir"] = img_dir
    kraken_freq, kraken_r, g_fr = build_dataset_current_config_kraken(**call_kwargs)

    g_fr_full = _pad_to_full_frequency_grid(g_fr, kraken_freq, all_arg_dict["freq"])

    # Build xarray dataset and save
    # NOTE: see _to_float32()'s own docstring -- same write-time
    # downcast as build_sensitivity_dataset(), for the same reason.
    ds = xr.Dataset(
        data_vars=dict(gf=(["f", "r"], _to_float32(np.abs(g_fr_full)))),
        coords={"f": _to_float32(all_arg_dict["freq"]), "r": _to_float32(kraken_r)},
    )
    # Keep the FULL baseline config as attrs (including "r0"/"d12", not
    # part of 'all_arg_dict' used for the KRAKEN call above) -- needed
    # later by process_sensitivity()/dist_from_baseline(). 'z_ssp'/
    # 'c_p_ssp' (a realistic profile's own arrays, when 'env_overrides'
    # provides them) are excluded here -- they'd otherwise bloat the
    # saved file's global attrs with a full depth profile; nothing
    # downstream needs them back out of 'ds.attrs' (build_kraken() is
    # never re-run from a saved dataset's own attrs).
    ds.attrs = {
        **{k: v for k, v in all_arg_dict.items() if k not in ("z_ssp", "c_p_ssp")},
        "r0": baseline_src_rcv()["r0"],
        "d12": baseline_src_rcv()["d12"],
    }

    # Derive gamma from gf
    gamma = derive_gamma(ds, ds.attrs["d12"])
    ds["gamma"] = gamma

    # Plot
    plt.figure()
    gamma.plot(
        cmap="magma",
        cbar_kwargs={"label": r"$\gamma$ [dB]"},
        vmin=np.nanpercentile(gamma, 1),
        vmax=np.nanpercentile(gamma, 99),
    )
    plt.ylabel("Fréquence [Hz]")
    plt.xlabel("r [m]")
    plt.ylim(np.min(kraken_freq), np.max(kraken_freq))
    plt.savefig(os.path.join(img_dir, "gamma_baseline.png"))

    plt.figure()
    gamma.sel(r=ds.r0, method="nearest").plot()
    plt.xlabel("Fréquence [Hz]")
    plt.ylabel(r"$\gamma$ [dB]")
    # plt.ylim(np.min(kraken_freq), np.max(kraken_freq))
    plt.savefig(os.path.join(img_dir, "gamma_baseline_r0.png"))

    plt.close("all")

    fpath = os.path.join(result_dir, "gf_dataset_baseline.nc")
    ds.to_netcdf(fpath)
    ds.close()
    return fpath


def dist_from_baseline(ds_baseline, ds_test, d12, r0):
    """Compute the RTF distance (L1/L2/theta/wasserstein) between the
    baseline's gamma at r0 and each swept value's gamma at r0.

    NOTE (wasserstein added, per user request): unlike L1/L2/theta
    (which compare gamma directly), the Wasserstein distance compares
    the RTF MAGNITUDE itself (rtf = 10 ** (gamma / 20)) -- see
    calc_rtf_wasserstein_dist()'s own docstring for the full rationale.
    A genuinely different way of asking "how far is this RTF from the
    baseline's": L1/L2 are sensitive to a uniform dB offset across the
    whole band the same way regardless of WHERE in frequency it occurs,
    while the Wasserstein distance treats the RTF as a mass
    distribution over frequency and penalizes moving that mass FURTHER
    (in frequency) more than moving it a little -- closer to "how much
    of the RTF's energy changed position, and by how far" than "how
    much did each frequency bin change in isolation".

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
    dist_wasserstein = calc_gamma_dist(
        gamma_a=gamma_a, gamma_b=gamma_b, dist_type="wasserstein"
    )

    return dist_L1, dist_L2, dist_theta, dist_wasserstein


def dist_from_baseline_around_r0(ds_baseline, ds_test, d12, r0):
    """Compute the RTF distance (L1/L2/theta) between the BASELINE's
    gamma AT r0 (a single, fixed reference) and EACH swept value's
    gamma AS A FUNCTION OF r, over the whole r window 'ds_test' covers
    -- unlike dist_from_baseline() (which only ever evaluates this at
    r=r0 itself, discarding the rest), this keeps the full r-dependence
    needed to derive a -3dB mainlobe width for every swept value (see
    calc_mainlobe_width_3dB() / single_sensitivity_test_calc_dist_width()
    and process_sensitivity_mainlobe_width(), which is what this
    function is for).

    Each swept value's distance curve is normalized by its OWN maximum
    over r (not a single global maximum shared across values) -- this
    matches single_sensitivity_test_calc_dist()'s established
    convention (see its own docstring) and is what
    calc_mainlobe_width_3dB()'s ">1/2" (-3dB) threshold assumes.

    Args:
        ds_baseline (xr.Dataset): the baseline dataset (see
            build_baseline()), with a precomputed 'gamma(f, r)'.
        ds_test (xr.Dataset): the swept dataset (see
            build_sensitivity_dataset()/
            process_sensitivity_mainlobe_width()), with
            'gf(test_arg_name, f, r)' -- the raw Green's function, not
            yet turned into an RTF.
        d12 (float): receiver separation (m).
        r0 (float): reference range (m).

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray): r (the
        range grid the distance curves are evaluated on, meters --
        receiver-1 positions, matching r0's own convention), dist_L1,
        dist_L2, dist_theta -- each shape (n_r, n_test_values), one
        column per swept value.

    Raises:
        ValueError: if 'ds_test''s r window is too narrow for 'd12'
            (receiver-1 and receiver-2's grids end up different sizes).
    """
    # Baseline gamma at r0 -- a single reference vector across
    # frequency, independent of the swept parameter (already
    # precomputed for every r at build_baseline() time -- a ONE-TIME
    # run, not a per-value sweep, so computing it over the full r grid
    # there is fine, see derive_gamma()'s own docstring).
    gamma_baseline = ds_baseline.gamma.sel(r=r0, method="nearest")
    gamma_a = gamma_baseline.values.T[:, np.newaxis, np.newaxis]  # (nf, 1, 1)

    # Test gamma as a function of r (the whole window 'ds_test' covers,
    # for every swept value at once): receiver 1 at r, receiver 2 at
    # r+d12 (matching derive_gamma()'s own index-shift convention, just
    # done here via an explicit range selection instead of an index
    # count, since 'ds_test' may not start at r=0).
    g_fr_1 = ds_test.gf.sel(r=slice(ds_test.r.min(), ds_test.r.max() - d12))
    g_fr_2 = ds_test.gf.sel(r=slice(ds_test.r.min() + d12, ds_test.r.max()))
    if (
        g_fr_1.sizes["r"] == 0
        or g_fr_2.sizes["r"] == 0
        or g_fr_1.sizes["r"] != g_fr_2.sizes["r"]
    ):
        raise ValueError(
            f"dist_from_baseline_around_r0: receiver-1 and receiver-2 grids "
            f"came out empty or differently sized ({g_fr_1.sizes['r']} vs "
            f"{g_fr_2.sizes['r']}) -- 'ds_test' likely doesn't extend far "
            f"enough beyond its own range window for d12={d12} (see "
            f"build_dataset_current_config_kraken()'s 'd12_max')."
        )

    gamma_test_r0 = 20 * np.log10(
        np.abs(g_fr_2.values / g_fr_1.values)
    )  # (ntest, nf, nr)
    gamma_b = np.moveaxis(
        gamma_test_r0, source=[0, 1, 2], destination=[-1, 0, 1]
    )  # (nf, nr, ntest)
    r_dist = g_fr_1.r.values

    # Compute distance
    dist_L1 = calc_gamma_dist(
        gamma_a=gamma_a, gamma_b=gamma_b, dist_type="L1"
    )  # (nr, ntest)
    dist_L2 = calc_gamma_dist(
        gamma_a=gamma_a, gamma_b=gamma_b, dist_type="L2"
    )  # (nr, ntest)
    dist_theta = calc_gamma_dist(
        gamma_a=gamma_a, gamma_b=gamma_b, dist_type="theta"
    )  # (nr, ntest)

    # Normalize each swept value's OWN curve by its OWN max over r (see
    # docstring) -- axis=0 is 'r' here (shape (nr, ntest)).
    dist_L1 = dist_L1 / np.max(dist_L1, axis=0)
    dist_L2 = dist_L2 / np.max(dist_L2, axis=0)
    dist_theta = dist_theta / np.max(dist_theta, axis=0)

    return r_dist, dist_L1, dist_L2, dist_theta


def derive_gamma(ds, d12):
    """Compute gamma(freq, r) over the FULL r grid.

    NOTE: originally written for -- and still fine for --
    build_baseline()'s use (a single, ONE-TIME configuration). Using
    this on a full, WIDE receiver-range grid (spanning many thousands
    of points) for every value of a multi-value swept dataset used to
    be a real memory/performance problem (see dist_from_baseline()'s
    docstring) -- but is fine, and now used, on a multi-value dataset
    whose r window is narrow by construction (see
    dist_from_baseline_around_r0()'s and
    dist_within_test_around_r0()'s own NOTEs: baseline_src_rcv()'s
    r_rcv only spans a region around r0, not the whole receiver
    range). If you widen that window back up, reconsider whether this
    is still an appropriate thing to call per swept value.
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


def dist_within_test_around_r0(ds_test, d12, r0):
    """Compute the RTF distance (L1/L2/theta) between EACH swept
    value's OWN gamma at r0 and that SAME value's gamma as a function
    of r -- entirely self-referential, no baseline involved at all
    (unlike dist_from_baseline_around_r0(), which compares against a
    FIXED, external baseline reference instead).

    This measures how sharply resolved/peaked each configuration's own
    RTF mainlobe is around r0 on its own terms -- i.e. for THIS
    parameter value, if you were trying to localize a source using
    this exact RTF, how much would the range estimate degrade as you
    move away from r0? -- independent of any mismatch against a
    baseline/"true" configuration (that mismatch is what
    dist_from_baseline_around_r0() measures instead). This generalizes
    this project's original, single-run
    single_sensitivity_test_calc_dist() (see its own docstring) to
    operate on every swept value in 'ds_test' at once, matching
    dist_from_baseline_around_r0()'s output shape/normalization
    convention so both can feed
    single_sensitivity_test_calc_dist_width() unchanged.

    Args:
        ds_test (xr.Dataset): the swept dataset (see
            build_sensitivity_dataset()), with 'gf(test_arg_name, f, r)'
            -- the raw Green's function, not yet turned into an RTF.
        d12 (float): receiver separation (m).
        r0 (float): reference range (m).

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray): r (the
        range grid the distance curves are evaluated on, meters),
        dist_L1, dist_L2, dist_theta -- each shape (n_r,
        n_test_values), one column per swept value, normalized by its
        own max over r (same convention as
        dist_from_baseline_around_r0() -- see its own docstring for
        why, and calc_mainlobe_width_3dB()'s ">1/2" threshold, which
        assumes it).
    """
    # gamma(f, r) for every swept value at once -- feasible here
    # because 'ds_test''s r window is narrow by construction (see
    # derive_gamma()'s own updated NOTE).
    gamma_test = derive_gamma(ds_test, d12)  # dims (test_arg_name, f, r)
    r = gamma_test.r.values

    # Each value's OWN gamma at r0 -- the reference THIS function
    # compares against is different for every swept value, unlike
    # dist_from_baseline_around_r0()'s single, shared baseline
    # reference.
    gamma_test_r0 = gamma_test.sel(r=r0, method="nearest")  # dims (test_arg_name, f)
    gamma_a = np.moveaxis(gamma_test_r0.values, [0, 1], [-1, 0])[
        :, np.newaxis, :
    ]  # (nf, 1, ntest)
    gamma_b = np.moveaxis(gamma_test.values, [0, 1, 2], [-1, 0, 1])  # (nf, nr, ntest)

    # Compute distance
    dist_L1 = calc_gamma_dist(
        gamma_a=gamma_a, gamma_b=gamma_b, dist_type="L1"
    )  # (nr, ntest)
    dist_L2 = calc_gamma_dist(
        gamma_a=gamma_a, gamma_b=gamma_b, dist_type="L2"
    )  # (nr, ntest)
    dist_theta = calc_gamma_dist(
        gamma_a=gamma_a, gamma_b=gamma_b, dist_type="theta"
    )  # (nr, ntest)

    # Normalize each swept value's OWN curve by its OWN max over r
    # (axis=0 is 'r' here, shape (nr, ntest)) -- see docstring.
    dist_L1 = dist_L1 / np.max(dist_L1, axis=0)
    dist_L2 = dist_L2 / np.max(dist_L2, axis=0)
    dist_theta = dist_theta / np.max(dist_theta, axis=0)

    return r, dist_L1, dist_L2, dist_theta


def _select_r0_pair(ds, r0, d12):
    """xr.open_mfdataset's 'preprocess' callback (see
    process_sensitivity()): reduce a single per-value file's 'gf'
    variable down to just the 2 r-values the RTF at r0 actually needs
    (r0 and r0+d12), IMMEDIATELY as that one file is opened, before any
    multi-file concatenation.

    This is what actually fixes process_sensitivity()'s slowness/memory
    usage (see dist_from_baseline()'s docstring for the full
    explanation): 'preprocess' runs on each file while it is still
    small and separate, so the (f, r) array with r spanning the whole,
    possibly huge receiver-range grid -- built by
    build_sensitivity_dataset(), one such file per swept value -- gets
    cut down to (f, 2) before dask ever needs to combine it with any
    other file's data. Selecting r0/r0+d12 AFTER concatenation instead
    (as a plain '.sel()' on the combined dataset) would still require
    reading every file's FULL r range from disk first, since
    xr.open_mfdataset's own lazy dask arrays are chunked per-file --
    'preprocess' avoids that entirely by never letting the full arrays
    become part of the combined dataset's dask graph in the first place.
    """
    return ds.sel(r=[r0, r0 + d12], method="nearest")


def save_sensitivity_distance_results(
    test_arg_name,
    test_values,
    dist_L1,
    dist_L2,
    dist_theta,
    dist_wasserstein=None,
    result_dir=RESULT_DIR,
    file_prefix="dist_",
):
    """Save the RTF distance-from-baseline results (L1/L2/theta[/
    wasserstein]) for ONE swept parameter to a small, dedicated CSV
    file -- small enough (a handful of floats per swept value) that a
    plain text format is the simplest, most portable choice; no need
    for NetCDF/xarray here, unlike the much larger raw Green's
    function datasets this is derived from (see
    build_sensitivity_dataset()).

    Args:
        test_arg_name (str): the swept parameter's name.
        test_values, dist_L1, dist_L2, dist_theta (array-like): equal-
            length 1D arrays, as returned by dist_from_baseline() (plus
            the parameter values themselves) -- or, for
            process_sensitivity_mainlobe_width(), the mainlobe WIDTHS
            (see single_sensitivity_test_calc_dist_width()) rather than
            raw distances.
        dist_wasserstein (array-like|None): the Wasserstein distance
            (see dist_from_baseline()'s own docstring), same length as
            the others. None (the default): omitted from the saved
            file entirely -- callers that only ever have L1/L2/theta
            (e.g. process_sensitivity_mainlobe_width()/
            process_sensitivity_intrinsic_mainlobe_width(), whose
            "distance AS A FUNCTION OF r" this hasn't been extended to
            -- see calc_gamma_dist()'s own NOTE) keep writing the
            original 4-column ("<test_arg_name>,L1,L2,theta") file
            unchanged; only dist_from_baseline()'s own callers
            (process_sensitivity()/process_celerity_sensitivity())
            pass it.
        result_dir (str): directory to write into, as
            '<result_dir>/<file_prefix><test_arg_name>.csv'.
        file_prefix (str): distinguishes what these metrics actually
            are when several kinds of results are saved side by side
            in the same 'result_dir' -- e.g. "dist_" for the raw RTF
            distance at r0 (process_sensitivity()) vs
            "mainlobe_width_" for the -3dB mainlobe width
            (process_sensitivity_mainlobe_width()). Using the same
            prefix for two DIFFERENT quantities would silently
            overwrite one with the other.

    Returns:
        str: path to the written file.
    """
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f"{file_prefix}{test_arg_name}.csv")
    if dist_wasserstein is None:
        data = np.column_stack([test_values, dist_L1, dist_L2, dist_theta])
        header = f"{test_arg_name},L1,L2,theta"
    else:
        data = np.column_stack(
            [test_values, dist_L1, dist_L2, dist_theta, dist_wasserstein]
        )
        header = f"{test_arg_name},L1,L2,theta,wasserstein"
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    return path


def load_sensitivity_distance_results(
    test_arg_name, result_dir=RESULT_DIR, file_prefix="dist_"
):
    """Reload a results file written by save_sensitivity_distance_results().

    Args:
        test_arg_name (str): the swept parameter's name.
        result_dir (str): same directory passed to
            save_sensitivity_distance_results().
        file_prefix (str): same prefix passed to
            save_sensitivity_distance_results() (e.g. "dist_" or
            "mainlobe_width_" -- see its own docstring).

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray|None):
        test_values, dist_L1, dist_L2, dist_theta, dist_wasserstein (or
        the mainlobe width equivalents, depending on 'file_prefix').
        dist_wasserstein is None when the file was saved without it
        (e.g. "mainlobe_width_"/"intrinsic_mainlobe_width_" files --
        see save_sensitivity_distance_results()'s own docstring).
    """
    path = os.path.join(result_dir, f"{file_prefix}{test_arg_name}.csv")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)
    dist_wasserstein = data[:, 4] if data.shape[1] > 4 else None
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], dist_wasserstein


def process_sensitivity(test_arg_names=None, result_dir=RESULT_DIR, save_dir=None):
    """Read back every parameter's per-value result files (see
    build_sensitivity_dataset()), compute each value's RTF distance
    from the baseline (see build_baseline()), and save the resulting
    L1/L2/theta distance-vs-parameter-value arrays to a dedicated file
    per parameter (see save_sensitivity_distance_results()), then plot
    them (see plot_sensitivity_curves()).

    NOTE (new, per user request): this function used to plot directly,
    inline, right after computing each parameter's distances, without
    saving them anywhere -- meaning the (expensive: see the NOTE below)
    computation had to be repeated from scratch just to see the plot
    again, e.g. after tweaking its styling. Every parameter's L1/L2/
    theta arrays are now saved to their own small file as soon as
    they're computed; plot_sensitivity_curves() is a separate function
    that only ever needs to read those small files back, not
    reprocess anything.

    NOTE (memory usage improved): the original opened each parameter's
    single, already-combined multi-value NetCDF file with a plain
    `xr.open_dataset(fpath)` -- eagerly loading the WHOLE file (every
    swept value's full g_fr grid) into RAM, for every parameter, all
    at once (nothing closed any of them either). Given
    build_sensitivity_dataset() now writes one small file per value
    instead (see its own docstring), this function reads them back
    with `xr.open_mfdataset(..., preprocess=_select_r0_pair)`: the
    'preprocess' callback (see its own docstring) cuts each file down
    to just the 2 r-values (r0, r0+d12) the RTF distance actually needs
    BEFORE combining files, so the full, possibly-huge receiver-range
    grid each file holds is never read in its entirety at all -- this
    is what actually fixes process_sensitivity() taking several minutes
    and a lot of RAM (see dist_from_baseline()'s docstring for exactly
    where the old code forced the full computation). Each parameter's
    dataset is also opened in its own 'with' block, so it is released
    before moving on to the next parameter, rather than accumulating
    every parameter's dataset handle for the rest of the run.

    Args:
        test_arg_names (list[str]|None): which parameters to process
            (each must have a '<result_dir>/<name>/' folder from
            build_sensitivity_dataset()). None discovers every such
            folder under 'result_dir' automatically.
        result_dir (str): where to read the per-value/baseline files
            from, and where to save the distance results.
        save_dir (str|None): forwarded to plot_sensitivity_curves() --
            if given, also save the returned figure there (e.g.
            RESILIENCE_IMG_DIR for a resilience study). None (the
            default): the figure is only returned, not saved.

    Returns:
        matplotlib.figure.Figure (see plot_sensitivity_curves()).
    """
    import functools

    baseline_src_rcv_param = baseline_src_rcv()
    d12 = baseline_src_rcv_param["d12"]
    r0 = baseline_src_rcv_param["r0"]
    preprocess = functools.partial(_select_r0_pair, r0=r0, d12=d12)

    if test_arg_names is None:
        test_arg_names = sorted(
            name
            for name in os.listdir(result_dir)
            if os.path.isdir(os.path.join(result_dir, name))
        )

    fpath_baseline = os.path.join(result_dir, "gf_dataset_baseline.nc")

    from time import time

    t0 = time()
    # NOTE: the baseline is small (a single configuration, not a
    # sweep) -- safe to keep open for the whole loop, unlike the
    # per-parameter sweep datasets below.
    with xr.open_dataset(fpath_baseline) as ds_baseline:
        for test_arg_name in test_arg_names:
            value_files = sorted(
                glob.glob(
                    os.path.join(result_dir, test_arg_name, f"{test_arg_name}_*.nc")
                )
            )
            if not value_files:
                continue

            # NOTE: 'join="override"' silences a real xarray
            # FutureWarning (confirmed by the user) -- open_mfdataset()
            # needs to align every NON-concat coordinate shared across
            # the files being combined (here "f"/"r", and "z" for the
            # celerity study's own per-profile files -- see
            # process_celerity_sensitivity()); its default is changing
            # from 'outer' (pad mismatches with NaN) to 'exact' (raise
            # ValueError on any mismatch) in a future xarray version.
            # "override" is the right choice specifically here (rather
            # than silencing the warning by switching to "exact"):
            # every one of these coordinates is written IDENTICALLY
            # into each per-value/per-profile file by
            # build_sensitivity_dataset()/
            # build_celerity_sensitivity_dataset() (same source array,
            # just repeated per file) -- "override" skips the
            # (redundant, since they're already known to match) index-
            # equality check entirely and just uses the first file's
            # own coordinate values, which is also the cheapest option
            # when combining many files.
            with xr.open_mfdataset(
                value_files,
                combine="nested",
                concat_dim=test_arg_name,
                join="override",
                preprocess=preprocess,
            ) as ds_test:
                dist_L1, dist_L2, dist_theta, dist_wasserstein = dist_from_baseline(
                    ds_baseline, ds_test, d12, r0
                )
                test_values = ds_test[test_arg_name].values
                print(f"Ellapsed time for var {test_arg_name} = {time()-t0:.1f}")

            save_sensitivity_distance_results(
                test_arg_name,
                test_values,
                dist_L1,
                dist_L2,
                dist_theta,
                dist_wasserstein,
                result_dir=result_dir,
            )

    return plot_sensitivity_curves(
        test_arg_names, result_dir=result_dir, save_dir=save_dir
    )


def plot_sensitivity_curves(
    test_arg_names=None,
    result_dir=RESULT_DIR,
    distance="theta",
    file_prefix="dist_",
    ylabel=None,
    save_dir=None,
):
    """Plot the RTF distance/mainlobe-width-vs-parameter-value curves
    (L1/L2/theta), reading them back from the small per-parameter files
    saved by process_sensitivity()/process_sensitivity_mainlobe_width()
    (see save_sensitivity_distance_results()) -- does NOT open or
    process any '.nc' file, so this is fast and cheap to call
    repeatedly (e.g. while tweaking the plot itself), unlike those
    functions (which need a full pass over the raw Green's function
    files to (re)compute these curves in the first place).

    Args:
        test_arg_names (list[str]|None): which parameters to plot
            (each must have a '<result_dir>/<file_prefix><name>.csv'
            file). None discovers every such file under 'result_dir'
            automatically.
        result_dir (str): where to read the saved results from.
        distance (str|list[str]): distance metric(s) to plot -- any of
            "L1", "L2", "theta", "wasserstein" (the last one only
            available for files saved with a 'dist_wasserstein' -- see
            save_sensitivity_distance_results()'s own docstring;
            requesting it for a 'file_prefix' that doesn't have it,
            e.g. "mainlobe_width_", raises a clear ValueError rather
            than a cryptic one).
        file_prefix (str): which saved results to read -- "dist_" (the
            default, matching process_sensitivity()'s raw RTF distance
            at r0) or "mainlobe_width_" (matching
            process_sensitivity_mainlobe_width()'s -3dB mainlobe
            width) -- see save_sensitivity_distance_results()'s own
            docstring.
        ylabel (str|None): y-axis label override, used when
            'distance' has more than one entry (otherwise the metric's
            own label, e.g. r"$L_1$", is used directly). Defaults to
            "Distance", appropriate for 'file_prefix="dist_"'; pass
            e.g. "Mainlobe width [m]" for 'file_prefix="mainlobe_width_"'.
        save_dir (str|None): if given, save the figure to
            '<save_dir>/<filename>.png' (see
            _sensitivity_figure_filename()) -- directory created if it
            doesn't exist yet. None (the default): the figure is only
            returned, not saved to disk. Pass IMG_DIR for this
            project's own conventional image folder.

    Returns:
        matplotlib.figure.Figure

    Raises:
        ValueError: if "wasserstein" is requested in 'distance' but
            the saved files don't have it (see 'distance' above).
    """

    from matplotlib.lines import Line2D

    if test_arg_names is None:
        suffix = ".csv"
        test_arg_names = sorted(
            name[len(file_prefix) : -len(suffix)]
            for name in os.listdir(result_dir)
            if name.startswith(file_prefix) and name.endswith(suffix)
        )

    n_tests = len(test_arg_names)
    fig, axs = plt.subplots(
        1, max(n_tests, 1), figsize=(16, 8), squeeze=False, sharey=True
    )
    axs = axs[0]

    arg_label = ARG_LABEL
    dist_label = METRIC_LABEL
    # NOTE (bug fixed): np.array(distance) leaves a 0-dimensional array
    # for the default single-string usage (distance="theta"), and
    # `distance[0]` a few lines below raises `IndexError: too many
    # indices for array: array is 0-dimensional` on that -- confirmed
    # reachable via the function's own default argument.
    # np.atleast_1d() always gives an indexable 1D array instead
    # (size 1 for a single string, same as before for the `.size > 1`
    # check just below).
    distance = np.atleast_1d(distance)

    def _require_wasserstein(dist_wasserstein, test_arg_name):
        if dist_wasserstein is None:
            raise ValueError(
                f"plot_sensitivity_curves: 'wasserstein' was requested but "
                f"'{file_prefix}{test_arg_name}.csv' was saved without it "
                f"(see save_sensitivity_distance_results()'s own docstring "
                f"-- only dist_from_baseline()'s own callers save it)."
            )

    if distance.size > 1:
        d_L1, d_L2, d_theta, d_wasserstein = [], [], [], []
        for i, test_arg_name in enumerate(test_arg_names):
            test_values, dist_L1, dist_L2, dist_theta, dist_wasserstein = (
                load_sensitivity_distance_results(
                    test_arg_name, result_dir=result_dir, file_prefix=file_prefix
                )
            )
            d_L1.append(np.max(dist_L1))
            d_L2.append(np.max(dist_L2))
            d_theta.append(np.max(dist_theta))
            if "wasserstein" in distance:
                _require_wasserstein(dist_wasserstein, test_arg_name)
                d_wasserstein.append(np.max(dist_wasserstein))

        norm_factor_L1 = np.max(d_L1)
        norm_factor_L2 = np.max(d_L2)
        norm_factor_theta = np.max(d_theta)
        norm_factor_wasserstein = np.max(d_wasserstein) if d_wasserstein else 1
    else:
        norm_factor_L1 = 1
        norm_factor_L2 = 1
        norm_factor_theta = 1
        norm_factor_wasserstein = 1

    for i, test_arg_name in enumerate(test_arg_names):
        test_values, dist_L1, dist_L2, dist_theta, dist_wasserstein = (
            load_sensitivity_distance_results(
                test_arg_name, result_dir=result_dir, file_prefix=file_prefix
            )
        )
        icol = 0

        # NOTE : not relevant
        # if test_arg_name == "depth":
        #     axs[i].set_xscale("log")

        if "L1" in distance:
            axs[i].plot(test_values, dist_L1 / norm_factor_L1, color=color(icol))
            icol += 1
        if "L2" in distance:
            axs[i].plot(test_values, dist_L2 / norm_factor_L2, color=color(icol))
            icol += 1
        if "theta" in distance:
            axs[i].plot(test_values, dist_theta / norm_factor_theta, color=color(icol))
            icol += 1
        if "wasserstein" in distance:
            _require_wasserstein(dist_wasserstein, test_arg_name)
            axs[i].plot(
                test_values,
                dist_wasserstein / norm_factor_wasserstein,
                color=color(icol),
            )
            icol += 1

        axs[i].set_xlabel(arg_label.get(test_arg_name, test_arg_name))
        # axs[i].legend()

    if distance.size > 1:
        fig.supylabel(ylabel or "Distance")
        legend_handles = [
            Line2D([0], [0], color=color(i), linestyle="-", label=dist_label[dist])
            for i, dist in enumerate(distance)
        ]

        fig.legend(
            handles=legend_handles, loc="outside upper center", ncols=distance.size
        )
    else:
        # theta is bounded between 0 and 1, so it is a good idea to set the y limit to [0, 1] for better visualization
        if "theta" in distance and not "intrinsic" in file_prefix:
            axs[0].set_ylim([0, 1])

        fig.supylabel(ylabel or dist_label[distance[0]])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = _sensitivity_figure_filename(
            "curves", test_arg_names, file_prefix=file_prefix
        )
        fig.savefig(os.path.join(save_dir, fname))

        plt.close(fig)

    return fig


def consolidate_sensitivity_dataset(
    test_arg_name, r_window, result_dir=RESULT_DIR, delete_originals=False
):
    """Combine every per-value '.nc' file for 'test_arg_name' (written
    by build_sensitivity_dataset(), one per swept value -- see its own
    docstring for why there are many small files rather than one big
    one) into a SINGLE combined file, keeping only a WINDOW of the
    range grid centered on the baseline's r0 (from r0-r_window to
    r0+r_window) rather than each file's full, much wider extended
    range grid -- to limit the disk space the combined file uses.

    Uses the same 'preprocess' trick as process_sensitivity() (see
    _select_r0_pair()'s docstring): the window is selected on EACH file
    individually, before they are combined, so the parts of the range
    grid outside the window are never read in full for any file either.

    Args:
        test_arg_name (str): the swept parameter whose per-value files
            (under '<result_dir>/<test_arg_name>/') should be combined.
        r_window (float): half-width (m) of the r-window to keep,
            centered on the baseline's r0 (i.e. the kept range is
            [r0 - r_window, r0 + r_window]).
        result_dir (str): where the per-value folder lives, and where
            the combined file is written
            ('<result_dir>/<test_arg_name>_combined.nc').
        delete_originals (bool): if True, delete the per-value '.nc'
            files (and their now-empty directory) once the combined
            file has been written successfully -- this is the only way
            this function actually FREES disk space rather than merely
            adding a smaller extra file; left as an explicit, off-by-
            default opt-in since it's destructive.

    Returns:
        str: path to the written combined file.

    Raises:
        FileNotFoundError: if no per-value files exist for 'test_arg_name'.
    """

    r0 = baseline_src_rcv()["r0"]
    value_dir = os.path.join(result_dir, test_arg_name)
    value_files = sorted(glob.glob(os.path.join(value_dir, f"{test_arg_name}_*.nc")))
    if not value_files:
        raise FileNotFoundError(
            f"No per-value files found for '{test_arg_name}' under {value_dir} "
            f"-- run build_sensitivity_dataset('{test_arg_name}', ...) first."
        )

    def _select_r_window(ds):
        return ds.sel(r=slice(r0 - r_window, r0 + r_window))

    with xr.open_mfdataset(
        value_files,
        combine="nested",
        concat_dim=test_arg_name,
        join="override",
        preprocess=_select_r_window,
    ) as ds_combined:
        fpath = os.path.join(result_dir, f"{test_arg_name}_combined.nc")
        ds_combined.to_netcdf(fpath)

    if delete_originals:
        for f in value_files:
            os.remove(f)
        try:
            os.rmdir(value_dir)  # only removes it if now empty
        except OSError:
            pass

    return fpath


def consolidate_all_sensitivity_datasets(
    r_window, test_arg_names=None, result_dir=RESULT_DIR, delete_originals=False
):
    """Apply consolidate_sensitivity_dataset() to every swept parameter
    found under 'result_dir' (or just 'test_arg_names', if given).

    Args: see consolidate_sensitivity_dataset().

    Returns:
        dict[str, str]: test_arg_name -> path to its combined file.
    """
    if test_arg_names is None:
        test_arg_names = sorted(
            name
            for name in os.listdir(result_dir)
            if os.path.isdir(os.path.join(result_dir, name))
        )

    fpaths = {}
    for test_arg_name in test_arg_names:
        fpaths[test_arg_name] = consolidate_sensitivity_dataset(
            test_arg_name,
            r_window,
            result_dir=result_dir,
            delete_originals=delete_originals,
        )
    return fpaths


def build_tests(use_debug_config=False):
    all_arg_dict = load_all_arg_dict(
        drop_keys=("fs", "fmax", "r0", "d12"), d12_max=5000
    )

    # npt = 20
    # nb_param = 4
    # size_per_test_Ko = 110000
    # total_size = size_per_test_Ko * nb_param * npt
    # print(
    #     f"Total memory size (if it were all held at once) = {total_size * 1e-6} Go "
    #     f"-- no longer applicable: build_sensitivity_dataset() now writes each "
    #     f"value's result as soon as it's computed (see its own docstring)."
    # )

    if use_debug_config:
        ndebug = 10
        sweeps = {
            "depth": np.linspace(70, 130, ndebug),
            "c1": np.linspace(1460, 1540, ndebug),
            "rho2": np.linspace(1.0 * 1e3, 2.5 * 1e3, ndebug),
            "attn2": np.linspace(0.0, 1.0, ndebug),
        }

    else:
        sweeps = {
            "depth": np.linspace(30, 5000, 300),
            "c1": np.linspace(1460, 1540, 160),
            # "c2": np.linspace(1550.0, 1900.0, npt),
            "rho2": np.linspace(1.0 * 1e3, 2.5 * 1e3, 250),
            "attn2": np.linspace(0.0, 1.0, 100),
        }

    for test_arg_name, test_arg_values in sweeps.items():
        build_sensitivity_dataset(
            test_arg_name, test_arg_values, all_arg_dict, model="kraken"
        )


# ======================================================================================================================
# Library resilience tests
# ======================================================================================================================
def build_resilience_tests(
    env_types=None, depth_var_tide=10, npt=200, ssp_data_dir=SSP_DATA_DIR
):
    """Build the depth-tide resilience test for one or several
    environment types.

    NOTE (env_type support added, per user request): this used to only
    ever build ONE (implicit) environment, at the classic sensitivity
    study's own baseline depth (100 m). Illustrating how the SAME
    absolute tidal elevation range affects a shallow- vs a deep-water
    waveguide differently needed a SECOND environment -- environment
    types reuse CELERITY_ENV_TYPES's own "sw"/"dw" depths (100 m /
    2000 m) for consistency with the celerity study's own naming, and
    each now gets its own '<RESILIENCE_RESULT_DIR>/<env_type>/depth/'
    subfolder (and its own baseline) instead of a single, implicit
    'RESILIENCE_RESULT_DIR/depth/' -- see process_resilience_tests()'s
    own matching update, and _resilience_study_dirs() (shared with the
    celerity sub-study's own '.../celerity/' folder -- see this
    module's own NOTE on RESILIENCE_RESULT_DIR for why the two are
    harmonized this way). This CHANGES the on-disk layout from before
    (results used to live directly under 'RESILIENCE_RESULT_DIR/depth/',
    then under 'RESILIENCE_RESULT_DIR/<env_type>/'); re-run this to
    regenerate under the new, harmonized one.

    Also fixed while at it: the baseline's own receiver depth is now
    set to 'nominal_depth - 1' for each environment type (matching
    build_sensitivity_dataset()'s own "z_rcv tracks 1 m above the
    seafloor" convention already used for every SWEPT depth value --
    see its own "depth" special-case), rather than leaving it at the
    classic baseline's fixed 99.5 m. For "sw" this is a negligible
    difference (99 m vs 99.5 m); for "dw" specifically, leaving it at
    99.5 m would put the baseline's receiver in the upper water column
    while every swept value's own receiver sits near 1999 m -- a
    receiver-DEPTH mismatch between baseline and sweep that has
    nothing to do with tide sensitivity, but would dominate the
    "distance from baseline" signal (see dist_from_baseline()) anyway
    if left unaddressed.

    NOTE (realistic celerity profile, per user request): the water
    column now uses each environment type's own MEAN ("all" situation
    -- see CELERITY_SITUATIONS/_ssp_filename()) real celerity profile
    (see load_mean_celerity_profile()) instead of the classic study's
    isovelocity one -- the same profile the celerity sub-study's own
    "all" baseline uses (see build_celerity_baseline()). Since the
    water depth itself is what's being swept here, that profile is
    automatically RE-ADAPTED (extended/truncated) to reach each swept
    depth value exactly -- see build_sensitivity_dataset()'s own
    "depth" special-case, which does this whenever 'z_ssp'/'c_p_ssp'
    are present in 'all_arg_dict' (as they now are here).

    Unlike the main sensitivity study (build_tests(), which asks "how
    much does the RTF's -3dB LOCALIZATION AMBIGUITY around r0 change as
    a waveguide parameter varies over a wide, largely unrealistic
    range" -- see process_sensitivity_mainlobe_width()), this asks a
    narrower, more practical question: "how much does GAMMA AT r0
    ITSELF change under REALISTIC, expected environmental variability"
    -- here, water depth fluctuating with the tide (the Bay of Fundy's
    ~20 m range is used as a realistic worst case). See
    process_resilience_tests() (dist_from_baseline(), evaluated
    directly AT r0 -- not a mainlobe width) for the matching analysis.

    Both each environment type's baseline (the nominal, unperturbed
    configuration -- see build_baseline()) and its swept dataset (see
    build_sensitivity_dataset()) are written under
    '<RESILIENCE_RESULT_DIR>/<env_type>/depth/' (NOT RESULT_DIR, the
    main sensitivity study's own directory) -- entirely separate from
    the main sensitivity study's own files, and self-contained (each
    environment type's baseline is (re)built here too, rather than
    assuming it already exists).

    Args:
        env_types (list[str]|None): which environment types to build
            -- see CELERITY_ENV_TYPES. None (the default): both "sw"
            and "dw".
        depth_var_tide (float): tidal range (m) to sweep depth over,
            +/- this value around each environment type's own nominal
            depth.
        npt (int): number of depth values to sweep per environment
            type.
        ssp_data_dir (str): forwarded to load_mean_celerity_profile().

    Raises:
        KeyError: if any entry of 'env_types' is not one of
            CELERITY_ENV_TYPES.
    """
    if env_types is None:
        env_types = list(CELERITY_ENV_TYPES.keys())

    for env_type in env_types:
        nominal_depth = CELERITY_ENV_TYPES[env_type]["depth"]
        result_dir, img_dir = _resilience_study_dirs(env_type, "depth")

        # NOTE: the "all" (whole-dataset mean) profile -- SAME one the
        # celerity sub-study's own "all" baseline uses (see
        # build_celerity_baseline()) -- adapted to this environment
        # type's nominal depth. Re-adapted per SWEPT depth value inside
        # build_sensitivity_dataset() itself (see its own "depth"
        # special-case), so this is only the STARTING point.
        z_ssp, c_p_ssp = load_mean_celerity_profile(
            _ssp_filename(env_type, "all"),
            target_depth=nominal_depth,
            ssp_data_dir=ssp_data_dir,
        )

        all_arg_dict = load_all_arg_dict(
            drop_keys=("fs", "fmax", "r0", "d12"), d12_max=5000
        )
        all_arg_dict["depth"] = nominal_depth
        all_arg_dict["z_ssp"] = z_ssp
        all_arg_dict["c_p_ssp"] = c_p_ssp

        # Self-contained: this environment type's own baseline, in its
        # own directory (see build_baseline()'s own 'result_dir'/
        # 'img_dir'/'env_overrides' docstring) -- does not depend on
        # any other environment type's or study's baseline existing.
        build_baseline(
            result_dir=result_dir,
            img_dir=img_dir,
            env_overrides={
                "depth": nominal_depth,
                "z_rcv": nominal_depth - 1,
                "z_ssp": z_ssp,
                "c_p_ssp": c_p_ssp,
            },
        )

        # Test sensibility to free surface elevation of the order of tide variations
        test_arg_name = "depth"
        test_arg_values = np.linspace(
            nominal_depth - depth_var_tide,
            nominal_depth + depth_var_tide,
            npt,
        )
        build_sensitivity_dataset(
            test_arg_name,
            test_arg_values,
            all_arg_dict,
            model="kraken",
            result_dir=result_dir,
        )


def process_resilience_tests(
    env_type="sw", test_arg_names=None, result_dir=None, save_dir=None
):
    """Read back build_resilience_tests()'s own result files for ONE
    environment type, compute the RTF distance BETWEEN THE BASELINE'S
    GAMMA AND EACH SWEPT VALUE'S GAMMA, BOTH EVALUATED AT r0 (see
    dist_from_baseline()) -- i.e. gamma's OWN sensitivity AT the
    reference position itself, as the swept parameter (water depth,
    for the tide test) varies -- and save/plot the result, entirely
    under the resilience study's own directories (see
    build_resilience_tests()'s own docstring and this module's NOTE on
    RESILIENCE_RESULT_DIR).

    This reuses process_sensitivity()'s own analysis
    (dist_from_baseline(): a single L1/L2/theta value per swept value,
    evaluated directly AT r0) -- NOT the -3dB mainlobe-width family
    (process_sensitivity_mainlobe_width()/
    process_sensitivity_intrinsic_mainlobe_width()), which asks a
    different question about localization ambiguity AROUND r0, not
    gamma's own value AT r0 -- just pointed at the resilience study's
    own directories instead of the main sensitivity study's.

    Args:
        env_type (str): "sw" or "dw" -- see CELERITY_ENV_TYPES/
            build_resilience_tests(). Each environment type has its
            OWN baseline (a different nominal depth entirely), so this
            only ever processes one at a time; see
            process_all_resilience_tests() to process every
            environment type at once.
        test_arg_names (list[str]|None): which resilience tests to
            process (each must have a
            '<RESILIENCE_RESULT_DIR>/<env_type>/depth/<name>/' folder
            from build_resilience_tests()). None discovers every such
            folder automatically (currently just "depth").
        result_dir (str|None): where to read the per-value/baseline
            files from, and where to save the distance results. None
            (the default): '<RESILIENCE_RESULT_DIR>/<env_type>/depth/'
            (see _resilience_study_dirs()).
        save_dir (str|None): forwarded to plot_sensitivity_curves().
            None (the default): '<RESILIENCE_IMG_DIR>/<env_type>/depth/'.

    Returns:
        matplotlib.figure.Figure (see plot_sensitivity_curves()).

    Raises:
        KeyError: if 'env_type' is not one of CELERITY_ENV_TYPES.
    """
    # NOTE: env_type must be a valid key -- fail fast with a clear
    # KeyError (matching build_celerity_baseline()'s own behaviour)
    # rather than a confusing FileNotFoundError further down.
    CELERITY_ENV_TYPES[env_type]

    default_result_dir, default_img_dir = _resilience_study_dirs(env_type, "depth")
    if result_dir is None:
        result_dir = default_result_dir
    if save_dir is None:
        save_dir = default_img_dir

    return process_sensitivity(
        test_arg_names=test_arg_names,
        result_dir=result_dir,
        save_dir=save_dir,
    )


def process_all_resilience_tests(env_types=None, test_arg_names=None):
    """Run process_resilience_tests() for every environment type
    requested -- the resilience study's own equivalent of
    process_celerity_tests() applied across environment types.

    Args:
        env_types (list[str]|None): which environment types to
            process. None (the default): every key of
            CELERITY_ENV_TYPES ("sw" and "dw").
        test_arg_names (list[str]|None): forwarded to
            process_resilience_tests().

    Returns:
        dict[str, matplotlib.figure.Figure]: env_type -> figure.
    """
    if env_types is None:
        env_types = list(CELERITY_ENV_TYPES.keys())

    return {
        env_type: process_resilience_tests(env_type, test_arg_names=test_arg_names)
        for env_type in env_types
    }


# ======================================================================================================================
# Library celerity (sound-speed profile) sensitivity tests
# ======================================================================================================================
def _get_ssp_variable(ds, fpath):
    """Return 'ds''s sound-speed profile variable, robust to an older,
    known-buggy naming issue.

    NOTE (bug fixed, defensive fallback): illustration_rtf/ssp/
    ssp_process_eof.py's own convert_synthetic_to_xarray() used to
    build the synthetic-profile DataArray without a 'name=' argument;
    saving an UNNAMED DataArray to NetCDF silently stores it under the
    generic variable name "__xarray_dataarray_variable__" instead of
    "ssp" -- confirmed by a real run (`ds.ssp` raised
    `AttributeError: 'Dataset' object has no attribute 'ssp'`). Fixed
    at the SOURCE (that function now names it "ssp" explicitly, like
    load_ssp_data.py's own real profiles already were) -- re-running
    process_ssp_profiles() regenerates correctly-named files. This
    fallback additionally lets ALREADY-generated (mis-named) files
    keep working without needing to regenerate them first: if "ssp"
    isn't found but there is EXACTLY ONE data variable in the file,
    that one is used instead (with a printed note), since a SSP file
    only ever holds the one profile variable to begin with.

    Args:
        ds (xr.Dataset): as opened from 'fpath'.
        fpath (str): only used to name the file in the printed note/
            error message.

    Returns:
        xr.DataArray

    Raises:
        ValueError: if "ssp" is not the variable's name AND the file
            has zero or more than one data variable (nothing safe to
            fall back to).
    """
    if "ssp" in ds.data_vars:
        return ds.ssp

    if len(ds.data_vars) == 1:
        (only_name,) = ds.data_vars
        print(
            f"Note: '{fpath}' has no 'ssp' variable (found '{only_name}' "
            f"instead) -- using it anyway, since it's the only data "
            f"variable in the file. Re-run "
            f"illustration_rtf/ssp/ssp_process_eof.py to regenerate this "
            f"file with the correct variable name."
        )
        return ds[only_name]

    raise ValueError(
        f"'{fpath}' has no 'ssp' variable, and {len(ds.data_vars)} data "
        f"variables in total -- cannot tell which one holds the sound-"
        f"speed profile."
    )


def _drop_depths_with_any_nan(z, c_p):
    """Drop every depth that is NaN for AT LEAST one profile, keeping
    every remaining profile fully NaN-free -- shared by
    load_mean_celerity_profile()/load_synthetic_celerity_profiles()
    (see get_ssp_eof()'s own NOTE on why checking every sample, not
    just one, matters). Also sorts by depth ascending.

    Args:
        z (np.ndarray): shape (n_depth,).
        c_p (np.ndarray): shape (n_depth,) for a single profile, or
            (n_profiles, n_depth) for a batch of them.

    Returns:
        tuple(np.ndarray, np.ndarray): z, c_p filtered/sorted the same
        way, same number of dimensions as given.
    """
    not_nan = ~np.isnan(c_p).any(axis=tuple(range(c_p.ndim - 1)))
    z = z[not_nan]
    c_p = c_p[..., not_nan]
    order = np.argsort(z)
    return z[order], c_p[..., order]


def _ensure_profile_starts_at_surface(z, c_p):
    """Extend a profile (or a batch of them, see
    _drop_depths_with_any_nan()'s own shape convention) flat down from
    the sea surface (z=0) if its shallowest point isn't already there
    -- see load_mean_celerity_profile()'s own NOTE for the rationale.
    """
    if z[0] > 0:
        z = np.concatenate([[0.0], z])
        c_p = np.concatenate([c_p[..., :1], c_p], axis=-1)
    return z, c_p


def _adapt_profile_to_depth(z, c_p, target_depth):
    """Adapt a profile (or a BATCH of profiles, c_p shape
    (n_profiles, n_depth), depth as the LAST axis) to reach EXACTLY
    'target_depth': linearly extended (using each profile's OWN last-2-
    points slope) if shallower, or truncated with an exact interpolated
    value AT 'target_depth' if deeper -- see
    load_mean_celerity_profile()'s own docstring for the full
    rationale (per-profile, since different profiles -- e.g. different
    synthetic samples -- can have different shapes/slopes near the
    boundary, even though they share the same 'z').

    Args:
        z (np.ndarray): shape (n_depth,), ascending, NaN-free, starting
            at 0 (see _drop_depths_with_any_nan()/
            _ensure_profile_starts_at_surface()).
        c_p (np.ndarray): shape (n_depth,) or (n_profiles, n_depth).
        target_depth (float)

    Returns:
        tuple(np.ndarray, np.ndarray): z, c_p adapted the same way,
        same number of dimensions as given.
    """
    if target_depth > z[-1]:
        slope = (c_p[..., -1] - c_p[..., -2]) / (z[-1] - z[-2])
        c_p_target = c_p[..., -1] + slope * (target_depth - z[-1])
        z = np.append(z, target_depth)
        c_p = np.concatenate([c_p, c_p_target[..., np.newaxis]], axis=-1)
    elif target_depth < z[-1]:
        keep = z < target_depth
        if c_p.ndim == 1:
            c_p_target = np.interp(target_depth, z, c_p)
        else:
            c_p_target = np.array([np.interp(target_depth, z, row) for row in c_p])
        z = np.append(z[keep], target_depth)
        c_p = np.concatenate([c_p[..., keep], c_p_target[..., np.newaxis]], axis=-1)
    # else: target_depth == z[-1] already, nothing to adjust.
    return z, c_p


def load_mean_celerity_profile(ssp_filename, target_depth, ssp_data_dir=SSP_DATA_DIR):
    """Load a saved SSP dataset (see illustration_rtf/ssp/
    load_ssp_data.py and ssp_process_eof.py), average it over time to
    get a single representative profile, and adapt it to
    'target_depth': linearly extended (using the slope of the last 2
    valid points) if the profile is shallower, or truncated
    (interpolating an exact value right at 'target_depth', rather than
    just dropping to whichever real data point happens to sit just
    above it) if deeper.

    Args:
        ssp_filename (str): the '.nc' file's name, e.g.
            "ssp_profiles_sw.nc" (see load_ssp_data.py's own output
            filenames -- CELERITY_ENV_TYPES names the ones this module
            actually uses).
        target_depth (float): the waveguide depth this profile must
            reach exactly (m) -- e.g. CELERITY_ENV_TYPES[env_type]["depth"].
        ssp_data_dir (str): directory 'ssp_filename' lives in.

    Returns:
        tuple(np.ndarray, np.ndarray): z (m, ascending, NaN-free,
        z[0] == 0 and z[-1] == target_depth exactly), c_p (m/s,
        same length as z) -- ready to pass as build_kraken()'s own
        'z_ssp'/'c_p_ssp'.

    Raises:
        ValueError: if fewer than 2 valid (non-NaN, after averaging)
            depths remain to build a profile from.
    """
    fpath = os.path.join(ssp_data_dir, ssp_filename)
    with xr.open_dataset(fpath) as ds:
        c_p = _get_ssp_variable(ds, fpath).mean(dim="time").values
        z = ds.depth.values

    # Drop NaN depths (below this profile's own real-data coverage --
    # see get_ssp_eof()'s own NOTE on why NaN can appear here) and sort
    # by depth ascending (real datasets normally already are, but this
    # makes no assumption about it).
    z, c_p = _drop_depths_with_any_nan(z, c_p)

    if z.size < 2:
        raise ValueError(
            f"load_mean_celerity_profile: '{ssp_filename}' has fewer than 2 "
            f"valid (non-NaN) depths after averaging over time -- cannot "
            f"build a profile from it."
        )

    # KRAKEN expects the water column's SSP to start right at the sea
    # surface (z=0); real CMEMS-style depth grids typically start at
    # (or extremely close to) 0 already, but if the shallowest
    # remaining point is not EXACTLY 0, extend flat down from the
    # surface using that shallowest point's own value (a negligible
    # approximation for the small gap this would ever realistically be).
    z, c_p = _ensure_profile_starts_at_surface(z, c_p)

    # NOTE: linear extension uses the LAST 2 points' own slope (per
    # user request: "prolongées linéairement jusqu'à la profondeur
    # nominale") -- e.g. the shallow-water profile, which the real
    # CMEMS extraction only reaches to ~50 m for, extended to the
    # classic study's own 100 m baseline depth. Truncation (per user
    # request: "sera donc limité à z < D") interpolates an exact value
    # AT the target depth -- e.g. the deep-water profile, which reaches
    # to ~2500 m, cut down to the 2000 m waveguide depth actually used.
    z, c_p = _adapt_profile_to_depth(z, c_p, target_depth)

    return z, c_p


def load_synthetic_celerity_profiles(
    ssp_filename, target_depth, ssp_data_dir=SSP_DATA_DIR, n_profiles=None
):
    """Load synthetic profiles from a '.nc' file (see
    illustration_rtf/ssp/ssp_process_eof.py's own
    process_ssp_profiles(), which produces these via PCA/EOF sampling),
    and adapt EACH ONE to 'target_depth' the same way
    load_mean_celerity_profile() adapts the single mean profile (see
    its own docstring for the full rationale) -- applied independently
    to each profile here, since different synthetic samples can have
    different shapes/slopes near the boundary even though they all
    share the same underlying depth grid.

    Args:
        ssp_filename (str): e.g. "synthetic_ssp_profiles_sw_1000.nc"
            (see _synthetic_ssp_filename(), which builds this name for
            a given (env_type, situation) pair).
        target_depth (float): see load_mean_celerity_profile().
        ssp_data_dir (str): directory 'ssp_filename' lives in.
        n_profiles (int|None): if given, only return the FIRST
            'n_profiles' profiles in the file (out of however many
            were generated -- see CELERITY_N_SYNTHETIC_SAMPLES) rather
            than every one of them -- reduces the computation time/
            memory of the KRAKEN sweep this feeds into (see
            build_celerity_sensitivity_dataset()), at the cost of a
            less complete sample of the underlying distribution.
            Taking the FIRST N (rather than, say, a random subset) is
            equivalent in practice: the synthetic profiles are
            themselves i.i.d. draws from a fitted distribution (see
            ssp_process_eof.py's own generate_new_ssp_profiles()),
            with no meaningful order to their own index, so any prefix
            of them is just as representative as a random subset would
            be -- while staying deterministic/reproducible. Applied
            AFTER the NaN-filtering/depth-adaptation steps below (not
            before), so the returned depth grid 'z' does not depend on
            'n_profiles' -- it always reflects every profile actually
            IN the file, keeping it consistent across different
            'n_profiles' choices (and matching the baseline profile's
            own grid the same way regardless). None (the default): use
            every profile in the file.

    Returns:
        tuple(np.ndarray, np.ndarray): z (m, shape (n_depth,), shared
        by every profile -- ascending, NaN-free, z[0] == 0, z[-1] ==
        target_depth exactly), c_p (m/s, shape (n_profiles_used,
        n_depth) -- n_profiles_used is min(n_profiles, however many
        are actually in the file) when 'n_profiles' is given).

    Raises:
        ValueError: if fewer than 2 valid (non-NaN for every profile)
            depths remain to build the profiles from.
    """
    fpath = os.path.join(ssp_data_dir, ssp_filename)
    with xr.open_dataset(fpath) as ds:
        c_p_all = _get_ssp_variable(ds, fpath).values  # (n_profiles, n_depth)
        z = ds.depth.values

    z, c_p_all = _drop_depths_with_any_nan(z, c_p_all)

    if z.size < 2:
        raise ValueError(
            f"load_synthetic_celerity_profiles: '{ssp_filename}' has fewer "
            f"than 2 depths that are valid (non-NaN) for EVERY profile -- "
            f"cannot build profiles from it."
        )

    z, c_p_all = _ensure_profile_starts_at_surface(z, c_p_all)
    z, c_p_all = _adapt_profile_to_depth(z, c_p_all, target_depth)

    if n_profiles is not None:
        if n_profiles > c_p_all.shape[0]:
            print(
                f"Note: '{ssp_filename}' only has {c_p_all.shape[0]} profiles "
                f"-- fewer than the requested n_profiles={n_profiles}; using "
                f"all of them."
            )
        c_p_all = c_p_all[:n_profiles]

    return z, c_p_all


def _celerity_profile_rmse(z_baseline, c_p_baseline, z_test, c_p_test):
    """RMSE (m/s) between the baseline (mean) celerity profile and one
    or several test profiles.

    NOTE: the baseline profile is interpolated onto the TEST profile's
    OWN depth grid before comparing, rather than assuming both share
    the exact same 'z' -- load_mean_celerity_profile() (used for the
    baseline) and load_synthetic_celerity_profiles() (used for the
    swept profiles) apply slightly DIFFERENT NaN-filtering criteria to
    the same underlying real dataset (the former drops a depth only if
    its OWN mean over time is NaN; the latter -- matching
    get_ssp_eof()'s own stricter check -- drops a depth if it is NaN
    at ANY single time), so the two CAN retain a different depth set
    in practice even for the same environment type. Interpolating
    avoids relying on them lining up exactly.

    Args:
        z_baseline (np.ndarray): shape (n_depth_baseline,).
        c_p_baseline (np.ndarray): shape (n_depth_baseline,).
        z_test (np.ndarray): shape (n_depth_test,).
        c_p_test (np.ndarray): shape (n_depth_test,) for a single
            profile, or (n_profiles, n_depth_test) for a batch.

    Returns:
        float|np.ndarray: RMSE (m/s) -- scalar for a single profile,
        shape (n_profiles,) for a batch.
    """
    c_p_baseline_interp = np.interp(z_test, z_baseline, c_p_baseline)
    return np.sqrt(np.mean((c_p_test - c_p_baseline_interp) ** 2, axis=-1))


def _celerity_profile_std(z_baseline, c_p_baseline, z_test, c_p_test):
    """std (m/s) between the baseline (mean) celerity profile and one
    or several test profiles.


    Args:
        z_baseline (np.ndarray): shape (n_depth_baseline,).
        c_p_baseline (np.ndarray): shape (n_depth_baseline,).
        z_test (np.ndarray): shape (n_depth_test,).
        c_p_test (np.ndarray): shape (n_depth_test,) for a single
            profile, or (n_profiles, n_depth_test) for a batch.

    Returns:
        float|np.ndarray: std (m/s) -- scalar for a single profile,
        shape (n_profiles,) for a batch.
    """
    c_p_baseline_interp = np.interp(z_test, z_baseline, c_p_baseline)
    c_p_bias = np.mean(c_p_test - c_p_baseline_interp)
    c_p_test_unbiased = c_p_test - c_p_bias
    unbiased_diff = c_p_test_unbiased - c_p_baseline_interp
    c_p_std = np.std(unbiased_diff, axis=-1)
    return c_p_std


### F1 score ###
def get_min_max_idx(arr: np.ndarray, axs: int = 1, pad: bool = True) -> np.ndarray:
    """Find local minima and maxima in array."""
    grad = np.diff(arr, axis=axs)
    grad_sign = np.sign(grad)
    min_max = np.abs(np.sign(np.diff(grad_sign, axis=axs)))
    if pad:
        pad_shape = list(min_max.shape)
        pad_shape[axs] = 1
        min_max = np.concatenate(
            [np.zeros(pad_shape), min_max, np.zeros(pad_shape)], axis=axs
        )
    return min_max


def get_f1_score(
    min_max_idx_truth: np.ndarray,
    min_max_idx_ae: np.ndarray,
    axs: int = 1,
    kernel_size: int = 10,
) -> np.ndarray:
    """Compute F1 score for extremum detection."""
    from scipy.ndimage import convolve

    kernel_shape = [1] * min_max_idx_truth.ndim
    kernel_shape[axs] = kernel_size
    kernel = np.ones(kernel_shape)
    truth_expanded = convolve(min_max_idx_truth, kernel, mode="constant", cval=0.0)
    ae_expanded = convolve(min_max_idx_ae, kernel, mode="constant", cval=0.0)

    true_positives = (truth_expanded > 0) & (min_max_idx_ae > 0)
    num_true_positives = np.sum(true_positives, axis=axs)
    false_positives = (truth_expanded == 0) & (min_max_idx_ae > 0)
    num_false_positives = np.sum(false_positives, axis=axs)
    false_negatives = (min_max_idx_truth > 0) & (ae_expanded == 0)
    num_false_negatives = np.sum(false_negatives, axis=axs)

    precision_den = num_true_positives + num_false_positives
    recall_den = num_true_positives + num_false_negatives
    precision_score = np.where(
        precision_den == 0, 0, num_true_positives / precision_den
    )
    recall_score = np.where(recall_den == 0, 0, num_true_positives / recall_den)
    sum_scores = precision_score + recall_score
    f1_score = np.where(
        sum_scores == 0, 0, 2 * (precision_score * recall_score) / sum_scores
    )

    return f1_score


def _celerity_profile_f1_score(
    z_baseline, c_p_baseline, z_test, c_p_test, kernel_size=1
):
    min_max_idx_truth = get_min_max_idx(c_p_baseline[np.newaxis, :], axs=1, pad=False)
    min_max_idx_ae = get_min_max_idx(c_p_test, axs=1, pad=False)
    f1_score = get_f1_score(
        min_max_idx_truth, min_max_idx_ae, axs=1, kernel_size=kernel_size
    )
    return f1_score


def _save_celerity_rmse_results(situation, profile_idx, rmse, result_dir):
    """Save the per-profile RMSE-from-baseline-profile results (see
    _celerity_profile_rmse()) for ONE situation to a small, dedicated
    CSV file, parallel to save_sensitivity_distance_results()'s own
    "dist_<situation>.csv" (joined later on 'profile_idx' -- see
    plot_celerity_distance_vs_rmse()).

    Args:
        situation (str): see CELERITY_SITUATIONS.
        profile_idx, rmse (array-like): equal-length 1D arrays.
        result_dir (str): directory to write into, as
            '<result_dir>/rmse_<situation>.csv'.

    Returns:
        str: path to the written file.
    """
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f"rmse_{situation}.csv")
    data = np.column_stack([profile_idx, rmse])
    np.savetxt(path, data, delimiter=",", header="profile,rmse", comments="")
    return path


def _save_celerity_f1_score_results(situation, profile_idx, f1_score, result_dir):
    """
    Args:
        situation (str): see CELERITY_SITUATIONS.
        profile_idx, rmse (array-like): equal-length 1D arrays.
        result_dir (str): directory to write into, as
            '<result_dir>/rmse_<situation>.csv'.

    Returns:
        str: path to the written file.
    """
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f"f1_score_{situation}.csv")
    data = np.column_stack([profile_idx, f1_score])
    np.savetxt(path, data, delimiter=",", header="profile,f1_score", comments="")
    return path


def _save_celerity_std_results(situation, profile_idx, std, result_dir):
    """Save the per-profile STD-from-baseline-profile results (see
    _celerity_profile_std()) for ONE situation to a small, dedicated
    CSV file, parallel to save_sensitivity_distance_results()'s own
    "dist_<situation>.csv" (joined later on 'profile_idx' -- see
    plot_celerity_distance_vs_std()).

    Args:
        situation (str): see CELERITY_SITUATIONS.
        profile_idx, std (array-like): equal-length 1D arrays.
        result_dir (str): directory to write into, as
            '<result_dir>/std_<situation>.csv'.

    Returns:
        str: path to the written file.
    """
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f"std_{situation}.csv")
    data = np.column_stack([profile_idx, std])
    np.savetxt(path, data, delimiter=",", header="profile,std", comments="")
    return path


def _load_celerity_rmse_results(situation, result_dir):
    """Reload a results file written by _save_celerity_rmse_results().

    Returns:
        tuple(np.ndarray, np.ndarray): profile_idx, rmse.
    """
    path = os.path.join(result_dir, f"rmse_{situation}.csv")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)
    return data[:, 0], data[:, 1]


def _load_celerity_f1_score_results(situation, result_dir):
    """Reload a results file written by _save_celerity_f1_score_results().

    Returns:
        tuple(np.ndarray, np.ndarray): profile_idx, rmse.
    """
    path = os.path.join(result_dir, f"f1_score_{situation}.csv")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)
    return data[:, 0], data[:, 1]


def _load_celerity_std_results(situation, result_dir):
    """Reload a results file written by _save_celerity_std_results().

    Returns:
        tuple(np.ndarray, np.ndarray): profile_idx, rmse.
    """
    path = os.path.join(result_dir, f"std_{situation}.csv")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)
    return data[:, 0], data[:, 1]


def _ssp_filename(env_type, situation):
    """Build the filename of the REAL (non-synthetic) SSP '.nc' file
    illustration_rtf/ssp/load_ssp_data.py produces for one
    (env_type, situation) combination -- the file
    load_mean_celerity_profile() averages over time to build a
    baseline profile from (see build_celerity_baseline()).

    Args:
        env_type (str): "sw" or "dw" -- see CELERITY_ENV_TYPES.
        situation (str): "all" (the whole, multi-decade dataset) or one
            of "winter"/"spring"/"summer"/"automn" (that season only)
            -- see CELERITY_SITUATIONS.

    Returns:
        str

    Raises:
        KeyError: if 'env_type' is not one of CELERITY_ENV_TYPES.
    """
    base = CELERITY_ENV_TYPES[env_type]["ssp_filename"]
    if base.endswith(".nc"):
        base = base[: -len(".nc")]
    suffix = "" if situation == "all" else f"_{situation}"
    return f"{base}{suffix}.nc"


def build_celerity_baseline(
    env_type, situation="all", result_dir=None, img_dir=None, ssp_data_dir=SSP_DATA_DIR
):
    """Build the celerity-profile-sensitivity study's baseline for ONE
    (environment type, situation) combination: the Green's function /
    gamma(f, r) for a waveguide using the MEAN (over time) REAL sound-
    speed profile for that combination (see
    load_mean_celerity_profile()), adapted to the waveguide's own
    depth -- everything else (bottom halfspace, source/receiver
    geometry, frequencies...) stays at the classic sensitivity study's
    own baseline values (see load_all_arg_dict()/build_baseline()).

    NOTE (bug fixed, per user request): every situation used to be
    compared against the SAME "all" baseline (built from the whole,
    multi-decade dataset's own mean profile), even the seasonal ones
    -- confirmed not what was intended: a "winter" sweep's synthetic
    profiles are themselves EOF-sampled from a PCA fit on winter data
    only (see ssp_process_eof.py's own process_ssp_profiles()), so
    comparing them against the whole-dataset mean profile conflates
    "how much does this winter profile deviate from a typical winter
    profile" with "how much does winter typically differ from the
    year-round average" -- two different questions. Each situation now
    gets its OWN baseline, built from THAT SAME situation's own mean
    profile (e.g. "winter"'s baseline uses ssp_profiles_sw_winter.nc's
    own mean, not ssp_profiles_sw.nc's) -- see 'situation' below, and
    process_celerity_sensitivity()'s own matching update.

    NOTE: 'z_rcv'/'z_s' are NOT adjusted per environment type here --
    they stay at the classic baseline's own values (z_s=5 m,
    z_rcv=99.5 m), which remain valid depths for BOTH "sw" (100 m) and
    "dw" (2000 m), but are not necessarily where you'd want them
    physically for "dw" specifically (99.5 m is shallow relative to a
    2000 m column, not "near the bottom" the way it is for "sw"'s own
    100 m). Left as-is since this wasn't asked for; override via
    load_all_arg_dict()/build_baseline()'s own 'env_overrides' if you
    want different source/receiver depths for "dw".

    Args:
        env_type (str): "sw" or "dw" -- see CELERITY_ENV_TYPES.
        situation (str): "all" (the default -- baseline built from the
            WHOLE dataset's own mean profile) or one of "winter"/
            "spring"/"summer"/"automn" -- see CELERITY_SITUATIONS. Each
            situation's own synthetic sweep (see
            build_celerity_sensitivity_dataset()) should be compared
            against THIS SAME situation's baseline, not "all"'s.
        result_dir (str|None): directory 'gf_dataset_baseline.nc' is
            written into. None (the default):
            '<RESILIENCE_RESULT_DIR>/<env_type>/celerity/<situation>/'
            (see _resilience_study_dirs()) -- the SAME folder
            build_celerity_sensitivity_dataset() writes that
            situation's own swept profiles into.
        img_dir (str|None): directory the 2 diagnostic figures are
            saved into. None (the default):
            '<RESILIENCE_IMG_DIR>/<env_type>/celerity/<situation>/'.
        ssp_data_dir (str): forwarded to load_mean_celerity_profile().

    Returns:
        str: path to the written '.nc' file.

    Raises:
        KeyError: if 'env_type' is not one of CELERITY_ENV_TYPES.
    """
    env_config = CELERITY_ENV_TYPES[env_type]
    base_result_dir, base_img_dir = _resilience_study_dirs(env_type, "celerity")

    if result_dir is None:
        result_dir = os.path.join(base_result_dir, situation)
    if img_dir is None:
        img_dir = os.path.join(base_img_dir, situation)

    z_ssp, c_p_ssp = load_mean_celerity_profile(
        _ssp_filename(env_type, situation),
        target_depth=env_config["depth"],
        ssp_data_dir=ssp_data_dir,
    )

    return build_baseline(
        result_dir=result_dir,
        img_dir=img_dir,
        env_overrides={
            "depth": env_config["depth"],
            "z_ssp": z_ssp,
            "c_p_ssp": c_p_ssp,
        },
    )


def build_celerity_baselines(
    env_types=None, situations=None, ssp_data_dir=SSP_DATA_DIR
):
    """Call build_celerity_baseline() for every (environment type,
    situation) combination (see CELERITY_ENV_TYPES/CELERITY_SITUATIONS)
    -- each situation needs its OWN, situation-specific baseline (see
    build_celerity_baseline()'s own NOTE for why).

    Args:
        env_types (list[str]|None): which environment types to build.
            None (the default): every key of CELERITY_ENV_TYPES
            ("sw" and "dw").
        situations (list[str]|None): which situations to build. None
            (the default): every entry of CELERITY_SITUATIONS.
        ssp_data_dir (str): forwarded to build_celerity_baseline().

    Returns:
        dict[tuple(str, str), str]: (env_type, situation) -> path to
        its written '.nc' file.
    """
    if env_types is None:
        env_types = list(CELERITY_ENV_TYPES.keys())
    if situations is None:
        situations = list(CELERITY_SITUATIONS)

    return {
        (env_type, situation): build_celerity_baseline(
            env_type, situation, ssp_data_dir=ssp_data_dir
        )
        for env_type in env_types
        for situation in situations
    }


def _synthetic_ssp_filename(
    env_type, situation, n_samples=CELERITY_N_SYNTHETIC_SAMPLES
):
    """Build the filename of the synthetic-profile '.nc' file
    illustration_rtf/ssp/ssp_process_eof.py's own
    process_ssp_profiles() produces for one (env_type, situation)
    combination -- see its own
    f"synthetic_{filename_ssp}_{n_new_samples}.nc" naming.

    Args:
        env_type (str): "sw" or "dw" -- see CELERITY_ENV_TYPES.
        situation (str): "all" (EOFs fit on the whole, multi-decade
            dataset) or one of "winter"/"spring"/"summer"/"automn"
            (EOFs fit on that season only) -- see CELERITY_SITUATIONS.
        n_samples (int): how many synthetic profiles were generated
            (ssp_process_eof.py's own 'n_new_samples').

    Returns:
        str

    Raises:
        KeyError: if 'env_type' is not one of CELERITY_ENV_TYPES.
    """
    base = CELERITY_ENV_TYPES[env_type]["ssp_filename"]
    if base.endswith(".nc"):
        base = base[: -len(".nc")]
    suffix = "" if situation == "all" else f"_{situation}"
    return f"synthetic_{base}{suffix}_{n_samples}.nc"


def build_celerity_sensitivity_dataset(
    env_type,
    situation,
    result_dir=None,
    ssp_data_dir=SSP_DATA_DIR,
    n_samples=CELERITY_N_SYNTHETIC_SAMPLES,
    n_profiles=None,
):
    """Sweep over synthetic celerity profiles generated for ONE
    (env_type, situation) combination (see illustration_rtf/ssp/
    ssp_process_eof.py, which produces them via PCA/EOF sampling),
    running KRAKEN once per profile and saving its Green's function.

    This is the celerity-sensitivity study's own equivalent of
    build_sensitivity_dataset() (which instead sweeps a single SCALAR
    parameter over a range of values, e.g. "depth" from 90 to 110 m):
    here, each swept "value" is an entire depth-varying profile, not a
    scalar -- see build_kraken()'s own 'z_ssp'/'c_p_ssp'. Follows the
    exact same memory-conscious pattern (see
    build_sensitivity_dataset()'s own NOTE): one small file written PER
    PROFILE, immediately after it's computed, rather than accumulating
    every profile's Green's function in memory before writing a single
    combined file.

    File layout: '<result_dir>/<situation>/<situation>_<i>.nc', one
    file per profile (0-based index 'i'), holding that one profile's
    Green's function magnitude ('gf', dims (profile, f, r)) AND the
    profile itself ('c_p_ssp', dims (profile, z)) for traceability --
    the shared depth grid 'z' is saved as a coordinate once per file
    (small; every profile in one (env_type, situation) sweep shares
    the exact same 'z', see load_synthetic_celerity_profiles()).

    Args:
        env_type (str): "sw" or "dw" -- see CELERITY_ENV_TYPES.
        situation (str): "all"/"winter"/"spring"/"summer"/"automn" --
            see CELERITY_SITUATIONS/_synthetic_ssp_filename().
        result_dir (str|None): parent directory for the per-profile
            files (written under '<result_dir>/<situation>/'). None
            (the default): '<RESILIENCE_RESULT_DIR>/<env_type>/celerity/'
            (matching build_celerity_baseline()'s own default).
        ssp_data_dir (str): directory the synthetic '.nc' file lives in.
        n_samples (int): forwarded to _synthetic_ssp_filename().
        n_profiles (int|None): forwarded to
            load_synthetic_celerity_profiles() -- if given, only run
            KRAKEN for (and save) the FIRST 'n_profiles' profiles
            rather than every one available, to reduce this sweep's
            own computation time/memory (see its own docstring for why
            taking the first N is representative regardless). None
            (the default): use every available profile.

    Returns:
        str: the directory the per-profile files were written into.

    Raises:
        KeyError: if 'env_type' is not one of CELERITY_ENV_TYPES.
    """
    env_config = CELERITY_ENV_TYPES[env_type]
    depth = env_config["depth"]

    if result_dir is None:
        result_dir, _ = _resilience_study_dirs(env_type, "celerity")
    out_dir = os.path.join(result_dir, situation)
    os.makedirs(out_dir, exist_ok=True)

    # NOTE (per user request): clear stale per-profile files from a
    # PREVIOUS sweep before writing new ones -- see _clear_dir()'s own
    # docstring for why this matters. Deliberately NOT a blanket
    # _clear_dir(out_dir) here, unlike build_sensitivity_dataset():
    # this SAME folder also holds this (env_type, situation)'s own
    # 'gf_dataset_baseline.nc' (see build_celerity_baseline()), which
    # must survive -- only this sweep's own '<situation>_*.nc' files
    # are removed.
    for stale_fpath in glob.glob(os.path.join(out_dir, f"{situation}_*.nc")):
        os.remove(stale_fpath)

    ssp_filename = _synthetic_ssp_filename(env_type, situation, n_samples=n_samples)
    z_ssp, c_p_ssp_all = load_synthetic_celerity_profiles(
        ssp_filename,
        target_depth=depth,
        ssp_data_dir=ssp_data_dir,
        n_profiles=n_profiles,
    )
    n_profiles = c_p_ssp_all.shape[0]

    all_arg_dict = load_all_arg_dict(
        drop_keys=("fs", "fmax", "r0", "d12"), d12_max=5000
    )
    all_arg_dict["depth"] = depth

    prev_progress = 0
    for i_profile in range(n_profiles):
        prev_progress = progression_bar(
            index=i_profile + 1,
            index0=0,
            indexf=n_profiles,
            prev_progress=prev_progress,
        )

        args = dict(all_arg_dict)
        args["z_ssp"] = z_ssp
        args["c_p_ssp"] = c_p_ssp_all[i_profile]

        call_kwargs = _extract_kwargs(build_dataset_current_config_kraken, args)
        kraken_freq, kraken_r, g_fr = build_dataset_current_config_kraken(**call_kwargs)

        g_fr_full = _pad_to_full_frequency_grid(g_fr, kraken_freq, all_arg_dict["freq"])

        # Write THIS profile's dataset immediately, then let g_fr/
        # g_fr_full go out of scope (freed before the next profile's
        # KRAKEN run) -- see build_sensitivity_dataset()'s own NOTE.
        ds_value = xr.Dataset(
            data_vars=dict(
                gf=(
                    ["profile", "f", "r"],
                    _to_float32(np.abs(g_fr_full))[np.newaxis, ...],
                ),
                c_p_ssp=(
                    ["profile", "z"],
                    _to_float32(c_p_ssp_all[i_profile])[np.newaxis, :],
                ),
            ),
            coords={
                "profile": [i_profile],
                "f": _to_float32(all_arg_dict["freq"]),
                "r": _to_float32(kraken_r),
                "z": _to_float32(z_ssp),
            },
        )
        fpath = os.path.join(out_dir, f"{situation}_{i_profile:04d}.nc")
        ds_value.to_netcdf(fpath)

    return out_dir


def build_celerity_tests(
    env_types=None,
    situations=None,
    n_samples=CELERITY_N_SYNTHETIC_SAMPLES,
    n_profiles=None,
):
    """Run build_celerity_sensitivity_dataset() for every
    (env_type, situation) combination requested -- the celerity-
    sensitivity study's own equivalent of build_tests() (which instead
    sweeps the classic study's 5 scalar parameters one at a time).

    Args:
        env_types (list[str]|None): which environment types to build.
            None (the default): every key of CELERITY_ENV_TYPES ("sw"
            and "dw").
        situations (list[str]|None): which situations to build. None
            (the default): every entry of CELERITY_SITUATIONS (5 of
            them) -- e.g. pass ["all"] to build just the whole-dataset-
            EOF situation first, as a quick, single-combination check
            before committing to the full sweep.
        n_samples (int): forwarded to build_celerity_sensitivity_dataset().
        n_profiles (int|None): forwarded to
            build_celerity_sensitivity_dataset() -- caps how many
            profiles are actually run through KRAKEN for EACH
            (env_type, situation) combination. None (the default): use
            every available profile.

    Returns:
        dict[tuple(str, str), str]: (env_type, situation) -> the
        directory its per-profile files were written into.
    """
    if env_types is None:
        env_types = list(CELERITY_ENV_TYPES.keys())
    if situations is None:
        situations = list(CELERITY_SITUATIONS)

    results = {}
    for env_type in env_types:
        for situation in situations:
            print(f"Processing env_type={env_type}, situation={situation}...")
            results[(env_type, situation)] = build_celerity_sensitivity_dataset(
                env_type,
                situation,
                n_samples=n_samples,
                n_profiles=n_profiles,
            )
    return results


def process_celerity_sensitivity(
    env_type, situations=None, result_dir=None, save_dir=None
):
    """Read back ONE environment type's per-profile result files (see
    build_celerity_sensitivity_dataset()), compute each profile's RTF
    distance from that (environment type, situation) combination's OWN
    baseline (see build_celerity_baseline()) EVALUATED AT r0 (see
    dist_from_baseline()), save the resulting L1/L2/theta distance-vs-
    profile-index arrays to a dedicated file per situation (see
    save_sensitivity_distance_results()), then plot them (see
    plot_sensitivity_curves()).

    NOTE (bug fixed, per user request): every situation used to be
    compared against the SAME "all" baseline -- see
    build_celerity_baseline()'s own NOTE for why that's wrong for the
    seasonal situations. Each situation's own baseline (built by
    build_celerity_baseline(env_type, situation)) is now opened
    separately, per situation.

    ALSO computes and saves each profile's RMSE (m/s) from ITS OWN
    situation's baseline (mean) celerity profile (see
    _celerity_profile_rmse()), to
    '<result_dir>/<situation>/rmse_<situation>.csv' (see
    _save_celerity_rmse_results()) -- the profile index alone carries
    no information about how different a profile actually IS from the
    reference one; this RMSE is what
    plot_celerity_distance_vs_rmse() plots distance against instead
    (the study's main diagnostic, per user request). Computed here
    (rather than in a separate pass) because 'c_p_ssp'/'z' -- unlike
    'gf' -- have no "r" dimension, so they come out of the SAME
    already-open per-situation dataset unaffected by
    '_select_r0_pair''s r-only reduction, at no extra file-read cost.

    This is process_sensitivity()'s own analysis
    (dist_from_baseline()/_select_r0_pair()'s memory-conscious
    preprocessing -- see their own docstrings), pointed at the
    celerity-sensitivity study's directories/file layout instead of
    the classic study's: "situation" (see CELERITY_SITUATIONS) plays
    the role 'test_arg_name' plays there, except each swept "value" is
    an entire profile rather than a scalar (see
    build_celerity_sensitivity_dataset()'s own docstring) -- so the
    saved/plotted "value" for each profile is its 0-based INDEX within
    that situation's sweep, not a physical quantity (the actual
    profile shape each index corresponds to is saved alongside 'gf' in
    its own per-profile file, as 'c_p_ssp', for traceability).

    Args:
        env_type (str): "sw" or "dw" -- see CELERITY_ENV_TYPES. Each
            environment type has its OWN baselines (a different
            waveguide depth and real profile entirely -- see
            build_celerity_baseline()), so this only ever processes
            ONE at a time (unlike 'situations' below); see
            process_celerity_tests() to process every environment type.
        situations (list[str]|None): which situations to process (each
            must have a '<result_dir>/<situation>/' folder from
            build_celerity_sensitivity_dataset(), holding both that
            situation's own baseline and its swept profiles). None
            (the default): every such folder under 'result_dir'
            automatically.
        result_dir (str|None): where to read the per-profile/baseline
            files from, and where to save the distance results. None
            (the default): '<RESILIENCE_RESULT_DIR>/<env_type>/celerity/'
            (matching build_celerity_baseline()'s own default parent).
        save_dir (str|None): forwarded to plot_sensitivity_curves() --
            if given, also save the returned figure there (e.g.
            '<RESILIENCE_IMG_DIR>/<env_type>/celerity/'). None (the
            default, matching process_sensitivity()'s own convention):
            the figure is only returned, not saved.

    Returns:
        matplotlib.figure.Figure (see plot_sensitivity_curves()).

    Raises:
        KeyError: if 'env_type' is not one of CELERITY_ENV_TYPES.
    """
    # NOTE: env_type must be a valid key -- fail fast with a clear
    # KeyError (matching build_celerity_baseline()'s own behaviour)
    # rather than a confusing FileNotFoundError further down.
    CELERITY_ENV_TYPES[env_type]
    env_config = CELERITY_ENV_TYPES[env_type]

    if result_dir is None:
        result_dir, _ = _resilience_study_dirs(env_type, "celerity")

    import functools

    baseline_src_rcv_param = baseline_src_rcv()
    d12 = baseline_src_rcv_param["d12"]
    r0 = baseline_src_rcv_param["r0"]
    preprocess = functools.partial(_select_r0_pair, r0=r0, d12=d12)

    if situations is None:
        situations = sorted(
            name
            for name in os.listdir(result_dir)
            if os.path.isdir(os.path.join(result_dir, name))
        )

    for situation in situations:
        situation_dir = os.path.join(result_dir, situation)
        value_files = sorted(
            glob.glob(os.path.join(situation_dir, f"{situation}_*.nc"))
        )
        if not value_files:
            continue

        # NOTE: THIS situation's own baseline (see
        # build_celerity_baseline()'s own NOTE for why -- "winter" is
        # compared against winter's own mean profile, not "all"'s).
        fpath_baseline = os.path.join(situation_dir, "gf_dataset_baseline.nc")
        z_baseline, c_p_baseline = load_mean_celerity_profile(
            _ssp_filename(env_type, situation), target_depth=env_config["depth"]
        )

        with xr.open_dataset(fpath_baseline) as ds_baseline:
            # NOTE: concat_dim="profile" -- matches
            # build_celerity_sensitivity_dataset()'s own saved
            # dimension name (every swept "value" here is a profile,
            # not a scalar named after 'situation' the way
            # build_sensitivity_dataset()'s own 'test_arg_name' is).
            with xr.open_mfdataset(
                value_files,
                combine="nested",
                concat_dim="profile",
                join="override",
                preprocess=preprocess,
            ) as ds_test:
                dist_L1, dist_L2, dist_theta, dist_wasserstein = dist_from_baseline(
                    ds_baseline, ds_test, d12, r0
                )
                profile_idx = ds_test["profile"].values

                # NOTE: 'c_p_ssp'/'z' have no "r" dimension, so
                # 'preprocess' (_select_r0_pair(), which only ever
                # touches "r") leaves them fully intact here -- no
                # extra file read needed to also get the RMSE-from-
                # baseline-profile diagnostic (see
                # plot_celerity_distance_vs_rmse(), the main reason
                # this exists at all per user request).
                # RMSE
                rmse = _celerity_profile_rmse(
                    z_baseline,
                    c_p_baseline,
                    ds_test["z"].values,
                    ds_test["c_p_ssp"].values,
                )
                # F1 score
                f1_score = _celerity_profile_f1_score(
                    z_baseline,
                    c_p_baseline,
                    ds_test["z"].values,
                    ds_test["c_p_ssp"].values,
                    kernel_size=5,
                )
                # Std
                std = _celerity_profile_std(
                    z_baseline,
                    c_p_baseline,
                    ds_test["z"].values,
                    ds_test["c_p_ssp"].values,
                )

        save_sensitivity_distance_results(
            situation,
            profile_idx,
            dist_L1,
            dist_L2,
            dist_theta,
            dist_wasserstein,
            result_dir=result_dir,
        )
        # RMSE
        _save_celerity_rmse_results(situation, profile_idx, rmse, result_dir)
        # F1 score
        _save_celerity_f1_score_results(situation, profile_idx, f1_score, result_dir)
        # STD
        _save_celerity_std_results(situation, profile_idx, std, result_dir)

    return plot_sensitivity_curves(situations, result_dir=result_dir, save_dir=save_dir)


def process_celerity_tests(env_types=None, situations=None, save=True):
    """Run process_celerity_sensitivity() for every environment type
    requested -- the celerity-sensitivity study's own equivalent of
    process_sensitivity() applied across BOTH environment types at
    once (each with its own baseline -- see
    process_celerity_sensitivity()'s own docstring for why it only
    ever processes one at a time on its own).

    Args:
        env_types (list[str]|None): which environment types to
            process. None (the default): every key of
            CELERITY_ENV_TYPES ("sw" and "dw").
        situations (list[str]|None): forwarded to
            process_celerity_sensitivity() -- None discovers every
            situation with saved sweep data, for each environment type
            independently.
        save (bool): if True (the default), each environment type's
            figure is saved to its own
            '<RESILIENCE_IMG_DIR>/<env_type>/celerity/' folder (see
            process_celerity_sensitivity()'s own 'save_dir'). False:
            figures are only returned, matching
            process_celerity_sensitivity()'s own 'save_dir=None'.

    Returns:
        dict[str, matplotlib.figure.Figure]: env_type -> figure.
    """
    if env_types is None:
        env_types = list(CELERITY_ENV_TYPES.keys())

    return {
        env_type: process_celerity_sensitivity(
            env_type,
            situations=situations,
            save_dir=_resilience_study_dirs(env_type, "celerity")[1] if save else None,
        )
        for env_type in env_types
    }


def plot_celerity_distance_vs_rmse(
    env_type,
    situations=None,
    metric="theta",
    result_dir=None,
    save_dir=None,
):
    """Plot the RTF distance from baseline (see
    process_celerity_sensitivity()) AS A FUNCTION OF each profile's own
    RMSE deviation from the baseline (mean) celerity profile (see
    _celerity_profile_rmse()), one panel per situation -- this is the
    celerity-sensitivity study's KEY diagnostic, per user request: the
    profile INDEX plot_sensitivity_curves() would otherwise show on the
    x-axis carries no information about how DIFFERENT a profile
    actually is from the reference one, whereas the RMSE does.

    Args:
        env_type (str): "sw" or "dw" -- see CELERITY_ENV_TYPES.
        situations (list[str]|None): which situations to plot (each
            must have a '<result_dir>/rmse_<situation>.csv' file from
            process_celerity_sensitivity()). None (the default):
            every such file under 'result_dir'.
        metric (str): "L1", "L2", or "theta" -- which distance metric
            to plot on the y-axis.
        result_dir (str|None): where to read the saved distance/RMSE
            results from. None (the default):
            '<RESILIENCE_RESULT_DIR>/<env_type>/celerity/'.
        save_dir (str|None): if given, save the figure to
            '<save_dir>/<filename>.png' (see
            _sensitivity_figure_filename()). None (the default): the
            figure is only returned, not saved.

    Returns:
        matplotlib.figure.Figure

    Raises:
        KeyError: if 'env_type' is not one of CELERITY_ENV_TYPES.
    """
    CELERITY_ENV_TYPES[env_type]

    if result_dir is None:
        result_dir, _ = _resilience_study_dirs(env_type, "celerity")

    if situations is None:
        prefix, suffix = "rmse_", ".csv"
        situations = sorted(
            name[len(prefix) : -len(suffix)]
            for name in os.listdir(result_dir)
            if name.startswith(prefix) and name.endswith(suffix)
        )

    n_situations = len(situations)
    fig, axs = plt.subplots(
        1, max(n_situations, 1), figsize=(16, 8), squeeze=False, sharey=True
    )
    axs = axs[0]

    for i, situation in enumerate(situations):
        profile_idx_dist, dist_L1, dist_L2, dist_theta, dist_wasserstein = (
            load_sensitivity_distance_results(
                situation, result_dir=result_dir, file_prefix="dist_"
            )
        )
        profile_idx_rmse, rmse = _load_celerity_rmse_results(situation, result_dir)

        # NOTE: both files were written from the SAME 'ds_test' within
        # a single process_celerity_sensitivity() call, in the same
        # order -- this re-sort by profile index is a cheap safety net
        # against that assumption ever breaking (e.g. a future
        # refactor reading them from separate passes), not something
        # expected to actually reorder anything today.
        order_dist = np.argsort(profile_idx_dist)
        order_rmse = np.argsort(profile_idx_rmse)
        dist = {
            "L1": dist_L1,
            "L2": dist_L2,
            "theta": dist_theta,
            "wasserstein": dist_wasserstein,
        }[metric][order_dist]
        rmse = rmse[order_rmse]

        axs[i].scatter(rmse, dist, s=12)
        axs[i].set_xlabel(r"RMSE from baseline profile [m s$^{-1}$]")
        axs[i].set_title(situation)

    axs[0].set_ylabel(f"Distance ({METRIC_LABEL[metric]})")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = _sensitivity_figure_filename(
            "dist_vs_rmse", situations, file_prefix="dist_", metric=metric
        )
        fig.savefig(os.path.join(save_dir, fname))

        plt.close(fig)

    return fig


def plot_celerity_distance_vs_f1_score(
    env_type,
    situations=None,
    metric="theta",
    result_dir=None,
    save_dir=None,
):
    """Plot the RTF distance from baseline (see
    process_celerity_sensitivity()) AS A FUNCTION OF each profile's own
    F1 score deviation from the baseline (mean) celerity profile (see
    _celerity_profile_f1_score()), one panel per situation -- this is the
    celerity-sensitivity study's KEY diagnostic, per user request: the
    profile INDEX plot_sensitivity_curves() would otherwise show on the
    x-axis carries no information about how DIFFERENT a profile
    actually is from the reference one, whereas the F1 does.

    Args:
        env_type (str): "sw" or "dw" -- see CELERITY_ENV_TYPES.
        situations (list[str]|None): which situations to plot (each
            must have a '<result_dir>/f1_score_<situation>.csv' file from
            process_celerity_sensitivity()). None (the default):
            every such file under 'result_dir'.
        metric (str): "L1", "L2", or "theta" -- which distance metric
            to plot on the y-axis.
        result_dir (str|None): where to read the saved distance/F1 score
            results from. None (the default):
            '<RESILIENCE_RESULT_DIR>/<env_type>/celerity/'.
        save_dir (str|None): if given, save the figure to
            '<save_dir>/<filename>.png' (see
            _sensitivity_figure_filename()). None (the default): the
            figure is only returned, not saved.

    Returns:
        matplotlib.figure.Figure

    Raises:
        KeyError: if 'env_type' is not one of CELERITY_ENV_TYPES.
    """
    CELERITY_ENV_TYPES[env_type]

    if result_dir is None:
        result_dir, _ = _resilience_study_dirs(env_type, "celerity")

    if situations is None:
        prefix, suffix = "f1_score_", ".csv"
        situations = sorted(
            name[len(prefix) : -len(suffix)]
            for name in os.listdir(result_dir)
            if name.startswith(prefix) and name.endswith(suffix)
        )

    n_situations = len(situations)
    fig, axs = plt.subplots(
        1, max(n_situations, 1), figsize=(16, 8), squeeze=False, sharey=True
    )
    axs = axs[0]

    for i, situation in enumerate(situations):
        profile_idx_dist, dist_L1, dist_L2, dist_theta, dist_wasserstein = (
            load_sensitivity_distance_results(
                situation, result_dir=result_dir, file_prefix="dist_"
            )
        )
        profile_idx_f1_score, f1_score = _load_celerity_f1_score_results(
            situation, result_dir
        )

        # NOTE: both files were written from the SAME 'ds_test' within
        # a single process_celerity_sensitivity() call, in the same
        # order -- this re-sort by profile index is a cheap safety net
        # against that assumption ever breaking (e.g. a future
        # refactor reading them from separate passes), not something
        # expected to actually reorder anything today.
        order_dist = np.argsort(profile_idx_dist)
        order_f1_score = np.argsort(profile_idx_f1_score)
        dist = {
            "L1": dist_L1,
            "L2": dist_L2,
            "theta": dist_theta,
            "wasserstein": dist_wasserstein,
        }[metric][order_dist]
        f1_score = f1_score[order_f1_score]

        axs[i].scatter(f1_score, dist, s=12)
        axs[i].set_xlabel(r"F1 score from baseline profile")
        axs[i].set_title(situation)

    axs[0].set_ylabel(f"Distance ({METRIC_LABEL[metric]})")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = _sensitivity_figure_filename(
            "dist_vs_f1_score", situations, file_prefix="dist_", metric=metric
        )
        fig.savefig(os.path.join(save_dir, fname))

        plt.close(fig)

    return fig


def plot_celerity_distance_vs_std(
    env_type,
    situations=None,
    metric="theta",
    result_dir=None,
    save_dir=None,
):
    """Plot the RTF distance from baseline (see
    process_celerity_sensitivity()) AS A FUNCTION OF each profile's own
    STD deviation from the baseline (mean) celerity profile (see
    _celerity_profile_std()), one panel per situation -- this is the
    celerity-sensitivity study's KEY diagnostic, per user request: the
    profile INDEX plot_sensitivity_curves() would otherwise show on the
    x-axis carries no information about how DIFFERENT a profile
    actually is from the reference one, whereas the STD does.

    Args:
        env_type (str): "sw" or "dw" -- see CELERITY_ENV_TYPES.
        situations (list[str]|None): which situations to plot (each
            must have a '<result_dir>/std_<situation>.csv' file from
            process_celerity_sensitivity()). None (the default):
            every such file under 'result_dir'.
        metric (str): "L1", "L2", or "theta" -- which distance metric
            to plot on the y-axis.
        result_dir (str|None): where to read the saved distance/STD
            results from. None (the default):
            '<RESILIENCE_RESULT_DIR>/<env_type>/celerity/'.
        save_dir (str|None): if given, save the figure to
            '<save_dir>/<filename>.png' (see
            _sensitivity_figure_filename()). None (the default): the
            figure is only returned, not saved.

    Returns:
        matplotlib.figure.Figure

    Raises:
        KeyError: if 'env_type' is not one of CELERITY_ENV_TYPES.
    """
    CELERITY_ENV_TYPES[env_type]

    if result_dir is None:
        result_dir, _ = _resilience_study_dirs(env_type, "celerity")

    if situations is None:
        prefix, suffix = "std_", ".csv"
        situations = sorted(
            name[len(prefix) : -len(suffix)]
            for name in os.listdir(result_dir)
            if name.startswith(prefix) and name.endswith(suffix)
        )

    n_situations = len(situations)
    fig, axs = plt.subplots(
        1, max(n_situations, 1), figsize=(16, 8), squeeze=False, sharey=True
    )
    axs = axs[0]

    for i, situation in enumerate(situations):
        profile_idx_dist, dist_L1, dist_L2, dist_theta, dist_wasserstein = (
            load_sensitivity_distance_results(
                situation, result_dir=result_dir, file_prefix="dist_"
            )
        )
        profile_idx_std, std = _load_celerity_std_results(situation, result_dir)

        # NOTE: both files were written from the SAME 'ds_test' within
        # a single process_celerity_sensitivity() call, in the same
        # order -- this re-sort by profile index is a cheap safety net
        # against that assumption ever breaking (e.g. a future
        # refactor reading them from separate passes), not something
        # expected to actually reorder anything today.
        order_dist = np.argsort(profile_idx_dist)
        order_std = np.argsort(profile_idx_std)
        dist = {
            "L1": dist_L1,
            "L2": dist_L2,
            "theta": dist_theta,
            "wasserstein": dist_wasserstein,
        }[metric][order_dist]
        std = std[order_std]

        axs[i].scatter(std, dist, s=12)
        axs[i].set_xlabel(r"STD from baseline profile [m s$^{-1}$]")
        axs[i].set_title(situation)

    axs[0].set_ylabel(f"Distance ({METRIC_LABEL[metric]})")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = _sensitivity_figure_filename(
            "dist_vs_std", situations, file_prefix="dist_", metric=metric
        )
        fig.savefig(os.path.join(save_dir, fname))

        plt.close(fig)

    return fig


def plot_extremal_celerity_configs(
    env_type,
    situations=None,
    metric="theta",
    result_dir=None,
    save_dir=None,
):
    """For each situation, plot:
      - gamma(f) AT r0 for the baseline and for the TWO profiles
        (swept values) with the SMALLEST and the LARGEST RTF distance
        from that baseline (mirrors
        plot_extremal_resilience_dist_configs(), adapted to the
        celerity study's own "profile"-indexed sweep);
      - the celerity profiles (c_p vs depth) themselves for those SAME
        two extremal profiles, alongside the baseline (mean) profile
        -- per user request: since the distance extrema are profile-
        SHAPE-dependent, seeing the actual profile shapes behind them
        is as informative as seeing gamma itself.

    One figure per kind (not combined), one PER SITUATION for each
    kind (i.e. 2 * len(situations) figures total).

    Args:
        env_type (str): "sw" or "dw" -- see CELERITY_ENV_TYPES.
        situations (list[str]|None): which situations to plot. None
            (the default): every situation with a saved distance
            result under 'result_dir'.
        metric (str): "L1", "L2", or "theta" -- which distance metric
            decides which profiles count as "smallest"/"largest"
            (neither gamma(f) nor the profile itself depends on
            'metric' -- only which 2 profiles get plotted does).
        result_dir (str|None): where the per-profile files, the
            baseline file, and the saved distance results live. None
            (the default): '<RESILIENCE_RESULT_DIR>/<env_type>/celerity/'.
        save_dir (str|None): if given, save every figure to
            '<save_dir>/<filename>.png'. None (the default): figures
            are only returned, not saved.

    Returns:
        dict[str, dict[str, matplotlib.figure.Figure]]:
        {situation: {"gamma": <fig>, "profile": <fig>}}.

    Raises:
        KeyError: if 'env_type' is not one of CELERITY_ENV_TYPES.
    """
    env_config = CELERITY_ENV_TYPES[env_type]

    if result_dir is None:
        result_dir, _ = _resilience_study_dirs(env_type, "celerity")

    baseline_src_rcv_param = baseline_src_rcv()
    d12 = baseline_src_rcv_param["d12"]
    r0 = baseline_src_rcv_param["r0"]

    if situations is None:
        prefix, suffix = "dist_", ".csv"
        situations = sorted(
            name[len(prefix) : -len(suffix)]
            for name in os.listdir(result_dir)
            if name.startswith(prefix) and name.endswith(suffix)
        )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    figures = {}

    for situation in situations:
        # NOTE (per user request): THIS situation's own baseline (see
        # build_celerity_baseline()'s own NOTE) -- "winter" is compared
        # against, and plotted alongside, winter's own mean profile,
        # not "all"'s.
        z_baseline, c_p_baseline = load_mean_celerity_profile(
            _ssp_filename(env_type, situation), target_depth=env_config["depth"]
        )

        profile_idx, dist_L1, dist_L2, dist_theta, dist_wasserstein = (
            load_sensitivity_distance_results(
                situation, result_dir=result_dir, file_prefix="dist_"
            )
        )
        dist = {
            "L1": dist_L1,
            "L2": dist_L2,
            "theta": dist_theta,
            "wasserstein": dist_wasserstein,
        }[metric]

        idx_min = int(np.nanargmin(dist))
        idx_max = int(np.nanargmax(dist))
        profile_min = profile_idx[idx_min]
        profile_max = profile_idx[idx_max]

        label_min = f"profile {profile_min:g} (min dist)"
        label_max = f"profile {profile_max:g} (max dist)"

        situation_dir = os.path.join(result_dir, situation)
        value_files = sorted(
            glob.glob(os.path.join(situation_dir, f"{situation}_*.nc"))
        )
        with xr.open_dataset(
            os.path.join(situation_dir, "gf_dataset_baseline.nc")
        ) as ds_baseline:
            baseline_gamma_r0 = ds_baseline.gamma.sel(r=r0, method="nearest")

            with xr.open_mfdataset(
                value_files,
                combine="nested",
                concat_dim="profile",
                join="override",
            ) as ds_test_full:
                ds_test_sel = ds_test_full.sel(
                    profile=[profile_min, profile_max], method="nearest"
                )
                gamma_sel = derive_gamma(ds_test_sel, d12)  # dims (profile, f, r)
                c_p_sel = ds_test_sel["c_p_ssp"].values  # (2, n_depth)
                z_sel = ds_test_sel["z"].values

                fig_gamma, ax_gamma = plt.subplots(figsize=(16, 8))
                baseline_gamma_r0.plot(ax=ax_gamma, label="Baseline")
                gamma_sel.isel(profile=0).sel(r=r0, method="nearest").plot(
                    ax=ax_gamma, label=label_min
                )
                gamma_sel.isel(profile=1).sel(r=r0, method="nearest").plot(
                    ax=ax_gamma, label=label_max
                )

        ax_gamma.set_xlabel("Fréquence [Hz]")
        ax_gamma.set_ylabel(r"$\gamma$ [dB]")
        ax_gamma.set_title(situation)
        ax_gamma.legend()

        fig_profile, ax_profile = plt.subplots(figsize=(8, 10))
        ax_profile.plot(c_p_baseline, z_baseline, color="k", lw=2, label="Baseline")
        ax_profile.plot(c_p_sel[0], z_sel, label=label_min)
        ax_profile.plot(c_p_sel[1], z_sel, label=label_max)
        ax_profile.invert_yaxis()
        ax_profile.set_xlabel(r"Sound speed [m s$^{-1}$]")
        ax_profile.set_ylabel("Depth [m]")
        ax_profile.set_title(situation)
        ax_profile.legend()

        figures[situation] = {"gamma": fig_gamma, "profile": fig_profile}

        if save_dir is not None:
            fig_gamma.savefig(
                os.path.join(
                    save_dir,
                    _sensitivity_figure_filename(
                        "gamma_at_r0", situation, file_prefix="dist_", metric=metric
                    ),
                )
            )
            fig_profile.savefig(
                os.path.join(
                    save_dir,
                    _sensitivity_figure_filename(
                        "extremal_profiles",
                        situation,
                        file_prefix="dist_",
                        metric=metric,
                    ),
                )
            )

            plt.close(fig_gamma)
            plt.close(fig_profile)

    return figures


def process_sensitivity_mainlobe_width(test_arg_names=None, result_dir=RESULT_DIR):
    """Read back every parameter's per-value result files (see
    build_sensitivity_dataset()), compute each value's RTF distance
    from the baseline AS A FUNCTION OF r around r0 (see
    dist_from_baseline_around_r0()), derive the -3dB mainlobe width of
    that distance curve for every value (see
    single_sensitivity_test_calc_dist_width()), save the resulting
    width-vs-parameter-value arrays to a dedicated file per parameter
    (see save_sensitivity_distance_results()), then plot them (see
    plot_sensitivity_curves()) -- this is what lets you see how the
    waveguide's parameters affect the RTF's range-resolution (how
    narrow/wide the -3dB mainlobe around the true range r0 is).

    NOTE (renamed from 'process_sensitivity_v2', finished): this used
    to save 'dist_L1'/'dist_L2'/'dist_theta' directly -- but those are
    now 2D (n_r, n_values) arrays (one full distance-vs-r curve per
    swept value, from dist_from_baseline_around_r0()), not the 1D,
    one-value-per-parameter arrays save_sensitivity_distance_results()
    expects (np.column_stack([test_values, dist_L1, ...]) would raise
    a shape-mismatch error, or silently misalign columns, confirmed by
    inspection). The mainlobe WIDTHS computed right above (already 1D,
    one value per swept parameter value) are what actually needs
    saving here. Saved under the "mainlobe_width_" prefix (see
    save_sensitivity_distance_results()'s own docstring) so this
    doesn't collide with process_sensitivity()'s own "dist_" files,
    which hold a different quantity (the raw RTF distance at r0 alone,
    not a mainlobe width).

    Args:
        test_arg_names (list[str]|None): which parameters to process
            (each must have a '<result_dir>/<name>/' folder from
            build_sensitivity_dataset()). None discovers every such
            folder under 'result_dir' automatically.
        result_dir (str): where to read the per-value/baseline files
            from, and where to save the mainlobe-width results.

    Returns:
        matplotlib.figure.Figure (see plot_sensitivity_curves()).
    """
    baseline_src_rcv_param = baseline_src_rcv()
    d12 = baseline_src_rcv_param["d12"]
    r0 = baseline_src_rcv_param["r0"]

    if test_arg_names is None:
        test_arg_names = sorted(
            name
            for name in os.listdir(result_dir)
            if os.path.isdir(os.path.join(result_dir, name))
        )

    fpath_baseline = os.path.join(result_dir, "gf_dataset_baseline.nc")

    # NOTE: no 'preprocess' here (unlike process_sensitivity()'s own
    # '_select_r0_pair') -- this function needs the FULL r window each
    # per-value file holds (to derive a mainlobe width), not just the
    # single r0 point. This is feasible because that window is now
    # narrow by construction (baseline_src_rcv()'s own r_rcv already
    # only spans a region around r0, not the whole receiver range),
    # unlike the much wider grids process_sensitivity()'s optimization
    # was written for.
    with xr.open_dataset(fpath_baseline) as ds_baseline:
        for test_arg_name in test_arg_names:
            value_files = sorted(
                glob.glob(
                    os.path.join(result_dir, test_arg_name, f"{test_arg_name}_*.nc")
                )
            )
            if not value_files:
                continue

            with xr.open_mfdataset(
                value_files,
                combine="nested",
                concat_dim=test_arg_name,
                join="override",
            ) as ds_test:
                # NOTE: no downcast needed here anymore -- 'gf' (real-
                # valued, a magnitude -- see build_sensitivity_dataset()'s
                # own np.abs()) is already saved as float32 on disk (see
                # _to_float32()'s docstring for why this now happens at
                # write time instead of here).
                # Distance from the baseline, as a function of r around r0.
                r_dist, dist_L1, dist_L2, dist_theta = dist_from_baseline_around_r0(
                    ds_baseline, ds_test, d12, r0
                )
                # -3dB mainlobe width of that distance curve, PER swept
                # value (dist_L1/L2/theta are 2D here -- see
                # single_sensitivity_test_calc_dist_width()'s own NOTE).
                width_L1, width_L2, width_theta = (
                    single_sensitivity_test_calc_dist_width(
                        dist_L1, dist_L2, dist_theta, r_grid=r_dist, r0=r0
                    )
                )

                test_values = ds_test[test_arg_name].values

            save_sensitivity_distance_results(
                test_arg_name,
                test_values,
                width_L1,
                width_L2,
                width_theta,
                result_dir=result_dir,
                file_prefix="mainlobe_width_",
            )

    return plot_sensitivity_curves(
        test_arg_names,
        result_dir=result_dir,
        distance=["L1", "L2", "theta"],
        file_prefix="mainlobe_width_",
        ylabel="Mainlobe width [m]",
    )


def process_sensitivity_intrinsic_mainlobe_width(
    test_arg_names=None, result_dir=RESULT_DIR
):
    """Read back every parameter's per-value result files (see
    build_sensitivity_dataset()), compute each value's RTF distance
    between its OWN gamma at r0 and its OWN gamma as a function of r
    around r0 (see dist_within_test_around_r0() -- no baseline
    involved at all, unlike process_sensitivity_mainlobe_width()),
    derive the -3dB mainlobe width of that distance curve for every
    value (see single_sensitivity_test_calc_dist_width()), save the
    resulting width-vs-parameter-value arrays to a dedicated file per
    parameter (see save_sensitivity_distance_results()), then plot them
    (see plot_sensitivity_curves()).

    This is the INTRINSIC counterpart to
    process_sensitivity_mainlobe_width(): that one measures how a
    parameter MISMATCH against a baseline widens the RTF's mainlobe;
    this one measures how narrow/wide each configuration's mainlobe
    already is on its own terms, with no baseline comparison at all
    (see dist_within_test_around_r0()'s own docstring for the full
    rationale).

    Args:
        test_arg_names (list[str]|None): which parameters to process
            (each must have a '<result_dir>/<name>/' folder from
            build_sensitivity_dataset()). None discovers every such
            folder under 'result_dir' automatically.
        result_dir (str): where to read the per-value files from, and
            where to save the mainlobe-width results.

    Returns:
        matplotlib.figure.Figure (see plot_sensitivity_curves()).
    """
    baseline_src_rcv_param = baseline_src_rcv()
    d12 = baseline_src_rcv_param["d12"]
    r0 = baseline_src_rcv_param["r0"]

    if test_arg_names is None:
        test_arg_names = sorted(
            name
            for name in os.listdir(result_dir)
            if os.path.isdir(os.path.join(result_dir, name))
        )

    # NOTE: no baseline file needed here at all (unlike
    # process_sensitivity_mainlobe_width()) -- see
    # dist_within_test_around_r0()'s docstring: this is entirely
    # self-referential, comparing each swept value against its own
    # gamma at r0, not against an external baseline configuration.
    for test_arg_name in test_arg_names:
        value_files = sorted(
            glob.glob(os.path.join(result_dir, test_arg_name, f"{test_arg_name}_*.nc"))
        )
        if not value_files:
            continue

        with xr.open_mfdataset(
            value_files,
            combine="nested",
            concat_dim=test_arg_name,
            join="override",
        ) as ds_test:
            # NOTE: no downcast needed here anymore -- see the matching
            # NOTE in process_sensitivity_mainlobe_width() /
            # _to_float32()'s docstring.
            r_dist, dist_L1, dist_L2, dist_theta = dist_within_test_around_r0(
                ds_test, d12, r0
            )
            width_L1, width_L2, width_theta = single_sensitivity_test_calc_dist_width(
                dist_L1, dist_L2, dist_theta, r_grid=r_dist, r0=r0
            )

            test_values = ds_test[test_arg_name].values

        save_sensitivity_distance_results(
            test_arg_name,
            test_values,
            width_L1,
            width_L2,
            width_theta,
            result_dir=result_dir,
            file_prefix="intrinsic_mainlobe_width_",
        )

    return plot_sensitivity_curves(
        test_arg_names,
        result_dir=result_dir,
        distance=["L1", "L2", "theta"],
        file_prefix="intrinsic_mainlobe_width_",
        ylabel="Intrinsic mainlobe width [m]",
    )


def plot_extremal_width_configs(
    test_arg_names=None,
    result_dir=RESULT_DIR,
    metric="L1",
    mode="baseline",
    save_dir=None,
):
    """For each swept parameter, plot:
      - the RTF distance vs. (r - r0) curve for the TWO configurations
        (swept values) with the SMALLEST and the LARGEST -3dB mainlobe
        width, in its OWN figure (one figure per parameter -- NOT
        subplots sharing one combined figure);
      - gamma(f, r - r0) itself, as a 2D image, for those SAME two
        configurations, side by side in a second figure per parameter.

    This lets you actually look at what the narrowest- and widest-
    mainlobe configurations look like -- both the distance curve the
    width was derived from, and the underlying RTF itself -- rather
    than just their scalar width.

    This reads back the width results already saved by
    process_sensitivity_mainlobe_width()/
    process_sensitivity_intrinsic_mainlobe_width() (see 'mode' below)
    to find WHICH swept values are the min/max ones, then re-opens just
    those 2 values' per-value files (not the whole sweep) and
    recomputes their distance curve and their gamma(f, r) -- the width
    results only ever stored the scalar width itself (see
    single_sensitivity_test_calc_dist_width()), not the full curve or
    gamma behind it, so this is the cheapest way to get them back for
    exactly the 2 configurations that matter here, without re-deriving
    anything for every swept value again.

    Args:
        test_arg_names (list[str]|None): which parameters to plot.
            None discovers every parameter with a saved width result
            for the given 'mode' under 'result_dir'.
        result_dir (str): where the per-value files, the baseline file
            (for 'mode="baseline"'), and the saved width results live.
        metric (str): "L1", "L2", or "theta" -- which width metric
            decides which configurations count as "smallest"/"largest"
            (a value that is the narrowest by one metric need not be
            by another). The SAME metric's distance curve is what gets
            plotted (the gamma(f, r) image does not depend on 'metric'
            at all -- it is the raw RTF, not a distance).
        mode (str): "baseline" (process_sensitivity_mainlobe_width()'s
            distance from an external baseline reference -- needs
            'gf_dataset_baseline.nc', see dist_from_baseline_around_r0())
            or "intrinsic" (process_sensitivity_intrinsic_mainlobe_width()'s
            purely self-referential distance, no baseline needed --
            see dist_within_test_around_r0()).
        save_dir (str|None): if given, save every figure to
            '<save_dir>/<kind>_<test_arg_name>.png' ('kind' is "dist"
            or "gamma") -- directory created if it doesn't exist yet.
            None (the default): figures are only returned, not saved
            to disk. Pass IMG_DIR for this project's own conventional
            image folder.

    Returns:
        dict[str, dict[str, matplotlib.figure.Figure]]:
        {test_arg_name: {"distance": <fig>, "gamma": <fig>}}.

    Raises:
        ValueError: if 'mode' is neither "baseline" nor "intrinsic".
    """
    if mode == "baseline":
        file_prefix = "mainlobe_width_"
    elif mode == "intrinsic":
        file_prefix = "intrinsic_mainlobe_width_"
    else:
        raise ValueError(f"mode must be 'baseline' or 'intrinsic', got {mode!r}")

    baseline_src_rcv_param = baseline_src_rcv()
    d12 = baseline_src_rcv_param["d12"]
    r0 = baseline_src_rcv_param["r0"]

    if test_arg_names is None:
        suffix = ".csv"
        test_arg_names = sorted(
            name[len(file_prefix) : -len(suffix)]
            for name in os.listdir(result_dir)
            if name.startswith(file_prefix) and name.endswith(suffix)
        )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    figures = {}

    # Only opened for 'mode="baseline"' -- see dist_from_baseline_around_r0()'s
    # own docstring for why "intrinsic" needs no baseline at all.
    ds_baseline = (
        xr.open_dataset(os.path.join(result_dir, "gf_dataset_baseline.nc"))
        if mode == "baseline"
        else None
    )
    try:
        for test_arg_name in test_arg_names:
            test_values, width_L1, width_L2, width_theta, _ = (
                load_sensitivity_distance_results(
                    test_arg_name, result_dir=result_dir, file_prefix=file_prefix
                )
            )
            width = {"L1": width_L1, "L2": width_L2, "theta": width_theta}[metric]

            idx_min = int(np.nanargmin(width))
            idx_max = int(np.nanargmax(width))
            value_min = test_values[idx_min]
            value_max = test_values[idx_max]

            param_label = ARG_LABEL.get(test_arg_name, test_arg_name)
            # NOTE (per user request): explicitly names the parameter
            # AND its value (not just a bare number) -- e.g.
            # "D [m] = 30 (min width)" instead of the previous, more
            # ambiguous "min width (30)".
            label_min = f"{param_label} = {value_min:g} (min width)"
            label_max = f"{param_label} = {value_max:g} (max width)"

            value_files = sorted(
                glob.glob(
                    os.path.join(result_dir, test_arg_name, f"{test_arg_name}_*.nc")
                )
            )
            with xr.open_mfdataset(
                value_files,
                combine="nested",
                concat_dim=test_arg_name,
                join="override",
            ) as ds_test_full:
                # Reduce to JUST the 2 configurations that matter here
                # (order preserved: [value_min, value_max]) BEFORE
                # deriving anything, so nothing is recomputed for the
                # values we don't care about here.
                ds_test_sel = ds_test_full.sel(
                    **{test_arg_name: [value_min, value_max]}, method="nearest"
                )

                if mode == "baseline":
                    r_dist, dist_L1, dist_L2, dist_theta = dist_from_baseline_around_r0(
                        ds_baseline, ds_test_sel, d12, r0
                    )
                else:
                    r_dist, dist_L1, dist_L2, dist_theta = dist_within_test_around_r0(
                        ds_test_sel, d12, r0
                    )

                # gamma(f, r) itself for these 2 configs (mode-
                # independent: it's the raw RTF, not a distance from
                # anything) -- see derive_gamma()'s own docstring.
                gamma_sel = derive_gamma(ds_test_sel, d12)  # dims (test_arg_name, f, r)
                gamma_vals = gamma_sel.values  # (2, nf, nr)
                r_gamma = gamma_sel.r.values
                freq = gamma_sel.f.values

            dist = {"L1": dist_L1, "L2": dist_L2, "theta": dist_theta}[
                metric
            ]  # (n_r, 2)

            # --- Figure 1: distance vs. (r - r0), one figure per parameter ---
            fig_dist, ax_dist = plt.subplots(figsize=(16, 8))
            ax_dist.plot(r_dist - r0, dist[:, 0], label=label_min)
            ax_dist.plot(r_dist - r0, dist[:, 1], label=label_max)
            ax_dist.set_xlabel(r"$r - r_0$ [m]")
            ax_dist.set_ylabel(f"Distance ({METRIC_LABEL[metric]})")
            ax_dist.set_title(param_label)
            ax_dist.legend()

            # --- Figure 2: gamma(f, r - r0), one figure per parameter,
            # min/max side by side ---
            fig_gamma, axs_gamma = plt.subplots(
                1, 2, figsize=(16, 8), sharex=True, sharey=True
            )
            for j, lbl in enumerate((label_min, label_max)):
                # vmin = np.percentile(gamma_vals[j], q=5)
                # vmax = np.percentile(gamma_vals[j], q=95)
                im = axs_gamma[j].pcolormesh(
                    r_gamma - r0,
                    freq,
                    gamma_vals[j],
                    shading="auto",
                    cmap="magma",
                    # vmin=vmin,
                    # vmax=vmax,
                )
                axs_gamma[j].set_xlabel(r"$r - r_0$ [m]")
                axs_gamma[j].set_title(lbl)
            axs_gamma[0].set_ylabel("f [Hz]")
            fig_gamma.colorbar(im, ax=axs_gamma, label=r"$\gamma$ [dB]")

            figures[test_arg_name] = {"distance": fig_dist, "gamma": fig_gamma}

            if save_dir is not None:
                fig_dist.savefig(
                    os.path.join(
                        save_dir,
                        _sensitivity_figure_filename(
                            "dist",
                            test_arg_name,
                            file_prefix=file_prefix,
                            mode=mode,
                            metric=metric,
                        ),
                    )
                )
                fig_gamma.savefig(
                    os.path.join(
                        save_dir,
                        _sensitivity_figure_filename(
                            "gamma",
                            test_arg_name,
                            file_prefix=file_prefix,
                            mode=mode,
                            metric=metric,
                        ),
                    )
                )
                plt.close(fig_dist)
                plt.close(fig_gamma)
    finally:
        if ds_baseline is not None:
            ds_baseline.close()

    return figures


def plot_extremal_resilience_dist_configs(
    test_arg_names=None,
    result_dir=RESILIENCE_RESULT_DIR,
    metric="theta",
    save_dir=RESILIENCE_IMG_DIR,
):
    """For each resilience-tested parameter, plot gamma(f) AT r0 for
    the baseline and for the TWO configurations (swept values) with
    the SMALLEST and the LARGEST RTF distance from that baseline (see
    dist_from_baseline() / process_resilience_tests()) -- letting you
    actually see what gamma looks like for the best- and worst-case
    configurations (e.g. the tide state closest to, and furthest from,
    the baseline), rather than just their scalar distance.

    This mirrors plot_extremal_width_configs()'s own approach (reload
    the saved distance results to find which swept values are the
    min/max ones, then re-open just those 2 values' per-value files),
    adapted to the resilience study's "dist_" results and directory
    layout (see build_resilience_tests()/process_resilience_tests()).

    Args:
        test_arg_names (list[str]|None): which resilience tests to
            plot (each must have a '<result_dir>/dist_<name>.csv' file
            from process_resilience_tests()). None discovers every
            such file under 'result_dir' automatically.
        result_dir (str): where the per-value files, the baseline file,
            and the saved distance results live. Defaults to
            RESILIENCE_RESULT_DIR.
        metric (str): "L1", "L2", or "theta" -- which distance metric
            decides which configurations count as "smallest"/
            "largest" (gamma(f) itself does not depend on 'metric' --
            only which 2 configurations get plotted does).
        save_dir (str|None): if given, save each figure to
            '<save_dir>/<filename>.png' (see
            _sensitivity_figure_filename()) -- directory created if it
            doesn't exist yet. None: figures are only returned, not
            saved. Defaults to RESILIENCE_IMG_DIR.

    Returns:
        dict[str, matplotlib.figure.Figure]: {test_arg_name: figure}.
    """
    d12 = baseline_src_rcv()["d12"]
    r0 = baseline_src_rcv()["r0"]

    if test_arg_names is None:
        prefix, suffix = "dist_", ".csv"
        test_arg_names = sorted(
            name[len(prefix) : -len(suffix)]
            for name in os.listdir(result_dir)
            if name.startswith(prefix) and name.endswith(suffix)
        )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    figures = {}

    with xr.open_dataset(
        os.path.join(result_dir, "gf_dataset_baseline.nc")
    ) as ds_baseline:
        # The baseline doesn't depend on which parameter is being swept
        # -- computed once, reused for every test_arg_name below.
        baseline_gamma_r0 = ds_baseline.gamma.sel(r=r0, method="nearest")

        for test_arg_name in test_arg_names:
            test_values, dist_L1, dist_L2, dist_theta, dist_wasserstein = (
                load_sensitivity_distance_results(
                    test_arg_name, result_dir=result_dir, file_prefix="dist_"
                )
            )
            dist = {
                "L1": dist_L1,
                "L2": dist_L2,
                "theta": dist_theta,
                "wasserstein": dist_wasserstein,
            }[metric]

            idx_min = int(np.nanargmin(dist))
            idx_max = int(np.nanargmax(dist))
            value_min = test_values[idx_min]
            value_max = test_values[idx_max]

            param_label = ARG_LABEL.get(test_arg_name, test_arg_name)
            label_min = f"{param_label} = {value_min:g} (min dist)"
            label_max = f"{param_label} = {value_max:g} (max dist)"

            value_files = sorted(
                glob.glob(
                    os.path.join(result_dir, test_arg_name, f"{test_arg_name}_*.nc")
                )
            )
            with xr.open_mfdataset(
                value_files,
                combine="nested",
                concat_dim=test_arg_name,
                join="override",
            ) as ds_test_full:
                # Reduce to JUST the 2 configurations that matter here
                # (order preserved: [value_min, value_max]) before
                # deriving gamma, so nothing is recomputed for the
                # values we don't care about here.
                ds_test_sel = ds_test_full.sel(
                    **{test_arg_name: [value_min, value_max]}, method="nearest"
                )
                gamma_sel = derive_gamma(ds_test_sel, d12)  # dims (test_arg_name, f, r)

                fig, ax = plt.subplots(figsize=(16, 8))
                baseline_gamma_r0.plot(ax=ax, label="Baseline")
                gamma_sel.isel(**{test_arg_name: 0}).sel(r=r0, method="nearest").plot(
                    ax=ax, label=label_min
                )
                gamma_sel.isel(**{test_arg_name: 1}).sel(r=r0, method="nearest").plot(
                    ax=ax, label=label_max
                )

            ax.set_xlabel("Fréquence [Hz]")
            ax.set_ylabel(r"$\gamma$ [dB]")
            ax.set_title(param_label)
            ax.legend()

            figures[test_arg_name] = fig

            if save_dir is not None:
                fname = _sensitivity_figure_filename(
                    "gamma_at_r0", test_arg_name, file_prefix="dist_", metric=metric
                )
                fig.savefig(os.path.join(save_dir, fname))
                plt.close(fig)

    return figures


def generate_all_diagnostics(
    distance=["theta"],
    process_sensi=True,
    celerity_env_types=None,
    celerity_situations=None,
    build_baseline=False,
):
    """Regenerate every diagnostic figure this module produces, across
    all 4 studies (classic sensitivity, resilience, celerity), saving
    each under its own IMG_DIR-family directory.

    NOTE (renamed from 'generate_all_diag', bug fixed): Step 2 used to
    call process_sensitivity_mainlobe_width() (which saves its results
    under the "mainlobe_width_" prefix -- the BASELINE-relative
    mainlobe width), but then plot/select from "intrinsic_
    mainlobe_width_" files (process_sensitivity_intrinsic_
    mainlobe_width()'s own, SELF-referential prefix -- see both
    functions' own docstrings for the distinction) in every step
    right after -- confirmed those files are never produced by the
    call that precedes them, so plot_sensitivity_curves()/
    plot_extremal_width_configs() there would either raise
    FileNotFoundError (first run) or silently plot STALE data left
    over from a past process_sensitivity_intrinsic_mainlobe_width()
    call (later runs). Step 2.1 now calls the "intrinsic" variant,
    matching what the rest of Step 2 actually reads.

    Also added an entire Step 4 for the celerity study (per user
    request -- see plot_celerity_distance_vs_rmse()/
    plot_extremal_celerity_configs()'s own docstrings), which this
    function had no coverage for at all before. Each environment
    type's celerity diagnostics are skipped gracefully (with a printed
    note, not a crash) if that environment type has no sweep data
    built yet (see build_celerity_sensitivity_dataset()) -- the
    celerity study is still being built up incrementally one
    (environment type, situation) combination at a time as of this
    writing, unlike the other 3 studies this function otherwise
    assumes are fully built.

    Args:
        distance (list[str]): which distance metric(s) to plot for the
            multi-metric panels (see plot_sensitivity_curves()'s own
            'distance'); distance[0] alone is used wherever a single
            metric is needed (extremal-configuration selection).
        process_sensi (bool): if True (the default), re-run every
            "process_*" analysis step (re-reading and re-deriving
            results from the raw per-value/per-profile files) before
            plotting. False: skip straight to plotting from whatever
            results are already saved (faster -- use when only the
            FIGURES need refreshing, not the underlying numbers).
            Does NOT affect the baseline-rebuilding steps (0 and the
            celerity baselines in Step 4), which always run -- see
            their own NOTE for why.
        celerity_env_types (list[str]|None): which environment types'
            celerity diagnostics to (re)generate. None (the default):
            every key of CELERITY_ENV_TYPES ("sw" and "dw").
        celerity_situations (list[str]|None): forwarded to
            process_celerity_sensitivity()/plot_celerity_distance_vs_rmse()/
            plot_extremal_celerity_configs() for each environment type
            -- None (the default) discovers every situation with saved
            data, independently per environment type.
    """
    # Step 0 : re-build baseline to plot baseline diags
    if celerity_env_types is None:
        celerity_env_types = list(CELERITY_ENV_TYPES.keys())

    # build_baseline()

    # Step 1 : Distance from baseline configuration evaluate at r0
    # 1.1) read all files and compute distance from baseline
    if process_sensi:
        process_sensitivity()
    # 1.2) plot distance from baseline for all parameters
    plot_sensitivity_curves(
        distance=distance,
        ylabel="Distance from baseline at r=r0",
        save_dir=IMG_DIR,
    )
    plt.close("all")

    # 1.3) plot distance from baseline for each parameter
    for test_arg_name in ["depth", "c1", "rho2", "attn2"]:
        plot_sensitivity_curves(
            test_arg_names=[test_arg_name],
            distance=distance,
            ylabel="Distance from baseline at r=r0",
            save_dir=IMG_DIR,
        )
        plt.close("all")

    # Step 2 : Mainlobe width of distance around r0 for each configuration
    # 2.1) read all files and compute mainlobe width of distance around r0
    if process_sensi:
        process_sensitivity_intrinsic_mainlobe_width()
    # 2.2) plot mainlobe width of distance around r0 for all parameters
    plot_sensitivity_curves(
        distance=distance,
        file_prefix="intrinsic_mainlobe_width_",
        ylabel="Intrinsic mainlobe width [m]",
        save_dir=IMG_DIR,
    )
    plt.close("all")

    # 2.3) plot mainlobe width of distance around r0 for each parameter
    for test_arg_name in ["depth", "c1", "rho2", "attn2"]:
        plot_sensitivity_curves(
            test_arg_names=[test_arg_name],
            distance=distance,
            file_prefix="intrinsic_mainlobe_width_",
            ylabel="Intrinsic mainlobe width [m]",
            save_dir=IMG_DIR,
        )
        # Plot the two configurations with the smallest and largest mainlobe width
        plot_extremal_width_configs(
            test_arg_names=[test_arg_name],
            metric=distance[0],
            mode="intrinsic",
            save_dir=IMG_DIR,
        )
        plt.close("all")

    # Step 3 : Resilience tests
    # NOTE: now loops over environment types (see
    # build_resilience_tests()'s own NOTE for why "sw"/"dw" both matter
    # here) -- reuses 'celerity_env_types' rather than adding a THIRD
    # parameter that would always just duplicate it in practice.
    for env_type in celerity_env_types:
        resilience_result_dir, resilience_img_dir = _resilience_study_dirs(
            env_type, "depth"
        )

        if not os.path.isdir(resilience_result_dir):
            print(
                f"generate_all_diagnostics: no resilience sweep data found for "
                f"env_type={env_type!r} yet (see build_resilience_tests()) "
                f"-- skipping its diagnostics."
            )
            continue

        # 3.1) read all files and compute distance from baseline for resilience tests
        if process_sensi:
            process_resilience_tests(env_type)
        # 3.2) plot distance from baseline for resilience tests
        # 3.2.1) depth
        plot_sensitivity_curves(
            ["depth"],
            distance=distance,
            result_dir=resilience_result_dir,
            save_dir=resilience_img_dir,
        )
        plot_extremal_resilience_dist_configs(
            ["depth"],
            metric=distance[0],
            result_dir=resilience_result_dir,
            save_dir=resilience_img_dir,
        )
        plt.close("all")

    # Step 4 : Celerity (sound-speed profile) tests
    generate_celerity_diag(
        distance=distance,
        celerity_env_types=celerity_env_types,
        celerity_situations=celerity_situations,
        process_sensi=process_sensi,
        build_baseline=build_baseline,
    )


def generate_celerity_diag(
    distance=["theta"],
    celerity_env_types=None,
    celerity_situations=None,
    process_sensi=True,
    build_baseline=False,
):
    # 4.0) re-build each (environment type, situation)'s baseline, for
    # its own environment/mode-shape diagnostics AND because each
    # situation now needs its OWN baseline (see
    # build_celerity_baseline()'s own NOTE) -- this ALWAYS runs,
    # regardless of 'process_sensi', matching Step 0's own baseline
    # rebuild. NOTE: no longer "cheap" the way Step 0's single baseline
    # is -- this now runs one KRAKEN configuration PER (environment
    # type, situation) combination (10 by default: 2 env types x 5
    # situations), not one per environment type.

    if build_baseline:
        build_celerity_baselines(
            env_types=celerity_env_types, situations=celerity_situations
        )

    for env_type in celerity_env_types:
        result_dir, img_dir = _resilience_study_dirs(env_type, "celerity")

        # NOTE: unlike the other 3 studies, the celerity study is
        # still being built up incrementally (one environment type/
        # situation at a time, per user request) -- an environment
        # type with no sweep data yet is skipped here rather than
        # raised, so this function stays usable while that's ongoing.
        if not os.path.isdir(result_dir) or not any(
            os.path.isdir(os.path.join(result_dir, name))
            for name in os.listdir(result_dir)
        ):
            print(
                f"generate_all_diagnostics: no celerity sweep data found for "
                f"env_type={env_type!r} yet (see build_celerity_sensitivity_dataset()) "
                f"-- skipping its diagnostics."
            )
            continue

        # 4.1) read all profile files and compute distance from baseline
        if process_sensi:
            process_celerity_sensitivity(
                env_type,
                situations=celerity_situations,
                save_dir=img_dir,
            )
        # 4.2) the key diagnostic: distance vs. RMSE deviation from the
        # baseline profile (see plot_celerity_distance_vs_rmse()'s own
        # docstring for why this, rather than the raw profile index).
        # RMSE
        plot_celerity_distance_vs_rmse(
            env_type,
            situations=celerity_situations,
            metric=distance[0],
            result_dir=result_dir,
            save_dir=img_dir,
        )
        # F1 score
        plot_celerity_distance_vs_f1_score(
            env_type,
            situations=celerity_situations,
            metric=distance[0],
            result_dir=result_dir,
            save_dir=img_dir,
        )
        # STD
        plot_celerity_distance_vs_std(
            env_type,
            situations=celerity_situations,
            metric=distance[0],
            result_dir=result_dir,
            save_dir=img_dir,
        )

        # 4.3) gamma AND the profiles themselves, for the smallest-/
        # largest-distance configurations.
        plot_extremal_celerity_configs(
            env_type,
            situations=celerity_situations,
            metric=distance[0],
            result_dir=result_dir,
            save_dir=img_dir,
        )
        plt.close("all")


def run_debug_test():

    # General sensi test
    # build_baseline()
    # build_tests(use_debug_config=True)

    # Run all for SW env
    run_all_celerity(env_type="sw", n_profiles=1)
    # Run all for DW env
    # run_all_celerity(env_type="dw", n_profiles=1)

    # Depth resilience tests
    build_resilience_tests(env_types=None, depth_var_tide=10, npt=1)

    generate_all_diagnostics(distance=["theta"], process_sensi=True)


def run_all_plateform():
    # Baseline tests
    build_baseline()
    build_tests(use_debug_config=False)

    # Run all for SW env
    run_all_celerity(env_type="sw", n_profiles=1000)
    # Run all for DW env
    run_all_celerity(env_type="dw", n_profiles=1000)

    # Depth resilience tests for SW
    build_resilience_tests(env_types=["sw"], depth_var_tide=10, npt=200)
    # Depth resilience tests for DW
    build_resilience_tests(env_types=["dw"], depth_var_tide=10, npt=200)

    generate_all_diagnostics(
        distance=["theta"], process_sensi=True, build_baseline=False
    )


def run_all_celerity(env_type, n_profiles=1000):

    build_celerity_baselines(env_types=[env_type])
    build_celerity_tests(env_types=[env_type], situations=None, n_profiles=n_profiles)
    # Diag for each seasons
    for situation in ["all", "winter", "spring", "summer", "automn"]:
        generate_celerity_diag(
            distance=["theta"],
            celerity_env_types=[env_type],
            celerity_situations=[situation],
            process_sensi=True,
            build_baseline=False,
        )

    # Diag for variations all
    generate_celerity_diag(
        distance=["theta"],
        celerity_env_types=[env_type],
        celerity_situations=["all"],
        process_sensi=False,
        build_baseline=False,
    )
    # Winter vs summer
    generate_celerity_diag(
        distance=["theta"],
        celerity_env_types=[env_type],
        celerity_situations=["summer", "winter"],
        process_sensi=False,
        build_baseline=False,
    )


if __name__ == "__main__":

    # run_all_plateform()

    # # Run all for SW env
    # build_celerity_baselines(env_types=["sw"])
    # build_celerity_tests(env_types=["sw"], situations=None, n_profiles=1000)
    # # # Diag for each seasons
    # for situation in ["all", "winter", "spring", "summer", "automn"]:
    #     generate_celerity_diag(
    #         distance=["theta"],
    #         celerity_env_types=["sw"],
    #         celerity_situations=[situation],
    #         process_sensi=False,
    #         build_baseline=False,
    #     )

    # build_baseline()
    # build_tests(use_debug_config=True)
    # process_sensitivity()
    # plot_sensitivity_curves(
    #     distance=["wasserstein"],
    #     ylabel="Distance from baseline at r=r0",
    #     save_dir=IMG_DIR,
    # )

    # generate_all_diagnostics(
    #     distance=["wasserstein"], process_sensi=True, build_baseline=False
    # )
    generate_all_diagnostics(distance=["theta"], process_sensi=True)

    # process_sensitivity_intrinsic_mainlobe_width()
    # # 2.2) plot mainlobe width of distance around r0 for all parameters
    # plot_sensitivity_curves(
    #     distance="theta",
    #     file_prefix="intrinsic_mainlobe_width_",
    #     ylabel="Intrinsic mainlobe width [m]",
    #     save_dir=IMG_DIR,
    # )

    # # Diag for each seasons
    # env_type = "sw"
    # # Winter vs summer
    # generate_celerity_diag(
    #     distance=["theta"],
    #     celerity_env_types=[env_type],
    #     celerity_situations=["summer", "winter"],
    #     process_sensi=False,
    #     build_baseline=False,
    # )

    # env_type = "dw"
    # # Winter vs summer
    # generate_celerity_diag(
    #     distance=["theta"],
    #     celerity_env_types=[env_type],
    #     celerity_situations=["summer", "winter"],
    #     process_sensi=False,
    #     build_baseline=False,
    # )
    # Illustration of the std indicator behavior

    # fpath_ssp = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration_rtf\data\ssp\ssp_profiles_sw.nc"
    # ds_ssp = xr.open_dataset(fpath_ssp)

    # z_baseline = ds_ssp.depth.values
    # c_p_baseline = ds_ssp.ssp.mean(dim="time").values
    # z_baseline, c_p_baseline = _drop_depths_with_any_nan(z_baseline, c_p_baseline)
    # z_baseline, c_p_baseline = _ensure_profile_starts_at_surface(
    #     z_baseline, c_p_baseline
    # )
    # z_baseline, c_p_baseline = _adapt_profile_to_depth(z_baseline, c_p_baseline, 100)

    # z = ds_ssp.depth.values
    # c_p = ds_ssp.ssp.values
    # z, c_p = _drop_depths_with_any_nan(z, c_p)
    # z, c_p = _ensure_profile_starts_at_surface(z, c_p)
    # z, c_p = _adapt_profile_to_depth(z, c_p, 100)

    # std = _celerity_profile_std(
    #     z_baseline,
    #     c_p_baseline,
    #     z,
    #     c_p,
    # )

    # # 2 worst profiles
    # idx_largest_std = np.argsort(std)[-2:]
    # largest_std = std[idx_largest_std]
    # # 2 best
    # idx_smallest_std = np.argsort(std)[:2]
    # smallest_std = std[idx_smallest_std]

    # plt.figure()
    # plt.plot(c_p_baseline, z_baseline, color="k")
    # plt.plot(
    #     c_p[idx_largest_std[0], :],
    #     z,
    #     color=color(0),
    #     label=rf"Largest std ($\sigma$ = {{{largest_std[0]:.2f}}} m/s)",
    # )
    # plt.plot(
    #     c_p[idx_largest_std[1], :],
    #     z,
    #     color=color(1),
    #     label=rf"2nd largest std ($\sigma$ = {{{largest_std[1]:.2f}}} m/s)",
    # )
    # plt.plot(
    #     c_p[idx_smallest_std[0], :],
    #     z,
    #     color=color(2),
    #     label=rf"Smallest std ($\sigma$ = {{{smallest_std[0]:.2f}}} m/s)",
    # )
    # plt.plot(
    #     c_p[idx_smallest_std[1], :],
    #     z,
    #     color=color(3),
    #     label=rf"2nd smallest std ($\sigma$ = {{{smallest_std[1]:.2f}}} m/s)",
    # )
    # plt.xlabel("Celerity [m/s]")
    # plt.ylabel("Depth [m]")
    # plt.gca().invert_yaxis()
    # plt.legend()
    # # plt.savefig("test")

    # # std = _celerity_profile_std(
    # #     z_baseline,
    # #     c_p_baseline,
    # #     z,
    # #     c_p[idx_smallest_std[0], :],
    # # )
    # # std = _celerity_profile_std(
    # #     z_baseline,
    # #     c_p_baseline,
    # #     z,
    # #     c_p[idx_largest_std[0], :],
    # # )

    # plt.show()
