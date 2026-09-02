#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   sensibility.py
@Desc    :   Generic machinery for a KRAKEN parameter-sensibility study
             on a Pekeris waveguide: vary ONE named parameter at a time
             around a baseline (everything else held fixed), run
             KRAKEN/FIELD for every value, and extract a scalar
             quantity derived from the resulting Green's function.

Naming and conventions (parameter dicts with 'c1'/'c2'/'rho1'/'rho2'/
'attn2'/'depth' keys, the run_sensibility_study(test_arg_name,
test_arg_values, all_arg_dict, ...) signature, the Green's function
normalization G = pressure * exp(i k0) / (4 pi)) intentionally match
this project's own existing RTF sensibility study, so the two can share
a single 'all_arg_dict' baseline and this module can be used either
standalone or as a building block for that more elaborate study.

Filesystem layout (see run_sensibility_study()):
    <computation_root>/<test_arg_name>/  -- ONE reused working
        directory per swept parameter (not per value): each value's
        KRAKEN run overwrites the previous one's '.env'/'.mod'/'.shd'
        files there, since only the extracted scalar QoI needs to
        survive past that run -- keeps disk usage bounded regardless
        of how many values are swept.
    <result_root>/<test_arg_name>.csv    -- one CSV per swept
        parameter: columns [test_arg_name, qoi_dB], one row per value.

Usage sketch (see examples/pekeris_sensibility_study/ for a complete,
runnable script covering all 5 Pekeris parameters):

    from propa.kraken_toolbox.sensibility import (
        baseline_arg_dict, run_sensibility_study,
    )

    all_arg_dict = baseline_arg_dict()
    depths = np.linspace(10.0, 5000.0, 30)
    results = run_sensibility_study(
        test_arg_name="depth",
        test_arg_values=depths,
        all_arg_dict=all_arg_dict,
        computation_root="io_files",
        result_root="data/sensibility",
    )
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os

import numpy as np

from propa.kraken_toolbox.src.kraken_env import (
    KrakenEnv,
    KrakenMedium,
    KrakenBottomHalfspace,
    KrakenField,
    KrakenFlp,
)
from propa.kraken_toolbox.src.kraken_manager import KrakenManager
from propa.kraken_toolbox.utils import default_nb_rcv_z

TITLE = "Pekeris sensibility study"

# Keys read from 'all_arg_dict' by build_kraken_env() -- every other
# key is ignored by this module (a caller sharing a bigger dict, e.g.
# with the RTF study's own extra keys, does not need to strip them).
_ENV_KEYS = ("c1", "c2", "rho1", "rho2", "attn2", "depth")
_SRC_RCV_KEYS = ("z_s", "z_rcv", "r0", "r_rcv")
_FREQ_KEY = "freq"


# ======================================================================================================================
# Baseline Pekeris waveguide
# ======================================================================================================================
def baseline_env_param():
    """Default Pekeris waveguide parameters.

    Returns:
        dict: 'c1' (water sound speed, m/s), 'c2' (bottom sound speed,
        m/s), 'rho1' (water density, g/cm3), 'rho2' (bottom density,
        g/cm3), 'attn2' (bottom compressional attenuation,
        dB/wavelength), 'depth' (water depth, m).
    """
    return {
        "c1": 1500.0,
        "c2": 1600.0,
        "rho1": 1.0,
        "rho2": 1.5,
        "attn2": 0.2,
        "depth": 100.0,
    }


def baseline_freq_param():
    """Default frequency for the sensibility study: a single frequency
    (range-independent Pekeris runs, one per swept parameter value).

    Returns:
        dict: 'freq' (Hz).
    """
    return {"freq": 100.0}


def baseline_src_rcv_param():
    """Default source/receiver configuration: a full receiver-range
    grid (for plotting TL(r) if wanted) plus a single reference range
    'r0' at which the scalar QoI is evaluated.

    NOTE: 'z_s'/'z_rcv' are kept shallow (5 m) so they remain valid
    (strictly inside the water column) across a wide sweep of 'depth'
    -- e.g. down to 10 m, as in this project's default depth sweep. If
    you sweep 'depth' shallower than that, or sweep 'z_s'/'z_rcv'
    themselves deeper than the shallowest 'depth' you use elsewhere,
    adjust accordingly; nothing here re-checks this automatically.

    Returns:
        dict: 'z_s' (source depth, m), 'z_rcv' (receiver depth, m),
        'r0' (reference range, m), 'r_rcv' (full receiver range grid,
        m).
    """
    return {
        "z_s": 5.0,
        "z_rcv": 5.0,
        "r0": 5.0 * 1e3,
        "r_rcv": np.arange(0.1, 10.0 + 0.1, 0.1) * 1e3,
    }


def baseline_arg_dict():
    """Merge baseline_env_param()/baseline_freq_param()/
    baseline_src_rcv_param() into a single dict, matching the
    'all_arg_dict' this module's functions (and this project's own RTF
    sensibility study) expect.

    Returns:
        dict
    """
    all_args = {}
    all_args.update(baseline_env_param())
    all_args.update(baseline_freq_param())
    all_args.update(baseline_src_rcv_param())
    return all_args


# ======================================================================================================================
# Environment construction and Green's function extraction
# ======================================================================================================================
def build_kraken_env(all_arg_dict, root, filename, nmedia=None):
    """Build a Pekeris waveguide KrakenEnv + KrakenFlp from a parameter
    dict (see baseline_arg_dict()).

    Args:
        all_arg_dict (dict): must contain every key in _ENV_KEYS,
            _SRC_RCV_KEYS and _FREQ_KEY (see baseline_*_param()).
            'rho1'/'rho2' are in kg/m3 (matching this project's RTF
            study convention) and converted to g/cm3 internally for
            KRAKEN; 'r_rcv'/'r0' are in meters and converted to km
            internally.
        root (str): directory to write the '.env'/'.flp' files into.
        filename (str): base filename (without extension).
        nmedia (int|None): forwarded to KrakenEnv (None auto-derives
            it -> 1, since this is a direct half-space, no buffer --
            see KrakenBottomHalfspace's docstring).

    Returns:
        tuple(KrakenEnv, KrakenFlp)
    """
    c1 = all_arg_dict["c1"]
    c2 = all_arg_dict["c2"]
    rho1 = all_arg_dict["rho1"] 
    rho2 = all_arg_dict["rho2"]
    attn2 = all_arg_dict["attn2"]
    depth = all_arg_dict["depth"]

    freq = np.atleast_1d(all_arg_dict[_FREQ_KEY]).astype(float)

    z_s = all_arg_dict["z_s"]
    z_rcv = all_arg_dict["z_rcv"]
    r_rcv_km = np.atleast_1d(all_arg_dict["r_rcv"]).astype(float) / 1000.0

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
            "a_p": attn2,
            "a_s": 0.0,
        },
        add_sediment_buffer_layer=False,  # direct half-space -> classic Pekeris model
    )

    n_rcv_z = default_nb_rcv_z(fmax=freq.max(), max_depth=depth, n_per_l=5)
    field = KrakenField(
        phase_speed_limits=[0, 2000],
        src_depth=z_s,
        n_rcv_z=n_rcv_z,
        rcv_z_min=0.0,
        rcv_z_max=depth,
        rcv_r_max=r_rcv_km.max(),
    )

    env = KrakenEnv(
        title=TITLE,
        env_root=root,
        env_filename=filename,
        freq=freq,
        kraken_medium=medium,
        kraken_bottom_hs=bottom_hs,
        kraken_field=field,
        nmedia=nmedia,
    )

    flp = KrakenFlp(
        env=env,
        src_type="point_source",
        mode_theory="adiabatic",  # irrelevant for a range-independent run, kept simple
        mode_addition="coherent",
        nb_modes=9999,
        src_depth=z_s,
        n_rcv_z=1,
        rcv_z_min=z_rcv,
        rcv_z_max=z_rcv,
        n_rcv_r=r_rcv_km.size,
        rcv_r_min=r_rcv_km.min(),
        rcv_r_max=r_rcv_km.max(),
    )

    return env, flp


def compute_green_function(env, flp, manager=None):
    """Run KRAKEN/FIELD for 'env'/'flp' (writing their '.env'/'.flp'
    files first) and return the Green's function, i.e. the KRAKEN
    pressure field normalized the same way as this project's own RTF
    sensibility study: G(f, r) = pressure(f, r) * exp(i k0) / (4 pi),
    with k0 = 2*pi*f/c0 (c0 = 1500 m/s, a fixed reference sound speed
    used only for this normalization phase term -- not read from
    'env').

    Args:
        env (KrakenEnv): built by build_kraken_env() (or equivalent);
            write_env() is called here.
        flp (KrakenFlp): ditto; write_flp() is called here.
        manager (KrakenManager|None): reused across calls if given
            (recommended for a sweep, to avoid reconstructing it every
            iteration); a fresh, quiet one is created otherwise.

    Returns:
        tuple(freq, g_fr, field_pos): freq (float, Hz -- this module
        only supports a single frequency per run, see build_kraken_env),
        g_fr (np.ndarray, complex, shape (n_r,): the Green's function
        over the receiver range grid, at the single receiver depth
        'z_rcv'), field_pos (dict, see read_shd.readshd's 'Pos').
    """
    if manager is None:
        manager = KrakenManager(verbose=False)

    # env.write_env()
    # flp.write_flp()
    pressure_field, field_pos = manager.runkraken(env=env, flp=flp, frequencies=env.freq)
    pressure_field = np.squeeze(pressure_field)

    freq = float(np.atleast_1d(env.freq)[0])
    c0 = 1500.0
    k0 = 2 * np.pi * freq / c0
    norm_factor = np.exp(1j * k0) / (4 * np.pi)
    g_fr = norm_factor * pressure_field

    return freq, g_fr, field_pos


# ======================================================================================================================
# Quantity of interest (QoI), derived from the Green's function
# ======================================================================================================================
def compute_qoi_green_level_at_reference_range(g_fr, field_pos, r0):
    """Default QoI: 20*log10(|G(r0)|), i.e. the Green's function level
    (dB) at the reference range 'r0' -- the nearest available range on
    the receiver grid is used.

    Args:
        g_fr (np.ndarray): Green's function over range, as returned by
            compute_green_function().
        field_pos (dict): as returned by compute_green_function().
        r0 (float): reference range (m).

    Returns:
        float: 20*log10(|G(r0)|), in dB.
    """
    r = field_pos["r"]["r"]  # meters -- see read_shd.py's docstring
    ir = int(np.argmin(np.abs(r - r0)))
    with np.errstate(divide="ignore"):
        return float(20.0 * np.log10(np.abs(g_fr[ir]) + 1e-300))


# ======================================================================================================================
# Sensibility study orchestration
# ======================================================================================================================
def run_sensibility_study(
    test_arg_name,
    test_arg_values,
    all_arg_dict,
    computation_root,
    result_root,
    compute_qoi=None,
    run_kraken=True,
    verbose=True,
):
    """Sweep ONE named parameter across 'test_arg_values' (everything
    else held at its value in 'all_arg_dict'), running a full
    KRAKEN/FIELD simulation for each value, and save the (value, QoI)
    results to a single CSV file.

    Args:
        test_arg_name (str): name of the parameter to sweep -- one of
            'depth', 'c1', 'c2', 'rho1', 'rho2', 'attn2' (the
            environment), 'freq', or 'z_s'/'z_rcv'/'r0' (source/
            receiver) -- i.e. any key of 'all_arg_dict' that
            build_kraken_env() reads.
        test_arg_values (array-like): the values to sweep
            'test_arg_name' across.
        all_arg_dict (dict): baseline parameters (see
            baseline_arg_dict()); every key except 'test_arg_name' is
            held fixed at its value here. Not mutated (a copy is
            iterated over internally).
        computation_root (str): parent directory for the per-parameter
            working directory (computation_root/test_arg_name/),
            reused (overwritten) across every value in
            'test_arg_values' -- only the extracted QoI survives past
            each run, so disk usage stays bounded regardless of how
            many values are swept.
        result_root (str): directory the results CSV is written into,
            as '<result_root>/<test_arg_name>.csv'.
        compute_qoi (callable|None): function(g_fr, field_pos, r0) ->
            float, called once per swept value to extract the quantity
            of interest from the Green's function. Defaults to
            compute_qoi_green_level_at_reference_range (the Green's
            function level, in dB, at 'all_arg_dict["r0"]').
        run_kraken (bool): if False, only writes the '.env'/'.flp'
            files for the LAST value (a quick sanity/dry-run check) and
            returns None -- no KRAKEN/FIELD run, no results file,
            useful to verify the environment construction without
            needing the real binaries.
        verbose (bool): print progress for each value.

    Returns:
        np.ndarray|None: shape (len(test_arg_values), 2) array of
        [value, qoi] pairs (also written to the results CSV), sorted in
        the same order as 'test_arg_values'; or None if run_kraken is
        False.
    """
    if compute_qoi is None:
        compute_qoi = compute_qoi_green_level_at_reference_range

    test_arg_values = np.atleast_1d(test_arg_values)
    work_dir = os.path.join(computation_root, test_arg_name)
    os.makedirs(work_dir, exist_ok=True)
    filename = f"pekeris_{test_arg_name}"

    args = all_arg_dict.copy()

    if not run_kraken:
        args[test_arg_name] = test_arg_values[-1]
        env, flp = build_kraken_env(args, root=work_dir, filename=filename)
        env.write_env()
        flp.write_flp()
        if verbose:
            print(f"[{test_arg_name}] run_kraken=False: wrote {env.env_fpath} (dry run only).")
        return None

    manager = KrakenManager(verbose=False)
    results = []
    for i, value in enumerate(test_arg_values):
        args[test_arg_name] = value
        env, flp = build_kraken_env(args, root=work_dir, filename=filename)
        freq, g_fr, field_pos = compute_green_function(env, flp, manager=manager)
        qoi = compute_qoi(g_fr, field_pos, args["r0"])
        results.append((value, qoi))
        if verbose:
            print(
                f"[{test_arg_name}] {i + 1}/{test_arg_values.size}: "
                f"{test_arg_name}={value:g} -> QoI={qoi:.2f} dB"
            )

    results = np.array(results, dtype=float)

    os.makedirs(result_root, exist_ok=True)
    out_path = os.path.join(result_root, f"{test_arg_name}.csv")
    np.savetxt(out_path, results, delimiter=",", header=f"{test_arg_name},qoi_dB", comments="")
    if verbose:
        print(f"[{test_arg_name}] Wrote {out_path}")

    return results


def load_sensibility_result(result_root, test_arg_name):
    """Reload a results CSV written by run_sensibility_study(), without
    re-running anything.

    Args:
        result_root (str): same directory passed to
            run_sensibility_study().
        test_arg_name (str): same parameter name used there.

    Returns:
        np.ndarray: shape (n_values, 2), columns [value, qoi_dB].
    """
    path = os.path.join(result_root, f"{test_arg_name}.csv")
    return np.loadtxt(path, delimiter=",", skiprows=1)


# ======================================================================================================================
# Plotting
# ======================================================================================================================
def plot_sensibility_result(results, test_arg_name, param_label=None, param_units=None, ax=None):
    """Plot the QoI vs. swept parameter value from a results array (as
    returned by / saved by run_sensibility_study()).

    Args:
        results (np.ndarray): shape (n_values, 2), columns
            [value, qoi_dB] (see run_sensibility_study's return value,
            or load_sensibility_result()).
        test_arg_name (str): the swept parameter's name (used as the
            default x-axis label if 'param_label' is not given).
        param_label (str|None): x-axis label override (e.g. a nicer
            name/symbol than the raw parameter key).
        param_units (str|None): units to append to the x-axis label,
            e.g. "m" or "m/s".
        ax (matplotlib.axes.Axes|None): plot into this axis instead of
            creating a new figure.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    values, qois = results[:, 0], results[:, 1]
    ax.plot(values, qois, marker="o")

    label = param_label or test_arg_name
    if param_units:
        label = f"{label} [{param_units}]"
    ax.set_xlabel(label)
    ax.set_ylabel("Green's function level [dB]")
    ax.set_title(f"Sensibility to {param_label or test_arg_name}")
    ax.invert_yaxis()  # TL/level convention: higher up = less loss
    ax.grid(True)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    pass
