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
import numpy as np
import xarray as xr

from misc import progression_bar
from source.normal_modes import (
    pekeris_green_fct,
    pekeris_cutoff_frequency,
    pekeris_n_modes,
)
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
    SENSITIVITY_DIRECTORY = os.path.join(project_root, "illustration_rtf", "data", "sensitivity")

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


def baseline_env():
    # Waveguide parameters
    # rho1 = 1.0 * 1e3  # density in water (kg/m^3)
    # c1 = 1500  # sound celerity in water (m/s)
    # rho2 = 1.9 * 1e3  # density in fluid sediment (kg/m^3)
    # c2 = 1650  # sound celerity in fluid sediment (m/s)
    # attn2 = 0.0  # compressional wave attenuation in fluid sediment in dB / wavelength
    # d = 100  # waveguide depth (m)

    rho1 = 1.0 * 1e3  # density in water (kg/m^3)
    c1 = 1500  # sound celerity in water (m/s)
    rho2 = 1.5 * 1e3  # density in fluid sediment (kg/m^3)
    c2 = 1600  # sound celerity in fluid sediment (m/s)
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
    z_rcv = 99.5  # D-0.5
    dr = 5
    r0 = 50 * 1e3
    # r_rcv = np.arange(r0 - 5 * 1e3, r0 + 5 * 1e3 + dr, dr)
    r_rcv = np.arange(5 * 1e3, 100 * 1e3 + dr, dr)
    d12 = 5000

    src_rcv_param = {"z_s": z_s, "z_rcv": z_rcv, "r_rcv": r_rcv, "r0": r0, "d12": d12}
    return src_rcv_param


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
    r_d_pos_sup_3dB = r_pos[idx_d_pos_sup_3dB][0]

    # Negative r - r0
    d_neg = dist_r_r0[r_reduced < 0]
    r_neg = r_reduced[r_reduced < 0]
    # Reverse both arrays
    d_neg = d_neg[::-1]
    r_neg = r_neg[::-1]

    idx_d_neg_sup_3dB = d_neg > 1 / 2
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
            freq, c1, c2, rho1, rho2, attn2, depth, z_s, z_rcv, r_rcv
        )
    elif model == "analytic":
        freq, r_grid, gamma = single_sensitivity_test_generate_dataset_analytic(
            freq, c1, c2, rho1, rho2, attn2, depth, z_s, z_rcv, r_rcv, d12
        )

    return freq, r_grid, gamma


def single_sensitivity_test_generate_dataset_kraken(
    freq, c1, c2, rho1, rho2, attn2, d, z_s, z, r_grid, d12
):

    # Extend r_grid
    dr = r_grid[1] - r_grid[0]
    r_grid_add = np.arange(r_grid[-1] + dr, r_grid[-1] + dr + d12, dr)
    r_grid_ = np.append(r_grid, r_grid_add)

    # Convert r_grid to km for kraken
    r_grid_ = r_grid_ * 1e-3

    # Derive green's function at all pos
    freq, g_fr, field_pos = build_kraken(
        freq, c1, c2, rho1, rho2, attn2, d, z_s, z, r_grid_, d12
    )

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
    dr = r_grid[1] - r_grid[0]
    r_grid_add = np.arange(r_grid[-1] + dr, r_grid[-1] + dr + d12, dr)
    r_grid_ = np.append(r_grid, r_grid_add)
    # Derive green's function at all pos
    g_fr = pekeris_green_fct(freq, c1, c2, rho1, rho2, attn2, d, z_s, z, r_grid_)

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

    # Width defined by monotonicity
    # width_L1, r_domain_L1_inf,  r_domain_L1_sup = calc_monotonicity_domain(dist_r_r0=dist_L1, r=r_grid, r0=r0)
    # width_L2, r_domain_L2_inf, r_domain_L2_sup = calc_monotonicity_domain(
    #     dist_r_r0=dist_L2, r=r_grid, r0=r0
    # )
    # width_theta, r_domain_theta_inf, r_domain_theta_sup = calc_monotonicity_domain(
    #     dist_r_r0=dist_theta, r=r_grid, r0=r0
    # )

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
        # print(all_args)
        # Run test
        w_L1, w_L2, w_theta = run_single_sensitivity_test(**all_args)

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
        # fmin=kraken_freq.min(),
        # alpha_wavelength=10,
        # add_sediment_buffer_layer=True,
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
        # nmedia=2,
    )
    # assert env.nmedia == 1
    # env.write_env()
    # print(f"Wrote {env.env_fpath} (nmedia={env.nmedia})")

    flp = KrakenFlp(
        env=env,
        src_type="point_source",
        mode_theory="adiabatic",  # irrelevant for a range-independent run, kept simple
        mode_addition="coherent",
        nb_modes=nb_modes,
        src_depth=z_s,
        n_rcv_z=1,
        rcv_z_min=z,
        rcv_z_max=z,
        # n_rcv_z=100,
        # rcv_z_min=0,
        # rcv_z_max=d,
        n_rcv_r=r_grid.size,
        rcv_r_min=r_grid.min(),
        rcv_r_max=r_grid.max(),
    )
    # flp.write_flp()
    # print(f"Wrote {flp.flp_fpath}")

    # ----------------------------------------------------------------------
    # 2. Run KRAKEN + FIELD (requires real binaries -- see KrakenManager /
    #    propa.kraken_toolbox.params.KRAKEN_BIN_DIRECTORY).
    # ----------------------------------------------------------------------
    manager = KrakenManager(verbose=False)
    pressure_field, field_pos = manager.runkraken(
        env=env, flp=flp, frequencies=env.freq
    )
    # print("KRAKEN/FIELD run completed.")

    # Squeeze
    pressure_field = pressure_field.squeeze()

    # Get green's function
    c0 = 1500
    k0 = 2 * np.pi * kraken_freq / c0
    norm_factor = np.exp(1j * k0) / (4 * np.pi)
    # norm_factor[:] = 1
    g_fr = norm_factor[:, np.newaxis] * pressure_field  # (nf, nr)

    return kraken_freq, g_fr, field_pos


def build_dataset_current_config_kraken(
    freq, c1, c2, rho1, rho2, attn2, depth, z_s, z_rcv, r_rcv, d12_max
):

    # Extend r_grid to d12_max
    dr = r_rcv[1] - r_rcv[0]
    r_grid_add = np.arange(r_rcv[-1] + dr, r_rcv[-1] + dr + d12_max, dr)
    r_grid_ = np.append(r_rcv, r_grid_add)

    # Convert r_grid to km for kraken
    r_grid_ = r_grid_ * 1e-3

    # Derive green's function at all pos
    freq, g_fr, field_pos = build_kraken(
        freq, c1, c2, rho1, rho2, attn2, depth, z_s, z_rcv, r_grid_
    )

    # nr_shift = int(d12 / dr)
    # # Green function at receiver 1
    # g_fr_1 = g_fr[:, 0:-nr_shift]
    # # Green function at receiver 2
    # g_fr_2 = g_fr[:, nr_shift:]

    # # Build RTF
    # pi_21_fr = g_fr_2 / g_fr_1
    # Derive gamma
    # gamma = 20 * np.log10(np.abs(pi_21_fr))  # (nf, nr)

    return freq, field_pos["r"]["r"], g_fr


def build_sensitivity_dataset(
    test_arg_name, test_arg_values, all_arg_dict, model="kraken"
):
    
    print(f"Processing {test_arg_name}...")
    
    # We will build the args to pass to the test function at each iteration
    all_args = all_arg_dict.copy()

    # Add test variable to arg
    all_args.update({test_arg_name: None})

    i_test = 0
    prev_progress = 0

    # Init arrays
    g_fr_arr = []

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
        # print(all_args)
        # Build dataset
        kraken_freq, kraken_r, g_fr = build_dataset_current_config_kraken(
            **all_args
        )  # gamma is (nf, nr)

        # Pad with nan to get full size
        n_missing_freq = all_args["freq"].size - kraken_freq.size
        pad_width = ((n_missing_freq, 0), (0, 0))
        g_fr_full = np.pad(
            g_fr, pad_width=pad_width, mode="constant", constant_values=np.nan
        )
        # Store
        g_fr_arr.append(g_fr_full)

        # kraken_freq, r_grid, gamma = single_sensitivity_test_generate_dataset(
        #     **all_args
        # )  # gamma is (nf, nr)

        # # Pad with nan to get full size
        # n_missing_freq = all_args["freq"].size - kraken_freq.size
        # pad_width = ((n_missing_freq, 0), (0, 0))
        # gamma_full = np.pad(
        #     gamma, pad_width=pad_width, mode="constant", constant_values=np.nan
        # )
        # # Store
        # gamma_arr.append(gamma_full)

    # Build xarray dataset and save
    ds_sensi = xr.Dataset(
        data_vars=dict(gf=([test_arg_name, "f", "r"], np.abs(np.array(g_fr_arr)))),
        coords={
            test_arg_name: test_arg_values,
            "f": all_args["freq"],
            "r": kraken_r,
        },
    )
    fpath = os.path.join(RESULT_DIR, f"gf_dataset_{test_arg_name}.nc")
    ds_sensi.to_netcdf(fpath)


if __name__ == "__main__":
    # Environment
    env_param = baseline_env()
    print("Baseline environement properties:")
    for k in env_param.keys():
        print(f"\t{k} = {env_param[k]}")

    # Signal
    sig_param = baseline_sig()
    print("Baseline signal properties:")
    for k in sig_param.keys():
        print(f"\t{k} = {sig_param[k]}")

    # Source receiver config
    src_rcv_param = baseline_src_rcv()
    print("Baseline src / rcv configuration:")
    for k in src_rcv_param.keys():
        print(f"\t{k} = {src_rcv_param[k]}")

    all_arg_dict = env_param.copy()
    all_arg_dict.update(sig_param)
    all_arg_dict.update(src_rcv_param)
    # Remove fmax and fs that are not usefull
    all_arg_dict.pop("fs")
    all_arg_dict.pop("fmax")
    all_arg_dict.pop("r0")
    all_arg_dict.pop("d12")

    all_arg_dict["d12_max"] = 5000

    size_per_test_Ko = 110000
    nb_param = 4
    npt = 2
    total_size = size_per_test_Ko * nb_param * npt
    print(f"Total memory size = {total_size*1e-6} Go")

    # Depth
    test_arg_name = "depth"
    d_min = 30
    d_max = 5000
    d_test = np.linspace(d_min, d_max, npt)
    test_arg_values = d_test

    build_sensitivity_dataset(
        test_arg_name, test_arg_values, all_arg_dict, model="kraken"
    )

    # Water celerity
    test_arg_name = "c1"
    c1_min = 1450
    c1_max = 1550
    c1_test = np.linspace(c1_min, c1_max, npt)
    test_arg_values = c1_test

    build_sensitivity_dataset(
        test_arg_name, test_arg_values, all_arg_dict, model="kraken"
    )

    # Bottom celerity
    test_arg_name = "c2"
    c2_min = 1550.0
    c2_max = 1900.0
    c2_test = np.linspace(c2_min, c2_max, npt)
    test_arg_values = c2_test

    build_sensitivity_dataset(
        test_arg_name, test_arg_values, all_arg_dict, model="kraken"
    )

    # Bottom density
    test_arg_name = "rho2"
    rho2_min = 1.0 * 1e3
    rho2_max = 2.5 * 1e3
    rho2_test = np.linspace(rho2_min, rho2_max, npt)
    test_arg_values = rho2_test

    build_sensitivity_dataset(
        test_arg_name, test_arg_values, all_arg_dict, model="kraken"
    )

    # Bottom attenuation
    test_arg_name = "attn2"
    attn2_min = 0.0
    attn2_max = 1.0
    attn2_test = np.linspace(attn2_min, attn2_max, npt)
    test_arg_values = attn2_test

    build_sensitivity_dataset(
        test_arg_name, test_arg_values, all_arg_dict, model="kraken"
    )
