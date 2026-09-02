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
import numpy as np

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

TITLE = "RTF sensibility study"
SENSIBILITY_DIRECTORY = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration_rtf\data\sensibility"
)
ENV_FILENAME = "rtf_sensibility_study"

# ======================================================================================================================
# Sensibility study properties
# ======================================================================================================================


def baseline_env():
    # Waveguide parameters
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
    z_rcv = 99.5  # D-1
    dr = 5
    r_rcv = np.arange(27.5 * 1e3, 32.5 * 1e3 + dr, dr)
    r0 = 30 * 1e3
    d12 = 100

    src_rcv_param = {"z_s": z_s, "z_rcv": z_rcv, "r_rcv": r_rcv, "r0": r0, "d12": d12}
    return src_rcv_param


# ======================================================================================================================
# Sensibility study functions
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


def single_sensibility_test_generate_dataset(
    freq, c1, c2, rho1, rho2, depth, z_s, z_rcv, r_rcv, d12, model="kraken"
):
    if model == "kraken":
        freq, r_grid, gamma = single_sensibility_test_generate_dataset_kraken(
            freq, c1, c2, rho1, rho2, depth, z_s, z_rcv, r_rcv, d12
        )
    elif model == "analytic":
        freq, r_grid, gamma = single_sensibility_test_generate_dataset_analytic(
            freq, c1, c2, rho1, rho2, depth, z_s, z_rcv, r_rcv, d12
        )

    return freq, r_grid, gamma


def build_kraken(freq, c1, c2, rho1, rho2, d, z_s, z, r_grid, d12):

    # Keeps only frequencies above mode 1 cut-off
    kraken_freq = freq[freq > pekeris_cutoff_frequency(m=1, c1=c1, c2=c2, d=d)]
    # Use only propative modes
    nb_modes = pekeris_n_modes(f=freq.max(), c1=c1, c2=c2, d=d)
    clim_max = 2000

    # ----------------------------------------------------------------------
    # 1. Environment: Pekeris waveguide
    # ----------------------------------------------------------------------
    medium = KrakenMedium(
        ssp_interpolation_method="C_linear",
        z_ssp=[0.0, d],
        c_p=[c1, c1],  # isovelocity water column
        rho=rho1,
    )

    bottom_hs = KrakenBottomHalfspace(
        halfspace_properties={
            "z": d,
            "c_p": c2,
            "c_s": 0.0,  # fluid sediment: no shear waves
            "rho": rho2,
            "a_p": 0.2,  # dB/wavelength    # TODO pass as param
            "a_s": 0.0,  # fluid sediment: no shear waves
        },
        add_sediment_buffer_layer=False,  # direct half-space -> classic Pekeris model
        # fmin=kraken_freq.min(),
        # alpha_wavelength=10,
        # add_sediment_buffer_layer=True,
    )

    n_rcv_z = default_nb_rcv_z(fmax=freq.max(), max_depth=d, n_per_l=5)
    field = KrakenField(
        phase_speed_limits=[0, clim_max],
        src_depth=z_s,
        n_rcv_z=n_rcv_z,
        rcv_z_min=0.0,
        rcv_z_max=d,
        rcv_r_max=r_grid.max(),
    )

    env = KrakenEnv(
        title=TITLE,
        env_root=SENSIBILITY_DIRECTORY,
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

    return freq, g_fr, field_pos


def single_sensibility_test_generate_dataset_kraken(
    freq, c1, c2, rho1, rho2, d, z_s, z, r_grid, d12
):

    # Extend r_grid
    dr = r_grid[1] - r_grid[0]
    r_grid_add = np.arange(r_grid[-1] + dr, r_grid[-1] + dr + d12, dr)
    r_grid_ = np.append(r_grid, r_grid_add)

    # Convert r_grid to km for kraken
    r_grid_ = r_grid_ * 1e-3

    # Derive green's function at all pos
    freq, g_fr, field_pos = build_kraken(
        freq, c1, c2, rho1, rho2, d, z_s, z, r_grid_, d12
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


def single_sensibility_test_generate_dataset_analytic(
    freq, c1, c2, rho1, rho2, d, z_s, z, r_grid, d12
):

    # Extend r_grid
    dr = r_grid[1] - r_grid[0]
    r_grid_add = np.arange(r_grid[-1] + dr, r_grid[-1] + dr + d12, dr)
    r_grid_ = np.append(r_grid, r_grid_add)
    # Derive green's function at all pos
    g_fr = pekeris_green_fct(freq, c1, c2, rho1, rho2, d, z_s, z, r_grid_)

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


def single_sensibility_test_calc_dist(gamma, r_grid, r0):
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


def single_sensibility_test_calc_dist_width(dist_L1, dist_L2, dist_theta, r_grid, r0):

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


def run_single_sensibility_test(
    freq, c1, c2, rho1, rho2, attn2, depth, z_s, z_rcv, r_rcv, r0, d12, model="kraken"
):
    # 1) Generate dataset
    freq, r_grid, gamma = single_sensibility_test_generate_dataset(
        freq, c1, c2, rho1, rho2, depth, z_s, z_rcv, r_rcv, d12, model=model
    )
    # 2) Derive distance around r0
    dist_L1, dist_L2, dist_theta = single_sensibility_test_calc_dist(
        gamma=gamma, r_grid=r_grid, r0=r0
    )
    # 3) Derive characteristic metric
    width_L1, width_L2, width_theta = single_sensibility_test_calc_dist_width(
        dist_L1, dist_L2, dist_theta, r_grid=r_grid, r0=r0
    )

    return width_L1, width_L2, width_theta


def run_sensibility_study(test_arg_name, test_arg_values, all_arg_dict, model="kraken"):
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
        w_L1, w_L2, w_theta = run_single_sensibility_test(**all_args)

        width_L1.append(w_L1)
        width_L2.append(w_L2)
        width_theta.append(w_theta)

    return np.array(width_L1), np.array(width_L2), np.array(width_theta)


if __name__ == "__main__":
    pass
