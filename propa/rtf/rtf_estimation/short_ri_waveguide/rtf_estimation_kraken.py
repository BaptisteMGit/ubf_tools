#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_estimation_kraken.py
@Time    :   2024/10/17 09:15:19
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.interpolate as sp_int

from misc import *
from propa.rtf.ideal_waveguide import *
from propa.kraken_toolbox.run_kraken import runkraken
from propa.rtf.ideal_waveguide import waveguide_params
from localisation.verlinden.testcases.testcase_envs import TestCase1_0
from real_data_analysis.real_data_utils import get_csdm_snapshot_number

from propa.rtf.rtf_utils import (
    D_frobenius,
    D_hermitian_angle,
    true_rtf,
    interp_true_rtf,
)
from propa.rtf.rtf_estimation_const import *
from propa.rtf.rtf_estimation.rtf_estimation_utils import *
from propa.rtf.rtf_estimation.rtf_estimation_plot_tools import *

TAU_IR = 5  # Impulse response duration
N_RCV = 5  # Number of receivers


def derive_kraken_tf():

    # Load params
    depth, r_src, z_src, z_rcv, _ = waveguide_params()

    # Run kraken
    t0 = time()
    f, ts, h_kraken_dict = run_kraken_simulation(r_src, z_src, z_rcv, depth)
    print("Broadband kraken: ", time() - t0)

    # Some nan values can appear in the transfert function
    # h_kraken = np.nan_to_num(h_kraken)
    for i in range(N_RCV):
        h_kraken_dict[f"rcv{i}"] = np.nan_to_num(h_kraken_dict[f"rcv{i}"])
        h_kraken = h_kraken_dict[f"rcv{i}"]

        # Save transfert function as a csv
        fpath = os.path.join(ROOT_DATA, f"kraken_tf_rcv{i}.csv")
        np.savetxt(fpath, np.array([f, h_kraken.real, h_kraken.imag]).T, delimiter=",")

        ir_kraken = fft.irfft(h_kraken)
        t_kraken = np.arange(0, len(ir_kraken)) * ts

        # Save kraken ir
        fpath = os.path.join(ROOT_DATA, f"kraken_ir_rcv{i}.csv")
        np.savetxt(fpath, np.array([t_kraken, ir_kraken]).T, delimiter=",")


def derive_kraken_tf_surface_noise():

    # Load params
    depth, r_src, z_src, z_rcv, _ = waveguide_params()

    # Run kraken
    t0 = time()
    f, ts, r, h_kraken_surface_noise = run_kraken_simulation_surface_noise(
        r_src, z_src, z_rcv, depth
    )
    print("Broadband kraken: ", time() - t0)

    # Define xarray dataset for the transfert function
    h_kraken_surface_noise_xr = xr.Dataset(
        data_vars=dict(
            tf_real=(["f", "r"], np.real(h_kraken_surface_noise)),
            tf_imag=(["f", "r"], np.imag(h_kraken_surface_noise)),
        ),
        coords={"f": f, "r": r},
    )
    # Save transfert function as a csv
    fpath = os.path.join(ROOT_DATA, "kraken_tf_surface_noise.nc")
    h_kraken_surface_noise_xr.to_netcdf(fpath)


def derive_kraken_tf_loose_grid():

    # Load params
    depth, r_src, z_src, z_rcv, _ = waveguide_params()

    # Run kraken
    f, ts, r, z, h_kraken_loose_grid = run_kraken_simulation_loose_grid(
        r_src, z_src, z_rcv, depth
    )

    # Define xarray dataset for the transfert function
    h_kraken_surface_noise_xr = xr.Dataset(
        data_vars=dict(
            tf_real=(
                ["f", "z", "r"],
                np.real(h_kraken_loose_grid),
            ),
            tf_imag=(["f", "z", "r"], np.imag(h_kraken_loose_grid)),
        ),
        coords={"f": f, "r": r, "z": z},
    )
    # Save transfert function as a csv
    fpath = os.path.join(ROOT_DATA, "kraken_tf_loose_grid.nc")
    h_kraken_surface_noise_xr.to_netcdf(fpath)


def run_kraken_simulation(r_src, z_src, z_rcv, depth):

    delta_rcv = 500
    x_rcv = np.array([i * delta_rcv for i in range(N_RCV)])
    r_src_rcv = r_src - x_rcv

    # Create the frequency vector
    duration = 50 * TAU_IR
    ts = 1e-2
    nt = int(duration / ts)
    f = fft.rfftfreq(nt, ts)

    # Init env
    bott_hs_properties = testcase_bottom_properties()

    tc_varin = {
        "freq": f,
        "src_depth": z_src,
        "max_range_m": r_src,
        "mode_theory": "adiabatic",
        "flp_n_rcv_z": 1,
        "flp_rcv_z_min": z_rcv,
        "flp_rcv_z_max": z_rcv,
        "min_depth": depth,
        "max_depth": depth,
        "dr_flp": delta_rcv,
        "nb_modes": 200,
        "bottom_boundary_condition": "acousto_elastic",
        "nmedia": 2,
        "phase_speed_limits": [200, 20000],
        "bott_hs_properties": bott_hs_properties,
    }
    tc = TestCase1_0(mode="prod", testcase_varin=tc_varin)
    title = "Simple waveguide with short impulse response"
    tc.title = title
    tc.env_dir = os.path.join(ROOT_FOLDER, "tmp")
    tc.update(tc_varin)

    # For too long frequencies vector field fails to compute -> we will iterate over frequency subband to compute the transfert function
    fmax = 50
    fmin = cutoff_frequency(C0, depth, bottom_bc="pressure_release")
    n_subband = 500
    i_subband = 1
    f0 = fmin
    f1 = f[n_subband]
    # h_kraken = np.zeros_like(f, dtype=complex)
    h_kraken_dict = {f"rcv{i}": np.zeros_like(f, dtype=complex) for i in range(N_RCV)}

    while f0 < fmax:
        # Frequency subband
        f_kraken = f[(f < f1) & (f >= f0)]
        # print(i_subband, f0, f1, len(f_kraken))
        pad_before = np.sum(f < f0)
        pad_after = np.sum(f >= f1)

        # Update env
        varin_update = {"freq": f_kraken}
        tc.update(varin_update)

        pressure_field, field_pos = runkraken(
            env=tc.env,
            flp=tc.flp,
            frequencies=tc.env.freq,
            parallel=True,
            verbose=False,
        )

        idx_r = [
            np.argmin(np.abs(field_pos["r"]["r"] - r_src_rcv[i])) for i in range(N_RCV)
        ]
        h_kraken_subband = np.squeeze(pressure_field, axis=(1, 2, 3))[:, idx_r]
        # print(pad_before, pad_after)
        for i in range(N_RCV):
            # Zero padding of the transfert function to match the length of the global transfert function
            h_kraken_dict[f"rcv{i}"] += np.pad(
                h_kraken_subband[:, i], (pad_before, pad_after)
            )

        # Update frequency subband
        i_subband += 1
        f0 = f1
        f1 = f[min(n_subband * i_subband, len(f) - 1)]

    return f, ts, h_kraken_dict


def run_kraken_simulation_surface_noise(r_src, z_src, z_rcv, depth):

    # Noise source spacing (m): the minimal wavelength is 30m for f = 50Hz and c = 1500m/s, dr = 1m ensure at least 30 points per wavelength
    dr_noise = 10
    # z_src = 0.5  # Noise source just below surface
    rmax_noise = r_src + 10 * 1e3  # Maximal range for noise source

    # Reciprocity
    z_src = z_rcv
    z_rcv = 0.5  # Noise source just below surface

    delta_rcv = 500
    x_rcv = np.array([i * delta_rcv for i in range(N_RCV)])
    # r_src_rcv = r_src - x_rcv

    # Create the frequency vector
    duration = 50 * TAU_IR
    ts = 1e-2
    nt = int(duration / ts)
    f = fft.rfftfreq(nt, ts)

    bott_hs_properties = testcase_bottom_properties()

    tc_varin = {
        "freq": f,
        "src_depth": z_src,
        "max_range_m": rmax_noise,
        "mode_theory": "adiabatic",
        "flp_n_rcv_z": 1,
        "flp_rcv_z_min": z_rcv,
        "flp_rcv_z_max": z_rcv,
        "min_depth": depth,
        "max_depth": depth,
        "dr_flp": dr_noise,
        "nb_modes": 200,
        "bottom_boundary_condition": "acousto_elastic",
        "nmedia": 2,
        "phase_speed_limits": [200, 20000],
        "bott_hs_properties": bott_hs_properties,
    }
    tc = TestCase1_0(mode="prod", testcase_varin=tc_varin)
    title = "Simple waveguide with short impulse response"
    tc.title = title
    tc.env_dir = os.path.join(ROOT_FOLDER, "tmp")
    tc.update(tc_varin)

    # For too long frequencies vector field fails to compute -> we will iterate over frequency subband to compute the transfert function
    fmax = 50
    fmin = cutoff_frequency(C0, depth, bottom_bc="pressure_release")
    n_subband = 500
    i_subband = 1
    f0 = fmin
    f1 = f[n_subband]
    # h_kraken = np.zeros_like(f, dtype=complex)
    nr = int(rmax_noise / dr_noise + 1)
    nf = len(f)
    # h_kraken_dict = {f"rcv{i}": np.zeros((nf, nr), dtype=complex) for i in range(N_RCV)}
    h_kraken_surface_noise = np.zeros((nf, nr), dtype=complex)

    while f0 < fmax:
        # Frequency subband
        f_kraken = f[(f < f1) & (f >= f0)]
        # print(i_subband, f0, f1, len(f_kraken))
        pad_before = np.sum(f < f0)
        pad_after = np.sum(f >= f1)

        # Update env
        varin_update = {"freq": f_kraken}
        tc.update(varin_update)

        pressure_field, field_pos = runkraken(
            env=tc.env,
            flp=tc.flp,
            frequencies=tc.env.freq,
            parallel=True,
            verbose=True,
        )

        h_kraken_subband = np.squeeze(pressure_field, axis=(1, 2, 3))
        r = field_pos["r"]["r"]

        h_kraken_surface_noise += np.pad(
            h_kraken_subband, ((pad_before, pad_after), (0, 0))
        )

        # Update frequency subband
        i_subband += 1
        f0 = f1
        f1 = f[min(n_subband * i_subband, len(f) - 1)]

    return f, ts, r, h_kraken_surface_noise


def run_kraken_simulation_loose_grid(r_src, z_src, z_rcv, depth):

    # Noise source spacing (m): the minimal wavelength is 30m for f = 50Hz and c = 1500m/s, dr = 1m ensure at least 30 points per wavelength
    dr_loose_grid = 100
    rmax = r_src + 20 * 1e3  # Maximal range

    delta_rcv = 500
    x_rcv = np.array([i * delta_rcv for i in range(N_RCV)])
    r_src_rcv = r_src - x_rcv

    # Reciprocity
    z_src = z_rcv
    # Potential interferers located near the surface
    dz_loose_grid = 1
    z_min = 1
    z_max = 50
    nz = int((z_max - z_min) / dz_loose_grid) + 1

    # Create the frequency vector
    duration = 50 * TAU_IR
    ts = 1e-2
    nt = int(duration / ts)
    f = fft.rfftfreq(nt, ts)

    # Init env
    bott_hs_properties = testcase_bottom_properties()

    tc_varin = {
        "freq": f,
        "src_depth": z_src,
        "max_range_m": rmax,
        "mode_theory": "adiabatic",
        "flp_n_rcv_z": nz,
        "flp_rcv_z_min": z_min,
        "flp_rcv_z_max": z_max,
        "min_depth": depth,
        "max_depth": depth,
        "dr_flp": dr_loose_grid,
        "nb_modes": 200,
        "bottom_boundary_condition": "acousto_elastic",
        "nmedia": 2,
        "phase_speed_limits": [200, 20000],
        "bott_hs_properties": bott_hs_properties,
    }
    tc = TestCase1_0(mode="prod", testcase_varin=tc_varin)
    title = "Simple waveguide with short impulse response"
    tc.title = title
    tc.env_dir = os.path.join(ROOT_FOLDER, "tmp")
    tc.update(tc_varin)

    # For too long frequencies vector field fails to compute -> we will iterate over frequency subband to compute the transfert function
    fmax = 50
    fmin = cutoff_frequency(C0, depth, bottom_bc="pressure_release")
    n_subband = 500
    i_subband = 1
    f0 = fmin
    f1 = f[n_subband]
    # h_kraken = np.zeros_like(f, dtype=complex)
    nr = int(rmax / dr_loose_grid + 1)
    nf = len(f)
    h_kraken_loose_grid = np.zeros((nf, nz, nr), dtype=complex)

    while f0 < fmax:
        # Frequency subband
        f_kraken = f[(f < f1) & (f >= f0)]
        # print(i_subband, f0, f1, len(f_kraken))
        pad_before = np.sum(f < f0)
        pad_after = np.sum(f >= f1)

        # Update env
        varin_update = {"freq": f_kraken}
        tc.update(varin_update)

        pressure_field, field_pos = runkraken(
            env=tc.env,
            flp=tc.flp,
            frequencies=tc.env.freq,
            parallel=True,
            verbose=True,
        )

        h_kraken_subband = np.squeeze(pressure_field, axis=(1, 2))
        r = field_pos["r"]["r"]
        z = field_pos["r"]["z"]

        h_kraken_loose_grid += np.pad(
            h_kraken_subband, ((pad_before, pad_after), (0, 0), (0, 0))
        )

        # Update frequency subband
        i_subband += 1
        f0 = f1
        f1 = f[min(n_subband * i_subband, len(f) - 1)]

    return f, ts, r, z, h_kraken_loose_grid


# ======================================================================================================================
# Test cases
# ======================================================================================================================


def testcase_bottom_properties():
    # Environment with properties to minimize the impulse response duration
    bott_hs_properties = {
        "rho": 1.5 * RHO_W * 1e-3,  # Density (g/cm^3)
        # "c_p": 1500,  # P-wave celerity (m/s)
        "c_p": 1550,  # P-wave celerity (m/s)
        "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
        "a_p": 0.2,  # Compression wave attenuation (dB/wavelength)
        "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
        "z": None,
    }

    return bott_hs_properties


def testcase_1_unpropagated_whitenoise(snr_dB=10, plot=True):
    """
    Test case 1
        - Waveguide: simple waveguide with short impulse response.
        - Signal: ship signal propagated through the waveguide using Kraken.
        - Noise: independent white gaussian noise on each receiver.
        - RTF estimation: covariance substraction and covariance whitening methods.

    Args:
        snr_dB (int, optional): Signal-to-noise ratio in dB. Defaults to 10.
    """

    # Load propagated signal
    _, r_src, _, _, _ = waveguide_params()
    direct_delay = r_src / C0
    rcv_sig_data = derive_received_signal(tau_ir=TAU_IR, delay_correction=direct_delay)
    t = rcv_sig_data["t"]

    # Load noise
    ns = len(t)
    fs = 1 / (t[1] - t[0])
    rcv_noise_data = derive_received_noise(
        ns,
        fs,
        propagated=False,
        noise_model="gaussian",
        snr_dB=snr_dB,
        sigma_ref=np.std(rcv_sig_data[f"rcv{0}"]["sig"]),
    )

    # Convert to array
    rcv_sig = np.empty((ns, N_RCV))
    rcv_noise = np.empty((ns, N_RCV))
    for i in range(N_RCV):
        id_rcv = f"rcv{i}"
        rcv_sig[:, i] = rcv_sig_data[id_rcv]["sig"]
        rcv_noise[:, i] = rcv_noise_data[id_rcv]["sig"]

    alpha_tau_ir = 3
    seg_length = alpha_tau_ir * TAU_IR
    nperseg = int(seg_length / (t[1] - t[0]))
    # Find the nearest power of 2
    nperseg = 2 ** int(np.log2(nperseg) + 1)
    alpha_overlap = 1 / 2
    noverlap = int(nperseg * alpha_overlap)

    print(f"nperseg = {nperseg}, noverlap = {noverlap}")

    # Estimate RTF using covariance substraction method
    f_cs, rtf_cs, Rx, Rs, Rv = rtf_covariance_substraction(
        t, rcv_sig, rcv_noise, nperseg=nperseg, noverlap=noverlap
    )

    f_cw, rtf_cw, _, _, _ = rtf_covariance_whitening(
        t, rcv_sig, rcv_noise, nperseg=nperseg, noverlap=noverlap
    )

    # Create folder to save results
    tc_folder = os.path.join(
        ROOT_FOLDER, "testcase_1_unpropagated_whitenoise", f"snr_{snr_dB}dB"
    )

    # Set properties to pass to the plotting functions
    fig_props = {
        "folder_path": tc_folder,
        "L": get_csdm_snapshot_number(
            rcv_sig[:, 0], rcv_sig_data["fs"], nperseg, noverlap
        ),
        "alpha_tau_ir": alpha_tau_ir,
        "alpha_overlap": alpha_overlap,
        "tau_ir": TAU_IR,
    }

    # Plot estimation results
    if plot:
        if not os.path.exists(tc_folder):
            os.makedirs(tc_folder)

        plot_signal_components(fig_props, t, rcv_sig, rcv_noise)
        mean_Rx, mean_Rs, mean_Rv = plot_mean_csdm(fig_props, Rx, Rs, Rv)
        plot_rtf_estimation(fig_props, f_cs, rtf_cs, f_cw, rtf_cw)
        compare_rtf_vs_received_spectrum(
            fig_props, f_cs, rtf_cs, f_cw, rtf_cw, rcv_signal=rcv_sig_data
        )

        plt.close("all")

    testcase_results = {
        "cs": {
            "f": f_cs,
            "rtf": rtf_cs,
        },
        "cw": {
            "f": f_cw,
            "rtf": rtf_cw,
        },
        "signal": rcv_sig_data,
        "noise": rcv_noise_data,
        "Rx": Rx,
        "Rs": Rs,
        "Rv": Rv,
        "props": fig_props,
        "tc_name": "Testcase 1",
        "tc_label": "testcase_1_unpropagated_whitenoise",
    }

    return testcase_results


def testcase_2_propagated_whitenoise(snr_dB=10, plot=True):
    """
    Test case 2
        - Waveguide: simple waveguide with short impulse response.
        - Signal: ship signal propagated through the waveguide using Kraken.
        - Noise: gaussian noise from a set of multiple sources propagated through the waveguide.
        - RTF estimation: covariance substraction and covariance whitening methods.
    """

    # Load propagated signal
    _, r_src, _, _, _ = waveguide_params()
    direct_delay = r_src / C0
    rcv_sig_data = derive_received_signal(tau_ir=TAU_IR, delay_correction=direct_delay)
    t = rcv_sig_data["t"]

    # Load propagated noise from multiple sources
    ns = len(t)
    fs = 1 / (t[1] - t[0])
    propagated_args = {"rmin": 0}
    rcv_noise_data = derive_received_noise(
        ns,
        fs,
        propagated=True,
        noise_model="gaussian",
        snr_dB=snr_dB,
        propagated_args=propagated_args,
        sigma_ref=np.std(rcv_sig_data[f"rcv{0}"]["sig"]),
    )

    # Convert to array
    rcv_sig = np.empty((ns, N_RCV))
    rcv_noise = np.empty((ns, N_RCV))
    for i in range(N_RCV):
        id_rcv = f"rcv{i}"
        rcv_sig[:, i] = rcv_sig_data[id_rcv]["sig"]
        rcv_noise[:, i] = rcv_noise_data[id_rcv]["sig"]

    alpha_tau_ir = 3
    seg_length = alpha_tau_ir * TAU_IR
    nperseg = int(seg_length / (t[1] - t[0]))
    # Find the nearest power of 2
    nperseg = 2 ** int(np.log2(nperseg) + 1)
    alpha_overlap = 1 / 2
    noverlap = int(nperseg * alpha_overlap)

    # print(f"nperseg = {nperseg}, noverlap = {noverlap}")

    # Estimate RTF using covariance substraction method
    f_cs, rtf_cs, Rx, Rs, Rv = rtf_covariance_substraction(
        t, rcv_sig, rcv_noise, nperseg=nperseg, noverlap=noverlap
    )
    f_cw, rtf_cw, _, _, _ = rtf_covariance_whitening(
        t, rcv_sig, rcv_noise, nperseg=nperseg, noverlap=noverlap
    )

    # Set properties to pass to the plotting functions
    # Create folder to save results
    tc_folder = os.path.join(
        ROOT_FOLDER, "testcase_2_propagated_whitenoise", f"snr_{snr_dB}dB"
    )

    fig_props = {
        "folder_path": tc_folder,
        "L": get_csdm_snapshot_number(
            rcv_sig[:, 0], rcv_sig_data["fs"], nperseg, noverlap
        ),
        "alpha_tau_ir": alpha_tau_ir,
        "alpha_overlap": alpha_overlap,
        "tau_ir": TAU_IR,
    }
    # Plot estimation results
    if plot:
        if not os.path.exists(tc_folder):
            os.makedirs(tc_folder)

        plot_signal_components(fig_props, t, rcv_sig, rcv_noise)
        mean_Rx, mean_Rs, mean_Rv = plot_mean_csdm(fig_props, Rx, Rs, Rv)
        plot_rtf_estimation(fig_props, f_cs, rtf_cs, f_cw, rtf_cw)
        compare_rtf_vs_received_spectrum(
            fig_props, f_cs, rtf_cs, f_cw, rtf_cw, rcv_signal=rcv_sig_data
        )
        plt.close("all")

    testcase_results = {
        "cs": {
            "f": f_cs,
            "rtf": rtf_cs,
        },
        "cw": {
            "f": f_cw,
            "rtf": rtf_cw,
        },
        "signal": rcv_sig,
        "noise": rcv_noise,
        "Rx": Rx,
        "Rs": Rs,
        "Rv": Rv,
        "props": fig_props,
        "tc_name": "Testcase 2",
        "tc_label": "testcase_2_propagated_whitenoise",
    }

    return testcase_results


def testcase_3_propagated_interference(snr_dB=0, plot=True, interference_type="z_call"):
    """
    Test case 2
        - Waveguide: simple waveguide with short impulse response.
        - Signal: ship signal propagated through the waveguide using Kraken.
        - Noise: interference signal propagated through the waveguide.
        - RTF estimation: covariance substraction and covariance whitening methods.
    """

    # Load propagated signal
    _, r_src, _, _, _ = waveguide_params()
    direct_delay = r_src / C0
    t0 = time()
    rcv_sig_data = derive_received_signal(tau_ir=TAU_IR, delay_correction=direct_delay)
    print(f"derive_received_signal: {time() - t0:.2f} s")
    t = rcv_sig_data["t"]

    # Load propagated noise from multiple sources
    ns = len(t)
    fs = 1 / (t[1] - t[0])

    interferer_r = [5 * 1e3, 15 * 1e3, 50 * 1e3]
    # interferer_z = [30, 25]
    # interferer_r = np.arange(5, 50, 5) * 1e3
    interferer_z = [25, 35, 30]

    n_src = len(interferer_r)
    interference_arg = {
        "signal_type": interference_type,
        "src_position": {
            "r": interferer_r,
            "z": interferer_z,
        },
    }
    t0 = time()
    rcv_interference_data = derive_received_interference(ns, fs, interference_arg)
    print(f"derive_received_interference: {time() - t0:.2f} s")

    additive_noise = derive_received_noise(
        ns,
        fs,
        propagated=False,
        noise_model="gaussian",
        snr_dB=snr_dB,
        sigma_ref=np.std(rcv_sig_data[f"rcv{0}"]["sig"]),
    )

    # Convert to array
    rcv_sig = np.empty((ns, N_RCV))
    rcv_noise = np.empty((ns, N_RCV))
    for i in range(N_RCV):
        id_rcv = f"rcv{i}"
        rcv_sig[:, i] = rcv_sig_data[id_rcv]["sig"]
        rcv_noise[:, i] = (
            rcv_interference_data[id_rcv]["sig"] + additive_noise[id_rcv]["sig"]
        )

    alpha_tau_ir = 3
    seg_length = alpha_tau_ir * TAU_IR
    nperseg = int(seg_length / (t[1] - t[0]))
    # Find the nearest power of 2
    nperseg = 2 ** int(np.log2(nperseg) + 1)
    alpha_overlap = 1 / 2
    noverlap = int(nperseg * alpha_overlap)

    # print(f"nperseg = {nperseg}, noverlap = {noverlap}")

    # Estimate RTF using covariance substraction method
    t0 = time()
    f_cs, rtf_cs, Rx, Rs, Rv = rtf_covariance_substraction(
        t, rcv_sig, rcv_noise, nperseg=nperseg, noverlap=noverlap
    )
    print(f"rtf_covariance_substraction: {time() - t0:.2f} s")
    t0 = time()
    f_cw, rtf_cw, _, _, _ = rtf_covariance_whitening(
        t, rcv_sig, rcv_noise, nperseg=nperseg, noverlap=noverlap
    )
    print(f"rtf_covariance_whitening: {time() - t0:.2f} s")

    # Set properties to pass to the plotting functions
    # Create folder to save results
    tc_folder = os.path.join(
        ROOT_FOLDER,
        "testcase_3_propagated_interference",
        f"{interference_type}_{n_src}_src",
        f"snr_{snr_dB}dB",
    )

    fig_props = {
        "folder_path": tc_folder,
        "L": get_csdm_snapshot_number(
            rcv_sig[:, 0], rcv_sig_data["fs"], nperseg, noverlap
        ),
        "alpha_tau_ir": alpha_tau_ir,
        "alpha_overlap": alpha_overlap,
        "tau_ir": TAU_IR,
    }
    # Plot estimation results
    if plot:
        if not os.path.exists(tc_folder):
            os.makedirs(tc_folder)

        t0 = time()
        plot_signal_components(fig_props, t, rcv_sig, rcv_noise, rcv_idx_to_plot=0)
        print(f"plot_signal_components: {time() - t0:.2f} s")
        mean_Rx, mean_Rs, mean_Rv = plot_mean_csdm(fig_props, Rx, Rs, Rv)
        plot_rtf_estimation(fig_props, f_cs, rtf_cs, f_cw, rtf_cw)
        compare_rtf_vs_received_spectrum(
            fig_props, f_cs, rtf_cs, f_cw, rtf_cw, rcv_signal=rcv_sig_data
        )
        # plt.show()
        plt.close("all")

    testcase_results = {
        "cs": {
            "f": f_cs,
            "rtf": rtf_cs,
        },
        "cw": {
            "f": f_cw,
            "rtf": rtf_cw,
        },
        "signal": rcv_sig,
        "noise": rcv_noise,
        "Rx": Rx,
        "Rs": Rs,
        "Rv": Rv,
        "props": fig_props,
        "tc_name": "Testcase 3",
        "tc_label": "testcase_3_propagated_interference",
    }

    return testcase_results


def dist_versus_snr(snrs, testcase=1, dist="frobenius"):

    # Select dist function to apply
    if dist == "frobenius":
        dist_func = D_frobenius
        dist_kwargs = {}
    elif dist == "hermitian_angle":
        dist_func = D_hermitian_angle
        dist_kwargs = {
            "unit": "deg",
            "apply_mean": True,
        }

    # Derive results for each snr
    rtf_cs = []
    rtf_cw = []
    for i_snr, snr_dB in enumerate(snrs):
        plot = False
        # plot = i_snr % 100 == 0
        print(f"i = {i_snr}, snr = {snr_dB}, plot = {plot}")
        if testcase == 1:
            # plot = False
            res_snr = testcase_1_unpropagated_whitenoise(snr_dB=snr_dB, plot=plot)
        elif testcase == 2:
            res_snr = testcase_2_propagated_whitenoise(snr_dB=snr_dB, plot=plot)
        elif testcase == 3:
            res_snr = testcase_3_propagated_interference(
                snr_dB=snr_dB, plot=plot, interference_type="z_call"
            )

        # Save rtfs into dedicated list
        rtf_cs.append(res_snr["cs"]["rtf"])
        rtf_cw.append(res_snr["cw"]["rtf"])

    f_cs = res_snr["cs"]["f"]
    f_cw = res_snr["cw"]["f"]
    f = f_cs

    # Load true RTF
    kraken_data = load_data()
    _, rtf_true_interp = interp_true_rtf(kraken_data, f)

    dist_cs = []
    dist_cw = []
    dist_cs_band = []
    dist_cw_band = []
    dist_cs_band_smooth = []
    dist_cw_band_smooth = []
    for i in range(len(snrs)):
        rtf_cs_i = rtf_cs[i]
        rtf_cw_i = rtf_cw[i]
        # Derive distance between estimated rtf and true rtf
        d_cs = dist_func(rtf_true_interp, rtf_cs_i, **dist_kwargs)
        d_cw = dist_func(rtf_true_interp, rtf_cw_i, **dist_kwargs)

        # Append to list
        dist_cs.append(d_cs)
        dist_cw.append(d_cw)

        # Same distance derivation for a restricted frequency band
        fmin_rtf = 5
        fmax_rtf = 50

        rtf_cs_i_band = rtf_cs_i[(f_cs >= fmin_rtf) & (f_cs <= fmax_rtf)]
        rtf_cw_i_band = rtf_cw_i[(f_cw >= fmin_rtf) & (f_cw <= fmax_rtf)]
        rtf_true_interp_band = rtf_true_interp[(f >= fmin_rtf) & (f <= fmax_rtf)]

        d_cs_band = dist_func(rtf_true_interp_band, rtf_cs_i_band, **dist_kwargs)
        d_cw_band = dist_func(rtf_true_interp_band, rtf_cw_i_band, **dist_kwargs)

        # Append to list
        dist_cs_band.append(d_cs_band)
        dist_cw_band.append(d_cw_band)

        # Distance for smoothed rtf
        window = 5
        rtf_cs_i_band_smooth = np.zeros_like(rtf_cs_i_band)
        rtf_cw_i_band_smooth = np.zeros_like(rtf_cw_i_band)
        for i in range(kraken_data["n_rcv"]):
            rtf_cs_i_band_smooth[:, i] = np.convolve(
                np.abs(rtf_cs_i_band[:, i]), np.ones(window) / window, mode="same"
            )
            rtf_cw_i_band_smooth[:, i] = np.convolve(
                np.abs(rtf_cw_i_band[:, i]), np.ones(window) / window, mode="same"
            )

        d_cs_band_smooth = dist_func(
            rtf_true_interp_band, rtf_cs_i_band_smooth, **dist_kwargs
        )
        d_cw_band_smooth = dist_func(
            rtf_true_interp_band, rtf_cw_i_band_smooth, **dist_kwargs
        )

        # Append to list
        dist_cs_band_smooth.append(d_cs_band_smooth)
        dist_cw_band_smooth.append(d_cw_band_smooth)

    # Plot distance versus snr
    props = res_snr["props"]
    title = (
        r"$\textrm{"
        + res_snr["tc_name"]
        + r"}\,"
        + " - "
        + " ["
        + f"{0}, {50}"
        + r"] \, \textrm{Hz}$"
        + f"\n({csdm_info_line(props)})"
    )
    fpath = os.path.join(ROOT_FOLDER, res_snr["tc_label"], "Df.png")
    plot_dist_vs_snr(
        snrs, dist_cs, dist_cw, title=title, dist_type=dist, savepath=fpath
    )
    # plt.figure()
    # plt.plot(snrs, 10 * np.log10(dist_cs), marker=".", label=r"$\mathcal{D}_F^{(CS)}$")
    # plt.plot(snrs, 10 * np.log10(dist_cw), marker=".", label=r"$\mathcal{D}_F^{(CW)}$")
    # plt.ylabel(r"$\mathcal{D}_F\, \textrm{[dB]}$")
    # plt.xlabel(r"$\textrm{snr} \, \textrm{[dB]}$")
    # plt.legend()
    # plt.grid()
    # plt.title(title)

    # # Save
    # plt.savefig(fpath)

    title = (
        r"$\textrm{"
        + res_snr["tc_name"]
        + r"}\,"
        + " - "
        + " ["
        + f"{fmin_rtf}, {fmax_rtf}"
        + r"] \, \textrm{Hz}$"
        + f"\n({csdm_info_line(props)})"
    )
    fpath = os.path.join(
        ROOT_FOLDER, res_snr["tc_label"], f"Df_band_{fmin_rtf}_{fmax_rtf}.png"
    )
    plot_dist_vs_snr(
        snrs, dist_cs_band, dist_cw_band, title=title, dist_type=dist, savepath=fpath
    )
    # plt.figure()
    # plt.plot(
    #     snrs, 10 * np.log10(dist_cs_band), marker=".", label=r"$\mathcal{D}_F^{(CS)}$"
    # )
    # plt.plot(
    #     snrs, 10 * np.log10(dist_cw_band), marker=".", label=r"$\mathcal{D}_F^{(CW)}$"
    # )
    # plt.ylabel(r"$\mathcal{D}_F\, \textrm{[dB]}$")  # TODO check unity  \textrm{[dB]}
    # plt.xlabel(r"$\textrm{snr} \, \textrm{[dB]}$")
    # plt.title(title)
    # plt.legend()
    # plt.grid()

    # Save
    # plt.savefig(fpath)

    title = (
        r"$\textrm{"
        + res_snr["tc_name"]
        + r"}\,"
        + " - "
        + " ["
        + f"{fmin_rtf}, {fmax_rtf}"
        + r"] \, \textrm{Hz}"
        + " - "
        + r"\textrm{smooth} \,"
        + f"(n = {window})$"
        + f"\n({csdm_info_line(props)})"
    )
    fpath = os.path.join(
        ROOT_FOLDER, res_snr["tc_label"], f"Df_band_{fmin_rtf}_{fmax_rtf}_smooth.png"
    )
    plot_dist_vs_snr(
        snrs,
        dist_cs_band_smooth,
        dist_cw_band_smooth,
        title=title,
        dist_type=dist,
        savepath=fpath,
    )
    # plt.figure()
    # plt.plot(
    #     snrs,
    #     10 * np.log10(dist_cs_band_smooth),
    #     marker=".",
    #     label=r"$\mathcal{D}_F^{(CS)}$",
    # )
    # plt.plot(
    #     snrs,
    #     10 * np.log10(dist_cw_band_smooth),
    #     marker=".",
    #     label=r"$\mathcal{D}_F^{(CW)}$",
    # )
    # plt.ylabel(r"$\mathcal{D}_F\, \textrm{[dB]}$")  # TODO check unity  \textrm{[dB]}
    # plt.xlabel(r"$\textrm{snr} \, \textrm{[dB]}$")
    # plt.title(title)
    # plt.legend()
    # plt.grid()

    # # Save

    # plt.savefig(fpath)


def check_interp():

    res_snr = testcase_2_propagated_whitenoise(snr_dB=0, plot=False)
    f_cs = res_snr["cs"]["f"]
    rtf_cs = res_snr["cs"]["rtf"]

    # Load true RTF
    kraken_data = load_data()
    f_true, rtf_true = true_rtf(kraken_data)
    rtf_true = rtf_true[:, -1]
    # Interpolate rtf_true to f_cs / f_cw
    interp_real = sp_int.interp1d(f_true, np.real(rtf_true))
    interp_imag = sp_int.interp1d(f_true, np.imag(rtf_true))
    rtf_true_interp = interp_real(f_cs) + 1j * interp_imag(f_cs)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(16, 10))
    axs[0].plot(f_true, np.abs(rtf_true))
    axs[0].scatter(f_cs, np.abs(rtf_true_interp), color="r", s=3)

    axs[1].plot(f_true, np.angle(rtf_true))
    axs[1].scatter(f_cs, np.angle(rtf_true_interp), color="r", s=3)

    plt.xlabel("f")


if __name__ == "__main__":
    # derive_kraken_tf()
    # derive_kraken_tf_surface_noise()
    # derive_kraken_tf_loose_grid()
    # kraken_data = load_data()
    # plot_ir(kraken_data, shift_ir=True)
    # plot_tf(kraken_data)

    # xr_surfnoise = xr.open_dataset(
    #     r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_estimation\short_ri_waveguide\data\kraken_tf_surface_noise.nc"
    # )
    # xr_surfnoise_rcv = xr.open_dataset(
    #     r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_estimation\short_ri_waveguide\data\kraken_tf_noise_rcv0.nc"
    # )

    # rcv_sig = derive_received_signal()
    # plot_signal(tau_ir=TAU_IR)
    # snrs = [-25, 25]
    # for snr_dB in snrs:
    #     testcase_1_unpropagated_whitenoise(snr_dB=snr_dB)

    # res = testcase_1_unpropagated_whitenoise(snr_dB=0)
    # compare_rtf_vs_received_spectrum(
    #     res["props"],
    #     res["cs"]["f"],
    #     res["cs"]["rtf"],
    #     res["cw"]["f"],
    #     res["cw"]["rtf"],
    #     rcv_signal=res["signal"],
    # )

    # testcase_2_propagated_whitenoise(snr_dB=0)
    # testcase_3_propagated_interference(plot=True, interference_type="dirac")
    # testcase_3_propagated_interference(snr_dB=0, plot=True, interference_type="z_call")
    # testcase_3_propagated_interference(snr_dB=10, plot=True, interference_type="z_call")
    # testcase_3_propagated_interference(plot=True, interference_type="ricker_pulse")

    # check_interp()
    snrs = [0, 10]
    # snrs = np.round(np.arange(-50, 50, 0.1), 1)
    # snrs = np.arange(-5, 5, 1)

    dist_versus_snr(snrs, testcase=1, dist="hermitian_angle")
    dist_versus_snr(snrs, testcase=2, dist="hermitian_angle")
    dist_versus_snr(snrs, testcase=3, dist="hermitian_angle")

    # dist_versus_snr(snrs, testcase=2)

    plt.show()
