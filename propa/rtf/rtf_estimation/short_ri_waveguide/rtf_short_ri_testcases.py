#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_estimation_testcases.py
@Time    :   2024/11/04 13:59:38
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
from propa.rtf.ideal_waveguide import waveguide_params
from propa.rtf.rtf_estimation.rtf_estimation_utils import *
from propa.rtf.rtf_estimation.rtf_estimation_plot_tools import *
from real_data_analysis.real_data_utils import get_csdm_snapshot_number
from propa.rtf.rtf_estimation.short_ri_waveguide.rtf_short_ri_consts import *
from propa.rtf.rtf_estimation.short_ri_waveguide.rtf_short_ri_kraken import (
    load_data,
    derive_received_signal,
    derive_received_noise,
    derive_received_interference,
)

# ======================================================================================================================
# Test cases
# ======================================================================================================================


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
    _, r_src, z_src, z_rcv, _ = waveguide_params()
    d = np.sqrt(r_src**2 + (z_rcv - z_src) ** 2)
    direct_delay = d / C0
    # print(f"direct_delay = {direct_delay}")

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

    # print(f"nperseg = {nperseg}, noverlap = {noverlap}")

    # Estimate RTF using covariance substraction method
    f_cs, rtf_cs, Rx, Rs, Rv = rtf_covariance_substraction(
        t, rcv_sig, rcv_noise, nperseg=nperseg, noverlap=noverlap
    )

    # Estimate RTF using covariance whitening method
    f_cw, rtf_cw, _, _, _ = rtf_covariance_whitening(
        t, rcv_sig, rcv_noise, nperseg=nperseg, noverlap=noverlap
    )

    # Create folder to save results
    tc_folder = os.path.join(
        ROOT_IMG, "testcase_1_unpropagated_whitenoise", f"snr_{snr_dB}dB"
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

        kraken_data = load_data()
        # Plot signal components
        plot_signal_components(fig_props, t, rcv_sig, rcv_noise)
        # Plot mean CSDM
        mean_Rx, mean_Rs, mean_Rv = plot_mean_csdm(fig_props, Rx, Rs, Rv)
        # Plot RTF estimation
        plot_rtf_estimation(fig_props, kraken_data, f_cs, rtf_cs, f_cw, rtf_cw)
        # Plot distance and SNR comparison
        compare_rtf_vs_received_spectrum(
            fig_props,
            kraken_data,
            f_cs,
            rtf_cs,
            f_cw,
            rtf_cw,
            rcv_signal=rcv_sig_data,
            rcv_noise=rcv_noise_data,
        )

        # Plot hermitian angle distribution
        plot_rtf_estimation_hermitian_angle_distribution(
            fig_props,
            kraken_data,
            f_cs,
            rtf_cs,
            rtf_cw,
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
    _, r_src, z_src, z_rcv, _ = waveguide_params()
    d = np.sqrt(r_src**2 + (z_rcv - z_src) ** 2)
    direct_delay = d / C0
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
        ROOT_IMG, "testcase_2_propagated_whitenoise", f"snr_{snr_dB}dB"
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

        kraken_data = load_data()
        plot_signal_components(fig_props, t, rcv_sig, rcv_noise)
        mean_Rx, mean_Rs, mean_Rv = plot_mean_csdm(fig_props, Rx, Rs, Rv)
        plot_rtf_estimation(fig_props, kraken_data, f_cs, rtf_cs, f_cw, rtf_cw)
        compare_rtf_vs_received_spectrum(
            fig_props,
            kraken_data,
            f_cs,
            rtf_cs,
            f_cw,
            rtf_cw,
            rcv_signal=rcv_sig_data,
            rcv_noise=rcv_noise_data,
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


def testcase_3_propagated_interference(
    snr_dB=0,
    plot=True,
    interference_type="z_call",
    interferer_r=None,
    interferer_z=None,
    force_snr_interference=False,
    snr_interference_dB=None,
):
    """
    Test case 3
        - Waveguide: simple waveguide with short impulse response.
        - Signal: ship signal propagated through the waveguide using Kraken.
        - Noise: interference signal propagated through the waveguide.
        - RTF estimation: covariance substraction and covariance whitening methods.
    """

    # Load propagated signal
    _, r_src, z_src, z_rcv, _ = waveguide_params()
    d = np.sqrt(r_src**2 + (z_rcv - z_src) ** 2)
    direct_delay = d / C0
    t0 = time()
    rcv_sig_data = derive_received_signal(tau_ir=TAU_IR, delay_correction=direct_delay)
    print(f"derive_received_signal: {time() - t0:.2f} s")
    t = rcv_sig_data["t"]

    # Load propagated noise from multiple sources
    ns = len(t)
    fs = 1 / (t[1] - t[0])

    # interferer_z = [30, 25]
    # interferer_r = np.arange(5, 50, 5) * 1e3
    if interferer_r is None or interferer_z is None:
        interferer_r = [5 * 1e3, 15 * 1e3, 50 * 1e3]
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

    if force_snr_interference:
        # Derive std at first receiver
        sigma_interference = np.std(rcv_interference_data[f"rcv0"]["sig"])
        # Normalize received interference to have the desired SNR
        if snr_interference_dB is None:
            snr_interference_dB = snr_dB

        sigma_2 = 10 ** (-snr_interference_dB / 10)
        alpha = (
            np.sqrt(sigma_2) / sigma_interference * np.std(rcv_sig_data["rcv0"]["sig"])
        )
        for i in range(len(rcv_interference_data.keys())):
            rcv_interference_data[f"rcv{i}"]["sig"] *= alpha

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
    rcv_noise_data = {key: rcv_interference_data[key] for key in rcv_interference_data}
    for i in range(N_RCV):
        id_rcv = f"rcv{i}"
        rcv_sig[:, i] = rcv_sig_data[id_rcv]["sig"]
        rcv_noise[:, i] = (
            rcv_interference_data[id_rcv]["sig"] + additive_noise[id_rcv]["sig"]
        )
        rcv_noise_data[id_rcv]["sig"] = rcv_noise[:, i]
        rcv_noise_data[id_rcv]["sig"] = scipy.signal.welch(
            rcv_noise[:, i], fs=fs, nperseg=2**12, noverlap=2**11
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
    if len(interferer_r) > 1:
        src_range_label = f"{interferer_r[0] // 1e3}to{interferer_r[-1] // 1e3}km"
    else:
        src_range_label = f"{interferer_r[0] // 1e3}km"

    tc_folder = os.path.join(
        ROOT_IMG,
        "testcase_3_propagated_interference",
        f"{interference_type}_{n_src}_src_{src_range_label}",
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
        kraken_data = load_data()
        plot_signal_components(fig_props, t, rcv_sig, rcv_noise, rcv_idx_to_plot=0)
        print(f"plot_signal_components: {time() - t0:.2f} s")
        mean_Rx, mean_Rs, mean_Rv = plot_mean_csdm(fig_props, Rx, Rs, Rv)
        plot_rtf_estimation(fig_props, kraken_data, f_cs, rtf_cs, f_cw, rtf_cw)
        compare_rtf_vs_received_spectrum(
            fig_props,
            kraken_data,
            f_cs,
            rtf_cs,
            f_cw,
            rtf_cw,
            rcv_signal=rcv_sig_data,
            rcv_noise=rcv_noise_data,
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
    pass
