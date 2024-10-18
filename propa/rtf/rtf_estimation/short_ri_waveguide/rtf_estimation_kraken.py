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
import matplotlib.pyplot as plt
import copy

from misc import *
from propa.rtf.ideal_waveguide import *
from propa.kraken_toolbox.run_kraken import runkraken
from propa.rtf.ideal_waveguide import waveguide_params
from localisation.verlinden.testcases.testcase_envs import TestCase1_0
from propa.rtf.rtf_estimation.rtf_estimation_utils import *
from propa.rtf.rtf_estimation.rtf_estimation_plot_tools import *
from real_data_analysis.real_data_utils import get_csdm_snapshot_number


from propa.rtf.rtf_estimation.rtf_estimation_const import *

TAU_IR = 5  # Impulse response duration
N_RCV = 5  # Number of receivers


def derive_kraken_tf():

    # Load params
    depth, r_src, z_src, z_rcv, _ = waveguide_params()

    # N_RCV = 5
    delta_rcv = 500
    x_rcv = np.array([i * delta_rcv for i in range(N_RCV)])
    r_src_rcv = r_src - x_rcv

    # Create the frequency vector
    duration = 50 * TAU_IR
    ts = 1e-2
    nt = int(duration / ts)
    f = fft.rfftfreq(nt, ts)

    # Init env
    bott_hs_properties = {
        "rho": 1.5 * RHO_W * 1e-3,  # Density (g/cm^3)
        "c_p": 1500,  # P-wave celerity (m/s)
        "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
        "a_p": 0.2,  # Compression wave attenuation (dB/wavelength)
        "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
        "z": None,
    }

    tc_varin = {
        "freq": f,
        "src_depth": z_src,
        "max_range_m": r_src,
        "mode_theory": "adiabatic",
        "flp_N_RCV_z": 1,
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
            verbose=True,
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

        # h_kraken += np.pad(h_kraken_subband, (pad_before, pad_after))

        # Update frequency subband
        i_subband += 1
        f0 = f1
        f1 = f[min(n_subband * i_subband, len(f) - 1)]

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


def testcase_1_unpropagated_whitenoise(snr_dB=10):

    # Create folder to save results
    tc_folder = os.path.join(
        ROOT_FOLDER, "testcase_1_unpropagated_whitenoise", f"snr_{snr_dB}dB"
    )
    if not os.path.exists(tc_folder):
        os.makedirs(tc_folder)

    # Load propagated signal
    rcv_sig_data = derive_received_signal(tau_ir=TAU_IR)
    t = rcv_sig_data["t"]

    ns = len(t)
    rcv_noise = np.empty((len(t), N_RCV))
    rcv_sig = np.empty((len(t), N_RCV))
    # Generate independent gaussian white noise on each receiver
    for i in range(N_RCV):
        sigma_v2 = 10 ** (-snr_dB / 10)
        v = np.random.normal(loc=0, scale=np.sqrt(sigma_v2), size=ns)
        rcv_noise[:, i] = v
        rcv_sig[:, i] = rcv_sig_data[f"rcv{i}"]["sig"] / np.std(
            rcv_sig_data[f"rcv{0}"]["sig"]
        )  # Normalize signal to unit variance

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

    # f_cw, rtf_cw, _, _, _ = rtf_covariance_whitening(t, rcv_sig, rcv_noise)

    # Set figure properties to pass to the plotting functions
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
    plot_signal_components(fig_props, t, rcv_sig, rcv_noise)
    mean_Rx, mean_Rs, mean_Rv = plot_mean_csdm(fig_props, Rx, Rs, Rv)
    plot_rtf_estimation(fig_props, f_cs, rtf_cs)

    plt.close("all")

    # # Define the recorded signal = received signal + noise
    # rcv_sig_noisy = np.empty((len(t), N_RCV))
    # for i in range(N_RCV):
    #     rcv_sig_noisy[:, i] = rcv_sig[f"rcv{i}"]["sig"] + rcv_noise[:, i]


if __name__ == "__main__":
    # derive_kraken_tf()
    # kraken_data = load_data()

    # plot_ir(kraken_data)
    # plot_tf(kraken_data)

    # rcv_sig = derive_received_signal()
    # plot_signal()
    # snrs = [-20, -10, 0, 10, 20, 30]
    # for snr_dB in snrs:
    #     testcase_1_unpropagated_whitenoise(snr_dB=snr_dB)

    testcase_1_unpropagated_whitenoise(snr_dB=0)

    plt.show()
