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
import scipy.signal as signal

from misc import *
from propa.rtf.ideal_waveguide import *
from signals.signals import generate_ship_signal
from propa.kraken_toolbox.run_kraken import runkraken
from propa.rtf.ideal_waveguide import waveguide_params
from localisation.verlinden.testcases.testcase_envs import TestCase1_0
from propa.rtf.rtf_estimation.rtf_estimation_utils import (
    rtf_covariance_substraction,
    rtf_covariance_whitening,
)
from real_data_analysis.real_data_utils import get_csdm_snapshot_number
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


from cst import RHO_W, C0

ROOT_FOLDER = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_estimation\short_ri_waveguide"
ROOT_DATA = os.path.join(ROOT_FOLDER, "data")
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


def derive_received_signal():
    """
    Derive the received signal at the receivers.
    The signal is modeled as a ship signal propagating in the ideal waveguide.
    """

    # Load params
    # depth, r_src, z_src, z_rcv, _ = waveguide_params()
    duration = 50 * TAU_IR

    # Load kraken data
    kraken_data = load_data()

    # Define useful params
    n_rcv = len(kraken_data.keys() - ["t", "f"])
    ts = kraken_data["t"][1] - kraken_data["t"][0]
    fs = 1 / ts

    # Create source signal
    f0 = 4.5
    # std_fi = 1e-2 * f0
    std_fi = 0.1 * f0
    tau_corr_fi = 0.1 * 1 / f0

    s, t = generate_ship_signal(
        Ttot=duration,
        f0=f0,
        std_fi=std_fi,
        tau_corr_fi=tau_corr_fi,
        fs=fs,
        normalize="max",
    )
    s *= np.hanning(len(s))
    # s /= np.std(s)

    src_spectrum = np.fft.rfft(s)

    # Derive psd
    psd = signal.welch(s, fs=fs, nperseg=2**12, noverlap=2**11)

    received_signal = {
        "t": t,
        "src": s,
        "f": kraken_data["f"],
        "spect": src_spectrum,
        "psd": psd,
        "std_fi": std_fi,
        "tau_corr_fi": tau_corr_fi,
        "f0": f0,
        "fs": fs,
    }

    for i in range(n_rcv):
        # Get transfert function
        h_kraken = kraken_data[f"rcv{i}"]["h_f"]

        # Received signal spectrum resulting from the convolution of the source signal and the impulse response
        transmited_sig_field_f = h_kraken * src_spectrum
        rcv_sig = np.fft.irfft(transmited_sig_field_f)

        # psd
        psd_rcv = signal.welch(rcv_sig, fs=fs, nperseg=2**12, noverlap=2**11)

        received_signal[f"rcv{i}"] = {
            "sig": rcv_sig,
            "spect": transmited_sig_field_f,
            "psd": psd_rcv,
        }

    return received_signal


def load_data():

    kraken_data = {}
    for i in range(N_RCV):
        # Load tf
        fpath = os.path.join(ROOT_DATA, f"kraken_tf_rcv{i}.csv")
        f_kraken, h_kraken_real, h_kraken_imag = np.loadtxt(
            fpath, delimiter=",", unpack=True
        )
        h_kraken = h_kraken_real + 1j * h_kraken_imag

        # Load ir kraken
        fpath = os.path.join(ROOT_DATA, f"kraken_ir_rcv{i}.csv")
        t_kraken, ir_kraken = np.loadtxt(fpath, delimiter=",", unpack=True)

        kraken_data[f"rcv{i}"] = {"ir": ir_kraken, "h_f": h_kraken}

    kraken_data.update({"t": t_kraken, "f": f_kraken})

    return kraken_data


def plot_ir(kraken_data, shift_ir=True):
    plt.figure()
    t = kraken_data["t"]
    delta_rcv = 500

    if shift_ir:
        xlims = [19, 24]
        fname = "kraken_ir_shifted"
    else:
        xlims = [16, 24]
        fname = "kraken_ir"

    for i in range(N_RCV):
        ir = kraken_data[f"rcv{i}"]["ir"]
        if shift_ir:
            tau = i * delta_rcv / C0
            shift = int(tau / (t[1] - t[0]))
            ir = np.roll(ir, shift)

        plt.plot(t, ir, label=f"rcv{i}")

    plt.xlabel(r"$t \, \textrm{[s]}$")
    plt.ylabel(r"$h(t)$")
    plt.title("Kraken impulse response")
    plt.grid()
    plt.xlim(xlims)
    plt.legend()
    plt.savefig(os.path.join(ROOT_DATA, f"{fname}.png"))


def plot_tf(kraken_data):
    plt.figure()
    for i in range(N_RCV):
        plt.plot(
            kraken_data["f"], np.abs(kraken_data[f"rcv{i}"]["h_f"]), label=f"rcv{i}"
        )
    plt.xlabel(r"$f \, \textrm{[Hz]}$")
    plt.ylabel(r"$|H(f)|$")
    plt.title("Kraken transfert function")
    plt.grid()
    plt.xlim([0, 50])
    plt.legend()
    plt.savefig(os.path.join(ROOT_DATA, "kraken_tf.png"))


def plot_signal():
    rcv_sig = derive_received_signal()
    t = rcv_sig["t"]
    idx_window_center = int(len(t) / 2)

    # Received signal
    plt.figure()
    for i in range(N_RCV):
        plt.plot(t, rcv_sig[f"rcv{i}"]["sig"], label=f"rcv{i}")
    plt.xlabel(r"$t \, \textrm{[s]}$")
    plt.ylabel(r"$s(t)$")
    plt.title("Received signal at the receivers")
    plt.grid()
    plt.xlim([t[idx_window_center] - TAU_IR / 2, t[idx_window_center] + TAU_IR / 2])
    plt.legend()
    plt.savefig(os.path.join(ROOT_DATA, "received_signal.png"))

    for i in range(N_RCV):
        # Received time serie
        plt.figure()
        plt.xlabel(r"$t \, \textrm{[s]}$")
        plt.ylabel(r"$s(t)$")
        plt.title(f"Received signal - rcv{i}")
        plt.grid()
        plt.xlim([t[idx_window_center] - TAU_IR / 2, t[idx_window_center] + TAU_IR / 2])
        plt.plot(t, rcv_sig[f"rcv{i}"]["sig"])
        plt.savefig(os.path.join(ROOT_DATA, f"received_signal_rcv{i}.png"))

        # Received spectrum
        plt.figure()
        plt.plot(rcv_sig["f"], np.abs(rcv_sig[f"rcv{i}"]["spect"]))
        plt.xlabel(r"$f \, \textrm{[Hz]}$")
        plt.ylabel(r"$|S(f)|$")
        plt.title(f"Received spectrum - rcv{i}")
        plt.grid()
        plt.xlim([0, 50])
        plt.savefig(os.path.join(ROOT_DATA, f"received_spectrum_rcv{i}.png"))

    # Source signal
    plt.figure()
    plt.plot(t, rcv_sig["src"])
    plt.xlabel(r"$t \, \textrm{[s]}$")
    plt.ylabel(r"$s(t)$")
    plt.title("Source signal")
    plt.grid()
    plt.xlim([t[idx_window_center] - TAU_IR / 2, t[idx_window_center] + TAU_IR / 2])
    plt.savefig(os.path.join(ROOT_DATA, "source_signal.png"))

    # Source spectrum
    plt.figure()
    plt.plot(rcv_sig["f"], np.abs(rcv_sig["spect"]))
    plt.xlabel(r"$f \, \textrm{[Hz]}$")
    plt.ylabel(r"$|S(f)|$")
    plt.title("Source spectrum")
    plt.grid()
    plt.xlim([0, 50])
    plt.savefig(os.path.join(ROOT_DATA, "source_spectrum.png"))

    # Source psd
    plt.figure()
    plt.plot(rcv_sig["psd"][0], 10 * np.log10(rcv_sig["psd"][1]))
    plt.xlabel(r"$f \, \textrm{[Hz]}$")
    plt.ylabel(r"$S_{xx} (f) \, \textrm{[dB]}$")
    plt.title("Source PSD")
    plt.grid()
    plt.xlim([0, 50])
    plt.savefig(os.path.join(ROOT_DATA, "source_psd.png"))


def plot_mean_csdm(fig_props, Rx, Rs, Rv):
    # Plot CSDM matrices
    mean_Rx = np.mean(Rx, axis=0)
    mean_Rs = np.mean(Rs, axis=0)
    mean_Rv = np.mean(Rv, axis=0)

    # Define pad
    pad = 0.3
    shrink = 1

    # plt.figure(figsize=(18, 6))  # Taille globale de la figure (plus grande)
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    ax1, ax2, ax3 = axs

    # Subplot 1
    im1 = ax1.imshow(np.abs(mean_Rx), aspect="equal")
    ax1.set_title(r"$R_{xx}$")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=pad)
    cbar1 = plt.colorbar(im1, cax=cax1, shrink=shrink)
    cbar1.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    cbar1.ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Subplot 2
    im2 = ax2.imshow(np.abs(mean_Rs), aspect="equal")
    ax2.set_title(r"$R_{ss}$")
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=pad)
    cbar2 = plt.colorbar(im2, cax=cax2, shrink=shrink)
    cbar2.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    cbar2.ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Subplot 3
    im3 = ax3.imshow(np.abs(mean_Rv), aspect="equal")
    ax3.set_title(r"$R_{nn}$")
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=pad)
    cbar3 = plt.colorbar(im3, cax=cax3, shrink=shrink)
    cbar3.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    cbar3.ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Add suptitle with information about the CSDM matrices
    fig.suptitle(csdm_info_line(fig_props))

    plt.savefig(os.path.join(fig_props["folder_path"], "csdm_matrices.png"))

    Rs_tilde = Rx - Rv
    mean_Rs_tilde = np.mean(Rs_tilde, axis=0)
    delta_Rs = Rs - Rs_tilde
    mean_delta_Rs = np.mean(delta_Rs, axis=0)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    ax1, ax2, ax3 = axs

    im1 = ax1.imshow(np.abs(mean_Rs_tilde), aspect="equal")
    ax1.set_title(r"$\tilde{R}_{ss} = R_{xx} - R_{nn}$")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=pad)
    cbar1 = plt.colorbar(im1, cax=cax1, shrink=shrink)
    cbar1.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    cbar1.ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Subplot 2
    im2 = ax2.imshow(np.abs(mean_Rs), aspect="equal")
    ax2.set_title(r"$R_{ss}$")
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=pad)
    cbar2 = plt.colorbar(im2, cax=cax2, shrink=shrink)
    cbar2.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    cbar2.ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Subplot 3
    im3 = ax3.imshow(np.abs(mean_delta_Rs), aspect="equal", cmap="bwr")
    ax3.set_title(r"$R_{ss} - \tilde{R}_{ss}$")
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=pad)
    cbar3 = plt.colorbar(im3, cax=cax3, shrink=shrink)
    cbar3.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    cbar3.ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    fig.suptitle(csdm_info_line(fig_props))
    plt.savefig(os.path.join(fig_props["folder_path"], "Rss_Rss_tilde_diff.png"))

    return mean_Rx, mean_Rs, mean_Rv


def csdm_info_line(fig_props):
    p1 = (
        r"$T_{\textrm{snap}} = "
        + str(fig_props["alpha_tau_ir"])
        + r"\tau_{\textrm{ir}},\quad$"
    )
    p2 = r"$\alpha_{\textrm{overlap}} = " + str(fig_props["alpha_overlap"]) + r",\quad$"
    p3 = r"$L_{\textrm{snap}} = " + str(fig_props["L"]) + r"$"
    info = p1 + p2 + p3

    return info


def plot_signal_components(fig_props, t, rcv_sig, rcv_noise):
    # Plot signal, noise and signal + noise in time domain for the first receiver (subplots)
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    ax1, ax2, ax3 = axs

    ax1.plot(t, rcv_sig[:, 0])
    ax1.set_title(r"$s(t)$")
    ax2.plot(t, rcv_noise[:, 0])
    ax2.set_title(r"$n(t)$")
    ax3.plot(t, rcv_sig[:, 0] + rcv_noise[:, 0])
    ax3.set_title(r"$x(t) = s(t) + n(t)$")
    ax3.set_xlabel(r"$t \, \textrm{[s]}$")

    plt.savefig(os.path.join(fig_props["folder_path"], "signal_noise.png"))

    # Same plot on a short time window
    idx_window_center = int(len(t) / 2)
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    ax1, ax2, ax3 = axs

    ax1.plot(t, rcv_sig[:, 0])
    ax1.set_title(r"$s(t)$")
    ax1.set_xlim([t[idx_window_center] - TAU_IR / 2, t[idx_window_center] + TAU_IR / 2])
    ax2.plot(t, rcv_noise[:, 0])
    ax2.set_title(r"$n(t)$")
    ax2.set_xlim([t[idx_window_center] - TAU_IR / 2, t[idx_window_center] + TAU_IR / 2])
    ax3.plot(t, rcv_sig[:, 0] + rcv_noise[:, 0])
    ax3.set_title(r"$x(t) = s(t) + n(t)$")
    ax3.set_xlabel(r"$t \, \textrm{[s]}$")
    ax3.set_xlim([t[idx_window_center] - TAU_IR / 2, t[idx_window_center] + TAU_IR / 2])

    plt.savefig(os.path.join(fig_props["folder_path"], "signal_noise_zoom.png"))


def testcase_1_unpropagated_whitenoise(snr_dB=10):

    # Create folder to save results
    tc_folder = os.path.join(
        ROOT_FOLDER, "testcase_1_unpropagated_whitenoise", f"snr_{snr_dB}dB"
    )
    if not os.path.exists(tc_folder):
        os.makedirs(tc_folder)

    # Load propagated signal
    rcv_sig_data = derive_received_signal()
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


def plot_rtf_estimation(fig_props, f_cs, rtf_cs, f_cw=None, rtf_cw=None):

    # Load true RTF
    f_true, rtf_true = true_rtf()

    # Plot RTF
    # ylim = 1.1 * np.nanmax(np.abs(rtf_true[:, i]))
    ylim = np.nanpercentile(np.abs(rtf_true), 99.9)
    for i in range(N_RCV):
        plt.figure()
        # Normalize RTFs for comparison
        # rtf_cs[:, i] /= np.max(np.abs(rtf_cs[:, i]))
        # rtf_true[:, i] /= np.nanmax(np.abs(rtf_true[:, i]))

        plt.plot(f_cs, np.abs(rtf_cs[:, i]), label=r"$\Pi_{" + str(i) + r"}^{(CS)}$")
        # plt.plot(
        #     f_cw, np.abs(rtf_cw[:, i]), "--", label=r"$\Pi_{" + str(i) + r"}^{(CW)}$"
        # )
        plt.plot(f_true, np.abs(rtf_true[:, i]), "--", label=r"$\Pi_{" + str(i) + r"}$")

        plt.ylim([0, ylim])

        plt.xlabel(r"$f \, \textrm{[Hz]}$")
        plt.ylabel(r"$|\Pi(f)|$")
        plt.legend()
        plt.title(
            r"$\textrm{RTF estimation with unpropagated white noise}$"
            + f"\n({csdm_info_line(fig_props)})"
        )

        plt.savefig(
            os.path.join(fig_props["folder_path"], f"rtf_estimation_rcv{i}.png")
        )

    # All RTF in the 0 - 30 Hz band
    fmin_rtf = 0
    fmax_rtf = 30

    rtf_true = rtf_true[(f_true >= fmin_rtf) & (f_true <= fmax_rtf), :]
    rtf_cs = rtf_cs[(f_cs >= fmin_rtf) & (f_cs <= fmax_rtf), :]
    # rtf_cw = rtf_cw[(f_cw >= fmin_rtf) & (f_cw <= fmax_rtf), :]
    f_true = f_true[(f_true >= fmin_rtf) & (f_true <= fmax_rtf)]
    f_cs = f_cs[(f_cs >= fmin_rtf) & (f_cs <= fmax_rtf)]
    # f_cw = f_cw[(f_cw >= fmin_rtf) & (f_cw <= fmax_rtf)]

    ylim = np.nanpercentile(np.abs(rtf_true), 99.9)
    for i in range(N_RCV):
        plt.figure()

        plt.plot(f_cs, np.abs(rtf_cs[:, i]), label=r"$\Pi_{" + str(i) + r"}^{(CS)}$")
        # plt.plot(
        #     f_cw, np.abs(rtf_cw[:, i]), "--", label=r"$\Pi_{" + str(i) + r"}^{(CW)}$"
        # )
        plt.plot(f_true, np.abs(rtf_true[:, i]), "--", label=r"$\Pi_{" + str(i) + r"}$")

        plt.ylim([0, ylim])

        plt.xlabel(r"$f \, \textrm{[Hz]}$")
        plt.ylabel(r"$|\Pi(f)|$")
        plt.legend()
        plt.title(
            r"$\textrm{RTF estimation with unpropagated white noise}$"
            + f"\n({csdm_info_line(fig_props)})"
        )
        plt.savefig(
            os.path.join(
                fig_props["folder_path"], f"rtf_estimation_limited_bandwith_rcv{i}.png"
            )
        )


def true_rtf():
    kraken_data = load_data()
    tf_ref = kraken_data[f"rcv{0}"]["h_f"]
    rtf = np.zeros((len(kraken_data["f"]), N_RCV), dtype=complex)
    for i in range(N_RCV):
        rtf[:, i] = kraken_data[f"rcv{i}"]["h_f"] / tf_ref

    return kraken_data["f"], rtf


if __name__ == "__main__":
    # derive_kraken_tf()
    # kraken_data = load_data()

    # plot_ir(kraken_data)
    # plot_tf(kraken_data)

    # rcv_sig = derive_received_signal()
    # plot_signal()
    snrs = [-20, -10, 0, 10, 20, 30]
    for snr_dB in snrs:
        testcase_1_unpropagated_whitenoise(snr_dB=snr_dB)

    plt.show()
