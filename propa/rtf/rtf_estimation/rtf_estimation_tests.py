#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_estimation_tests.py
@Time    :   2024/11/04 13:56:19
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
import scipy as sp

from misc import *
from propa.rtf.ideal_waveguide import *
from real_data_analysis.real_data_utils import *
from propa.rtf.rtf_estimation.rtf_estimation_utils import *
from propa.rtf.rtf_estimation.rtf_estimation_ideal_waveguide_misc import *


########################################################################################################################
# Tests
########################################################################################################################


def test_rcv_noise():
    # dur = 20
    # fs = 100
    # depth = 1000
    # max_range = 3 * 1e4

    # n_rcv = 5
    # z_rcv = depth - 1
    # delta_rcv = 1000
    # x_rcv = np.array([i * delta_rcv for i in range(n_rcv)])

    dur, fs, depth, n_rcv, z_rcv, x_rcv, _, _ = testcase_params()

    t, rcv_noise = received_noise(
        duration=dur,
        fs=fs,
        z_rcv=z_rcv,
        x_rcv=x_rcv,
        depth=depth,
        max_range=max_range,
    )

    # Plot signals
    plt.figure()
    for i in range(n_rcv):
        plt.plot(t, rcv_noise[:, i], label=f"Receiver {i}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Received noise at the receivers")
    plt.legend()

    # # Derive covariance matrix of the received noise
    # cov = np.cov(rcv_noise.T)
    # cov /= np.max(cov)
    # plt.figure()
    # plt.imshow(cov, cmap="jet")
    # plt.colorbar()
    # plt.title("Covariance matrix of the received noise")

    # Derive CSDM
    fs = 1 / (t[1] - t[0])
    nperseg = 256
    noverlap = 128
    # Step 1 -  build the list of stft of the received noise at each receiver
    stft_list = []
    for i in range(n_rcv):
        ff, tt, stft = sp.stft(
            rcv_noise[:, i],
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
        )
        stft_list.append(stft)
    # Step 2 - build the CSDM
    csdm = compute_csd_matrix_fast(stft_list, n_seg_cov=0)
    csdm = np.abs(np.mean(csdm, axis=0))
    csdm /= np.max(csdm)
    # Step 3 - plot
    plt.figure()
    plt.imshow(csdm, cmap="jet")
    plt.colorbar()
    plt.title("CSDM of the received noise")

    # Plot spectrograms
    for i in range(n_rcv):
        plt.figure()
        stft_i = stft_list[i]
        vmin = np.percentile(20 * np.log10(np.abs(stft_i)), 20)
        vmax = np.percentile(20 * np.log10(np.abs(stft_i)), 99)
        plt.pcolormesh(
            tt, ff, 20 * np.log10(np.abs(stft_i)), cmap="jet", vmin=vmin, vmax=vmax
        )
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [s]")
        plt.title(f"Spectrogram of the received noise at receiver {i}")
        plt.colorbar()
    plt.show()


def test_rcv_signal():

    dur, fs, depth, n_rcv, z_rcv, x_rcv, z_src, r_src = testcase_params()

    t, rcv_sig = received_signal(
        duration=dur,
        fs=fs,
        z_src=z_src,
        r_src=r_src,
        z_rcv=z_rcv,
        x_rcv=x_rcv,
        depth=depth,
    )

    # Plot signals
    plt.figure()
    for i in range(n_rcv):
        plt.plot(t, rcv_sig[:, i], label=f"Receiver {i}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Received signal at the receivers")
    plt.legend()

    # # Derive covariance matrix of the received signal
    # cov = np.cov(rcv_sig.T)
    # cov /= np.max(cov)
    # plt.figure()
    # plt.imshow(cov, cmap="jet")
    # plt.colorbar()
    # plt.title("Covariance matrix of the received signal")

    # Derive CSDM
    fs = 1 / (t[1] - t[0])
    nperseg = 256
    noverlap = 128
    # Step 1 -  build the list of stft of the received signal at each receiver
    stft_list = []
    for i in range(n_rcv):
        ff, tt, stft = sp.stft(
            rcv_sig[:, i],
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
        )
        stft_list.append(stft)
    # Step 2 - build the CSDM
    csdm = compute_csd_matrix_fast(stft_list, n_seg_cov=0)
    csdm = np.abs(np.mean(csdm, axis=0))
    # csdm = np.abs(np.fft.irfft(csdm, axis=0)[0])
    csdm /= np.max(csdm)
    # Step 3 - plot
    plt.figure()
    plt.imshow(csdm, cmap="jet")
    plt.colorbar()
    plt.title("Mean CSDM over frequency band of the received signal")

    # Plot spectrograms
    for i in range(n_rcv):
        plt.figure()
        stft_i = stft_list[i]
        vmin = np.percentile(20 * np.log10(np.abs(stft_i)), 20)
        vmax = np.percentile(20 * np.log10(np.abs(stft_i)), 99)
        plt.pcolormesh(
            tt, ff, 20 * np.log10(np.abs(stft_i)), cmap="jet", vmin=vmin, vmax=vmax
        )
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [s]")
        plt.title(f"Spectrogram of the received signal at receiver {i}")
        plt.colorbar()

    plt.show()


def test_all_signals(snr_dB=10):

    dur, fs, depth, n_rcv, z_rcv, x_rcv, z_src, r_src = testcase_params()

    t, rcv_sig = received_signal(
        duration=dur,
        fs=fs,
        z_src=z_src,
        r_src=r_src,
        z_rcv=z_rcv,
        x_rcv=x_rcv,
        depth=depth,
    )

    t, rcv_noise = received_noise(
        duration=dur,
        fs=fs,
        z_rcv=z_rcv,
        x_rcv=x_rcv,
        depth=depth,
        max_range=max_range,
        snr_dB=snr_dB,
    )

    x = rcv_sig + rcv_noise

    # Plot signals
    f, axs = plt.subplots(n_rcv, 1, figsize=(10, 10), sharex=True)

    for i in range(n_rcv):
        axs[i].plot(t, x[:, i], color="k")
        axs[i].plot(t, rcv_sig[:, i], color="b", linestyle="--")
        axs[i].plot(t, rcv_noise[:, i], color="r", linestyle="--")

        # Add annotation at the top right of the subplot with the receiver number
        axs[i].annotate(
            f"Receiver {i}",
            xy=(0.005, 0.945),
            xycoords="axes fraction",
            fontsize=12,
            ha="left",
            va="top",
            bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
        )

    axs[0].legend([r"$x(t) = s(t) + v(t)$", r"$s(t)$", r"$v(t)$"])

    f.supxlabel(r"$t \textrm{[s]}$")
    f.supylabel(r"$y(t)$")
    # plt.title("Received signal and noise at the receivers")
    plt.legend()

    plt.show()


def test_rtf_cs(snr_dB=10):

    dur, fs, depth, n_rcv, z_rcv, x_rcv, z_src, r_src = testcase_params()

    t, rcv_sig = received_signal(
        duration=dur,
        fs=fs,
        z_src=z_src,
        r_src=r_src,
        z_rcv=z_rcv,
        x_rcv=x_rcv,
        depth=depth,
    )

    t, rcv_noise = received_noise(
        duration=dur,
        fs=fs,
        z_rcv=z_rcv,
        x_rcv=x_rcv,
        depth=depth,
        max_range=max_range,
        snr_dB=snr_dB,
    )

    f, rtf = rtf_covariance_substraction(t, rcv_sig, rcv_noise)
    # first component of rtf is the reference receiver -> we can remove it (equal to 1)
    rtf = rtf[:, 1:]

    # Restrict to propagating frequencies
    idx_above_cutoff = f > cutoff_frequency(c0, depth)
    f = f[idx_above_cutoff]
    rtf = rtf[idx_above_cutoff, :]

    # Derive the true RTF
    r_ref = r_src - x_rcv
    f, g_ref = g_mat(
        f,
        z_src,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        D=depth,
        r_rcv_ref=r_ref[0],
        r=r_ref[1:],
    )
    # g_ref is of shape (nf, nr, nrcv, nz) -> we can remove second and third dimensions
    g_ref = np.squeeze(g_ref, axis=(1, 3))

    # Plot RTF
    plt.figure()
    for i in range(n_rcv - 1):
        plt.plot(f, np.abs(rtf[:, i]), label=f"Receiver {i}")
        plt.plot(f, np.abs(g_ref[:, i]), "--", label=f"True RTF Receiver {i}")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title("RTF estimation using covariance substraction method")
    plt.legend()

    plt.show()


def test_rtf_cw(snr_dB=10):

    dur, fs, depth, n_rcv, z_rcv, x_rcv, z_src, r_src = testcase_params()

    t, rcv_sig = received_signal(
        duration=dur,
        fs=fs,
        z_src=z_src,
        r_src=r_src,
        z_rcv=z_rcv,
        x_rcv=x_rcv,
        depth=depth,
    )

    t, rcv_noise = received_noise(
        duration=dur,
        fs=fs,
        z_rcv=z_rcv,
        x_rcv=x_rcv,
        depth=depth,
        max_range=max_range,
        snr_dB=snr_dB,
    )

    f, rtf = rtf_covariance_whitening(t, rcv_sig, rcv_noise)
    # first component of rtf is the reference receiver -> we can remove it (equal to 1)
    rtf = rtf[:, 1:]

    # Restrict to propagating frequencies
    idx_above_cutoff = f > cutoff_frequency(c0, depth)
    f = f[idx_above_cutoff]
    rtf = rtf[idx_above_cutoff, :]

    # Derive the true RTF
    r_ref = r_src - x_rcv
    f, g_ref = g_mat(
        f,
        z_src,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        D=depth,
        r_rcv_ref=r_ref[0],
        r=r_ref[1:],
    )
    # g_ref is of shape (nf, nr, nrcv, nz) -> we can remove second and third dimensions
    g_ref = np.squeeze(g_ref, axis=(1, 3))

    # Plot RTF
    plt.figure()
    for i in range(n_rcv - 1):
        plt.plot(f, np.abs(rtf[:, i]), label=f"Receiver {i}")
        plt.plot(f, np.abs(g_ref[:, i]), "--", label=f"True RTF Receiver {i}")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title("RTF estimation using covariance whitening method")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    test_rcv_noise()
    test_rcv_signal()
    test_all_signals()
    test_rtf_cs()
    test_rtf_cw()
