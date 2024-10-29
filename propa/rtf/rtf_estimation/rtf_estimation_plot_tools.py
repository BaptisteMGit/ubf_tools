#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_estimation_plot_tools.py
@Time    :   2024/10/18 11:00:36
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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mpl_toolkits.axes_grid1 import make_axes_locatable
from propa.rtf.ideal_waveguide import print_arrivals
from propa.rtf.rtf_estimation.rtf_estimation_utils import *
from propa.rtf.rtf_estimation.rtf_estimation_const import *
from propa.rtf.rtf_utils import (
    D_frobenius,
    D_hermitian_angle,
    true_rtf,
    interp_true_rtf,
)


# ======================================================================================================================
# Plot tools
# ======================================================================================================================


def plot_ir(kraken_data, shift_ir=True, plot_arrivals=True):
    plt.figure()
    t = kraken_data["t"]
    delta_rcv = 500

    depth, r_src, z_src, z_rcv, _ = waveguide_params()
    arrivals = print_arrivals(z_src=z_src, z_rcv=z_rcv, r=r_src, depth=depth, n=4)

    if shift_ir:
        xlims = [19, 24]
        fname = "kraken_ir_shifted"
    else:
        xlims = [16, 24]
        fname = "kraken_ir"

    title = waveguide_info_line()

    if plot_arrivals:
        for k in range(arrivals.shape[0]):
            plt.axvline(
                arrivals[k, 0],
                color="k",
                linestyle="--",
                linewidth=1,
            )
            plt.text(
                arrivals[k, 0] - 0.1,
                np.max(kraken_data[f"rcv{0}"]["ir"]) * 0.75,
                r"$t_{\textrm{arr}"
                + f"{k+1}"
                + r"} ="
                + f"{np.round(arrivals[k, 0], 2)}"
                + r"\, \textrm{s}$",
                rotation=90,
                color="k",
                fontsize=12,
            )

    for i in range(kraken_data["n_rcv"]):
        ir = kraken_data[f"rcv{i}"]["ir"]
        if shift_ir:
            tau = i * delta_rcv / C0
            shift = int(tau / (t[1] - t[0]))
            ir = np.roll(ir, shift)

        plt.plot(t, ir, label=r"$rcv_{" + f"{i}" + r"}$")

    plt.xlabel(r"$t \, \textrm{[s]}$")
    plt.ylabel(r"$h(t)$")
    plt.title(title)
    plt.grid()
    plt.xlim(xlims)
    plt.legend()
    plt.savefig(os.path.join(ROOT_DATA, f"{fname}.png"))


def plot_tf(kraken_data):
    plt.figure()
    for i in range(kraken_data["n_rcv"]):
        plt.plot(
            kraken_data["f"],
            np.abs(kraken_data[f"rcv{i}"]["h_f"]),
            label=r"$rcv_{" + f"{i}" + r"}$",
        )
    plt.xlabel(r"$f \, \textrm{[Hz]}$")
    plt.ylabel(r"$|H(f)|$")
    plt.title(waveguide_info_line())
    plt.grid()
    plt.xlim([0, 50])
    plt.legend()
    plt.savefig(os.path.join(ROOT_DATA, "kraken_tf.png"))


def plot_signal(tau_ir):
    rcv_sig = derive_received_signal(tau_ir=tau_ir)
    t = rcv_sig["t"]
    idx_window_center = int(len(t) / 2)

    # Received signal
    plt.figure()
    for i in range(rcv_sig["n_rcv"]):
        plt.plot(t, rcv_sig[f"rcv{i}"]["sig"], label=f"rcv{i}")
    plt.xlabel(r"$t \, \textrm{[s]}$")
    plt.ylabel(r"$s(t)$")
    plt.title("Received signal at the receivers")
    plt.grid()
    plt.xlim(
        [
            t[idx_window_center] - rcv_sig["tau_ir"] / 2,
            t[idx_window_center] + rcv_sig["tau_ir"] / 2,
        ]
    )
    plt.legend()
    plt.savefig(os.path.join(ROOT_DATA, "received_signal.png"))

    for i in range(rcv_sig["n_rcv"]):
        # Received time serie
        plt.figure()
        plt.xlabel(r"$t \, \textrm{[s]}$")
        plt.ylabel(r"$s(t)$")
        plt.title(f"Received signal - rcv{i}")
        plt.grid()
        plt.xlim(
            [
                t[idx_window_center] - rcv_sig["tau_ir"] / 2,
                t[idx_window_center] + rcv_sig["tau_ir"] / 2,
            ]
        )
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
    plt.xlim(
        [
            t[idx_window_center] - rcv_sig["tau_ir"] / 2,
            t[idx_window_center] + rcv_sig["tau_ir"] / 2,
        ]
    )
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


def waveguide_info_line():
    depth, r_src, z_src, z_rcv, _ = waveguide_params()

    p0 = r"$r_{\textrm{src}} = " + str(r_src * 1e-3) + r"\, \textrm{km},\quad$"
    p1 = r"$z_{\textrm{src}} = " + str(z_src) + r"\, \textrm{m},\quad$"
    p2 = r"$z_{\textrm{rcv}} = " + str(z_rcv) + r"\, \textrm{m},\quad$"
    p3 = r"$D = " + str(depth) + r"\, \textrm{m}$"
    info = p0 + p1 + p2 + p3

    return info


def plot_rcv_signals(fig_props, t, rcv_sig, rcv_noise, rcv_idx_to_plot=0):
    # Plot signal, noise and signal + noise in time domain for the first receiver (subplots)
    rcv_sig_0 = rcv_sig[:, rcv_idx_to_plot]
    rcv_noise_0 = rcv_noise[:, rcv_idx_to_plot]
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    ax1, ax2, ax3 = axs

    ax1.plot(t, rcv_sig_0)
    ax1.set_title(r"$s(t)$")
    ax2.plot(t, rcv_noise_0)
    ax2.set_title(r"$n(t)$")
    ax3.plot(t, rcv_sig_0 + rcv_noise_0)
    ax3.set_title(r"$x(t) = s(t) + n(t)$")
    ax3.set_xlabel(r"$t \, \textrm{[s]}$")

    plt.savefig(os.path.join(fig_props["folder_path"], "signal_noise.png"))

    # Same plot on a short time window
    idx_window_center = int(len(t) / 2)
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    ax1, ax2, ax3 = axs

    ax1.plot(t, rcv_sig_0)
    ax1.set_title(r"$s(t)$")
    ax1.set_xlim(
        [
            t[idx_window_center] - fig_props["tau_ir"] / 2,
            t[idx_window_center] + fig_props["tau_ir"] / 2,
        ]
    )
    ax2.plot(t, rcv_noise_0)
    ax2.set_title(r"$n(t)$")
    ax2.set_xlim(
        [
            t[idx_window_center] - fig_props["tau_ir"] / 2,
            t[idx_window_center] + fig_props["tau_ir"] / 2,
        ]
    )
    ax3.plot(t, rcv_sig_0 + rcv_noise_0)
    ax3.set_title(r"$x(t) = s(t) + n(t)$")
    ax3.set_xlabel(r"$t \, \textrm{[s]}$")
    ax3.set_xlim(
        [
            t[idx_window_center] - fig_props["tau_ir"] / 2,
            t[idx_window_center] + fig_props["tau_ir"] / 2,
        ]
    )

    plt.savefig(os.path.join(fig_props["folder_path"], "signal_noise_zoom.png"))


def plot_rcv_stfts(fig_props, t, rcv_sig, rcv_noise, rcv_idx_to_plot=0):

    rcv_sig_0 = rcv_sig[:, rcv_idx_to_plot]
    rcv_noise_0 = rcv_noise[:, rcv_idx_to_plot]

    # Plot stft
    # nperseg = 2**12
    # noverlap = int(nperseg * 3 / 4)
    nperseg = 2**7
    # noverlap = int(0.8 * nperseg)
    noverlap = int(0.8 * nperseg)

    fs = 1 / (t[1] - t[0])
    stft_sig_0 = sp.stft(
        rcv_sig_0,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )
    stft_noise_0 = sp.stft(
        rcv_noise_0,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )
    stft_x_0 = sp.stft(
        rcv_sig_0 + rcv_noise_0,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )

    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    ax1, ax2, ax3 = axs

    vmin = np.round(np.percentile(20 * np.log10(np.abs(stft_x_0[2])), 25) / 10) * 10
    vmax = np.round(np.percentile(20 * np.log10(np.abs(stft_x_0[2])), 99) / 10) * 10
    # vmin = np.round(np.percentile(20 * np.log10(np.abs(stft_noise_0[2])), 25) / 10) * 10
    # vmax = (
    #     np.round(np.percentile(20 * np.log10(np.abs(stft_noise_0[2])), 99.99) / 10) * 10
    # )
    cmap = "jet"

    im1 = ax1.pcolormesh(
        stft_sig_0[1],
        stft_sig_0[0],
        20 * np.log10(np.abs(stft_sig_0[2])),
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax1.set_title(r"$S(f, t)$")

    im2 = ax2.pcolormesh(
        stft_noise_0[1],
        stft_noise_0[0],
        # np.abs(stft_noise_0[2]),
        20 * np.log10(np.abs(stft_noise_0[2])),
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax2.set_title(r"$N(f, t)$")

    im3 = ax3.pcolormesh(
        stft_x_0[1],
        stft_x_0[0],
        # np.abs(stft_x_0[2]),
        20 * np.log10(np.abs(stft_x_0[2])),
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax3.set_title(r"$X(f, t) = S(f, t) + N(f, t)$")

    fig.supxlabel(r"$t \, \textrm{[s]}$")
    fig.supylabel(r"$f \, \textrm{[Hz]}$")
    fig.colorbar(
        im3,
        ax=axs.ravel().tolist(),
        orientation="vertical",
        pad=0.05,
        shrink=0.95,
        aspect=40,
        label=r"$\textrm{Amplitude [dB]}$",
    )
    # plt.colorbar(im3)

    plt.savefig(os.path.join(fig_props["folder_path"], "stft_signal_noise.png"))


def plot_rcv_autocorr(fig_props, t, rcv_sig, rcv_noise, rcv_idx_to_plot=0):

    fs = 1 / (t[1] - t[0])
    rcv_sig_0 = rcv_sig[:, rcv_idx_to_plot]
    rcv_noise_0 = rcv_noise[:, rcv_idx_to_plot]

    # Autocorrelation of the three signals
    acf_sig_0 = sp.correlate(rcv_sig_0, rcv_sig_0, mode="full")
    acf_sig_0 /= acf_sig_0.max()
    acf_noise_0 = sp.correlate(rcv_noise_0, rcv_noise_0, mode="full")
    acf_noise_0 /= acf_noise_0.max()
    acf_x_0 = sp.correlate(
        rcv_sig_0 + rcv_noise_0, rcv_sig_0 + rcv_noise_0, mode="full"
    )
    acf_x_0 /= acf_x_0.max()

    lag_acf = sp.correlation_lags(len(rcv_sig_0), len(rcv_sig_0), mode="full")
    t_acf = lag_acf / fs

    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    ax1, ax2, ax3 = axs

    ax1.plot(t_acf, acf_sig_0)
    ax1.set_title(r"$R_{ss}(\tau)$")

    ax2.plot(t_acf, acf_noise_0)
    ax2.set_title(r"$R_{nn}(\tau)$")

    ax3.plot(t_acf, acf_x_0)
    ax3.set_title(r"$R_{xx}(\tau)$")

    fig.supxlabel(r"$\tau \, \textrm{[s]}$")
    fig.supylabel(r"$R(\tau)$")

    plt.savefig(os.path.join(fig_props["folder_path"], "acf_signal_noise.png"))


def plot_signal_components(fig_props, t, rcv_sig, rcv_noise, rcv_idx_to_plot=0):
    plot_rcv_signals(fig_props, t, rcv_sig, rcv_noise, rcv_idx_to_plot=rcv_idx_to_plot)
    plot_rcv_stfts(fig_props, t, rcv_sig, rcv_noise, rcv_idx_to_plot=rcv_idx_to_plot)
    plot_rcv_autocorr(fig_props, t, rcv_sig, rcv_noise, rcv_idx_to_plot=rcv_idx_to_plot)


def compare_rtf_vs_received_spectrum(
    fig_props, f_cs, rtf_cs, f_cw=None, rtf_cw=None, rcv_signal=None
):

    # Load true RTF
    kraken_data = load_data()
    f_true, rtf_true = interp_true_rtf(kraken_data, f_cs)

    # Derive Hermitian angle distance
    dist_cs = D_hermitian_angle(
        rtf_ref=rtf_true, rtf=rtf_cs, unit="deg", apply_mean=False
    )
    dist_cw = D_hermitian_angle(
        rtf_ref=rtf_true, rtf=rtf_cw, unit="deg", apply_mean=False
    )

    # Plot Hermitian angle
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Twin axis for received spectrum
    spec2 = np.abs(rcv_signal[f"rcv{0}"]["spect"]) ** 2
    val = -10 * np.log10(spec2)
    f = rcv_signal["f"]

    ax1.plot(f, val, color="k", linestyle="-", linewidth=0.2)
    # ax1.scatter(f, val, color="k", marker=".", s=1)
    ax1.set_ylabel(r"$-10 log_{10}(|S(f)|^2) \, \textrm{[dB]}$")

    ax2.plot(
        f_cs,
        dist_cs,
        label=r"$\theta_{\textrm{CS}}$",
        linestyle="-",
        color="b",
        marker="o",
        linewidth=0.2,
        markersize=2,
    )
    ax2.plot(
        f_cw,
        dist_cw,
        label=r"$\theta_{\textrm{CW}}$",
        linestyle="-",
        color="r",
        marker="o",
        linewidth=0.2,
        markersize=2,
    )
    ax2.set_xlabel(r"$f \, \textrm{[Hz]}$")
    ax2.set_ylabel(r"$\theta \, \textrm{[°]}$")

    ax2.legend()
    plt.title(
        # r"$\textrm{RTF estimation with unpropagated white noise}$" +
        f"\n({csdm_info_line(fig_props)})"
    )

    plt.savefig(
        os.path.join(fig_props["folder_path"], f"hermitian_angle_vs_spectrum.png")
    )

    print("Mean Hermitian angle distance CS: ", np.nanmean(dist_cs))
    print("Mean Hermitian angle distance CW: ", np.nanmean(dist_cw))


def plot_rtf_estimation(fig_props, f_cs, rtf_cs, f_cw=None, rtf_cw=None):

    # Load true RTF
    kraken_data = load_data()
    f_true, rtf_true = true_rtf(kraken_data)

    # Plot RTF
    # ylim = 1.1 * np.nanmax(np.abs(rtf_true[:, i]))
    ylim = np.nanpercentile(np.abs(rtf_true), 99.9)
    for i in range(kraken_data["n_rcv"]):
        plt.figure()
        plt.plot(
            f_cs,
            np.abs(rtf_cs[:, i]),
            label=r"$\Pi_{" + str(i) + r"}^{(CS)}$",
            linestyle="-",
            color="b",
            marker="o",
            linewidth=0.2,
            markersize=2,
        )
        plt.plot(
            f_cw,
            np.abs(rtf_cw[:, i]),
            label=r"$\Pi_{" + str(i) + r"}^{(CW)}$",
            linestyle="-",
            color="r",
            marker="o",
            linewidth=0.2,
            markersize=2,
        )
        plt.plot(
            f_true, np.abs(rtf_true[:, i]), label=r"$\Pi_{" + str(i) + r"}$", color="k"
        )
        plt.yscale("log")

        # plt.ylim([0, ylim])

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
    rtf_cw = rtf_cw[(f_cw >= fmin_rtf) & (f_cw <= fmax_rtf), :]
    f_true = f_true[(f_true >= fmin_rtf) & (f_true <= fmax_rtf)]
    f_cs = f_cs[(f_cs >= fmin_rtf) & (f_cs <= fmax_rtf)]
    f_cw = f_cw[(f_cw >= fmin_rtf) & (f_cw <= fmax_rtf)]

    # ylim = np.nanpercentile(np.abs(rtf_true), 99.5)
    for i in range(kraken_data["n_rcv"]):
        plt.figure()

        plt.plot(
            f_cs,
            np.abs(rtf_cs[:, i]),
            label=r"$\Pi_{" + str(i) + r"}^{(CS)}$",
            linestyle="-",
            color="b",
            marker="o",
            linewidth=0.2,
            markersize=2,
        )
        plt.plot(
            f_cw,
            np.abs(rtf_cw[:, i]),
            label=r"$\Pi_{" + str(i) + r"}^{(CW)}$",
            linestyle="-",
            color="r",
            marker="o",
            linewidth=0.2,
            markersize=2,
        )
        plt.plot(
            f_true, np.abs(rtf_true[:, i]), label=r"$\Pi_{" + str(i) + r"}$", color="k"
        )
        plt.yscale("log")

        # plt.ylim([0, ylim])

        plt.xlabel(r"$f \, \textrm{[Hz]}$")
        plt.ylabel(r"$|\Pi(f)|$")
        plt.legend()
        plt.title(
            r"$\textrm{RTF estimation with unpropagated white noise}$"
            + f"\n({csdm_info_line(fig_props)})"
        )
        plt.savefig(
            os.path.join(
                fig_props["folder_path"], f"rtf_estimation_limited_bandwidth_rcv{i}.png"
            )
        )

    # Rolling mean to smooth high freq variations
    window = 5
    rtf_cs_smooth = np.zeros_like(rtf_cs)
    rtf_cw_smooth = np.zeros_like(rtf_cw)
    for i in range(kraken_data["n_rcv"]):
        rtf_cs_smooth[:, i] = np.convolve(
            np.abs(rtf_cs[:, i]), np.ones(window) / window, mode="same"
        )
        rtf_cw_smooth[:, i] = np.convolve(
            np.abs(rtf_cw[:, i]), np.ones(window) / window, mode="same"
        )

    # ylim = np.nanpercentile(np.abs(rtf_true), 99.5)

    for i in range(kraken_data["n_rcv"]):
        plt.figure()

        plt.plot(
            f_cs,
            rtf_cs_smooth[:, i],
            label=r"$\Pi_{" + str(i) + r"}^{(CS)}$",
            linestyle="-",
            color="b",
            marker="o",
            linewidth=0.2,
            markersize=2,
        )
        plt.plot(
            f_cw,
            rtf_cw_smooth[:, i],
            label=r"$\Pi_{" + str(i) + r"}^{(CW)}$",
            linestyle="-",
            color="r",
            marker="o",
            linewidth=0.2,
            markersize=2,
        )
        plt.plot(
            f_true, np.abs(rtf_true[:, i]), label=r"$\Pi_{" + str(i) + r"}$", color="k"
        )

        plt.yscale("log")
        # plt.ylim([0, ylim])

        plt.xlabel(r"$f \, \textrm{[Hz]}$")
        plt.ylabel(r"$|\Pi(f)|$")
        plt.legend()
        plt.title(
            r"$\textrm{RTF estimation with unpropagated white noise}$"
            + f"\n({csdm_info_line(fig_props)})"
        )
        plt.savefig(
            os.path.join(fig_props["folder_path"], f"rtf_estimation_smooth_rcv{i}.png")
        )


def plot_dist_vs_snr(
    snrs, dist_cs, dist_cw, title="", dist_type="hermitian_angle", savepath=None
):
    if dist_type == "hermitian_angle":
        plot_dist_vs_snr_hermitian_angle(snrs, dist_cs, dist_cw, title, savepath)
    elif dist_type == "frobenius":
        plot_dist_vs_snr_frobenius(snrs, dist_cs, dist_cw, title, savepath)


def plot_dist_vs_snr_frobenius(snrs, dist_cs, dist_cw, title, savepath=None):
    plt.figure()
    plt.plot(snrs, 10 * np.log10(dist_cs), marker=".", label=r"$\mathcal{D}_F^{(CS)}$")
    plt.plot(snrs, 10 * np.log10(dist_cw), marker=".", label=r"$\mathcal{D}_F^{(CW)}$")
    plt.ylabel(r"$\mathcal{D}_F\, \textrm{[dB]}$")
    plt.xlabel(r"$\textrm{snr} \, \textrm{[dB]}$")
    plt.title(title)
    plt.legend()
    plt.grid()

    if savepath is not None:
        plt.savefig(savepath)


def plot_dist_vs_snr_hermitian_angle(snrs, dist_cs, dist_cw, title, savepath=None):
    plt.figure()
    plt.plot(
        snrs,
        dist_cs,
        label=r"$\theta_{\textrm{CS}}$",
        linestyle="-",
        color="b",
        marker="o",
        linewidth=0.5,
        markersize=2,
    )
    plt.plot(
        snrs,
        dist_cw,
        label=r"$\theta_{\textrm{CW}}$",
        linestyle="-",
        color="r",
        marker="o",
        linewidth=0.5,
        markersize=2,
    )
    plt.xlabel(r"$\textrm{snr} \, \textrm{[dB]}$")
    plt.ylabel(r"$\theta \, \textrm{[°]}$")
    plt.title(title)
    plt.legend()
    plt.grid()

    if savepath is not None:
        plt.savefig(savepath)


if __name__ == "__main__":
    pass
