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
from propa.rtf.rtf_estimation.rtf_estimation_utils import *
from propa.rtf.rtf_estimation.rtf_estimation_const import *

# ======================================================================================================================
# Plot tools
# ======================================================================================================================


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

    for i in range(kraken_data["n_rcv"]):
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
    for i in range(kraken_data["n_rcv"]):
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
    ax1.set_xlim(
        [
            t[idx_window_center] - fig_props["tau_ir"] / 2,
            t[idx_window_center] + fig_props["tau_ir"] / 2,
        ]
    )
    ax2.plot(t, rcv_noise[:, 0])
    ax2.set_title(r"$n(t)$")
    ax2.set_xlim(
        [
            t[idx_window_center] - fig_props["tau_ir"] / 2,
            t[idx_window_center] + fig_props["tau_ir"] / 2,
        ]
    )
    ax3.plot(t, rcv_sig[:, 0] + rcv_noise[:, 0])
    ax3.set_title(r"$x(t) = s(t) + n(t)$")
    ax3.set_xlabel(r"$t \, \textrm{[s]}$")
    ax3.set_xlim(
        [
            t[idx_window_center] - fig_props["tau_ir"] / 2,
            t[idx_window_center] + fig_props["tau_ir"] / 2,
        ]
    )

    plt.savefig(os.path.join(fig_props["folder_path"], "signal_noise_zoom.png"))


def plot_rtf_estimation(fig_props, f_cs, rtf_cs, f_cw=None, rtf_cw=None):

    # Load true RTF
    kraken_data = load_data()
    f_true, rtf_true = true_rtf(kraken_data)

    # Plot RTF
    # ylim = 1.1 * np.nanmax(np.abs(rtf_true[:, i]))
    ylim = np.nanpercentile(np.abs(rtf_true), 99.9)
    for i in range(kraken_data["n_rcv"]):
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
    for i in range(kraken_data["n_rcv"]):
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


def true_rtf(kraken_data):
    tf_ref = kraken_data[f"rcv{0}"]["h_f"]
    rtf = np.zeros((len(kraken_data["f"]), kraken_data["n_rcv"]), dtype=complex)
    for i in range(kraken_data["n_rcv"]):
        rtf[:, i] = kraken_data[f"rcv{i}"]["h_f"] / tf_ref

    return kraken_data["f"], rtf


if __name__ == "__main__":
    pass
