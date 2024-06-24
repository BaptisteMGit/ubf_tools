#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/06/20 07:42:52
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
import pandas as pd
import scipy.signal as sp
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

from sklearn.decomposition import PCA, FastICA
from skimage.metrics import structural_similarity as ssim


# ======================================================================================================================
# ICA
# ======================================================================================================================


def apply_ica(X, n_components=3, verbose=True):

    # Compute ICA
    ica = FastICA(n_components=n_components, whiten="arbitrary-variance", max_iter=300)
    S_ = ica.fit_transform(X)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix

    # We can `prove` that the ICA model applies by reverting the unmixing.
    if verbose:
        print(f"ICA model applies : {np.allclose(X, np.dot(S_, A_.T) + ica.mean_)}")
        print(f"Components : {ica.components_}")
        print(f"Mixing : {ica.mixing_}")
        print(f"Mean : {ica.mean_}")

    return S_, A_


def analyse_ica(X, S_, fs, nperseg, noverlap):
    time = np.arange(X.shape[0]) / fs
    reconstructed_signals = {
        "name": "ICA recovered signals",
        "signal": [],
        "freq": [],
        "tt": [],
        "stft": [],
    }
    mixed_signals = {
        "name": "Observations (mixed signal)",
        "signal": [],
        "freq": [],
        "tt": [],
        "stft": [],
    }

    for i in range(S_.shape[1]):
        freq, tt, stft = sp.stft(
            X[:, i], fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap
        )
        mixed_signals["tt"] = tt
        mixed_signals["freq"] = freq
        mixed_signals["stft"].append(stft)
        mixed_signals["signal"].append(X[:, i])

        # s_ = S_[:, i] / np.max(
        #     S_[:, i]
        # )  # Normalize reconstructed signal to compare with original
        s_ = S_[:, i]
        freq, tt, stft = sp.stft(
            s_, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap
        )
        reconstructed_signals["freq"] = freq
        reconstructed_signals["tt"] = tt
        reconstructed_signals["stft"].append(stft)
        reconstructed_signals["signal"].append(s_)

    return time, mixed_signals, reconstructed_signals


def plot_ica_results(
    time, mixed_signals, reconstructed_signals, separated_data=None, vmin=-80, vmax=0
):
    # Plot time series
    n_src = len(mixed_signals["signal"])
    if separated_data is not None:
        n_col = 3
    else:
        n_col = 2

    f, axs = plt.subplots(n_src, n_col, sharex=True, figsize=(16, 12))
    f.suptitle("Time series")
    for i in range(n_src):
        axs[i, 0].plot(time, mixed_signals["signal"][i])
        axs[i, 1].plot(time, reconstructed_signals["signal"][i])
        if separated_data is not None:
            axs[i, 2].plot(time, separated_data["signal"][i])

    axs[0, 0].set_title(mixed_signals["name"])
    axs[0, 1].set_title(reconstructed_signals["name"])
    if separated_data is not None:
        axs[0, 2].set_title("Separated signals")

    plt.tight_layout()
    plt.show()

    # Plot spectrograms
    f, axs = plt.subplots(n_src, n_col, sharex=True, sharey=True, figsize=(16, 12))
    f.suptitle("Spectrogram")
    for i in range(n_src):
        axs[i, 0].pcolormesh(
            mixed_signals["tt"],
            mixed_signals["freq"],
            20 * np.log10(abs(mixed_signals["stft"][i])),
            # shading="gouraud",
            vmin=vmin,
            vmax=vmax,
        )
        axs[i, 1].pcolormesh(
            reconstructed_signals["tt"],
            reconstructed_signals["freq"],
            20 * np.log10(abs(reconstructed_signals["stft"][i])),
            # shading="gouraud",
            vmin=vmin,
            vmax=vmax,
        )
        if separated_data is not None:
            axs[i, 2].pcolormesh(
                separated_data["tt"],
                separated_data["freq"],
                20 * np.log10(abs(separated_data["stft"][i])),
                # shading="gouraud",
                vmin=vmin,
                vmax=vmax,
            )

    axs[0, 0].set_title(mixed_signals["name"])
    axs[0, 1].set_title(reconstructed_signals["name"])
    if separated_data is not None:
        axs[0, 2].set_title("Separated signals")

    plt.tight_layout()
    plt.show()


def ica_perf_analysis(reconstructed_signals, separated_data, verbose=False):
    n_src = len(separated_data["signal"])
    # Intercorrelation
    intercorr_mat = np.zeros((n_src, n_src))
    for i in range(n_src):
        for j in range(n_src):
            a = separated_data["signal"][i]
            b = reconstructed_signals["signal"][j]
            intercorr_mat[i, j] = np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2))
    if verbose:
        print(f"Inter-correlation matrix :\n", intercorr_mat)

    # SSIM
    ssim_mat = np.zeros((n_src, n_src))
    for i in range(n_src):
        for j in range(n_src):
            a = abs(separated_data["stft"][i])
            b = abs(reconstructed_signals["stft"][j])
            L = np.max(b) - np.min(b)
            ssim_mat[i, j] = ssim(a, b, data_range=L)
    if verbose:
        print(f"SSIM matrix :\n", ssim_mat)

    return intercorr_mat, ssim_mat


# ======================================================================================================================
# Load data
# ======================================================================================================================


def load_synthetic_data(fs, nperseg, noverlap, snr=10):
    # Load synthetic data
    root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\signaux_ica_jl"
    vars = ["ondeT", "navire", "Zcall"]

    data = {
        "vars": vars,
        "fpaths": [],
        "stft": [],
        "tt": [],
        "freq": [],
        "noise": [],
        "signal": [],
    }

    for iv, v in enumerate(vars):
        data["fpaths"].append(os.path.join(root, f"{v}.txt"))
        s = pd.read_csv(data["fpaths"][iv])
        s = s.values.flatten()

        # roll to have two signals
        if v == "ondeT":
            # idx_t_s0 = np.argmax(s)
            # t_s1 = 250
            # shift_s1 = int(t_s1 * fs) - idx_t_s0
            # s1 = np.roll(s, shift_s1, axis=0)

            # t_s2 = 1400
            # shift_s2 = int(t_s2 * fs) - idx_t_s0
            # s2 = np.roll(s, shift_s2, axis=0)
            # s = s1 + s2

            # Jean Lecoulant 19/06/2024
            s = 0.5 * np.roll(s, 130000) + np.roll(s, 132000) + np.roll(s, 20000)

        # Normalize data
        s = s / np.max(abs(s))

        # Derive sigma noise
        if snr is not None:
            P_sig = np.mean(s**2)
            sigma_noise = np.sqrt(P_sig / (10 ** (snr / 10)))
            data["noise"].append(np.random.normal(0, sigma_noise, len(s)))
        else:
            data["noise"].append(np.zeros(len(s)))

        # Add noise
        s += data["noise"][iv]
        data["signal"].append(s)
        # data[v] = s

        # Derive spectro
        # s_spec = data[v]
        freq, tt, stft = sp.stft(
            s, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap
        )
        # stft[stft == 0] = np.min(stft[stft != 0])
        data["stft"].append(stft)
        data["freq"] = freq
        data["tt"] = tt

    nt = len(data["signal"][0])
    time = np.arange(0, nt * 1 / fs, 1 / fs)
    data["time"] = time

    return data


def load_real_data(nperseg, noverlap):
    # Load data
    root_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\wav\RHUMRUM"

    data = {
        "fpaths": [],
        "vars": ["BDH", "BHZ", "BH1", "BH2"],
        "stft": [],
        "tt": [],
        "freq": [],
        "signal": [],
    }

    for chnl in data["vars"]:
        fname = f"signal_{chnl}_RR44_2013-05-3.wav"
        data["fpaths"].append(os.path.join(root_data, fname))
        fs, sig = wavfile.read(data["fpaths"][-1])

        # Normalize data
        sig = sig / np.max(sig)
        data["signal"].append(sig)

        f, tt, stft = sp.stft(
            sig, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap
        )
        data["stft"].append(stft)

    data["freq"] = f
    data["tt"] = tt
    time = np.arange(0, len(sig) * 1 / fs, 1 / fs)
    data["time"] = time

    return data


def mix_synthetic_signals(
    data, snr, mixing_noise_level=0.01, verbose=True, a0=1, a1=1, a2=1, n_rcv=3
):
    v_tuple = tuple([data["signal"][iv] for iv in range(len(data["vars"]))])
    S = np.c_[v_tuple]

    # Mix data
    shape = (n_rcv, S.shape[1])
    v = np.array([[a0, a1, a2]])
    A = v.repeat(repeats=n_rcv, axis=0) + mixing_noise_level * np.random.normal(
        size=shape
    )
    X = np.dot(S, A.T)  # Observations

    # Add noise
    for i in range(X.shape[1]):
        X[:, i] += 0.1 * np.random.normal(0, 1, X[:, i].shape)

    # if snr is not None:
    #     for i in range(X.shape[1]):
    #         P_sig = np.mean(X[:, i] ** 2)
    #         sigma_noise = np.sqrt(P_sig / (10 ** (snr / 10)))
    #         X[:, i] += np.random.normal(0, sigma_noise, X[:, i].shape)

    if verbose:
        print(f"Mixing matrix : {A}")
        # print(f"Observations : {X}")

    return X, S, A


# ======================================================================================================================
# Plot
# ======================================================================================================================


def plot_input_signals(data, vmin=-80, vmax=0):
    # Plot time series
    f, axs = plt.subplots(len(data["signal"]), 1, sharex=True)
    for iv, v in enumerate(data["signal"]):
        axs[iv].plot(data["time"], data["signal"][iv])

        if "vars" in data.keys():
            axs[iv].set_title(data["vars"][iv])
        elif "channels" in data.keys():
            axs[iv].set_title(data["channels"][iv])

    f.supxlabel("Time [s]")
    f.supylabel("Normalized amplitude")
    plt.tight_layout()

    # Plot spectrograms
    f, axs = plt.subplots(len(data["stft"]), 1, sharex=True)
    for iv, v in enumerate(data["stft"]):
        im = axs[iv].pcolormesh(
            data["tt"],
            data["freq"],
            20 * np.log10(abs(data["stft"][iv])),
            # shading="gouraud",
            vmin=vmin,
            vmax=vmax,
        )
        if "vars" in data.keys():
            axs[iv].set_title(data["vars"][iv])
        elif "channels" in data.keys():
            axs[iv].set_title(data["channels"][iv])

    f.supxlabel("Time [s]")
    f.supylabel("Frequency [Hz]")
    plt.tight_layout()
    f.colorbar(im, ax=axs.ravel().tolist())
    plt.show()


def plot_ica_perf(inter_corr_mat, ssim_mat, vars=None):
    inter_corr_mat = np.abs(inter_corr_mat)
    f, axs = plt.subplots(1, 2, sharex=True)
    axs[0].imshow(inter_corr_mat, cmap="gray")
    axs[0].set_title("Inter-correlation matrix")
    axs[1].imshow(ssim_mat, cmap="gray")
    axs[1].set_title("SSIM matrix")

    # Add text
    for i in range(inter_corr_mat.shape[0]):
        for j in range(inter_corr_mat.shape[1]):
            axs[0].text(
                j, i, f"{inter_corr_mat[i, j]:.3f}", ha="center", va="center", color="r"
            )
            axs[1].text(
                j, i, f"{ssim_mat[i, j]:.3f}", ha="center", va="center", color="r"
            )

    # Add labels
    separated_labels = ["s1", "s2", "s3"]
    if vars is not None:
        axs[0].set_yticks(np.arange(len(vars)))
        axs[0].set_xticks(np.arange(len(separated_labels)))
        axs[0].set_yticklabels(vars)
        axs[0].set_xticklabels(separated_labels)
        axs[1].set_yticks(np.arange(len(vars)))
        axs[1].set_xticks(np.arange(len(separated_labels)))
        axs[1].set_yticklabels(vars)
        axs[1].set_xticklabels(separated_labels)

    plt.tight_layout()
    plt.show()
