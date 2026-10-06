#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   segmentation_nmf.py
@Time    :   2025/07/10 13:58:02
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy import signal
import numpy as np
from numpy import savetxt
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from sklearn.decomposition import NMF

# ======================================================================================================================
# Functions
# ======================================================================================================================


# ======================================================================================================================
# NMF segmentation utilities
# ======================================================================================================================


def rollav(x, n):
    ra = np.convolve(x, np.ones((n,)), "same") / n
    return ra


def cwidth(x, toll):
    n = len(x)
    minx = np.min(x)
    maxx = np.max(x)
    iamax = np.argmax(x)
    ip = 0
    if iamax <= n:
        while (ip < n - iamax) and (x[iamax + ip] - minx > toll * (maxx - minx)):
            # print(ip)
            ip += 1
    im = 0
    if iamax > 0:
        while (im <= iamax) and (x[iamax - im] - minx > toll * (maxx - minx)):
            im += 1
    w = ip + im
    return w


def isZcall(h, t_):  # detecteur de Z call
    nt, nc = h.shape[1], h.shape[0]
    f_ = np.fft.fftfreq(nt, t_[2] - t_[1])
    ms = abs(np.fft.fft(h, axis=1))
    ms[:, 0:2] = 0
    ms[:, nt - 2 : nt] = 0
    y, n = [], []
    for ic in range(nc):
        imp = np.argmax(rollav(ms[ic, :], 2))
        if (f_[imp] >= 1 / 72) and (f_[imp] <= 1 / 60):
            y.append(ic)
        else:
            n.append(ic)
    return y, n


def isT(w, f_, lwin, toll, Zchor=True):  # detecteur de seismes/ondes T

    if Zchor:
        w = w[abs(f_ - 22.0) >= 4.0, :]

    nc = w.shape[1]
    ilwin = np.where(f_ >= lwin)[0][0]
    y, n = [], []
    for ic in range(nc):
        if cwidth(rollav(w[:, ic], 4), toll) >= ilwin:
            y.append(ic)
        else:
            n.append(ic)
    return y, n


def pctend(x):
    nc, nt = x.shape[0], x.shape[1]
    X0 = abs(np.fft.fft(x, axis=1))
    X = X0[:, 0 : int(nt / 2)]
    s = np.sum(X, axis=1)
    n = np.sum(X[:, 0:2], axis=1)
    p = n / s
    return p


def isNav(w, h, t_, f_, fwin, toll):
    nt, nc = h.shape[1], h.shape[0]
    f2 = np.fft.fftfreq(nt, t_[2] - t_[1])
    ifwin = np.where(f_ >= fwin)[0][0]
    s = h.shape

    ms = abs(
        np.fft.fft(h, axis=1)
    )  # on fait la tranformee de Fourier des fct d'activation
    ms[:, 0:2] = 0
    ms[:, nt - 2 : nt] = 0

    pc = pctend(h)
    y, n = [], []

    for ic in range(nc):
        imp = np.argmax(ms[ic, :])
        if (pc[ic] >= 0.1) and (cwidth(rollav(w[:, ic], 5), toll) <= ifwin):
            y.append(ic)
        else:
            n.append(ic)
    return y, n


def apply_nmf(
    sgn,
    n_fft=1024,
    win="hann",
    lwin=180000,  # taille de la fenetre pour la separation de source
    Ncomp=50,  # nombre de composantes pour la NMF
):
    N = len(sgn)
    hop_length = n_fft * 0.5
    Nwin = int(N / lwin)  # nbre de fenêtres

    nmf = NMF(
        n_components=Ncomp,
        max_iter=10000,
        beta_loss="frobenius",
        solver="mu",
        init="nndsvda",
    )  # NMF setup

    for iw in range(Nwin):
        sgn30 = sgn[lwin * iw : lwin * (iw + 1)]  # Extract current signal window
        f, t, spectro = signal.stft(
            sgn30, fs=100, window=win, nperseg=n_fft, noverlap=n_fft - hop_length
        )  # Derive STFT
        W = nmf.fit_transform(abs(spectro))  # Apply NMF to current signal portion
        H = nmf.components_

        zy, zn = isZcall(H, t)
        spz = W[:, zy] @ H[zy, :]

        ty, tn = isT(W, f, 10.0, 0.1)
        spt = W[:, ty] @ H[ty, :]

        ny, nn = isNav(W, H, t, f, 2.0, 0.1)
        spn = W[:, ny] @ H[ny, :]

        # Store NMF results
        if iw == 0:
            spZ, spT, spN = spz, spt, spn
        else:
            spZ = np.concatenate((spZ, spz), axis=1)
            spT = np.concatenate((spT, spt), axis=1)
            spN = np.concatenate((spN, spn), axis=1)

    # end = clock.time()
    # print(end - start)

    ftrait = np.linspace(0, 50, num=spZ.shape[0])
    ttrait = np.linspace(0, 24 * 3600, num=spZ.shape[1])
    spZ[spZ == 0] = 1e-50
    spT[spT == 0] = 1e-50
    spN[spN == 0] = 1e-50

    return ftrait, ttrait, spZ, spT, spN


def apply_nmf_v2(
    t_stft,
    f_stft,
    x_stft,
    n_stftwin_nmf,  # Number of STFT windows to process in each NMF iteration
    n_nmf_comp=50,  # nombre de composantes pour la NMF
):
    n_stft_windows = t_stft.shape[0]  # Number of STFT windows (along time axis)
    n_win_nfm = int(n_stft_windows / n_stftwin_nmf)  # Number of NMF windows to process

    # NMF setup
    nmf = NMF(
        n_components=n_nmf_comp,
        max_iter=10000,
        beta_loss="frobenius",
        solver="mu",
        init="nndsvda",
    )

    f_win = f_stft  # Frequency vector remains the same for all windows
    for iw in range(n_win_nfm):
        # Extract STFT portion
        start_idx = iw * n_stftwin_nmf
        end_idx = start_idx + n_stftwin_nmf
        x_stft_win = x_stft[:, start_idx:end_idx]  # Extract current signal window
        t_win = t_stft[
            start_idx:end_idx
        ]  # Corresponding time vector for the STFT window

        # Apply NMF to current signal portion
        W = nmf.fit_transform(abs(x_stft_win))
        H = nmf.components_

        zy, zn = isZcall(H, t_win)
        spz = W[:, zy] @ H[zy, :]

        ty, tn = isT(W, f_win, 10.0, 0.1)
        spt = W[:, ty] @ H[ty, :]

        ny, nn = isNav(W, H, t_win, f_win, 2.0, 0.1)
        spn = W[:, ny] @ H[ny, :]

        # Store NMF results
        if iw == 0:
            spZ, spT, spN = spz, spt, spn
        else:
            spZ = np.concatenate((spZ, spz), axis=1)
            spT = np.concatenate((spT, spt), axis=1)
            spN = np.concatenate((spN, spn), axis=1)

    # Process a last window to get the same output shape as the input STFT
    n_missing_windows = n_stft_windows % n_stftwin_nmf

    if n_missing_windows != 0:
        # Handle the last window if it doesn't fit evenly
        # Keep the same portion size as the previous windows (NMF params are tuned for this window size)
        start_idx = -n_stftwin_nmf
        x_stft_win = x_stft[:, start_idx:]  # Extract current signal window
        t_win = t_stft[start_idx:]  # Corresponding time vector for the STFT window

        # Apply NMF to current signal portion
        W = nmf.fit_transform(abs(x_stft_win))
        H = nmf.components_

        zy, zn = isZcall(H, t_win)
        spz = W[:, zy] @ H[zy, :]

        ty, tn = isT(W, f_win, 10.0, 0.1)
        spt = W[:, ty] @ H[ty, :]

        ny, nn = isNav(W, H, t_win, f_win, 2.0, 0.1)
        spn = W[:, ny] @ H[ny, :]

        # Crop to keep only the relevant information
        spz = spz[:, -n_missing_windows:]
        spt = spt[:, -n_missing_windows:]
        spn = spn[:, -n_missing_windows:]

        # Store NMF results
        spZ = np.concatenate((spZ, spz), axis=1)
        spT = np.concatenate((spT, spt), axis=1)
        spN = np.concatenate((spN, spn), axis=1)

    # end = clock.time()
    # print(end - start)

    spZ[spZ == 0] = 1e-50
    spT[spT == 0] = 1e-50
    spN[spN == 0] = 1e-50

    return spZ, spT, spN


# def apply_nmf(
#     sgn,
#     n_fft=1024,
#     win="hann",
#     lwin=180000,  # taille de la fenetre pour la separation de source
#     Ncomp=50,  # nombre de composantes pour la NMF
# ):
#     N = len(sgn)
#     hop_length = n_fft * 0.5
#     Nwin = int(N / lwin)  # nbre de fenêtres

#     nmf = NMF(
#         n_components=Ncomp,
#         max_iter=10000,
#         beta_loss="frobenius",
#         solver="mu",
#         init="nndsvda",
#     )  # On calcule la NMF

#     for iw in range(Nwin):
#         sgn30 = sgn[lwin * iw : lwin * (iw + 1)]  # on prend 1/2 heure de signal
#         f, t, spectro = signal.stft(
#             sgn30, fs=100, window=win, nperseg=n_fft, noverlap=n_fft - hop_length
#         )  # on fait le spectro
#         W = nmf.fit_transform(abs(spectro))  # on applique la NMF
#         H = nmf.components_

#         zy, zn = isZcall(H, t)
#         spz = W[:, zy] @ H[zy, :]

#         ty, tn = isT(W, f, 10.0, 0.1)
#         spt = W[:, ty] @ H[ty, :]

#         ny, nn = isNav(W, H, t, f, 2.0, 0.1)
#         spn = W[:, ny] @ H[ny, :]

#         if iw == 0:
#             spZ, spT, spN = spz, spt, spn
#         else:
#             spZ = np.concatenate((spZ, spz), axis=1)
#             spT = np.concatenate((spT, spt), axis=1)
#             spN = np.concatenate((spN, spn), axis=1)

#     # end = clock.time()
#     # print(end - start)

#     ftrait = np.linspace(0, 50, num=spZ.shape[0])
#     ttrait = np.linspace(0, 24 * 3600, num=spZ.shape[1])
#     spZ[spZ == 0] = 1e-50
#     spT[spT == 0] = 1e-50
#     spN[spN == 0] = 1e-50

#     return ftrait, ttrait, spZ, spT, spN


# def apply_nmf_v2(
#     sgn,
#     nfft=1024,
#     fs=100,
#     sep_window_s=30 * 60,  # Length of the separation window in seconds
#     Ncomp=50,  # nombre de composantes pour la NMF
# ):

#     # ns = len(sgn)  # Number of samples in the signal
#     # stft_win_duration = nfft / fs  # Duration of the STFT window in seconds
#     # n_max_analysis_windows = int(
#     #     ns / (stft_win_duration * fs)
#     # )  # Max number of analysis windows

#     # t_max_analysis_windows = (
#     #     n_max_analysis_windows * stft_win_duration
#     # )  # Max time duration for analysis windows
#     # n_max_stft_windows = int(
#     #     t_max_analysis_windows / stft_win_duration
#     # )  # Max number of STFT windows
#     # # t_max = n_max_stft_windows * stft_win_duration  # Max time duration for STFT windows

#     # # idx_max = int(t_max * fs)  # Max number of samples

#     # # Crop signal to get exact number of STFT windows
#     # signal_for_nmf = sgn[:idx_max]  # Crop signal to get exact number of STFT windows

#     # alpha_overlap = 0.5  # Overlap factor used in STFT

#     # # Number of samples in a sep_window_s analysis window
#     sep_window_ns = sep_window_s * fs
#     # n_sep_win = int(
#     #     len(sgn) // sep_window_ns
#     # )  # Number of separation analysis windows in the signal

#     # noverlap = int(nfft * alpha_overlap)  # Overlap in samples for STFT
#     # nfft_win = int(len(sgn) / noverlap)
#     # # Crop signal to get exact number of STFT windows
#     # # signal_for_nmf = sgn[: nfft_win * noverlap]

#     # # Number of STFT windows that fit into the analysis window
#     # n_win_stft_in_sep_window = int(sep_window_ns / noverlap)
#     # # Number of separation windows that fit in the total signal duration
#     # n_sep_win = int(nfft_win / n_win_stft_in_sep_window)

#     # # Crop signal to get exact number of separation windows
#     # idx_max = n_sep_win * n_win_stft_in_sep_window * nfft
#     # signal_for_nmf = sgn[:idx_max]

#     # # Derive STFT
#     # f_stft, t_stft, signal_stft = signal.stft(
#     #     signal_for_nmf,
#     #     fs=100,
#     #     window="hann",
#     #     nperseg=nfft,
#     #     noverlap=int(nfft * alpha_overlap),
#     #     padded=False,
#     # )
#     # n_win_stft = signal_stft.shape[1]

#     nmf = NMF(
#         n_components=Ncomp,
#         max_iter=10000,
#         beta_loss="frobenius",
#         solver="mu",
#         init="nndsvda",
#     )  # On calcule la NMF

#     lwin = sep_window_ns
#     win = "hann"
#     n_fft = nfft

#     noverlap = nfft * 0.5  # Hop length for STFT
#     n_win_stft_in_sep_window = (
#         np.ceil(sep_window_ns / noverlap).astype(int) + 1
#     )  # Number of STFT windows in a separation window
#     t_segment = t_stft[0:n_win_stft_in_sep_window]
#     for iw in range(n_sep_win):

#         sgn30 = sgn[lwin * iw : lwin * (iw + 1)]  # on prend 1/2 heure de signal
#         f, t, spectro = signal.stft(
#             sgn30, fs=100, window=win, nperseg=nfft, noverlap=noverlap
#         )  # on fait le spectro

#         # Extract the STFT segment for the current separation window
#         spectro_segment = signal_stft[
#             :, n_win_stft_in_sep_window * iw : n_win_stft_in_sep_window * (iw + 1)
#         ]
#         W = nmf.fit_transform(abs(spectro_segment))  # on applique la NMF
#         H = nmf.components_

#         zy, zn = isZcall(H, t_segment)
#         spz = W[:, zy] @ H[zy, :]

#         ty, tn = isT(W, f_stft, 10.0, 0.1)
#         spt = W[:, ty] @ H[ty, :]

#         ny, nn = isNav(W, H, t_segment, f_stft, 2.0, 0.1)
#         spn = W[:, ny] @ H[ny, :]

#         if iw == 0:
#             spZ, spT, spN = spz, spt, spn
#         else:
#             spZ = np.concatenate((spZ, spz), axis=1)
#             spT = np.concatenate((spT, spt), axis=1)
#             spN = np.concatenate((spN, spn), axis=1)

#     # Crop input STFT and t_stft to match the output spectrograms
#     signal_stft = signal_stft[:, : n_win_stft_in_sep_window * n_sep_win]
#     t_stft = t_stft[: n_win_stft_in_sep_window * n_sep_win]

#     # end = clock.time()
#     # print(end - start)

#     spZ[spZ == 0] = 1e-50
#     spT[spT == 0] = 1e-50
#     spN[spN == 0] = 1e-50

#     return f_stft, t_stft, signal_stft, spZ, spT, spN


if __name__ == "__main__":
    pass
