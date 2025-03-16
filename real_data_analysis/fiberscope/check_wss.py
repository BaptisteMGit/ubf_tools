#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   check_wss.py
@Time    :   2025/03/05 12:09:34
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import nptdms
import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal as sp
import matplotlib.pyplot as plt

from signals.signals import lfm_chirp
from publication.PublicationFigure import PubFigure
from real_data_analysis.fiberscope.read_tdms import load_fiberscope_data
from real_data_analysis.signal_processing_utils import get_bifrequency_spectrum

PubFigure()


def check_wss(file_path):
    # Load data
    ds = load_fiberscope_data(file_path)

    # Limit frequency band
    f0 = 5 * 1e3
    f1 = 18 * 1e3

    # # Derive stft
    ds = ds.sel(h_index=1)
    # Select time window
    ds = ds.sel(time=slice(0, 0.5))
    nperseg = 2**17
    noverlap = 2**16
    ff, tt, stft = sp.stft(
        ds.signal,
        fs=ds.fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )
    # stft = ds.stft_amp * np.exp(1j * ds.stft_phase)
    ff_in_band = np.logical_and((ff >= f0), (ff <= f1))
    stft = stft[ff_in_band, :]
    ff = ff[ff_in_band]

    # # For a single hydrophone
    # stft = ds.stft_amp * np.exp(1j * ds.stft_phase)
    # stft = stft.sel(h_index=1)
    # L = stft.shape[1]

    # stft = stft.sel(ff=slice(f0, f1))

    # Bi-frequency spectum
    # S_f1f2 = np.zeros((len(stft.ff), len(stft.ff)), dtype=complex)

    # for i1, f1 in enumerate(stft.ff.values):
    #     for i2, f2 in enumerate(stft.ff.values):
    #         S_f1f2[i1, i2] = np.mean(
    #             stft.sel(ff=f1).values * np.conj(stft.sel(ff=f2).values)
    #         )

    # Faster implementation with matrix multiplication
    L = stft.shape[1]
    S_f1f2 = stft @ np.conj(stft).T
    S_f1f2 = 1 / L * S_f1f2

    # Normalize
    S_f1f2 /= np.max(np.abs(S_f1f2))

    S_f1f2 = xr.DataArray(
        S_f1f2,
        dims=["ff1", "ff2"],
        # coords={"ff1": stft.ff.values, "ff2": stft.ff.values},
        coords={"ff1": ff, "ff2": ff},
    )
    S_f1f2_log_magnitude = 20 * np.log10(np.abs(S_f1f2))
    # Plot
    fig, ax = plt.subplots()
    S_f1f2_log_magnitude.plot(
        ax=ax,
        x="ff1",
        y="ff2",
        vmin=-30,
        vmax=0,
        cmap="viridis",
        add_colorbar=True,
        cbar_kwargs={"label": "Magnitude [dB]"},
    )
    # im = ax.imshow(
    #     stft.ff.values,
    #     stft.ff.values,
    #     20 * np.log10(np.abs(S_f1f2)),
    #     origin="lower",
    #     aspect="auto",
    #     vmin=-30,
    #     vmax=0,
    # )
    # plt.colorbar(im, label=f"Magnitude [dB]")
    ax.set_xlabel("$f_1$" + " [Hz]")
    ax.set_ylabel("$f_2$" + " [Hz]")
    ax.set_title("Bi-frequency spectrum")
    plt.savefig("bi_frequency_spectrum.png")

    plt.show()


def check_analytical_lfm_bifreq():

    root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope\imgs\lfm"

    ### LFM Chirp ###
    f0 = 8 * 1e3
    f1 = 15 * 1e3
    fs = 10 * f1
    T = 0.1
    phi = 0

    # Signal
    s, t = lfm_chirp(f0, f1, fs, T, phi)

    # Plot lfm chirp
    fig, ax = plt.subplots()
    ax.plot(t, s)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("LFM Chirp")
    # plt.show()
    plt.savefig(os.path.join(root_img, "lfm_chirp.png"))

    # Plot spectrogram
    nperseg = 2**10
    noverlap = 2**9

    ff, tt, stft = sp.stft(
        s,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )
    plt.figure()
    plt.pcolormesh(tt, ff, 20 * np.log10(np.abs(stft)), shading="gouraud")
    plt.ylim([0, f1 + 2 * 1e3])
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram")
    plt.colorbar(label="Magnitude [dB]")
    plt.savefig(os.path.join(root_img, "spectrogram.png"))

    # Bispectrum from spectrum
    X = np.fft.rfft(s)
    freq = np.fft.rfftfreq(len(s), 1 / fs)

    # Plot spectrum
    plt.figure()
    plt.plot(freq, np.abs(X))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("Spectrum")
    plt.savefig(os.path.join(root_img, "spectrum.png"))

    X = np.atleast_2d(X)
    S_f1f2 = np.conj(X).T @ X

    plt.figure()
    plt.pcolormesh(freq, freq, np.abs(S_f1f2) / np.max(np.abs(S_f1f2)))
    plt.xlabel("$f_1$ [Hz]")
    plt.ylabel("$f_2$ [Hz]")
    plt.title("Bispectrum")
    plt.colorbar(label="Magnitude")
    plt.xlim(f0 * 0.9, f1 * 1.1)
    plt.ylim(f0 * 0.9, f1 * 1.1)
    plt.savefig(os.path.join(root_img, "bispectrum_fromspec.png"))

    fmin = f0
    fmax = f1
    plot = True
    # root_img = "."
    get_bifrequency_spectrum(
        s,
        fs,
        fmin=fmin,
        fmax=fmax,
        nperseg=nperseg,
        noverlap=noverlap,
        plot=plot,
        root_img=root_img,
    )


if __name__ == "__main__":
    data_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\Fiberscope_campagne_oct_2024"
    # date = "09-10-2024"
    date = "10-10-2024"
    data_path = os.path.join(data_root, f"Campagne_{date}")

    # file_name = "09-10-2024T09-44-51-713655_P1_N1_Burst_2.tdms"
    # file_name = "09-10-2024T11-03-11-806485_P1_N1_Sweep_49.tdms"
    # file_name = "09-10-2024T10-34-58-394627_P1_N1_Sweep_34.tdms"
    file_name = "10-10-2024T12-03-02-201689_P4_N1_Sweep_211.tdms"
    file_path = os.path.join(data_path, file_name)

    # # Plot spectrogram
    # ds = load_fiberscope_data(file_path)

    # # plt.figure()
    # # ds.signal.plot(x="time", hue="h_index")
    # # plt.show()

    # ds = ds.sel(h_index=1)
    # ds["signal"] = ds.signal - ds.signal.mean("time")

    # plt.figure()
    # ds.signal.plot(x="time")
    # # plt.xlim(1, 1.2)
    # plt.savefig("signal.png")

    # # # Derive stft
    # nperseg = 2**14
    # noverlap = 2**13
    # ff, tt, stft = sp.stft(
    #     ds.signal,
    #     fs=ds.fs,
    #     window="hann",
    #     nperseg=nperseg,
    #     noverlap=noverlap,
    # )
    # stft = ds.stft_amp * np.exp(1j * ds.stft_phase)
    # stft_abs = np.abs(stft)
    # stft_magnitude = 20 * np.log10(stft_abs)

    # plt.figure()
    # plt.pcolormesh(tt, ff, stft_magnitude, cmap="viridis")
    # plt.colorbar(label="Magnitude [dB]")
    # plt.ylim(0, 20e3)
    # plt.xlim(2, 4)
    # # stft_magnitude.plot(x="tt", y="ff", cmap="viridis", add_colorbar=True)
    # plt.savefig("spectrogram.png")
    # plt.show()

    # check_wss(file_path)

    check_analytical_lfm_bifreq()
