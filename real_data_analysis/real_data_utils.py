#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   real_data_analysis_1.ipynb
@Time    :   2024/09/06 12:01:19
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
# %matplotlib ipympl

import sys

# sys.path.append(r"C:\Users\baptiste.menetrier\Desktop\devPy\phd")

import os
from numba import jit, njit
import pandas as pd
import numpy as np
import scipy.fft as sf
import scipy.signal as sp
import matplotlib.pyplot as plt

from get_data.ais.ais_tools import *
from get_data.wav.get_data_from_rhumrum import *
from publication.PublicationFigure import PubFigure
from localisation.verlinden.misc.verlinden_utils import load_rhumrum_obs_pos

PubFigure(label_fontsize=22, title_fontsize=24, legend_fontsize=16, ticks_fontsize=20)


def load_wav_data(
    date, duration_s, rcv_id, ch, freq_properties, save=True, root_wav=None
):

    fmin = freq_properties["fmin"]
    fmax = freq_properties["fmax"]
    filter_type = freq_properties["filter_type"]
    filter_corners = freq_properties["filter_corners"]
    nperseg = freq_properties["nperseg"]
    noverlap = freq_properties["noverlap"]

    data = {}
    data["date"] = date
    data["duration_s"] = duration_s
    data["rcv_id"] = rcv_id
    data["ch"] = ch

    raw_sig, filt_sig, corr_sig = get_rhumrum_data(
        station_id=rcv_id,
        date=date,
        duration_sec=duration_s,
        channels=ch,
        plot=False,
        fmin=fmin,
        fmax=fmax,
        filter_type=filter_type,
        filter_corners=filter_corners,
        save=save,
        root_wav=root_wav,
    )

    sig = corr_sig["BDH"]
    data["data"] = sig.data
    data["sig"] = sig

    # Derive stft
    f, tt, stft = sp.stft(
        data["data"],
        fs=sig.meta.sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )
    data["f"] = f
    data["tt"] = tt
    data["stft"] = stft

    # Create a time vector composed of datetime objects for the x-axis
    t0 = data["sig"].meta.starttime.datetime
    t1 = data["sig"].meta.endtime.datetime
    dtt = data["tt"][1] - data["tt"][0]
    tt_datetime = pd.date_range(start=t0, end=t1, periods=len(data["tt"]))

    data["tt_datetime"] = tt_datetime
    data["day_str"] = t0.strftime("%Y-%m-%d")

    root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\real_data"
    data["root_img"] = os.path.join(root, t0.strftime("%Y-%m"), t0.strftime("%Y-%m-%d"))

    if not os.path.exists(data["root_img"]):
        os.makedirs(data["root_img"])

    return data


def get_dsp(data, dsp_args):

    ff, Pxx = sp.welch(
        data["data"],
        fs=data["sig"].meta.sampling_rate,
        nperseg=dsp_args["nperseg"],
        noverlap=int(dsp_args["nperseg"] * dsp_args["overlap_coef"]),
        window=dsp_args["window"],
    )
    Pxx_in_band = Pxx[(ff >= dsp_args["fmin"]) & (ff <= dsp_args["fmax"])]
    ff_in_band = ff[(ff >= dsp_args["fmin"]) & (ff <= dsp_args["fmax"])]

    # Find peaks in coherence to locate first harmonic
    distance_samples = np.floor(1.5 / (ff_in_band[1] - ff_in_band[0]))
    idx_f_peaks, _ = sp.find_peaks(
        10 * np.log10(Pxx_in_band), distance=distance_samples
    )
    f_peaks = ff_in_band[idx_f_peaks]
    k = np.arange(1, len(f_peaks) + 1)
    harmonic_labels = []
    j = 0
    for i in range(0, len(f_peaks)):
        if i % 2 == 0:
            harmonic_labels.append(f"${k[j]}f_0$")
            j += 1
        else:
            harmonic_labels.append(r"$\frac{" + f"{2+i}" + r"}{2}f_0$")

    f0 = f_peaks[0]

    data["ff"] = ff
    data["Pxx"] = Pxx
    data["ff_in_band"] = ff_in_band
    data["Pxx_in_band"] = Pxx_in_band
    data["f_peaks"] = f_peaks
    data["idx_f_peaks"] = idx_f_peaks
    data["harmonic_labels"] = harmonic_labels
    data["f0"] = f0


def plot_stft(data, save=False):
    freq_inside = (data["f"] > 8) * (data["f"] < 42)
    vmin = np.percentile(20 * np.log10(np.abs(data["stft"][freq_inside])), 20)
    vmax = np.percentile(20 * np.log10(np.abs(data["stft"])), 99)
    plt.figure(figsize=(14, 6))
    plt.pcolormesh(
        data["tt_datetime"],
        data["f"],
        20 * np.log10(np.abs(data["stft"])),
        vmin=vmin,
        vmax=vmax,
        cmap="jet",
        shading="gouraud",
    )
    plt.ylabel(r"$f \, \textrm{[Hz]}$")
    plt.xlabel(r"$t \, \textrm{[s]}$")
    cbar = plt.colorbar()
    cbar.set_label(r"$\textrm{Amplitude [dB]}$")

    # Format x-axis with time and show only the date once
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.gca().xaxis.set_major_locator(
        mdates.MinuteLocator(interval=30)
    )  # Two tick per hour
    plt.gca().xaxis.set_minor_locator(
        mdates.MinuteLocator(interval=15)
    )  # Minor ticks every 15 minutes

    # Add the date as a single label above the axis
    plt.gcf().autofmt_xdate()
    date_str = data["tt_datetime"][0].strftime("%Y-%m-%d")

    title = f"{data['rcv_id']} - {date_str}"
    plt.title(title)
    # plt.annotate(
    #     date_str, xy=(0.05, -0.2), xycoords="axes fraction", ha="center", fontsize=18
    # )

    if save:
        # Create dedicated folder
        folder = os.path.join(data["root_img"], "stft")
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Build filename
        start_hour = data["tt_datetime"][0].strftime("%H-%M-%S")
        fname = f"{data['rcv_id']}_{data['day_str']}_{start_hour}_stft.png"
        fpath = os.path.join(folder, fname)
        plt.savefig(fpath)
        plt.close()


def plot_dsp(data, fmin, fmax, save=False):
    plt.figure()
    plt.plot(data["ff_in_band"], 10 * np.log10(data["Pxx_in_band"]))
    # Add detected peaks
    plt.scatter(
        data["f_peaks"],
        10 * np.log10(data["Pxx_in_band"][data["idx_f_peaks"]]),
        color="r",
    )

    for i in range(len(data["f_peaks"])):
        plt.annotate(
            f"{i}",
            (
                data["f_peaks"][i],
                10 * np.log10(data["Pxx_in_band"][data["idx_f_peaks"][i]]),
            ),
            xytext=(
                data["f_peaks"][i] + 0.1,
                10 * np.log10(data["Pxx_in_band"][data["idx_f_peaks"][i]]) + 0.5,
            ),
            ha="left",
        )

    # Add labels
    # for i, txt in enumerate(harmonic_labels):
    #     plt.annotate(
    #         txt,
    #         (f_peaks[i], 10 * np.log10(Pxx_in_band[idx_f_peaks[i]])),
    #         xytext=(f_peaks[i] + 0.1, 10 * np.log10(Pxx_in_band[idx_f_peaks[i]]) + 0.5),
    #         ha="left",
    #     )

    plt.xlabel(r"$f \, \textrm{[Hz]}$")
    plt.ylabel(r"$S_{xx}(f) \, \textrm{[dB (Pa$^2$ Hz$^{-1}$)]}$")
    plt.grid()
    plt.xlim(fmin, fmax)

    date_str = data["tt_datetime"][0].strftime("%Y-%m-%d")
    title = f"{data['rcv_id']} - {date_str}"
    plt.title(title)

    if save:

        # Create dedicated folder
        folder = os.path.join(data["root_img"], f"dsp_{fmin:.0f}_{fmax:.0f}")
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Build filename
        start_hour = data["tt_datetime"][0].strftime("%H-%M-%S")
        fname = f"{data['rcv_id']}_{data['day_str']}_{fmin:.2f}_{fmax:.2f}_{start_hour}_dsp.png"
        fpath = os.path.join(folder, fname)
        plt.savefig(fpath)
        plt.close()


def load_and_preprocess_ais_data():
    root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ais\extract-ais-pos-for-zone-ecole-navale-by-month-201305.csv"
    fname = "extract-ais-pos-for-zone-ecole-navale-by-month-201305.csv"
    fpath = os.path.join(root, fname)

    lon_min = 65
    lon_max = 66
    lat_min = -28
    lat_max = -27

    # Load and pre-filter
    df = extract_ais_area(fpath, lon_min, lon_max, lat_min, lat_max)
    # Remove ships with less than 2 points
    df = df.groupby("mmsi").filter(lambda x: len(x) > 1)
    # Interpolate trajectories to have a point every 5 minutes
    df_interp = interpolate_trajectories(df, time_step="5min")

    return df_interp


def compute_csd_matrix_fast(stfts, n_seg_cov):
    """
    Compute the Cross Spectral Density (CSD) matrix for a set of receivers using matrix operations.

    Args:
    - stfts: list of 2D STFT matrices (frequency bins x time snapshots), one per receiver.
    - n_seg_cov: Number of time snapshots to average over (number of segments per block).

    Returns:
    - csd_matrix: 3D CSD matrix (frequency bins x num_receivers x num_receivers).
    """
    num_receivers = len(stfts)
    num_freq_bins, num_snapshots = stfts[0].shape

    if n_seg_cov == 0:
        n_seg_cov = num_snapshots

    n_available_segments = num_snapshots // n_seg_cov

    # Convert list of arrays into a single array
    stacked_stfts = np.asarray(
        stfts
    )  # Shape: (num_receivers, num_freq_bins, num_snapshots)
    stacked_stfts = np.moveaxis(
        stacked_stfts, 0, -1
    )  # (num_freq_bins, num_snapshots, num_receivers)

    # Preallocate CSD matrix
    csd_matrix = np.empty(
        (num_freq_bins, num_receivers, num_receivers, n_available_segments),
        dtype=np.complex128,
    )

    # Compute CSD matrix using batch operations
    for k in range(n_available_segments):
        idx_start = k * n_seg_cov
        stft_block = stacked_stfts[
            :, idx_start : idx_start + n_seg_cov, :
        ]  # View-based slicing
        stft_block_conj = np.conj(stft_block)  # Precompute conjugate

        csd_matrix[..., k] = (
            np.einsum("ftr,fts->frs", stft_block, stft_block_conj) / n_seg_cov
        )

    return np.squeeze(csd_matrix, axis=-1) if n_available_segments == 1 else csd_matrix


def get_csdm_snapshot_number(y, fs, nperseg, noverlap):
    """Derive the number of snapshots in the stft used to compute the CSDM."""

    ff, tt, stft = sp.stft(
        y,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )

    k = len(tt)

    return k
