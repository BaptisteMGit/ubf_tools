#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2026/09/28 21:54:27
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import soundfile as sf
import scipy.signal as sp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta
from publication.publication_figure import PubFigure, set_subfigures_abc_labels

project_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
root_data = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\acouplane\data"
)
root_bathy_data = os.path.join(project_root, "data", "bathy")
root_metadata = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ACOUPLANE\METADATA"

PubFigure()

dict_corresp_before_verif = {
    4: "4",
    7: "7",
    1: "a",
    2: "b",
    5: "c",
    6: "d",
}

source_position_dataset_fpath = os.path.join(root_data, "acouplane_source_position.nc")
wav_dataset_fpath = os.path.join(root_data, "acouplane_sample_wav.nc")
obs_position_fpath = os.path.join(root_metadata, "obs_pos.txt")
datetime_format = "%Y-%m-%d_%H-%M-%S"
# ======================================================================================================================
# Functions
# ======================================================================================================================


def load_signal(fpath, center=True):
    # Load signal from wav file
    signal, fs = sf.read(fpath)
    ns = signal.size
    # Get time vector
    time = np.arange(signal.shape[0]) / fs
    if center:
        # Centre signal
        signal -= np.mean(signal)

    return time, signal, fs, ns


def plot_sequence(
    ds_wav, start_analysis_dt, analysis_duration_s, nperseg=2**8, alpha_overlap=0.75
):

    datetime_fmt = ds_wav.attrs["datetime_format"]

    # Analysis bounds
    t_start = datetime.strptime(start_analysis_dt, datetime_fmt)
    t_end = t_start + timedelta(seconds=analysis_duration_s)

    # Signal
    plot_signal(ds_wav, t_start, t_end)

    # Spectrogram
    noverlap = int(nperseg * alpha_overlap)
    plot_spectro(ds_wav, t_start, t_end, nperseg, noverlap)


def plot_spectro(ds_wav, t_start, t_end, nperseg, noverlap):

    n_obs = ds_wav.sizes["obs_id"]
    fig, axs = plt.subplots(n_obs, 1, sharex=True, figsize=(16, 12))
    axs = np.atleast_1d(axs)

    sxx_plot = []

    for i, id_ in enumerate(ds_wav.obs_id.values):

        obs_id = dict_corresp_before_verif[id_]  # UPDATE TODO after verif

        datetime_fmt = ds_wav.attrs["datetime_format"]

        sig_varname = f"signal_obs{obs_id}"
        time_coordsname = f"time{obs_id}"
        signal = ds_wav[sig_varname]

        # Select a window of the signal
        # fs = ds_wav.attrs[f"fs_obs{obs_id}"]
        fs = ds_wav.fs.sel(obs_id=id_).values

        # Start of recording
        t0 = ds_wav.attrs[f"start_datetime_obs{obs_id}"]
        t0 = datetime.strptime(t0, datetime_fmt)

        # Select window
        t_from_t0_start_s = (t_start - t0).total_seconds()
        n_start = int(t_from_t0_start_s * fs)
        t_from_t0_end_s = (t_end - t0).total_seconds()
        n_end = int(t_from_t0_end_s * fs)

        # Slice signal
        sig_win = signal.isel({time_coordsname: slice(n_start, n_end)})

        # Define datetime borders
        t0_slice = t0 + timedelta(seconds=n_start * 1 / fs)
        t1_slice = t0 + timedelta(seconds=n_end * 1 / fs)

        # Derive stft
        ff, tt, stft = sp.stft(
            sig_win.values,  # .values -> ici on charge les données en mémoire (un tout petit subset seulement)
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="psd",  # U^2 / Hz
        )
        sxx = 10 * np.log10(np.abs(stft))  # dB re 1uPa**2 / Hz
        # Associated datetime vector
        tt_datetime = pd.date_range(
            t0_slice,
            t0_slice + timedelta(seconds=tt[-1]),
            freq=f"{tt[1]-tt[0]}s",
            inclusive="both",
        )

        sxx_plot.append(sxx)

    # Plot
    cmap = "magma"
    sxx_plot = np.array(sxx_plot)
    vmin = np.percentile(sxx_plot, 10)
    vmax = np.percentile(sxx_plot, 99)

    for i, id_ in enumerate(ds_wav.obs_id.values):
        obs_id = dict_corresp_before_verif[id_]  # UPDATE TODO after verif

        im = axs[i].pcolormesh(
            tt_datetime, ff, sxx_plot[i, ...], cmap=cmap, vmin=vmin, vmax=vmax
        )
        # axs[i].set_title(f"OBS{obs_id}")
        # axs[i].set_ylim([fmin, fmax])

    clabel = r"dB re 1$\mu$Pa$^2$ / Hz"
    fig.colorbar(
        im,
        ax=axs.ravel().tolist(),
        label=clabel,
        orientation="vertical",
        fraction=1.0,
        pad=0.03,
    )

    formatter = mdates.DateFormatter("%H:%M:%S")
    axs[-1].xaxis.set_major_formatter(formatter)
    formatter = mdates.DateFormatter("%H:%M:%S")
    axs[-1].xaxis.set_major_formatter(formatter)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    axs[-1].xaxis.set_major_locator(locator)
    plt.setp(axs[-1].get_xticklabels(), rotation=15, ha="right")

    fig.supylabel("Frequency [Hz]")
    fig.supxlabel("Time (UTC)")

    set_subfigures_abc_labels(
        axs=axs,
        fontsize=18,
        x_pos=0.995,
        y_pos=1.02,
        ha="right",
        va="bottom",
    )


def plot_signal(ds_wav, t_start, t_end):

    n_obs = ds_wav.sizes["obs_id"]
    fig, axs = plt.subplots(n_obs, 1, sharex=True, figsize=(16, 12))
    axs = np.atleast_1d(axs)

    for i, id_ in enumerate(ds_wav.obs_id.values):

        obs_id = dict_corresp_before_verif[id_]  # UPDATE TODO after verif

        datetime_fmt = ds_wav.attrs["datetime_format"]

        sig_varname = f"signal_obs{obs_id}"
        time_coordsname = f"time{obs_id}"
        signal = ds_wav[sig_varname]

        # Select a window of the signal
        # fs = ds_wav.attrs[f"fs_obs{obs_id}"]
        fs = ds_wav.fs.sel(obs_id=id_).values

        # Start of recording
        t0 = ds_wav.attrs[f"start_datetime_obs{obs_id}"]
        t0 = datetime.strptime(t0, datetime_fmt)

        # Select window
        t_from_t0_start_s = (t_start - t0).total_seconds()
        n_start = int(t_from_t0_start_s * fs)
        t_from_t0_end_s = (t_end - t0).total_seconds()
        n_end = int(t_from_t0_end_s * fs)

        # Slice signal
        sig_win = signal.isel({time_coordsname: slice(n_start, n_end)})

        # Define datetime borders
        t0_slice = t0 + timedelta(seconds=n_start * 1 / fs)
        t1_slice = t0 + timedelta(seconds=n_end * 1 / fs)

        # Associated datetime vector
        tt_datetime = pd.date_range(
            t0_slice,
            t1_slice,
            freq=f"{1 / fs}s",
            inclusive="left",
        )

        # Plot
        axs[i].plot(tt_datetime, sig_win)
        # axs[i].set_title(f"OBS{obs_id}")

    formatter = mdates.DateFormatter("%H:%M:%S")
    axs[-1].xaxis.set_major_formatter(formatter)
    formatter = mdates.DateFormatter("%H:%M:%S")
    axs[-1].xaxis.set_major_formatter(formatter)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    axs[-1].xaxis.set_major_locator(locator)
    plt.setp(axs[-1].get_xticklabels(), rotation=15, ha="right")

    fig.supylabel("Frequency [Hz]")
    fig.supxlabel("Time (UTC)")

    set_subfigures_abc_labels(
        axs=axs,
        fontsize=18,
        x_pos=0.995,
        y_pos=1.02,
        ha="right",
        va="bottom",
    )


def apply_sta_lta(
    ds_wav,
    sta_duration,
    lta_duration,
    start_analysis_dt,
    analysis_duration_s,
    threshold_obs4,
    threshold_obs7,
):

    datetime_fmt = ds_wav.attrs["datetime_format"]

    # Analysis bounds
    t_start = datetime.strptime(start_analysis_dt, datetime_fmt)
    t_end = t_start + timedelta(seconds=analysis_duration_s)

    thresholds = [threshold_obs4, threshold_obs7]

    # Arrivals
    arrivals_dt = []
    sig_dt = []
    signals = []
    arrivals_dt_idx = []

    for i, id_ in enumerate(ds_wav.obs_id.values):

        obs_id = dict_corresp_before_verif[id_]  # UPDATE TODO after verif

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(16, 12))
        axs = np.atleast_1d(axs)

        datetime_fmt = ds_wav.attrs["datetime_format"]

        sig_varname = f"signal_obs{obs_id}"
        time_coordsname = f"time{obs_id}"
        signal = ds_wav[sig_varname]

        # Select a window of the signal
        # fs = ds_wav.attrs[f"fs_obs{obs_id}"]
        fs = ds_wav.fs.sel(obs_id=id_).values

        # Start of recording
        t0 = ds_wav.attrs[f"start_datetime_obs{obs_id}"]
        t0 = datetime.strptime(t0, datetime_fmt)

        # Select window
        t_from_t0_start_s = (t_start - t0).total_seconds()
        n_start = int(t_from_t0_start_s * fs)
        t_from_t0_end_s = (t_end - t0).total_seconds()
        n_end = int(t_from_t0_end_s * fs)

        # Slice signal
        sig_win = signal.isel({time_coordsname: slice(n_start, n_end)})
        signals.append(sig_win)

        # STA/LTA params
        sta_n = int(sta_duration * fs)
        lta_n = int(lta_duration * fs)
        # STA
        sig_sta = np.abs(sig_win).rolling({time_coordsname: sta_n}, center=True).mean()
        # LTA
        sig_lta = np.abs(sig_win).rolling({time_coordsname: lta_n}, center=True).mean()
        # STA / LTA
        sta_lta = sig_sta.values / sig_lta.values
        # Normalize
        sta_lta /= np.nanmax(sta_lta)

        # Detect

        # Define datetime borders
        t0_slice = t0 + timedelta(seconds=n_start * 1 / fs)
        t1_slice = t0 + timedelta(seconds=n_end * 1 / fs)

        # Associated datetime vector
        tt_datetime = pd.date_range(
            t0_slice,
            t1_slice,
            freq=f"{1 / fs}s",
            inclusive="left",
        )
        sig_dt.append(tt_datetime)

        # Plot
        axs[0].plot(tt_datetime, sig_win, color="k")
        axs[1].plot(tt_datetime, sig_sta, color="r", label="STA")
        axs[1].plot(tt_datetime, sig_lta, color="b", label="LTA")
        axs[2].plot(tt_datetime, sta_lta, color="k", label="STA/LTA")

        # Add threshold
        threshold = thresholds[i]
        axs[2].axhline(
            threshold, color="r", linestyle="--", label="Detection threshold"
        )
        detect_mask = sta_lta >= threshold
        axs[2].plot(
            tt_datetime,
            detect_mask,
            color="b",
        )

        # Detect arrivals
        diff_mask = np.diff(detect_mask.astype(int))
        up_front = diff_mask > 0
        up_front = np.append(up_front, False)
        arr_dt_idx = np.arange(tt_datetime.size)[up_front]
        arrivals_dt_idx.append(arr_dt_idx)
        arr_dt = tt_datetime[up_front]
        arrivals_dt.append(arr_dt)

        for arr in arr_dt:
            axs[0].axvline(arr, color="r", linestyle="--")

        for ax in axs:
            ax.legend()

        fig.suptitle(f"OBS{obs_id}")

        formatter = mdates.DateFormatter("%H:%M:%S")
        axs[-1].xaxis.set_major_formatter(formatter)
        formatter = mdates.DateFormatter("%H:%M:%S")
        axs[-1].xaxis.set_major_formatter(formatter)
        locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
        axs[-1].xaxis.set_major_locator(locator)
        plt.setp(axs[-1].get_xticklabels(), rotation=15, ha="right")

        fig.supylabel("Amplitude")
        fig.supxlabel("Time (UTC)")

        set_subfigures_abc_labels(
            axs=axs,
            fontsize=18,
            x_pos=0.995,
            y_pos=1.02,
            ha="right",
            va="bottom",
        )

    return sig_dt, signals, arrivals_dt, arrivals_dt_idx


def apply_sta_lta_only_first(
    ds_wav,
    sta_duration,
    lta_duration,
    start_analysis_dt,
    analysis_duration_s,
    thresholds,
    pulse_interval_s=3,
    fs=2000,
):

    pulse_sample_offset = int(pulse_interval_s * fs)
    datetime_fmt = ds_wav.attrs["datetime_format"]

    # Analysis bounds
    t_start = datetime.strptime(start_analysis_dt, datetime_fmt)
    t_end = t_start + timedelta(seconds=analysis_duration_s)

    # Arrivals
    arrivals_dt = []
    sig_dt = []
    signals = []
    arrivals_dt_idx = []

    for i, id_ in enumerate(ds_wav.obs_id.values):

        obs_id = dict_corresp_before_verif[id_]  # UPDATE TODO after verif

        fig, axs = plt.subplots(1, 1, sharex=True, figsize=(16, 12))
        axs = np.atleast_1d(axs)

        datetime_fmt = ds_wav.attrs["datetime_format"]

        sig_varname = f"signal_obs{obs_id}"
        time_coordsname = f"time{obs_id}"
        signal = ds_wav[sig_varname]

        # Select a window of the signal
        # fs = ds_wav.attrs[f"fs_obs{obs_id}"]
        fs = ds_wav.fs.sel(obs_id=id_).values

        # Start of recording
        t0 = ds_wav.attrs[f"start_datetime_obs{obs_id}"]
        t0 = datetime.strptime(t0, datetime_fmt)

        # Select window
        t_from_t0_start_s = (t_start - t0).total_seconds()
        n_start = int(t_from_t0_start_s * fs)
        t_from_t0_end_s = (t_end - t0).total_seconds()
        n_end = int(t_from_t0_end_s * fs)

        # Slice signal
        sig_win = signal.isel({time_coordsname: slice(n_start, n_end)})
        signals.append(sig_win)

        # STA/LTA params
        sig_win_sta_lta = signal.isel(
            {time_coordsname: slice(n_start, n_start + pulse_sample_offset * 3)}
        )
        sta_n = int(sta_duration * fs)
        lta_n = int(lta_duration * fs)
        # STA
        sig_sta = (
            np.abs(sig_win_sta_lta)
            .rolling({time_coordsname: sta_n}, center=True)
            .mean()
        )
        # LTA
        sig_lta = (
            np.abs(sig_win_sta_lta)
            .rolling({time_coordsname: lta_n}, center=True)
            .mean()
        )
        # STA / LTA
        sta_lta = sig_sta.values / sig_lta.values
        # Normalize
        sta_lta /= np.nanmax(sta_lta)

        # Define datetime borders
        t0_slice = t0 + timedelta(seconds=n_start * 1 / fs)
        t1_slice = t0 + timedelta(seconds=n_end * 1 / fs)

        # Associated datetime vector
        tt_datetime = pd.date_range(
            t0_slice,
            t1_slice,
            freq=f"{1 / fs}s",
            inclusive="left",
        )
        tt_datetime_sta_lta = pd.date_range(
            t0_slice,
            t0 + timedelta(seconds=(n_start + pulse_sample_offset * 3) * 1 / fs),
            freq=f"{1 / fs}s",
            inclusive="left",
        )

        sig_dt.append(tt_datetime)

        # Plot
        axs[0].plot(tt_datetime, sig_win, color="k")

        # Add threshold
        threshold = thresholds[i]
        detect_mask = sta_lta >= threshold

        # Detect arrivals
        diff_mask = np.diff(detect_mask.astype(int))
        up_front = diff_mask > 0
        up_front = np.append(up_front, False)
        arr_dt_idx = np.arange(tt_datetime_sta_lta.size)[up_front]
        # arr_dt = tt_datetime[up_front]

        # Keep only first detection and get the other using interval
        arr_dt_idx_0 = arr_dt_idx[0]
        k = 0
        arr_dt_idx_k = arr_dt_idx_0 + pulse_sample_offset * k
        arr_dt_idx = []
        while arr_dt_idx_k < tt_datetime.size:
            arr_dt_idx.append(arr_dt_idx_k)
            k += 1
            arr_dt_idx_k = arr_dt_idx_0 + pulse_sample_offset * k

        arr_dt_idx = np.array(arr_dt_idx)
        arr_dt = tt_datetime[arr_dt_idx]

        arrivals_dt_idx.append(arr_dt_idx)
        arrivals_dt.append(arr_dt)

        for arr in arr_dt:
            axs[0].axvline(arr, color="r", linestyle="--")

        for ax in axs:
            ax.legend()
            ax.set_xlim(tt_datetime_sta_lta[0], tt_datetime_sta_lta[-1])

        fig.suptitle(f"OBS{obs_id}")

        formatter = mdates.DateFormatter("%H:%M:%S")
        axs[-1].xaxis.set_major_formatter(formatter)
        formatter = mdates.DateFormatter("%H:%M:%S")
        axs[-1].xaxis.set_major_formatter(formatter)
        locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
        axs[-1].xaxis.set_major_locator(locator)
        plt.setp(axs[-1].get_xticklabels(), rotation=15, ha="right")

        fig.supylabel("Amplitude")
        fig.supxlabel("Time (UTC)")

        set_subfigures_abc_labels(
            axs=axs,
            fontsize=18,
            x_pos=0.995,
            y_pos=1.02,
            ha="right",
            va="bottom",
        )

    return sig_dt, signals, arrivals_dt, arrivals_dt_idx
