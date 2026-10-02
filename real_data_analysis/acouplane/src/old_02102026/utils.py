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
from scipy.ndimage import uniform_filter1d
from publication.publication_figure import set_subfigures_abc_labels

from real_data_analysis.acouplane.src.acouplane_cst import *

dict_corresp_before_verif = {
    4: "4",
    7: "7",
    1: "a",
    2: "b",
    5: "c",
    6: "d",
}
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


def extract_pulse_windows(
    ds_arr,
    signals,
    times,
    fs,
    offset_s,
    pulse_len_s,
    window_mode="common",
    ref_obs=None,
):
    """Extract the time window of each pulse for each sensor.

    Parameters
    ----------
    ds_arr : xr.Dataset
        Dataset containing the ``arr_dt`` variable (arrival times) with
        dimensions (obs_id, pulse_id).
    signals : dict
        {obs_id: 1D array} received signal of each OBS.
    times : dict
        {obs_id: 1D datetime64 array} time axis of each OBS signal.
    fs : float
        Sampling frequency [Hz].
    offset_s : float
        Time [s] by which the window start precedes the arrival.
    pulse_len_s : float
        Window duration [s].
    window_mode : {"common", "shifted"}
        - "common"  (option 1): a single window shared by all sensors,
          starting `offset_s` before the earliest arrival among `ref_obs`.
        - "shifted" (option 2): one window per sensor, starting `offset_s`
          before its own arrival.
    ref_obs : list of str, optional
        Sensors used to define the window start (required for "common").

    Returns
    -------
    windows : dict
        {obs_id: array (n_pulses, n_win)}; NaN for invalid pulses.
    idx_start : dict
        {obs_id: int array (n_pulses,)} start index of each window.

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """

    n_offset = int(offset_s * fs)
    n_win = int(pulse_len_s * fs)

    # Window start time for each sensor: {obs_id: datetime64 (n_pulses,)}
    if window_mode == "common":
        if not ref_obs:
            raise ValueError("ref_obs is required when window_mode='common'")
        t_first = (
            ds_arr["arr_dt"].sel(obs_id=list(ref_obs)).min("obs_id", skipna=True).values
        )
        t_start = {obs: t_first for obs in signals}
        # Time grid used to convert times to indices: first reference sensor
        t_grid = {obs: np.asarray(times[ref_obs[0]]) for obs in signals}
    elif window_mode == "shifted":
        t_start = {obs: ds_arr["arr_dt"].sel(obs_id=obs).values for obs in signals}
        t_grid = {obs: np.asarray(times[obs]) for obs in signals}
    else:
        raise ValueError(f"Unknown window_mode: {window_mode!r}")

    windows, idx_start = {}, {}
    for obs, sig in signals.items():
        i0_all = np.searchsorted(t_grid[obs], t_start[obs]) - n_offset
        idx_start[obs] = i0_all
        w = np.full((i0_all.size, n_win), np.nan)
        for j, i0 in enumerate(i0_all):
            if np.isnat(t_start[obs][j]) or i0 < 0 or i0 + n_win > len(sig):
                continue  # invalid pulse or window outside the signal -> NaN
            w[j] = sig[i0 : i0 + n_win]
        windows[obs] = w
    return windows, idx_start


def compute_rtf(
    ds_arr,
    signals,
    times,
    num_obs,
    den_obs,
    ref_obs=None,
    window_mode="common",
    fs=2000,
    offset_s=0.1,
    pulse_len_s=None,
    nfft=2**16,
    n_smooth=5,
):
    """Estimate the RTF Pi = S_{num,den} / S_{den,den} for each pulse.

    Non-stationary signal estimator with negligible noise.

    Parameters
    ----------
    ds_arr, signals, times, fs, offset_s, pulse_len_s, window_mode, ref_obs
        See `extract_pulse_windows`.
    num_obs : list of str
        Sensors in the numerator (e.g. [4, 1, 6]).
    den_obs : str
        Sensor in the denominator (e.g. 7).
    nfft : int
        FFT length (zero-padding if larger than the window).
    n_smooth : int
        Length of the moving average applied to the RTF magnitude in dB.

    Returns
    -------
    _type_
        _description_

    """

    all_obs = list(dict.fromkeys([*num_obs, den_obs, *(ref_obs or [])]))
    windows, idx_start = extract_pulse_windows(
        ds_arr,
        {o: signals[o] for o in all_obs},
        times,
        fs,
        offset_s,
        pulse_len_s,
        window_mode=window_mode,
        ref_obs=ref_obs,
    )

    freq = np.fft.rfftfreq(nfft, d=1 / fs)
    fft = {o: np.fft.rfft(windows[o], n=nfft, axis=-1) for o in all_obs}

    # Auto-spectrum of the denominator sensor and RTF for each numerator
    s_den = fft[den_obs] * np.conj(fft[den_obs])
    pi = np.stack([(fft[o] * np.conj(fft[den_obs])) / s_den for o in num_obs])

    gamma = 20 * np.log10(np.abs(pi))
    # Same behavior as np.convolve(..., mode="same") with a boxcar kernel
    gamma_smooth = uniform_filter1d(
        gamma, size=n_smooth, axis=-1, mode="constant", cval=0.0
    )

    rtf_id = [f"{o}/{den_obs}" for o in num_obs]
    dims = ("rtf_id", "pulse_id", "freq")
    return xr.Dataset(
        data_vars=dict(
            pi=(dims, pi),
            gamma=(dims, gamma),
            gamma_smooth=(dims, gamma_smooth),
            idx_start=(
                ("obs_id", "pulse_id"),
                np.stack([idx_start[o] for o in all_obs]),
            ),
        ),
        coords=dict(
            rtf_id=rtf_id, obs_id=all_obs, pulse_id=ds_arr.pulse_id.values, freq=freq
        ),
        attrs=dict(
            fs=fs,
            offset_s=offset_s,
            nfft=nfft,
            den_obs=den_obs,
            window_mode=window_mode,
            n_smooth=n_smooth,
        ),
    )


# ======================================================================================================================
# Plot functions
# ======================================================================================================================


### Positions ###
def plot_obs_pos(df_rcv_pos, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    for id in df_rcv_pos.id:
        rcv = df_rcv_pos.loc[df_rcv_pos["id"] == id]
        plt.scatter(rcv.lon, rcv.lat, marker="d", label=f"{rcv.id.values[0].upper()}")
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title("OBS positions")
    ax.legend()
    return ax


### Signal and spectrogram ###
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


def plot_rtf_map(
    ds_rtf,
    rtf_id,
    window_mode,
    freq_lim=(10, 800),
    percentiles=(5, 95),
    cmap="magma",
    ax=None,
):
    """Plot the RTF magnitude [dB] as a (pulse_id, frequency) map.

    Parameters
    ----------
    ds_rtf : xr.Dataset
        Output of `compute_rtf`, concatenated along "window_mode".
    rtf_id : str
        RTF to display (e.g. "4/7").
    window_mode : str
        Window option to display ("common" or "shifted").
    freq_lim : tuple of float
        Frequency limits [Hz] of the plot.
    percentiles : tuple of float
        Percentiles of the data used as color limits.
    """
    da = ds_rtf["gamma"].sel(rtf_id=rtf_id, window_mode=window_mode)
    vmin, vmax = np.nanpercentile(da.values, percentiles)

    if ax is None:
        _, ax = plt.subplots()
    im = ax.pcolormesh(
        da["freq"], da["pulse_id"], da.values, cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax.set_xlim(*freq_lim)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Replica ID")
    ax.set_title(f"RTF {rtf_id} - window mode: {window_mode}")
    ax.figure.colorbar(im, ax=ax, label="Magnitude [dB]")
    return ax


def plot_rtf_diff(
    ds_rtf,
    rtf_id,
    mode_a="shifted",
    mode_b="common",
    freq_lim=(10, 800),
    clim=10,
    cmap="bwr",
    ax=None,
):
    """Plot the RTF magnitude difference [dB] (mode_a - mode_b) as a map.

    Parameters
    ----------
    mode_a, mode_b : str
        Window modes to compare; the map shows mode_a minus mode_b.
    clim : float
        Symmetric color limit [dB].
    """
    gamma = ds_rtf["gamma"].sel(rtf_id=rtf_id)
    diff = gamma.sel(window_mode=mode_a) - gamma.sel(window_mode=mode_b)

    if ax is None:
        _, ax = plt.subplots()
    im = ax.pcolormesh(
        diff["freq"], diff["pulse_id"], diff.values, cmap=cmap, vmin=-clim, vmax=clim
    )
    ax.set_xlim(*freq_lim)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Replica ID")
    ax.set_title(f"RTF {rtf_id} - {mode_a} minus {mode_b}")
    ax.figure.colorbar(im, ax=ax, label="Difference [dB]")
    return ax
