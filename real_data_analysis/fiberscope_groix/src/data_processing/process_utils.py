#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   process_utils.py
@Time    :   2026/01/20 10:57:11
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Utils for data processing
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
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import butter, lfilter
import scipy.signal as sp

# ======================================================================================================================
# Defaults
# ======================================================================================================================
PROJECT_ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"

SBE39_OBS_FOLDER = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\fiberscope_groix_oct_2025\SBE39_OBS"

ROOT_GROIX_DATA = os.path.join(PROJECT_ROOT, "data", "fiberscope_groix_oct_2025")
ROOT_GROIX_METADATA = os.path.join(ROOT_GROIX_DATA, "metadata")
ROOT_GROIX_WAV = os.path.join(ROOT_GROIX_DATA, "wav")

ROOT_FOLDER = os.path.join(PROJECT_ROOT, "real_data_analysis", "fiberscope_groix")
DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")
IMG_FOLDER = os.path.join(ROOT_FOLDER, "img")


# ======================================================================================================================
# Functions
# ======================================================================================================================
def get_available_wav_files(obs_id, root_groix_wav=ROOT_GROIX_WAV):
    """
    Get available wav files for a given OBS id.

    Parameters:
    obs_id : int
        The OBS id (1-indexed, i.e., OBS1 -> obs_id=1).
    root_groix_wav : str
        The root directory where wav files are stored.
    Returns:
    wav_start_times : dict
        Dictionary mapping wav file start datetimes to their file paths.
    start_datetime_arr : np.ndarray of datetime.datetime
        Array of wav file start datetimes.
    """

    # Get the list of wav file available for selected obs_id
    obs_wav_dir = os.path.join(root_groix_wav, f"OBS{obs_id+3}", "vitesse")
    # List all wav files in the directory
    wav_files = os.listdir(obs_wav_dir)
    # Extract start times from filenames
    wav_start_times = {}
    for wav_file in wav_files:
        if wav_file.endswith(".wav"):
            date = wav_file.split("_")[2]
            time = wav_file.split("_")[3].split("-vel")[0]
            timestamp_str = f"{date}_{time}"
            date_fmt = "%Y-%m-%d_%H-%M-%S"
            file_start_datetime = datetime.strptime(timestamp_str, date_fmt)
            wav_start_times[file_start_datetime] = os.path.join(obs_wav_dir, wav_file)

    start_datetime_arr = np.array(list(wav_start_times.keys()))

    return wav_start_times, start_datetime_arr


def get_wav_file_for_emission(emission_datetime, start_datetime_arr, wav_start_times):
    """
    Get the wav file corresponding to a given emission time. The function finds the wav file that starts before the emission time.
    Note: we assume that the period of interest is fully contained in a single wav file.

    Parameters:
    emission_datetime : datetime.datetime
        The emission datetime.
    start_datetime_arr : np.ndarray of datetime.datetime
        Array of wav file start datetimes (from get_available_wav_files).
    wav_start_times : dict
        Dictionary mapping wav file start datetimes to their file paths (from get_available_wav_files).
    Returns:
    wav_fpath : str
        The file path of the corresponding wav file.
    wav_start_datetime : datetime.datetime
        The start datetime of the corresponding wav file.
    """

    # Compute time differences between emission and wav start times
    dt = emission_datetime - start_datetime_arr
    # Keep only positive time differences (we are only looking for wav files that start before the emission)
    dt_is_positive = dt >= pd.Timedelta(0)
    # Find closest
    wav_start_datetime = start_datetime_arr[dt_is_positive][-1]
    # Get corresponding wav file path
    wav_fpath = wav_start_times[wav_start_datetime]

    return wav_fpath, wav_start_datetime


def get_tr_apriori(
    emission_pos, ds_gps, wav_start_datetime, emission_datetime, obs_id, c0=1500
):
    """
    Get apriori reception time and travel time based on emission position and receiver apriori position.
    Parameters:
    emission_pos : lidt or tuple of float
        The emission position (easting, northing, up) in meters (in the local ENU reference frame).
    ds_gps : xarray.Dataset
        The GPS dataset containing receiver apriori positions.
    wav_start_datetime : datetime.datetime
        The start datetime of the wav file.
    emission_datetime : datetime.datetime
        The emission datetime.
    obs_id : int
        The OBS id (1-indexed, i.e., OBS1 -> obs_id=1).
    c0 : float
        The sound speed in m/s (default is 1500 m/s).
    Returns:
    reception_datetime : datetime.datetime
        The apriori reception datetime.
    tr : float
        The apriori reception time from wav start in seconds.
    prop_time : float
        The apriori propagation time in seconds.
    """

    # Extract emission position
    e_emission, n_emission, u_emission = emission_pos

    # Get receiver apriori position
    e_obs = ds_gps.attrs[f"obs{obs_id}_e_apriori"]
    n_obs = ds_gps.attrs[f"obs{obs_id}_n_apriori"]
    u_obs = ds_gps.attrs[f"obs{obs_id}_u_apriori"]

    # Derive apriori propagation distance
    prop_range = np.sqrt(
        (e_emission - e_obs) ** 2
        + (n_emission - n_obs) ** 2
        + (u_emission - u_obs) ** 2
    )

    # Derive apriori propagation time
    prop_time = prop_range / c0

    # Deduce apriori reception time
    reception_datetime = emission_datetime + pd.Timedelta(prop_time, "s")

    # Derive reception time from wav start
    tr = reception_datetime - wav_start_datetime
    # Convert to seconds
    tr = tr.total_seconds()

    return reception_datetime, tr, prop_time


def apply_match_filter(signal, fs, f0, f1, T):
    """
    Apply match filtering to a chirp signal.

    Parameters:
    signal : ndarray
        The received signal samples to be processed.
    fs : float
        Sampling frequency in Hz.
    f0 : float
        Start frequency of the chirp in Hz.
    f1 : float
        End frequency of the chirp in Hz.
    T : float
        Duration of the chirp in seconds.

    Returns:
    lags : ndarray
        The time lags corresponding to the matched filtered signal.
    sig_mf : ndarray
        The matched filtered signal.

    """

    # Generate the reference chirp signal
    t = np.arange(0, T, 1 / fs)
    reference_chirp = sp.chirp(t, f0=f0, f1=f1, t1=T, method="linear")

    # Perform matched filtering (cross-correlation)
    sig_mf = np.correlate(signal, reference_chirp, mode="same")
    lags = sp.correlation_lags(t.size, t.size, mode="same") * 1 / fs

    return lags, sig_mf


def plot_arrivals_detection(
    t_win,
    signal_win,
    sig_mf,
    t_arrivals,
    peaks_idx,
    peak_times,
    sequence_info,
    signal_params,
    first_emission_reception_datetime=None,
    last_emission_reception_datetime=None,
    plot_last_first=False,
    t_hydro_source_offset=31,
    save=False,
    img_root=None,
    fs=2000,
    nperseg=64,
    noverlap=32,
    first_emission_in_sequence_datetime=None,
    last_emission_in_sequence_datetime=None,
    verbose=False,
    plot_zoom=False,
):

    if verbose:
        print("Plotting arrivals...")

    if img_root is None:
        img_root = os.path.join(IMG_FOLDER, "reception", "arrivals_detection")

    # Unwrap sequence information
    seq_id = sequence_info.get("seq_id", None)
    obs_id = sequence_info.get("obs_id", None)
    vc_carte = sequence_info.get("vc_carte", None)
    signal_type = sequence_info.get("signal_type", None)
    emission_type = sequence_info.get("emission_type", None)

    # Normalize signal for better visualization
    signal_win = signal_win / np.max(np.abs(signal_win))

    # Compute spectrogram for visualization
    ff, tt, Sxx = sp.stft(
        signal_win,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="psd",
    )
    tt_datetime = pd.date_range(
        t_win[0],
        t_win[0] + pd.Timedelta(tt[-1], "s"),
        freq=f"{tt[1]-tt[0]}s",
        inclusive="both",
    )

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    # Plot raw signal
    axs[0].plot(t_win, signal_win, color="k")
    axs[0].set_ylim([-1, 1])
    # Plot matched filtered signal
    axs[1].plot(t_win, sig_mf, color="k")
    axs[1].set_ylim([-1, 1])

    # Add detected peaks
    if len(peaks_idx) > 0:
        axs[1].scatter(
            peak_times,
            sig_mf[peaks_idx],
            marker="o",
            color="r",
            label=r"$t_{\text{peak}}$",
            s=5,
            zorder=10,
        )

    axs[0].set_ylabel("s")
    axs[1].set_ylabel(r"$s_{mf}$")

    im = axs[2].pcolormesh(tt_datetime, ff, 10 * np.log10(np.abs(Sxx)), cmap="jet")
    # fig.colorbar(im, cax=axs[2], label=r"Pa$^2$ / Hz in dB")
    axs[2].set_ylabel("Fréquence [Hz]")

    # Add arrivals
    if len(t_arrivals) > 0:
        # Plot arrows for arrivals
        for iarr, t_arrival in enumerate(t_arrivals):
            axs[0].annotate(
                f"{iarr}",
                xy=(t_arrival, 0.5),
                xytext=(t_arrival, 0.9),
                arrowprops=dict(arrowstyle="->", color="red"),
                horizontalalignment="center",
                verticalalignment="center",
            )
            axs[2].annotate(
                f"{iarr}",
                xy=(t_arrival, np.max(ff) * 0.75),
                xytext=(t_arrival, np.max(ff) * 0.95),
                arrowprops=dict(arrowstyle="->", color="black"),
                horizontalalignment="center",
                verticalalignment="center",
            )

    if plot_last_first:
        for i in range(len(axs)):
            if (
                first_emission_reception_datetime is None
                or last_emission_reception_datetime is None
            ):
                raise ValueError(
                    "first_emission_reception_datetime and last_emission_reception_datetime must be provided when plot_last_first is True."
                )

            axs[i].axvline(
                first_emission_reception_datetime,
                color="g",
                linestyle="--",
                label=r"$T_{\text{reception}}(\text{first})$",
            )
            axs[i].axvline(
                last_emission_reception_datetime,
                color="b",
                linestyle="--",
                label=r"$T_{\text{reception}}(\text{last})$",
            )
            if first_emission_in_sequence_datetime is not None:
                axs[i].axvline(
                    first_emission_in_sequence_datetime,
                    color="red",
                    linestyle="--",
                    label=r"$T_{\text{emission}}(\text{first})$",
                )
            if last_emission_in_sequence_datetime is not None:
                axs[i].axvline(
                    last_emission_in_sequence_datetime,
                    color="red",
                    linestyle="--",
                    label=r"$T_{\text{emission}}(\text{last})$",
                )

        for i in range(len(axs)):
            axs[i].axvline(
                first_emission_reception_datetime
                + pd.Timedelta(t_hydro_source_offset, "s"),
                color="g",
                linestyle="-",
                label=r"$T_{\text{reception}}(\text{first}) + \tau_{H_{\text{source}}}$",
            )

            axs[i].axvline(
                last_emission_reception_datetime
                + pd.Timedelta(t_hydro_source_offset, "s"),
                color="b",
                linestyle="-",
                label=r"$T_{\text{reception}}(\text{last}) + \tau_{H_{\text{source}}}$",
            )
            if first_emission_in_sequence_datetime is not None:
                axs[i].axvline(
                    first_emission_in_sequence_datetime
                    + pd.Timedelta(t_hydro_source_offset, "s"),
                    color="red",
                    linestyle="-",
                    label=r"$T_{\text{emission}}(\text{first}) + \tau_{H_{\text{source}}}$",
                )
            if last_emission_in_sequence_datetime is not None:
                axs[i].axvline(
                    last_emission_in_sequence_datetime
                    + pd.Timedelta(t_hydro_source_offset, "s"),
                    color="red",
                    linestyle="-",
                    label=r"$T_{\text{emission}}(\text{last}) + \tau_{H_{\text{source}}}$",
                )

    for i in range(len(axs)):
        axs[i].legend(ncols=2, loc="lower left")

    fig.supxlabel("Temps UTC")
    # fig.supylabel("Signal")
    fig.suptitle(
        f"Sequence ID {seq_id} - OBS{obs_id} \n Vc carte = {vc_carte} V - Signal: {signal_type} - Source: {emission_type}"
    )

    if save:
        img_fname = f"arrival_detection_seqID{seq_id}_OBS{obs_id}.png"
        img_fpath = os.path.join(img_root, img_fname)
        fig.savefig(img_fpath, dpi=300)

    # Plot a zoom on the first emission reception
    if len(peaks_idx) > 0 and plot_zoom:

        Bw = signal_params["f1"] - signal_params["f0"]
        Tmf = 1 / Bw
        # Tmf = signal_params["chirp_T"]
        alpha_tmf = 30
        zoom_win_duration = pd.Timedelta(0.05, "s")
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axs[0].plot(t_win, signal_win)
        axs[0].set_ylabel("s")
        axs[1].plot(t_win, sig_mf)
        axs[1].vlines(
            peak_times - pd.Timedelta(alpha_tmf * Tmf / 2, "s"),
            ymin=-1,
            ymax=1,
            color="r",
            linestyle="--",
            label=rf"$t_{{\text{{peak}}}} - {alpha_tmf} \times \frac{{T_{{mf}}}}{{2}}$",
        )
        axs[1].vlines(
            peak_times + pd.Timedelta(alpha_tmf * Tmf / 2, "s"),
            ymin=-1,
            ymax=1,
            color="r",
            linestyle="--",
            label=rf"$t_{{\text{{peak}}}} + {alpha_tmf} \times \frac{{T_{{mf}}}}{{2}}$",
        )

        # Add detected peaks
        axs[0].scatter(
            peak_times,
            signal_win[peaks_idx],
            marker="o",
            color="r",
            label=r"$t_r$",
            s=5,
            zorder=10,
        )
        axs[1].scatter(
            peak_times,
            sig_mf[peaks_idx],
            marker="o",
            color="r",
            label=r"$t_{\text{peak}}$",
            s=5,
            zorder=10,
        )
        axs[1].set_ylabel(r"$s_{mf}$")

        # # Plot signal phase
        # sig_mf_fft = np.fft.fft(sig_mf)
        # sig_mf_phase = np.angle(sig_mf_fft)
        # axs[2].plot(t_win, sig_mf_phase)
        # axs[2].set_ylabel(r"$\phi$ [rad]")

        im = axs[2].pcolormesh(tt_datetime, ff, 10 * np.log10(np.abs(Sxx)), cmap="jet")
        axs[2].set_ylabel("Fréquence [Hz]")
        for i in range(len(axs)):
            axs[i].set_xlim(
                peak_times[0] - 1 / 2 * zoom_win_duration,
                peak_times[0] + 1 / 2 * zoom_win_duration,
            )
        axs[0].legend(ncols=2, loc="upper right")
        axs[1].legend(ncols=2, loc="upper right")

        fig.supxlabel("Temps UTC")
        fig.suptitle(
            f"Sequence ID {seq_id} - OBS{obs_id} \n Vc carte = {vc_carte} V - Signal: {signal_type} - Source: {emission_type}"
        )

        if save:
            img_fname = (
                f"arrival_detection_seqID{seq_id}_OBS{obs_id}_first_arrival_zoom.png"
            )
            img_fpath = os.path.join(img_root, img_fname)
            fig.savefig(img_fpath, dpi=300)


def get_arrivals(signal_win, t_win, df_sequence, fs, verbose=False):
    """
    Apply matched filtering to detect arrivals in a signal window.

    Parameters:
    signal_win : ndarray
        The signal samples in the time window to process.
    t_win : pd.DatetimeIndex
        The time vector corresponding to the signal window.
    df_sequence : pd.DataFrame
        DataFrame containing sequence parameters.
    fs : float
        Sampling frequency in Hz.
    verbose : bool
        If True, print progress messages.
    Returns:
    peaks_idx : ndarray
        Indices of detected peaks in the matched filtered signal.
    peak_times : pd.DatetimeIndex
        Times of detected peaks.
    t_arrivals : pd.DatetimeIndex
        Estimated arrival times corrected for chirp duration.
    sig_mf : ndarray
        The matched filtered signal.
    signal_params : dict
        Dictionary containing signal parameters.
    signal_win_filter : ndarray
        The bandpass filtered signal.

    """

    if verbose:
        print("Applying matched filtering to detect arrivals...")

    # Extract signal parameters
    f0 = df_sequence["Frequency min (Hz)"].iloc[0]
    f1 = df_sequence["Frequency max (Hz)"].iloc[0]
    chirp_T = df_sequence["Duration (s)"].iloc[0]
    T_repeat = df_sequence["Trepeat (s)"].iloc[0]
    signal_params = {
        "f0": f0,
        "f1": f1,
        "chirp_T": chirp_T,
        "T_repeat": T_repeat,
    }

    # Filter signal to keep only frequencies of interest
    order = 4
    bandwidth = f1 - f0
    lowcut = f0 + 0.1 * bandwidth
    highcut = f1 - 0.1 * bandwidth
    b, a = butter(order, [lowcut, highcut], fs=fs, btype="band")
    signal_win_filter = lfilter(b, a, signal_win)

    # Apply matched filtering
    lags, sig_mf = apply_match_filter(signal_win_filter, fs, f0, f1, chirp_T)
    # Normalize matched filtered signal
    sig_mf = sig_mf / np.max(np.abs(sig_mf))

    # Find peaks in matched filtered signal
    dist = 0.9 * T_repeat * fs
    peaks_idx, properties = sp.find_peaks(
        sig_mf, height=np.std(sig_mf) * 5, distance=dist
    )
    peak_times = t_win[peaks_idx]

    # Correct arrivals for chirp duration
    t_arrivals = peak_times - pd.Timedelta(1 / 2 * chirp_T, "s")

    return peaks_idx, peak_times, t_arrivals, sig_mf, signal_params, signal_win_filter


def detected_arrivals_psnr(sig_mf, peaks_idx, signal_params, fs, plot=False):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) for each detected arrival in the matched filtered signal.
    Parameters:
    sig_mf : ndarray
        The matched filtered signal.
    peaks_idx : ndarray
        Indices of detected peaks in the matched filtered signal.
    signal_params : dict
        Dictionary containing signal parameters.
    fs : float
        Sampling frequency in Hz.
    plot : bool
        If True, plot the signal and noise windows for each arrival.
    Returns:
    psnr_arrivals : ndarray
        Array of PSNR values for each detected arrival.
    """

    # Define a window around each arrival
    bandwidth = signal_params["f1"] - signal_params["f0"]
    Tmf = 1 / bandwidth
    alpha_tmf = 150
    # Proceed sequentially for each detected arrival
    mf_time = np.arange(len(sig_mf)) / fs
    psnr_arrivals = []
    for iarr, peak_idx in enumerate(peaks_idx):

        # Select signal portion around the arrival
        tmin = mf_time[peak_idx] - signal_params["T_repeat"] / 10
        tmin = max(tmin, 0)
        nmin = int(tmin * fs)
        tmax = mf_time[peak_idx] + signal_params["T_repeat"] / 10
        tmax = min(tmax, mf_time[-1])
        nmax = int(tmax * fs)
        sig_mf_portion = sig_mf[nmin:nmax]

        # Define masks for signal and noise windows
        mask_signal = np.zeros_like(sig_mf, dtype=bool)

        # Set mask to True in the signal window
        tmin_sig = mf_time[peak_idx] - alpha_tmf * Tmf / 2
        tmin_sig = max(tmin_sig, 0)
        nmin_sig = int(tmin_sig * fs)
        tmax_sig = mf_time[peak_idx] + alpha_tmf * Tmf / 2
        tmax_sig = min(tmax_sig, mf_time[-1])
        nmax_sig = int(tmax_sig * fs)

        mask_signal[nmin_sig:nmax_sig] = True
        mask_signal_win = mask_signal[nmin:nmax]

        noise_win_portion = sig_mf_portion[~mask_signal_win]
        noise_rms = np.sqrt(np.mean(noise_win_portion**2))
        sig_peak = sig_mf[peak_idx]
        peak_signal_to_noise_rms = sig_peak / noise_rms
        psnr_arrivals.append(peak_signal_to_noise_rms)

        if plot:
            plt.figure()
            plt.plot(
                mf_time[nmin:nmax],
                sig_mf_portion,
                label="Matched Filtered Signal Portion",
            )
            plt.plot(
                mf_time[nmin:nmax][mask_signal_win],
                sig_mf_portion[mask_signal_win],
                color="orange",
                label="Signal Window",
            )
            plt.axvline(
                mf_time[peak_idx], color="red", linestyle="--", label="Arrival Time"
            )
            plt.axvline(tmin_sig, color="green", linestyle="--", label="-Tmf")
            plt.axvline(tmax_sig, color="green", linestyle="--", label="+Tmf")
            plt.axhline(noise_rms, color="k", linestyle="--", label="noise rms")
            plt.legend()

            plt.show()

    return np.array(psnr_arrivals)


if __name__ == "__main__":
    pass
