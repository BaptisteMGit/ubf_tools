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

# import tracemalloc
from pympler import muppy, summary
import gc


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

# Convention Gen_Axes_D_V4 (Cf ELOBSBin2Wav.py)
CHANNELS_ORDER = {
    "Z": 0,
    "X": 1,
    "Y": 2,
    "H": 3,
}
HYDRO_CHANNEL = "H"


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
    wav_start_datetime,
    t_win_sec,
    signal_win,
    sig_mf,
    t_arrivals_sec,
    peaks_idx,
    peak_times_sec,
    sequence_info,
    signal_params,
    first_emission_reception_datetime=None,
    last_emission_reception_datetime=None,
    plot_last_first=False,
    t_hydro_source_offset=27,
    cmap="Greys",
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
        print("\t\tPlotting arrivals...")

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

    # Convert time vectors to datetime
    t_win = np.array([wav_start_datetime + pd.Timedelta(t, "s") for t in t_win_sec])
    peak_times = np.array(
        [wav_start_datetime + pd.Timedelta(t, "s") for t in peak_times_sec]
    )
    t_arrivals = np.array(
        [wav_start_datetime + pd.Timedelta(t, "s") for t in t_arrivals_sec]
    )

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
        axs[1].legend(ncols=1, loc="lower left")

    axs[0].set_ylabel("s")
    axs[1].set_ylabel(r"$s_{mf}$")

    im = axs[2].pcolormesh(tt_datetime, ff, 10 * np.log10(np.abs(Sxx)), cmap=cmap)
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
                arrowprops=dict(arrowstyle="->", color="red"),
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
        plt.close(fig)

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

        im = axs[2].pcolormesh(tt_datetime, ff, 10 * np.log10(np.abs(Sxx)), cmap=cmap)
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
            plt.close(fig)


def get_arrivals(signal_win, t_win_sec, df_sequence, fs, verbose=False):
    """
    Apply matched filtering to detect arrivals in a signal window.

    Parameters:
    signal_win : ndarray
        The signal samples in the time window to process.
    t_win_sec : ndarray
        The time vector corresponding to the signal window (in sec from start).
    df_sequence : pd.DataFrame
        DataFrame containing sequence parameters.
    fs : float
        Sampling frequency in Hz.
    verbose : bool
        If True, print progress messages.
    Returns:
    peaks_idx : ndarray
        Indices of detected peaks in the matched filtered signal.
    peak_times_sec : ndarray
        Times of detected peaks (in sec from start).
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
        print("\t\tMatch filtering...")

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
    peak_times_sec = t_win_sec[peaks_idx]

    # Correct arrivals for chirp duration
    # t_arrivals = peak_times - pd.Timedelta(1 / 2 * chirp_T, "s")
    t_arrivals_sec = peak_times_sec - 0.5 * chirp_T

    return (
        peaks_idx,
        peak_times_sec,
        t_arrivals_sec,
        sig_mf,
        signal_params,
        signal_win_filter,
    )


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


def select_dataframe_subset(df, subset_params):
    """
    Select a subset of a DataFrame based on specified parameters.
    Parameters:
    df : pd.DataFrame
        The input DataFrame to filter.
    subset_params : dict
        Dictionary containing parameters to filter the DataFrame.
        Keys are column names and values are the desired values to filter by.
        If a value is None, that parameter is not used for filtering.
    Returns:
    df_sel : pd.DataFrame
        The filtered DataFrame.
    """

    df_sel = df.copy()
    for param in subset_params.keys():
        if subset_params[param] is not None:
            df_sel = df_sel[df_sel[param] == subset_params[param]]

            df_sel.reset_index(drop=True, inplace=True)

    return df_sel


def build_arrivals_dataset(
    df,
    ds_gps,
    sel_sequence_id,
    pre_reception_time=5.0,
    post_reception_time=10.0,
    channels_order=CHANNELS_ORDER,
    used_channel=HYDRO_CHANNEL,
    t_hydro_source_offset=None,
    img_root=None,
    plot=False,
    plot_zoom=False,
    savefig=False,
    verbose=False,
):

    # Define image folder
    if img_root is None:
        img_root = os.path.join(
            IMG_FOLDER, "reception", "arrivals_detection", "preprocessing"
        )

    # --------------------------------------------------
    # Initialisation
    # --------------------------------------------------
    origin_keys = df.columns.to_list()
    processed_keys = origin_keys
    processed_data = {key: [] for key in processed_keys}

    # List available wav files for each OBS (to avoid reloading them for each sequence)
    wav_start_times_dict = {}
    start_datetime_arr_dict = {}
    for obs_id in [1, 2, 3]:
        wav_start_times, start_datetime_arr = get_available_wav_files(obs_id)
        wav_start_times_dict[obs_id] = wav_start_times
        start_datetime_arr_dict[obs_id] = start_datetime_arr

    print(f"Selected sequences: {len(sel_sequence_id)} -> {sel_sequence_id}")

    # --------------------------------------------------
    # Main loop over sequences
    # --------------------------------------------------
    for seq_id in sel_sequence_id:

        if verbose:
            print(f"Processing sequence ID: {seq_id}")

        # Select dataframe for the current sequence
        df_sequence = df.loc[df["Sequence_id"] == seq_id]

        # # Correct offset # TODO remove
        # # print(df_sequence["Emission datetime"])
        # df_sequence["Emission datetime"] = df_sequence["Emission datetime"] + pd.Timedelta(t_hydro_source_offset, "s")
        # print(df_sequence["Emission datetime"])

        new_data = {}
        # Lood over receivers
        for obs_id in [1, 2, 3]:

            if verbose:
                print(f"\tOBS{obs_id}")

            # Get available wav files for the selected OBS
            wav_start_times = wav_start_times_dict[obs_id]
            start_datetime_arr = start_datetime_arr_dict[obs_id]

            # -----------------------------------------
            # 1) Get theoretical arrivals
            # -----------------------------------------
            # Get all theoretical arrival time
            emissions_datetime = []
            th_arrivals_datetime = []
            th_propagation_delay = []
            th_arrivals_seconds_from_start = []
            for emission_id in range(df_sequence.shape[0]):
                emission_i = df_sequence.iloc[emission_id]
                emission_i_datetime = emission_i["Emission datetime"].to_pydatetime(
                    warn=False
                )
                emissions_datetime.append(emission_i_datetime)

                if emission_id == 0:  # First emission in sequence
                    # Find the corresponding wav file (for now we assume that the sequence fits in a single wav file)
                    wav_fpath, wav_start_datetime = get_wav_file_for_emission(
                        emission_datetime=emission_i_datetime,
                        start_datetime_arr=start_datetime_arr,
                        wav_start_times=wav_start_times,
                    )
                    if verbose:
                        print(f"\t\tWav file: {wav_fpath}")

                # Emission position
                emission_i_pos = [
                    emission_i["Emission interpolated E GPS"],
                    emission_i["Emission interpolated N GPS"],
                    emission_i["Emission interpolated U GPS"],
                ]
                # Theoretical time of arrival
                emission_i_reception_datetime, tr_i, prop_time_i = get_tr_apriori(
                    emission_pos=emission_i_pos,
                    ds_gps=ds_gps,
                    wav_start_datetime=wav_start_datetime,
                    emission_datetime=emission_i_datetime,
                    obs_id=obs_id,
                )
                th_propagation_delay.append(prop_time_i)
                th_arrivals_datetime.append(emission_i_reception_datetime)
                th_arrivals_seconds_from_start.append(tr_i)

            # Convert to numpy arrays
            th_propagation_delay = np.array(th_propagation_delay)
            th_arrivals_datetime = np.array(th_arrivals_datetime)
            th_arrivals_seconds_from_start = np.array(th_arrivals_seconds_from_start)

            # First emission in the sequence
            first_emission_in_sequence_datetime = emissions_datetime[0]
            first_emission_reception_datetime = th_arrivals_datetime[0]
            tr_first = th_arrivals_seconds_from_start[0]
            # Last emission in sequence
            last_emission_in_sequence_datetime = emissions_datetime[-1]
            last_emission_reception_datetime = th_arrivals_datetime[-1]
            tr_last = th_arrivals_seconds_from_start[-1]
            if verbose:
                print(
                    "\t\tEmission datetime (first pulse):",
                    first_emission_in_sequence_datetime,
                )
                print(
                    "\t\tEmission datetime (last pulse):",
                    last_emission_in_sequence_datetime,
                )

            # -----------------------------------------
            # 2) Load signal
            # -----------------------------------------
            # Load the wav file
            if verbose:
                print("\t\tLoading wav file...")

            # Read the wav file
            signal, fs = sf.read(wav_fpath)
            # Select the channel
            signal = signal[:, channels_order[used_channel]]
            # Center the signal
            signal = signal - np.mean(signal)

            # Compute source position considering the offset for the current emission   (Source, Longueur filée)
            # TODO : implement position correction if needed

            # Select the time window of interest for current sequence (all emissions in the sequence + pre/post times)
            t_start_win = tr_first - pre_reception_time
            t_end_win = tr_last + post_reception_time

            # Convert in samples
            n_samp_start_win = int(t_start_win * fs)
            n_samp_end_win = int(t_end_win * fs)
            # Slice signal
            signal_win = signal[n_samp_start_win:n_samp_end_win]
            wav_end_datetime = wav_start_datetime + pd.Timedelta(
                signal.shape[0] * 1 / fs, "s"
            )
            # t_dt = pd.date_range(
            #     wav_start_datetime, wav_end_datetime, freq=f"{1/fs}s", inclusive="left"
            # )
            # t_win = t_dt[n_samp_start_win:n_samp_end_win]

            t_sec = np.arange(signal.shape[0]) / fs
            t_win_sec = t_sec[n_samp_start_win:n_samp_end_win]

            if verbose:
                print(
                    f"\t\tWav file loaded (from {wav_start_datetime} to {wav_end_datetime})"
                )

            # -----------------------------------------
            # 3) Process signal to detect arrivals
            # -----------------------------------------
            (
                peaks_idx,
                peak_times_sec,
                t_arrivals_sec,
                sig_mf,
                signal_params,
                signal_win_filter,
            ) = get_arrivals(
                signal_win,
                t_win_sec,
                df_sequence,
                fs,
                verbose=verbose,
            )
            signal_win = signal_win_filter

            # -----------------------------------------
            # 4) Plot detected arrivals (optional)
            # -----------------------------------------
            if plot:
                sequence_info = {
                    "seq_id": seq_id,
                    "obs_id": obs_id,
                    "vc_carte": df_sequence["Vc carte (V)"].iloc[0],
                    "signal_type": df_sequence["Signal"].iloc[0],
                    "emission_type": df_sequence["Source"].iloc[0],
                }
                nperseg = 256
                noverlap = int(nperseg * 0.5)

                plot_arrivals_detection(
                    wav_start_datetime=wav_start_datetime,
                    t_win_sec=t_win_sec,
                    signal_win=signal_win,
                    sig_mf=sig_mf,
                    t_arrivals_sec=t_arrivals_sec,
                    peaks_idx=peaks_idx,
                    peak_times_sec=peak_times_sec,
                    sequence_info=sequence_info,
                    signal_params=signal_params,
                    plot_last_first=False,
                    first_emission_reception_datetime=first_emission_reception_datetime,
                    last_emission_reception_datetime=last_emission_reception_datetime,
                    t_hydro_source_offset=t_hydro_source_offset,
                    save=savefig,
                    img_root=img_root,
                    fs=fs,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    first_emission_in_sequence_datetime=None,
                    last_emission_in_sequence_datetime=None,
                    verbose=verbose,
                    plot_zoom=plot_zoom,
                )

                plt.close("all")

            # -----------------------------------------
            # 6) Derive PSNR for each detected arrival
            # -----------------------------------------
            # Derive peak signal to noise ratio (PSNR) on matched filtered signal
            psnr_arrivals = detected_arrivals_psnr(
                sig_mf, peaks_idx, signal_params, fs, plot=False
            )

            # -----------------------------------------
            # 7) Matching arrivals -> emissions
            # -----------------------------------------
            # Convert t_arrivals into datetime.datetime
            # t_arrivals_dt = np.array(
            #     [t_arr.to_pydatetime(warn=False) for t_arr in t_arrivals]
            # )
            t_arrivals_dt = np.array(
                [
                    wav_start_datetime + pd.Timedelta(t_arr_s, "s")
                    for t_arr_s in t_arrivals_sec
                ]
            )

            valid_detection = np.zeros_like(emissions_datetime, dtype=bool)

            if len(t_arrivals_dt) < len(emissions_datetime):
                print(
                    f"Warning: only {len(t_arrivals_dt)} arrivals detected for {len(emissions_datetime)} emissions in sequence {seq_id} OBS{obs_id}"
                )
                # Pad in case not all peaks are detected
                psnr_arrivals_full = np.full_like(
                    emissions_datetime, np.nan, dtype=float
                )
                # t_arrivals_full = np.full_like(emissions_datetime, pd.NaT)
                t_arrivals_dt_full = np.full_like(emissions_datetime, pd.NaT)

                # Associate arrivals to closest theoretical arrival
                th_arrivals_datetime_copy = th_arrivals_datetime.copy()
                for i_t_arr, t_arr_dt in enumerate(t_arrivals_dt):
                    # Find closest
                    closest_th_arr_idx = np.argmin(
                        np.abs(th_arrivals_datetime_copy - t_arr_dt)
                    )
                    # Remove this theoretical arrival from the copy to avoid double matching
                    th_arrivals_datetime_copy = np.delete(
                        th_arrivals_datetime_copy, closest_th_arr_idx
                    )
                    # Replace in padded arrays
                    t_arrivals_dt_full[closest_th_arr_idx] = t_arr_dt
                    # t_arrivals_full[closest_th_arr_idx] = t_arrivals[i_t_arr]
                    psnr_arrivals_full[closest_th_arr_idx] = psnr_arrivals[i_t_arr]

                    # Set valid_detection flag to true
                    valid_detection[closest_th_arr_idx] = True

                # Release memory
                del th_arrivals_datetime_copy

            else:
                # t_arrivals_full = t_arrivals
                t_arrivals_dt_full = t_arrivals_dt
                psnr_arrivals_full = psnr_arrivals
                valid_detection[:] = True

            # Derive propagation time
            try:
                meas_propagation_delay = t_arrivals_dt_full - np.array(
                    emissions_datetime
                )
                meas_propagation_delay = np.array(
                    [t.total_seconds() for t in meas_propagation_delay]
                )
            except:
                print("Wrong number of arrivals detected")

            # -----------------------------------------
            # 7) Store results
            # -----------------------------------------
            # new_data[f"Arrival datetime OBS{obs_id}"] = list(t_arrivals_full)
            new_data[f"Arrival datetime OBS{obs_id}"] = list(t_arrivals_dt_full)

            new_data[f"Theoretical propagation time OBS{obs_id}"] = list(
                th_propagation_delay
            )
            new_data[f"Measured propagation time OBS{obs_id}"] = list(
                meas_propagation_delay
            )
            new_data[f"PSNR OBS{obs_id}"] = list(psnr_arrivals_full)
            new_data[f"Valid detection OBS{obs_id}"] = list(valid_detection)

        # -----------------------------------------
        # 8) Aggregate results
        # -----------------------------------------
        new_data["pulse_id"] = list(
            np.arange(len(emissions_datetime))
        )  # Add an id for each pulse
        for key in origin_keys:
            processed_data[key].extend(df_sequence[key].values)
        for key in new_data:
            if key in processed_data.keys():
                processed_data[key].extend(new_data[key])
            else:
                processed_data[key] = new_data[key]

        # # Release memory
        # del t_dt, t_win, new_data
        # gc.collect()

        if verbose:
            print(f"\tSequence ID {seq_id} processed.")
            all_objects = muppy.get_objects()
            summary.print_(summary.summarize(all_objects))

    # Convert to dataframe
    df_processed = pd.DataFrame(processed_data)

    # Attribute f_score to each sequence
    df_processed = attribute_sequence_f_score(df_processed, verbose=verbose)

    return df_processed


def attribute_sequence_f_score(df_processed, verbose=False):
    for sel_id in df_processed["Sequence_id"].unique():
        df_seq = df_processed[df_processed["Sequence_id"] == sel_id]

        for obs_id in [1, 2, 3]:

            # First criterion: ratio of detected arrivals
            col_name = f"Valid detection OBS{obs_id}"
            n_detected = df_seq[col_name].sum()
            crit_1 = n_detected / df_seq.shape[0]

            # Second criterion: error relative to expected repetition period
            col_name = f"Arrival datetime OBS{obs_id}"
            t_diff_mean = df_seq[col_name].diff().mean().total_seconds()
            repeat_period_em = df_seq["Trepeat (s)"].iloc[0]
            crit_2 = 1 - abs(t_diff_mean - repeat_period_em) / repeat_period_em
            if t_diff_mean < 0 or crit_2 < 0 or np.isnan(crit_2):
                crit_2 = 0

            # Third criterion: normalized psnr
            col_name = f"PSNR OBS{obs_id}"
            psnr_mean = df_seq[col_name].mean()
            crit_3 = psnr_mean / df_processed[col_name].max()
            if np.isnan(crit_3):
                crit_3 = 0

            # Final score as sum of criteria
            final_score = (crit_1 + crit_2 + crit_3) / 3
            if verbose:
                print(
                    f"Sequence ID {sel_id} - OBS{obs_id} : Score = {final_score:.2f} (C1={crit_1:.2f}, C2={crit_2:.2f}, C3={crit_3:.2f})"
                )

            # Store final score in dataframe
            df_processed.loc[
                (df_processed["Sequence_id"] == sel_id),
                f"f_score OBS{obs_id}",
            ] = final_score

    return df_processed


if __name__ == "__main__":
    pass
