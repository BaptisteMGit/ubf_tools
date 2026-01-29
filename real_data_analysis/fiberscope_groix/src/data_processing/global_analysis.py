#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   global_analysis.py
@Time    :   2026/01/21 10:05:31
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Analyse data from fiberscope Groix 2025 experiment
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

from datetime import datetime, timedelta
from real_data_analysis.fiberscope_groix.src.data_processing.arrivals_utils import (
    get_available_wav_files,
)
from real_data_analysis.real_data_utils import V2uPa
import real_data_analysis.fiberscope_groix.src.params as p

# ======================================================================================================================
# Paths
# ======================================================================================================================
root_folder = p.root_folder
project_root = p.project_root
root_groix_data = p.root_groix_data

img_folder = p.root_img
data_folder = p.root_data
root_groix_wav = p.root_groix_wav

root_img_stft = os.path.join(img_folder, "signal")

# channels_order = {
#     "Z": 0,
#     "X": 1,
#     "Y": 2,
#     "H": 3,
# }
# hydro_channel = "H"


# ======================================================================================================================
# Functions
# ======================================================================================================================


def merge_wav_files(root_data=data_folder, output_format="nc", channels=p.hydro_channel, verbose=False):

    # Make sure channels is a 1D array
    channels = np.atleast_1d(channels)

    # Get available wav files for each OBS
    wav_start_times_dict = {}
    start_datetime_arr_dict = {}
    for obs_id in [1, 2, 3]:
        wav_start_times, start_datetime_arr = get_available_wav_files(obs_id, root_groix_wav=root_groix_wav)
        wav_start_times_dict[obs_id] = wav_start_times
        start_datetime_arr_dict[obs_id] = start_datetime_arr

    # print(wav_start_times_dict)

    data_obs = {}
    # Load entire signal for each OBS
    for obs_id in [1, 2, 3]:
        obs_wav_files_dict = wav_start_times_dict[obs_id]

        # # TODO : limitation for personnal computer memory -> remove this on TIM 
        # id0 = 12
        # id1 = 15
        # # keep_only_first = 2
        # obs_wav_files_dict = dict(list(obs_wav_files_dict.items())[id0:id1])

        full_signal = {ch: [] for ch in channels}
        full_time = []
                    
        for wav_start_dt in obs_wav_files_dict.keys():
            wav_fpath = obs_wav_files_dict[wav_start_dt]
            # Load signal from wav file
            signal, fs = sf.read(wav_fpath)
            # Get time vector
            t0 = full_time[-1] + 1 / fs if full_time else 0
            time = np.arange(signal.shape[0]) / fs + t0
            # Add 
            full_time.extend(time)

            for ch in channels:

                # Select the channel
                signal_ch = signal[:, p.channels_order[ch]]

                if ch == "H":
                    # Centre signal
                    signal_ch -= np.mean(signal_ch)
                    # Convert to uPa
                    signal_ch = V2uPa(signal_ch, p.obs_hydro_sensitivity, p.obs_hydro_gain)

                # Add to list
                full_signal[ch].extend(signal_ch)

        full_time = np.array(full_time)

        # Datetime info
        start_dt = min(obs_wav_files_dict.keys())
        end_dt = max(obs_wav_files_dict.keys()) + pd.Timedelta(signal.size / fs, "s")

        # Store data
        data_obs[obs_id] = dict(
            signal={},
            time=full_time,
            start_dt=start_dt,
            end_dt=end_dt,
            fs=fs,
        )
        for ch in channels:
            data_obs[obs_id]["signal"][ch] = np.array(full_signal[ch])

    # TODO : need to be updated -> this code is broken 
    date_fmt = "%Y-%m-%d_%H-%M-%S"
    if output_format == "wav":
        for obs_id in data_obs.keys():
            full_signal = data_obs[obs_id]["signal"]
            fs = data_obs[obs_id]["fs"]
            start_dt = data_obs[obs_id]["start_dt"]
            end_dt = data_obs[obs_id]["end_dt"]

            # Save full wav
            start_dt_str = start_dt.strftime(date_fmt)
            end_dt_str = end_dt.strftime(date_fmt)
            # Serial number from file name
            obs_serial_number = "".join(os.path.basename(wav_fpath).split("_")[0:2])
            # Path to save
            wav_name = f"{obs_serial_number}_{start_dt_str}_to_{end_dt_str}.wav"
            full_wav_fpath = os.path.join(root_data, wav_name)
            # Write file
            sf.write(full_wav_fpath, full_signal, fs)

            if verbose:
                print(f"Full wav saved at: {full_wav_fpath}")

    elif output_format == "nc":

        for ch in channels:

            # Storing time is useless since we can easily recompute it from fs
            ds_wav = xr.Dataset(
                data_vars=dict(
                    signal_obs1=(data_obs[1]["signal"].astype(np.float32)),
                    signal_obs2=(data_obs[2]["signal"].astype(np.float32)),
                    signal_obs3=(data_obs[3]["signal"].astype(np.float32)),
                ),
                attrs=dict(
                    description="Merged wav files from Fiberscope Groix Oct 2025 experiment",
                    used_channel=ch,
                    fs_obs1=data_obs[1]["fs"],
                    start_datetime_obs1=data_obs[1]["start_dt"].strftime(date_fmt),
                    end_datetime_obs1=data_obs[1]["end_dt"].strftime(date_fmt),
                    fs_obs2=data_obs[2]["fs"],
                    start_datetime_obs2=data_obs[2]["start_dt"].strftime(date_fmt),
                    end_datetime_obs2=data_obs[2]["end_dt"].strftime(date_fmt),
                    fs_obs3=data_obs[3]["fs"],
                    start_datetime_obs3=data_obs[3]["start_dt"].strftime(date_fmt),
                    end_datetime_obs3=data_obs[3]["end_dt"].strftime(date_fmt),
                    datetime_format=date_fmt,
                ),
            )

            for obs_id in [1, 2, 3]:
                ch_units = "uPa" if ch == "H" else "m/s"
                ds_wav[f"signal_obs{obs_id}"].attrs["units"] = ch_units
                ch_name = "Pressure" if ch == "H" else f"{ch} velocity"
                ds_wav[f"signal_obs{obs_id}"].attrs["long_name"] = ch_name


            # Save dataset
            fname = f"channel_{ch}_wav.nc"
            nc_fpath = os.path.join(root_data, fname)
            ds_wav.to_netcdf(nc_fpath)

            if verbose:
                print(f"NetCDF file saved at: {nc_fpath}")


def compute_spectrogram(
    window_duration,
    root_data=data_folder,
    root_img=img_folder,
    nperseg=4096,
    noverlap=2048,
):

    # Load wav data from netcdf
    nc_fpath = os.path.join(root_data, "channel_H_wav.nc")
    ds_wav = xr.open_dataset(nc_fpath)
    datetime_fmt = ds_wav.attrs["datetime_format"]

    window_duration_hour = window_duration / 3600

    for obs_id in [1, 2, 3]:

        # Img folder to use to store spectrograms
        root_img_obs = os.path.join(
            root_img, f"OBS{obs_id}", f"{window_duration_hour}H"
        )
        if not os.path.exists(root_img_obs):
            os.makedirs(root_img_obs)

        sig_varname = f"signal_obs{obs_id}"
        signal = ds_wav[sig_varname]
        # Select a window of the signal
        fs = ds_wav.attrs[f"fs_obs{obs_id}"]
        window_sample_size = int(window_duration * fs)
        n_start = 0
        n_end = window_sample_size

        # Start of recording
        t0 = ds_wav.attrs[f"start_datetime_obs{obs_id}"]
        t0 = datetime.strptime(t0, datetime_fmt)

        # Iterate over successive windows
        while n_end < signal.size:
            # Slice signal
            sig_win = signal.isel({sig_varname: slice(n_start, n_end)})
            # Define datetime borders
            t_start = n_start * 1 / fs
            t_end = n_end * 1 / fs
            t0_slice = t0 + timedelta(seconds=t_start)
            t1_slice = t0 + timedelta(seconds=t_end)
            # Update window bounds to the next one
            n_start = n_end
            n_end += window_sample_size

            # Derive stft
            ff, tt, stft = sp.stft(
                sig_win.values,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="psd",
            )
            sxx_0 = 1  # 1uPa**2 / Hz
            sxx = 10 * np.log10(np.abs(stft) / sxx_0)
            # Associated datetime vector
            tt_datetime = pd.date_range(
                t0_slice,
                t0_slice + timedelta(seconds=tt[-1]),
                freq=f"{tt[1]-tt[0]}s",
                inclusive="both",
            )

            # Plot
            cmap = "viridis"
            vmin = np.percentile(sxx, 10)
            vmax = np.percentile(sxx, 99)
            # fig, ax = plt.subplots()
            plt.figure()
            im = plt.pcolormesh(tt_datetime, ff, sxx, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, label=r"dB re 1$\mu$Pa$^2$ / Hz")
            plt.ylabel("Fréquence [Hz]")
            plt.xlabel("Temps UTC")
            # plt.show()

            # Save in dedicated folder
            start_dt_str = t0_slice.strftime(datetime_fmt)
            end_dt_str = t1_slice.strftime(datetime_fmt)
            fname = f"OBS{obs_id}_{start_dt_str}_to_{end_dt_str}.png"
            fpath = os.path.join(root_img_obs, fname)
            plt.savefig(fpath)

            plt.close("all")

    # print(stft_dict[1]["stft"])


if __name__ == "__main__":
    merge_wav_files(output_format="nc", channels=["H"], verbose=True)

    # window_duration_hour = 2
    # window_duration = window_duration_hour * 3600  # in seconds
    # nperseg = 2**14
    # noverlap = 2**13
    # compute_spectrogram(
    #     window_duration=window_duration,
    #     root_data=data_folder,
    #     root_img=root_img_stft,
    #     nperseg=nperseg,
    #     noverlap=noverlap,
    # )

    # pass
