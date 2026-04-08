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

from get_data.wav.get_data_from_rhumrum import get_rhumrum_data

# import real_data_analysis.fiberscope_groix.src.params as p

# ======================================================================================================================
# Paths
# ======================================================================================================================

PROJECT_ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"

ROOT_RHUMRUM_DATA = os.path.join(PROJECT_ROOT, "data", "rhumrum")
ROOT_RHUMRUM_WAV = os.path.join(ROOT_RHUMRUM_DATA, "wav")

if not os.path.exists(ROOT_RHUMRUM_WAV):
    os.makedirs(ROOT_RHUMRUM_WAV)

fmin = 2
fmax = 48
filter_type = "bandpass"
filter_corners = 4

default_freq_properties = {
    "fmin": fmin,
    "fmax": fmax,
    "filter_type": filter_type,
    "filter_corners": filter_corners,
}

# ais_spationav_fpath = os.path.join(root_groix_ais, "SPATIONAV_AIS_001.csv")


# root_folder = p.root_folder
# project_root = p.project_root
# root_groix_data = p.root_groix_data

# img_folder = p.root_img
# data_folder = p.root_data
# root_groix_wav = p.root_groix_wav

# root_img_stft = os.path.join(img_folder, "signal")

# # channels_order = {
# #     "Z": 0,
# #     "X": 1,
# #     "Y": 2,
# #     "H": 3,
# # }
# # hydro_channel = "H"


# ======================================================================================================================
# Functions
# ======================================================================================================================


def build_merged_wav_file(
    date,
    root_data,
    ch="BDH",
    duration_s=60 * 60 * 1,
    rcv_ids=["RR46"],
    freq_properties=default_freq_properties,
    verbose=False,
):
    fmin = freq_properties["fmin"]
    fmax = freq_properties["fmax"]
    filter_type = freq_properties["filter_type"]
    filter_corners = freq_properties["filter_corners"]

    data = {}
    for rcv_id in rcv_ids:
        data_obs = {}

        data_obs["date"] = date
        data_obs["duration_s"] = duration_s
        data_obs["rcv_id"] = rcv_id
        data_obs["ch"] = ch

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
            save=False,
            root_wav="",
        )

        sig = corr_sig["BDH"]
        data_obs["signal"] = sig.data
        data_obs["sig"] = sig

        # Start and end datetimes of signal
        data_obs["start_dt"] = sig.meta.starttime.datetime
        data_obs["end_dt"] = sig.meta.endtime.datetime
        data_obs["fs"] = sig.meta.sampling_rate

        # Store data for current receiver
        data[rcv_id] = data_obs

    ch = ch[0]

    # Build data_vars dict
    data_vars = {}
    date_fmt = "%Y-%m-%d_%H-%M-%S"
    attrs = dict(
        description="Merged wav files RHUMRUM data",
        channel=ch,
        datetime_format=date_fmt,
    )

    for i, rcv_id in enumerate(rcv_ids):
        var_name = f"signal_obs{i+1}"
        data_vars[var_name] = (
            [f"time{i+1}"],
            data[rcv_id]["signal"].astype(np.float32),
        )

        # Receiver names
        data_vars["obs_name"] = rcv_ids

        attrs[f"fs_obs{i+1}"] = data[rcv_id]["fs"]
        # Datetimes
        start_dt_name = f"start_datetime_obs{i+1}"
        attrs[start_dt_name] = data[rcv_id]["start_dt"].strftime(date_fmt)
        start_dt_name = f"end_datetime_obs{i+1}"
        attrs[start_dt_name] = data[rcv_id]["end_dt"].strftime(date_fmt)

    # Storing time is useless since we can easily recompute it from fs
    ds_wav = xr.Dataset(
        data_vars=data_vars,
        attrs=attrs,
    )

    for obs_id in range(1, len(rcv_ids) + 1):
        ch_units = "uPa" if ch == "BDH" else "m/s"
        ds_wav[f"signal_obs{obs_id}"].attrs["units"] = ch_units
        ch_name = "Pressure" if ch == "BDH" else f"{ch} velocity"
        ds_wav[f"signal_obs{obs_id}"].attrs["long_name"] = ch_name

    # Save dataset
    fname = f"channel_{ch}_wav.nc"
    nc_fpath = os.path.join(root_data, fname)
    ds_wav.to_netcdf(nc_fpath)

    if verbose:
        print(f"NetCDF file saved at: {nc_fpath}")


# def compute_spectrogram(
#     window_duration,
#     root_data=data_folder,
#     root_img=img_folder,
#     nperseg=4096,
#     noverlap=2048,
#     channel="H",
# ):

#     # Load wav data from netcdf
#     nc_fpath = os.path.join(root_data, f"channel_{channel}_wav.nc")
#     ds_wav = xr.open_dataset(nc_fpath)
#     datetime_fmt = ds_wav.attrs["datetime_format"]

#     window_duration_hour = window_duration / 3600

#     for obs_id in [1, 2, 3]:

#         # Img folder to use to store spectrograms
#         root_img_obs = os.path.join(
#             root_img, f"OBS{obs_id}", channel, f"{window_duration_hour}H"
#         )
#         if not os.path.exists(root_img_obs):
#             os.makedirs(root_img_obs)

#         time_coordsname = f"time{obs_id}"
#         sig_varname = f"signal_obs{obs_id}"
#         signal = ds_wav[sig_varname]
#         # Select a window of the signal
#         fs = ds_wav.attrs[f"fs_obs{obs_id}"]
#         window_sample_size = int(window_duration * fs)
#         n_start = 0
#         n_end = window_sample_size

#         # Start of recording
#         t0 = ds_wav.attrs[f"start_datetime_obs{obs_id}"]
#         t0 = datetime.strptime(t0, datetime_fmt)

#         # Iterate over successive windows
#         while n_end < signal.size:
#             # Slice signal
#             sig_win = signal.isel({time_coordsname: slice(n_start, n_end)})
#             # Define datetime borders
#             t_start = n_start * 1 / fs
#             t_end = n_end * 1 / fs
#             t0_slice = t0 + timedelta(seconds=t_start)
#             t1_slice = t0 + timedelta(seconds=t_end)
#             # Update window bounds to the next one
#             n_start = n_end
#             n_end += window_sample_size

#             # Derive stft
#             ff, tt, stft = sp.stft(
#                 sig_win.values,
#                 fs=fs,
#                 window="hann",
#                 nperseg=nperseg,
#                 noverlap=noverlap,
#                 scaling="psd",
#             )
#             sxx_0 = 1  # 1uPa**2 / Hz
#             sxx = 10 * np.log10(np.abs(stft) / sxx_0)
#             # Associated datetime vector
#             tt_datetime = pd.date_range(
#                 t0_slice,
#                 t0_slice + timedelta(seconds=tt[-1]),
#                 freq=f"{tt[1]-tt[0]}s",
#                 inclusive="both",
#             )

#             # Plot
#             cmap = "viridis"
#             vmin = np.percentile(sxx, 10)
#             vmax = np.percentile(sxx, 99)
#             # fig, ax = plt.subplots()

#             plt.figure()
#             im = plt.pcolormesh(tt_datetime, ff, sxx, cmap=cmap, vmin=vmin, vmax=vmax)

#             clabel = (
#                 r"dB re 1$\mu$Pa$^2$ / Hz"
#                 if channel == "H"
#                 else r"dB re 1$(m~s^{-1})^2$ / Hz"
#             )
#             plt.colorbar(im, label=clabel)
#             plt.ylabel("Fréquence [Hz]")
#             plt.xlabel("Temps UTC")
#             # plt.show()

#             # Save in dedicated folder
#             start_dt_str = t0_slice.strftime(datetime_fmt)
#             end_dt_str = t1_slice.strftime(datetime_fmt)
#             fname = f"OBS{obs_id}_{channel}_{start_dt_str}_to_{end_dt_str}.png"
#             fpath = os.path.join(root_img_obs, fname)
#             plt.savefig(fpath)

#             plt.close("all")

#     # print(stft_dict[1]["stft"])


if __name__ == "__main__":

    project_root = PROJECT_ROOT
    root_folder = os.path.join(project_root, "real_data_analysis", "rhumrum")
    data_folder = os.path.join(root_folder, "data")

    # Def frequency properties
    freq_properties = default_freq_properties

    date = "2013-05-09 00:00:00"
    duration_s = 60 * 60 * 24
    swir_rcv_ids = ["RR41", "RR43", "RR44", "RR47"]

    build_merged_wav_file(
        date,
        root_data=data_folder,
        ch=["BDH"],
        rcv_ids=swir_rcv_ids,
        duration_s=duration_s,
        freq_properties=default_freq_properties,
        verbose=True,
    )

    # pass
