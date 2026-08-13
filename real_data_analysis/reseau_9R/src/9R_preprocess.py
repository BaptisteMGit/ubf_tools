#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_mfp_preprocess.py
@Time    :   2026/05/18 13:59:22
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
import xarray as xr
import pandas as pd
import soundfile as sf
from datetime import datetime


from source.utils.utils_ais import interpolate_ais
from source.utils.utils_bathy import load_bathy, ROOT_BATHY_DATA

# 9R network: 36 stations
#  from 2022-12-05 00:00:00 to 2023-06-26 23:59:59.999900
#  mean location: (18.37940555555555, -81.75037527777778)


# ======================================================================================================================
# 1) Get AIS data for the area and period of interest and save it as xarray dataset : ds_ais
# ======================================================================================================================
def extract_subset_ais(
    box_center_lon=-81.7504, box_center_lat=18.3794, dlon_box=0.1, dlat_box=0.1
):
    # Path to data (aggregated AIS data for the world in 2023)
    fpath_ais = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ais\ais_monde_2023\ais_aggregated_pos.parquet"
    # Load
    df_ais = pd.read_parquet(fpath_ais)
    # Remove AIS lines with unique positions (at least two points to define a trajectory)
    df_ais = df_ais.groupby("mmsi").filter(lambda x: len(x) > 1)

    # Extract area of interest
    # box_center_lon = -81.7504
    # box_center_lat = 18.3794
    box_lon_min = box_center_lon - dlon_box / 2
    box_lon_max = box_center_lon + dlon_box / 2
    box_lat_min = box_center_lat - dlat_box / 2
    box_lat_max = box_center_lat + dlat_box / 2
    df_ais = df_ais.loc[
        (df_ais.longitude >= box_lon_min)
        & (df_ais.longitude <= box_lon_max)
        & (df_ais.latitude >= box_lat_min)
        & (df_ais.latitude <= box_lat_max)
    ]

    print(df_ais.head())
    print(df_ais.shape)

    df_ais.reset_index(drop=True, inplace=True)

    fname = "ais_9R.parquet"
    root_tmp = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\reseau_9R\data"
    fpath = os.path.join(root_tmp, fname)
    df_ais.to_parquet(fpath)


def load_subset_ais():
    fname = "ais_9R.parquet"
    root_tmp = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\reseau_9R\data"
    fpath = os.path.join(root_tmp, fname)
    df_ais = pd.read_parquet(fpath)

    df_ais = df_ais.groupby("mmsi").filter(lambda x: len(x) > 1)

    return df_ais


def format_ais_data(ais_interp, time_step):

    t_start = ais_interp["datetime"].min()
    t_end = ais_interp["datetime"].max()
    mmsi = ais_interp["mmsi"].unique()

    # Define common time vector
    common_time = pd.date_range(start=t_start, end=t_end, freq=time_step)

    ais_lon_mat = []
    ais_lat_mat = []

    for mmsi_i in mmsi:

        ais_mmsi = ais_interp.loc[ais_interp["mmsi"] == mmsi_i]
        ais_mmsi_arr = ais_mmsi.to_numpy(dtype=np.float32)[
            :, 1:3
        ]  # Exclude time and mmsi column

        # Pad lon lat with NaNs to match common_time length
        n_pad_start = (ais_mmsi["datetime"].iloc[0] - t_start) // pd.Timedelta(
            time_step
        )
        n_pad_end = (t_end - ais_mmsi["datetime"].iloc[-1]) // pd.Timedelta(time_step)
        ais_mmsi_arr = np.pad(
            ais_mmsi_arr,
            ((n_pad_start, n_pad_end), (0, 0)),
            mode="constant",
            constant_values=np.nan,
        )

        # Add to array
        ais_lon_mat.append(ais_mmsi_arr[:, 0])
        ais_lat_mat.append(ais_mmsi_arr[:, 1])

    # Convert to np arrays of shape (n_mmsi, n_time)
    ais_lon_mat = np.array(ais_lon_mat)
    ais_lat_mat = np.array(ais_lat_mat)

    return ais_lon_mat, ais_lat_mat, mmsi, common_time


def ais_dataset(
    mmsi,
    common_time,
    ais_lon_mat,
    ais_lat_mat,
):

    time_step = (common_time[1] - common_time[0]).total_seconds()
    ds_ais = xr.Dataset(
        data_vars=dict(
            lon=(["mmsi", "time"], ais_lon_mat),
            lat=(["mmsi", "time"], ais_lat_mat),
        ),
        coords=dict(
            time=common_time.tz_localize(
                None
            ),  # Remove timezone info for xarray compatibility
            mmsi=mmsi,
        ),
        attrs=dict(
            network="9R",
            description="AIS data for the 9R network area, interpolated to regular time steps",
            interpolation_time_step=time_step,
            geodesic_frame="WGS84",
        ),
    )

    # Add attributes to variables / coords
    ds_ais.mmsi.attrs["units"] = ""
    ds_ais.time.attrs["timezone"] = "UTC"
    ds_ais.lon.attrs["units"] = "°"
    ds_ais.lat.attrs["units"] = "°"
    ds_ais.lon.attrs["long_name"] = "Longitude"
    ds_ais.lat.attrs["long_name"] = "Latitude"

    return ds_ais


def preprocess_ais(root_data):

    # Load data
    df_ais = load_subset_ais()

    # Rename columns to match expected format for interpolation
    df_ais.rename(
        columns={"longitude": "lon", "latitude": "lat"},
        inplace=True,
    )
    # Interpolate AIS data to regular time steps (e.g. 10s) for each MMSI
    interpolation_time_step = "10s"
    ais_interp, mmsi = interpolate_ais(df_ais, time_step=interpolation_time_step)

    # Format AIS data in matrices of shape (n_mmsi, n_time) for longitude and latitude, and get common time vector
    ais_lon_mat, ais_lat_mat, mmsi, common_time = format_ais_data(
        ais_interp=ais_interp,
        time_step=interpolation_time_step,
    )

    # Convert to xarray
    ds_ais = ais_dataset(
        mmsi,
        common_time,
        ais_lon_mat,
        ais_lat_mat,
    )

    # Save to netCDF
    fname = "ais.nc"
    fpath = os.path.join(root_data, fname)
    ds_ais.to_netcdf(fpath)


# ======================================================================================================================
#  2) Get bathy data for the area of interest and save it as xarray dataset : ds_bathy
# ======================================================================================================================


def prepocess_bathy(
    root_data,
    box_center_lon=-81.7504,
    box_center_lat=18.3794,
    dlon_box=0.1,
    dlat_box=0.1,
    root_bathy_data=ROOT_BATHY_DATA,
):

    # Load bathy data
    bathy = load_bathy(
        box_center_lon=box_center_lon,
        box_center_lat=box_center_lat,
        dlon_box=dlon_box,
        dlat_box=dlat_box,
        root_bathy_data=root_bathy_data,
    )

    # Build bathy dataset
    ds_bathy = xr.Dataset(
        data_vars=dict(
            elevation=(["lat", "lon"], bathy.elevation.values),
        ),
        coords=dict(
            lon=bathy.lon.values,
            lat=bathy.lat.values,
        ),
        attrs=dict(
            description="Bathymetry data from GEBCO 2021",
            geodesic_frame="WGS84",
        ),
    )

    # Add attributes to variables
    ds_bathy.elevation.attrs["units"] = "m"
    ds_bathy.lon.attrs["units"] = "°"
    ds_bathy.lat.attrs["units"] = "°"
    ds_bathy.elevation.attrs["long_name"] = "Elevation (WGS84)"
    ds_bathy.lon.attrs["long_name"] = "Longitude"
    ds_bathy.lat.attrs["long_name"] = "Latitude"

    # Save to netCDF
    fname = "bathy.nc"
    fpath = os.path.join(root_data, fname)
    ds_bathy.to_netcdf(fpath)


# ======================================================================================================================
#  3) Get wav data save it as xarray dataset : ds_wav
# ======================================================================================================================


def get_wav_info(root_data):
    # Get the list of wav files available in the directory
    obs_wav_dir = os.path.join(
        root_data,
        "wav",
    )
    wav_files = os.listdir(obs_wav_dir)

    # Extract start times from filenames
    date_fmt = "%Y-%m-%d_%H-%M-%S"
    wav_file_info = []
    for wav_file in wav_files:
        if wav_file.endswith(".wav"):
            start_dt = datetime.strptime("_".join(wav_file.split("_")[3:5]), date_fmt)
            end_dt = datetime.strptime("_".join(wav_file.split("_")[5:7]), date_fmt)
            rcv_id = wav_file.split("_")[-1].split(".")[0]
            rcv_id_num = int(
                rcv_id[3:]
            )  # Extract number from rcv_id (e.g. "rcv_1" -> 1)
            rcv_id = f"RCV{rcv_id_num:02d}"  # Format rcv_id as "RCV01", "RCV02", etc.
            ch = wav_file.split("_")[2]

            # Store info
            wav_file_i = {
                "start_datetime": start_dt,
                "end_datetime": end_dt,
                "filepath": os.path.join(obs_wav_dir, wav_file),
                "receiver_id": rcv_id,
                "channel": ch,
            }
            wav_file_info.append(wav_file_i)

    return wav_file_info


def prepocess_wav(root_data):
    # Get information about wav files (start time, end time, filepath, receiver id)
    wav_info = get_wav_info(root_data=root_data)

    signal_mat = []  # n_rcv, nt
    sampling_freqs = []  # n_rcv
    # Load signal
    for wav_file_i in wav_info:
        # print()
        # Import wav file
        signal, fs = sf.read(wav_file_i["filepath"])
        # Centre signal
        signal -= np.mean(signal)
        # Add to list
        signal_mat.append(signal)
        sampling_freqs.append(fs)

    # Assert all sampling frequencies are the same
    assert (
        len(set(sampling_freqs)) == 1
    ), "All wav files must have the same sampling frequency"
    fs = sampling_freqs[0]
    # Assert all signals are from the same channel
    channels = [wav_file_i["channel"] for wav_file_i in wav_info]
    assert len(set(channels)) == 1, "All wav files must be from the same channel"
    ch = channels[0]

    # Storing time is useless since we can easily recompute it from fs
    rcv_ids = [wav_info_i["receiver_id"] for wav_info_i in wav_info]
    signal_labels = [f"signal_{rcv_id}" for rcv_id in rcv_ids]
    time_labels = [f"time_{rcv_id}" for rcv_id in rcv_ids]
    data_vars = {
        signal_label: ([time_labels[i]], signal_mat[i].astype(np.float32))
        for i, signal_label in enumerate(signal_labels)
    }
    start_dts = [wav_file_i["start_datetime"] for wav_file_i in wav_info]
    end_dts = [wav_file_i["end_datetime"] for wav_file_i in wav_info]
    data_vars.update(
        {
            "start_datetimes": (["receiver_id"], start_dts),
            "end_datetimes": (["receiver_id"], end_dts),
        }
    )

    ds_wav = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            receiver_id=rcv_ids,
        ),
        attrs=dict(
            description="Merged wav files.",
            channel=ch,
            fs=fs,
            datetime_format="%Y-%m-%d_%H-%M-%S",
        ),
    )

    ch_units = "uPa"
    ch_name = "Acoustic pressure"
    for rcv_id in rcv_ids:
        ds_wav[f"signal_{rcv_id}"].attrs["units"] = ch_units
        ds_wav[f"signal_{rcv_id}"].attrs["long_name"] = ch_name

    # Save dataset
    fname = f"channel_{ch}_wav.nc"
    nc_fpath = os.path.join(root_data, fname)
    ds_wav.to_netcdf(nc_fpath)


if __name__ == "__main__":
    root_data = rf"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\reseau_9R\data"
    box_center_lon = -81.7504
    box_center_lat = 18.3794

    dlon = 10
    dlat = 10

    # ### 1) AIS preprocessing ###
    # extract_subset_ais(
    #     box_center_lon=box_center_lon,
    #     box_center_lat=box_center_lat,
    #     dlon_box=dlon,
    #     dlat_box=dlat,
    # )

    # preprocess_ais(root_data=root_data)

    # ### 2) Bathy preprocessing ###
    # prepocess_bathy(
    #     box_center_lon=box_center_lon,
    #     box_center_lat=box_center_lat,
    #     dlon_box=dlon,
    #     dlat_box=dlat,
    #     root_data=root_data,
    # )

    ### 3) Wav preprocessing ###
    # prepocess_wav(root_data=root_data)
    # # Load
    # ds_wav = xr.open_dataset(os.path.join(root_data, "channel_EDH_wav.nc"))
    # print(ds_wav)
