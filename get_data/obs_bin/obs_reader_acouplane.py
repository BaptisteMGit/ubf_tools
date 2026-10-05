#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   obs_reader.py
@Time    :   2026/10/05 13:40:18
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Utility functions to read the OBS data from the ACOUPLANE project. Adapted from the original code
OBS_reader.py provided by Myriam Lajaunie (SHOM).

"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import pandas as pd

from datetime import datetime
from get_data.obs_bin.outils_shom.Elobs_bin_reader import (
    getSerialNumbers,
    fileListGet,
    parseELOBSfilename,
    ConvertGPSDate2String,
    FS_SIGDEL3_V,
)
from misc import progression_bar

CHANNEL_H = 3
FS = 2000
G = 0.0  # Gain in dB for the hydrophone channel (channel 3)


def read_obs_bin(bin_filepath):
    raw_obs_data = np.fromfile(bin_filepath, np.int32, -1, "")
    return raw_obs_data


def convert_raw_data(raw_data, fullScale):
    """Convert raw OBS data to physical units.

    Parameters
    ----------
    raw_data : np.ndarray
        Raw OBS data.
    fullScale : float
        Full scale value for the conversion.

    Returns
    -------
    np.ndarray
        Converted OBS data in physical units.
    """
    return raw_data * float(fullScale) / 2**32


def read_obs_channel_H(folder, read_raw=False, fs=FS, verbose=False):
    """Read OBS data from folder containing bin files assuming that the folder contains only one serial number and that the data type is digital (hydrophone channel).

    Parameters
    ----------
    folder : str
        Directory containing the bin files for a single OBS serial number.
    read_rawdata : bool, optional
        If True, read the raw data without conversion to physical units, by default False
    fs : int, optional
        Sampling frequency, by default FS
    verbose : bool, optional
        verbose option, by default False

    Returns
    -------
    (obs_data, obs_time)
    obs_data : np.ndarray
        Array containing the OBS data.
    obs_time : np.ndarray
        Array containing the corresponding datetimes for the OBS data.

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """

    # Adapted from the function readOBS() (original code OBS_reader.py provided by Myriam Lajaunie (SHOM)).

    # Here folder are should be popullated with single serial number
    # Using sn_user = -1 get all serial numbers in the folder
    list_serial_numbers = getSerialNumbers(folder, sn_user=-1)
    # Assert that there is only one serial number in the folder
    if len(list_serial_numbers) == 1:
        obs_serial_number = list_serial_numbers[0][0]
    else:
        raise ValueError("There should be only one serial number in the folder")

    # ts = get_min_timestamp(folder, obs_serial_number, ts=-1)

    # NOTE: sn[1] = "A" or "D" for data type (A: analogic, D: digital) -> here we only have digital data (hydrophone)
    # Assert that the data type is "D" for digital
    if list_serial_numbers[0][1] != "D":
        raise ValueError("Data type should be 'D' for digital (hydrophone)")

    # axes = Gen_Axes_D

    # NOTE: here we only have channel 3 (hydrophone) in the folder, so we can set fullScale directly
    # if sn[1] == "A" or chan == 3:
    #     fullScale = FS_SIGDEL3_V / np.power(10, g / 20)
    fullScale = FS_SIGDEL3_V / np.power(10, G / 20)

    # List files in folder
    obs_bin_files = fileListGet(folder, ext="bin")

    obs_data = []
    obs_time = []
    for fpath in obs_bin_files:

        if verbose:
            print(f"Reading file: {fpath}")
        # Get file information
        file_info = parseELOBSfilename(fpath, ext="bin")
        # Get start timestamp from file information
        # NOTE: B. Menetrier -> One can check that float(file_info["start"]) / 1e6 actually contains the entire timestamp
        # by using the following code:
        # start_gps_date = 1377271452000000
        # start_gps_date = float(start_timestamp) / 1e6
        # start_dt = ConvertGPSDate2String(gpsDate=start_gps_date) which gives 'Aug 28 2023 15:23:54 UTC'
        # This result matches with the example provided in the header of the detrame function from
        # the MATLAB code (provided by JM B.).
        start_gps_date = float(file_info["start"]) / 1e6

        # Le timestamp de stop n'est pas bon sur les anciennes versions => calcul sur taille du fichier
        # NOTE : B. Menetrier ->
        # os.path.getsize(f) donne la taille du fichier en octets (bytes)
        # Le facteur 4 correspond à la taille d'un int32 en octets, et le facteur fs correspond au nombre d'échantillons par seconde.
        # Donc, os.path.getsize(f) / 4 / fs calcule la durée du fichier en secondes.
        # NOTE: il faut que fs = 2000 Hz pour que stop_gps_date match avec float(file_info["stop"]) / 1e6
        # stop_gps_date = start_gps_date + os.path.getsize(fpath) / 4 / fs

        # Here we trust the stop timestamp in the file name, but we can also compute it from the file size if needed
        # NOTE: one can check that it matches start_dt and stop_dt provided by the MATLAB code (provided by JM B.)
        stop_gps_date = float(file_info["stop"]) / 1e6

        # Convert to string datetime
        dt_fmt = "%d/%m/%Y %H:%M:%S"
        start_str = ConvertGPSDate2String(gpsDate=start_gps_date, format=dt_fmt)
        stop_str = ConvertGPSDate2String(gpsDate=stop_gps_date, format=dt_fmt)
        if verbose:
            print(f"Recording from {start_str} to {stop_str}")

        # Convert to datetime
        start_dt = datetime.strptime(start_str, dt_fmt)
        stop_dt = datetime.strptime(stop_str, dt_fmt)

        # Read bin file
        data = read_obs_bin(fpath)
        if not read_raw:
            # Convert raw data to physical units
            data = convert_raw_data(data, fullScale)

        # NOTE: check the following lines -> ensure data as the required length
        # The extraction is copied from getDataFromBin() and thus assumes that the data to keep is the first samples of the file.
        # I don't know exactly why we are doing this
        data_duration_s = (stop_dt - start_dt).seconds
        last_data_sample = int(data_duration_s * fs)

        # Slice the data to the required length
        data = data[:last_data_sample]
        # Create a time vector for the data
        time = pd.date_range(
            start=start_dt, end=stop_dt, freq=f"{1/fs}s", inclusive="left"
        )

        # Store the data and time in the lists
        obs_data.append(data)
        obs_time.append(time)

    # Order arrays by time (in case the files are not in order)
    obs_start_times = [t[0] for t in obs_time]
    idx_sorted = np.argsort(obs_start_times)
    obs_data = [obs_data[i] for i in idx_sorted]
    obs_time = [obs_time[i] for i in idx_sorted]

    # Check that the time arrays are continuous (no gaps between files)
    for i in range(len(obs_time) - 1):
        step = (obs_time[i + 1][0] - obs_time[i][-1]).total_seconds()
        if step != 1 / fs:
            raise ValueError(
                f"Time arrays are not continuous between files {i} and {i+1}: "
                f"{obs_time[i][-1]} != {obs_time[i + 1][0]}"
            )

    # Concatenate all data and time
    obs_data = np.concatenate(obs_data)
    obs_time = np.concatenate(obs_time)

    return obs_data, obs_time, fullScale


def read_obs_data(data_folder, read_raw=False, verbose=False):

    # List all subfolders in the data folder
    obs_folders = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, f))
    ]
    obs_data = {}
    obs_time = {}

    # Test progress bar
    index0 = 0
    indexf = len(obs_folders) - 1
    prev_progress = 0

    for i, folder in enumerate(obs_folders):
        if verbose:
            print(f"Reading OBS data from folder: {folder}")
        else:
            prev_progress = progression_bar(i, index0, indexf, prev_progress)

        # Read data from the folder
        obsi_data, obsi_time, fullscale = read_obs_channel_H(
            folder, read_raw=read_raw, verbose=verbose
        )

        # Store
        obs_name = os.path.basename(folder)
        obs_data[obs_name] = obsi_data
        obs_time[obs_name] = obsi_time

    return obs_data, obs_time, fullscale


if __name__ == "__main__":
    # # Read single OBS data from a specific folder
    # obs1_folder = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ACOUPLANE\DATA\PRESSURE\BIN\OBS1"
    # obs1_data, obs1_time = read_obs_channel_H(obs1_folder, verbose=True)

    # Read all OBS data from a main folder containing subfolders for each OBS
    data_folder = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ACOUPLANE\DATA\PRESSURE\BIN"
    obs_data, obs_time = read_obs_data(
        data_folder=data_folder, read_raw=False, verbose=True
    )

    print(obs_data)
    print(obs_time)
