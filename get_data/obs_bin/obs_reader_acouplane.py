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

from datetime import datetime, timedelta
from get_data.obs_bin.outils_shom.Elobs_bin_reader import (
    getSerialNumbers,
    fileListGet,
    parseELOBSfilename,
    ConvertGPSDate2String,
    FS_SIGDEL3_V,
)
import get_data.obs_bin.outils_shom.Elobs_bin_reader as obsr

from misc import progression_bar

CHANNEL_H = 3
FS = 2000
G = 0.0  # Gain in dB for the hydrophone channel (channel 3)


def read_obs_bin(bin_filepath):
    raw_obs_data = np.fromfile(bin_filepath, np.int32, -1, "")
    return raw_obs_data


def convert_data_raw_to_physical(raw_data, fullScale):
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

def convert_data_physical_to_raw(physical_data, fullScale):
    """Convert OBS data in physical units back to raw data in int32.

    Parameters
    ----------
    physical_data : np.ndarray
        OBS data in physical units.
    fullScale : float
        Full scale value for the conversion.

    Returns
    -------
    np.ndarray
        Raw OBS data in int32.
    """
    raw_data = physical_data * 2**32 / float(fullScale) 
    return raw_data.astype(np.int32)


def read_obs_channel_H_with_obsr(
    folder,
    read_raw=False,
    fs=FS,
    utc_date_lower_bound="23/02/2026 00:00:00",
    utc_date_upper_bound="03/03/2026 00:00:00",
    verbose=False,
):
    """ Read OBS data from folder containing bin files using obsr class provided.

    WARNING : this code assumes two modification of Elobs_bin_reader.py :
        1) Hard coding of sampling frequency fs = 2000 (line 618)
        2) Modification of the default behavior of the getMinTimestamp() function: line 310 is commented so that it returns np.max(ts_int) instaed of -1 even if channels are missing

    Parameters
    ----------
    folder : str
        Directory containing the bin files for a single OBS serial number.
    read_rawdata : bool, optional
        If True, read the raw data without conversion to physical units, by default False
    fs : int, optional
        Sampling frequency, by default FS
    utc_date_lower_bound : str, optional
        Lower bound of the data period to read, by default "23/02/2026 00:00:00"
    utc_date_upper_bound : str, optional
        Upper bound of the data period to read, by default "03/03/2026 00:00:00"
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    (obs_data, obs_time)
    obs_data : np.ndarray
        Array containing the OBS data.
    obs_time : np.ndarray
        Array containing the corresponding UTC datetimes for the OBS data.
    """

    # NOTE : utc_date bounds must be chosen to englobe the entire data sequence to read.
    start_date = utc_date_lower_bound
    stop_date = utc_date_upper_bound

    sig, timestamps, lh, sn = obsr.readOBS(
        folder, start_date, stop_date, channel="H", calib=True, SeismoVar="acc"
    )

    ### Convert data units if needed ###
    if read_raw:
        # NOTE: here we only have channel 3 (hydrophone) in the folder, so we can set fullScale directly
        # if sn[1] == "A" or chan == 3:
        #     fullScale = FS_SIGDEL3_V / np.power(10, g / 20)
        fullScale = FS_SIGDEL3_V / np.power(10, G / 20)

        # obsr.readOBS return scaled data in Volts, we need to convert back to raw_data in int32
        obs_data = convert_data_physical_to_raw(sig, fullScale=fullScale)
    else:
        # Convertion not required, keep physical units (way higher memory consumption)
        obs_data = sig
        # Set fullscale to None
        fullScale = None 

    # NOTE :
    # using timestamps returned by obsr.readOBS() might lead to errors. Indeed, timestamps are given in seconds from GPS epoch (datetime(1980, 1, 6)), yet
    # timestamps do not perfectly match with sampling frequency fs = 2000 Hz.
    # To avoid any confusion, obs_time vector is built directly using sampling frequency

    ### Convert time ###
    # NOTE : timestamps are in GPS time in s (precision up to us according to filename timestamp)
    gps_time_s = timestamps
    # Get start date in UTC
    dt_fmt = "%d/%m/%Y %H:%M:%S UTC"
    gps_start_date_s = gps_time_s[0]
    # Get integer number of seconds
    gps_start_date_int_s = gps_start_date_s.astype(np.int64)
    # Convert to UTC seconds
    utc_start_date_int_s = ConvertGPSDate2String(gpsDate=gps_start_date_int_s, format=dt_fmt)

    # Add microseconds information
    start_gps_date_us = (gps_start_date_s%1 * 1e6).round(0)        # File timestamp is given in us, float is rounded 
    utc_start_dt = datetime.strptime(utc_start_date_int_s, dt_fmt) + timedelta(microseconds=start_gps_date_us)

    # Build associated datetimes
    obs_time = pd.date_range(
        start=utc_start_dt, freq=f"{1/fs}s", inclusive="left", periods=len(timestamps)
    )

    return obs_data, obs_time, fullScale


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
        stop_gps_date = start_gps_date + os.path.getsize(fpath) / 4 / fs

        # Here we trust the stop timestamp in the file name, but we can also compute it from the file size if needed
        # NOTE: one can check that it matches start_dt and stop_dt provided by the MATLAB code (provided by JM B.)
        # stop_gps_date = float(file_info["stop"]) / 1e6

        # Convert to string datetime
        dt_fmt = "%d/%m/%Y %H:%M:%S UTC"
        # dt_fmt = "%d/%m/%Y %H:%M:%S:%f UTC"

        # Warning, ConvertGPSDate2String convert to int and thus discard us info 
        start_str = ConvertGPSDate2String(gpsDate=start_gps_date, format=dt_fmt)
        stop_str = ConvertGPSDate2String(gpsDate=stop_gps_date, format=dt_fmt)
        if verbose:
            print(f"Recording from {start_str} to {stop_str}")

        # Convert to datetime
        start_gps_date_us = float(file_info["start"][-6:])      # According to the documentation (harvesting des données.pdf), timestemps represent GPS time in us
        start_dt = datetime.strptime(start_str, dt_fmt) + timedelta(microseconds=start_gps_date_us)
        stop_gps_date_us = float(file_info["start"][-6:])      # According to the documentation (harvesting des données.pdf), timestemps represent GPS time in us
        stop_dt = datetime.strptime(stop_str, dt_fmt) + timedelta(microseconds=stop_gps_date_us)

        # Read bin file
        data = read_obs_bin(fpath)
        if not read_raw:
            # Convert raw data to physical units
            data = convert_data_raw_to_physical(data, fullScale)


        # Create a time vector for the data
        time = pd.date_range(
            start=start_dt, end=stop_dt, freq=f"{1/fs}s", inclusive="left"
        )

        # NOTE: check the following lines -> ensure data as the required length
        # The extraction is copied from getDataFromBin() and thus assumes that the data to keep is the first samples of the file.
        # I don't know exactly why we are doing this
        data_duration_s = (stop_dt - start_dt).seconds
        last_data_sample = int(data_duration_s * fs)
        # Slice data and time to the required length
        last_data_sample = min(last_data_sample, data.size)
        data = data[:last_data_sample]
        time = time[:last_data_sample]

        # print(f"data size = {data.size}, time size = {time.size}, equal = {data.size == time.size}")
        # Store the data and time in the lists
        obs_data.append(data)
        obs_time.append(time)

    # Order arrays by time (in case the files are not in order)
    obs_start_times = [t[0] for t in obs_time]
    idx_sorted = np.argsort(obs_start_times)
    obs_data = [obs_data[i] for i in idx_sorted]
    obs_time = [obs_time[i] for i in idx_sorted]

    # for i in range(len(obs_data)):
    #     print(f"data size = {obs_data[i].size}, time size = {obs_time[i].size}, equal = {obs_data[i].size == obs_time[i].size}")

    # Check that the time arrays are continuous (no gaps between files)
    for i in range(len(obs_time) - 1):
        step = (obs_time[i + 1][0] - obs_time[i][-1]).total_seconds()
        if step != 1 / fs:
            # raise ValueError(
            #     f"Time arrays are not continuous between files {i} and {i+1}: "
            #     f"{obs_time[i][-1]} != {obs_time[i + 1][0]}"
            # )
            print(
                f"Warning : Time arrays are not continuous between files {i} and {i+1}: "
                f"{obs_time[i][-1]} != {obs_time[i + 1][0]}, holes will be filled with zeros !"
            )
            # Fill with zeros to avoid holes     
            fill_times = pd.date_range(start=obs_time[i][-1], end=obs_time[i+1][0], freq=f"{1/fs}s", inclusive="neither")        
            obs_data[i] = np.concatenate([obs_data[i], np.zeros(fill_times.size)])
            obs_time[i] = np.concatenate([obs_time[i], fill_times])
            # np.zeros(int((lts[i] - ts) * fs))

    # for i in range(len(obs_data)):
    #     print(f"data size = {obs_data[i].size}, time size = {obs_time[i].size}, equal = {obs_data[i].size == obs_time[i].size}")

    # Concatenate all data and time
    obs_data = np.concatenate(obs_data)
    obs_time = np.concatenate(obs_time)

    if read_raw:
        # Ensure raw data are stored as int32
        obs_data = obs_data.astype(np.int32)

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

        # # Read data from the folder -> BM implementation 
        # obsi_data, obsi_time, fullscale = read_obs_channel_H(
        #     folder, read_raw=read_raw, verbose=verbose
        # )

        # Read data from the folder -> adapted from obsr class 
        obsi_data, obsi_time, fullscale = read_obs_channel_H_with_obsr(
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
    
    if os.name == "nt":  # Windows
        data_folder = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ACOUPLANE\DATA\PRESSURE\BIN"
    else:  # Linux
        data_folder = '/home/data/ACOUPLANE/DATA/PRESSURE/BIN'

    obs_data, obs_time = read_obs_data(
        data_folder=data_folder, read_raw=False, verbose=True
    )

    print(obs_data)
    print(obs_time)
