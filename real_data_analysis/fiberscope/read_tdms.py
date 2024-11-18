#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   read_tdms.py
@Time    :   2024/11/12 15:43:22
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import nptdms
import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal as sp
import matplotlib.pyplot as plt

from publication.PublicationFigure import PubFigure

PubFigure()


def load_fiberscope_data(file_path):
    # Load data from tdms file
    group_of_interrest = "Acquisition Hydros - Données"

    # Load data into dataframe
    usefull_props = {
        "channel": [
            "wf_increment",
            "wf_start_offset",
            "wf_samples",
            "wf_start_time",
        ],
        "group": ["Freq. ech", "Freq. ech. Unité", "Gamme", "Gamme Unité"],
    }
    with nptdms.TdmsFile.open(file_path) as tdms_file:
        df = tdms_file[group_of_interrest].as_dataframe(time_index=True)
        usefull_attrs = {
            prop: tdms_file[group_of_interrest].properties[prop]
            for prop in usefull_props["group"]
        }
        usefull_attrs.update(
            {
                prop: tdms_file[group_of_interrest]["Hydro1"].properties[prop]
                for prop in usefull_props["channel"]
            }
        )

    # Rename columns
    df.columns = [f"H{i}" for i in range(1, len(df.columns) + 1)]

    # Convert to xarray
    ds = df.to_xarray()

    # Rename index dimension into time
    ds = ds.rename({"index": "time"})

    # ds.drop_attrs("wf_increment")

    # Concatenate H1 to H5 into a single new varaible 'signal'
    ds = xr.concat([ds.H1, ds.H2, ds.H3, ds.H4, ds.H5], dim="h_index")
    ds = ds.to_dataset(name="signal")
    # Add new coordinates
    ds["h_index"] = [1, 2, 3, 4, 5]

    # Remove mean from signal to ensure that the signal is centered on 0
    ds["signal"] = ds.signal - ds.signal.mean("time")

    # Store usefull attributes in the dataset
    for key, value in usefull_attrs.items():
        ds.attrs[key] = value

    # Convert start time from datetime to str (For serialization to netCDF files)
    ds.attrs["wf_start_time"] = pd.to_datetime(ds.attrs["wf_start_time"]).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )

    # Rename attrs
    ds.attrs["ts"] = np.float64(ds.attrs["wf_increment"])
    ds.attrs["fs"] = 1 / ds.ts

    # Add attributes to dimensions
    ds["time"].attrs["long_name"] = r"$t$"
    ds["time"].attrs["unit"] = r"$\textrm{s}$"
    ds["h_index"].attrs["long_name"] = "Hydrophone index"
    ds["h_index"].attrs["unit"] = "index"

    # Add attributes to variables
    ds["signal"].attrs["long_name"] = r"$u$"
    ds["signal"].attrs["unit"] = r"$\textrm{V}$"

    # Derive stft of the signal
    stft = []
    nperseg = 2**14
    noverlap = 2**13
    for idx in ds.h_index:
        ff, tt, stft_i = sp.stft(
            ds.signal.sel(h_index=idx),
            fs=ds.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
        )
        stft.append(stft_i)

    # Add stft to dataset as two new variables (amplitude and phase to avoid complex values)
    stft = np.array(stft)
    ds["ff"] = ff
    ds["tt"] = tt
    ds["stft_amp"] = (
        ["h_index", "ff", "tt"],
        np.abs(stft),
    )
    ds["stft_phase"] = (
        ["h_index", "ff", "tt"],
        np.angle(stft),
    )

    return ds


if __name__ == "__main__":
    data_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\Fiberscope_campagne_oct_2024"
    date = "09-10-2024"
    data_path = os.path.join(data_root, f"Campagne_{date}")

    # file_name = "09-10-2024T09-44-51-713655_P1_N1_Burst_2.tdms"
    # file_name = "09-10-2024T11-03-11-806485_P1_N1_Sweep_49.tdms"
    file_name = "09-10-2024T10-34-58-394627_P1_N1_Sweep_34.tdms"
    file_path = os.path.join(data_path, file_name)

    # file_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope\09-10-2024T09-34-36-733480_P1_N1_Burst_1.tdms"
    # group_of_interrest = "Acquisition Hydros - Données"
    # with nptdms.TdmsFile.open(file_path) as tdms_file:
    #     all_groups = tdms_file.groups()
    #     print(all_groups)

    #     group = tdms_file[group_of_interrest]
    #     # Iterate over all items in the group properties and print them
    #     for name, value in group.properties.items():
    #         print("{0}: {1}".format(name, value))

    #     hydro1 = group["Hydro1"]
    #     for name, value in hydro1.properties.items():
    #         print("{0}: {1}".format(name, value))

    #     time = hydro1.time_track()
    #     ts = hydro1.properties["wf_increment"]

    #     # Get the data
    #     channel = tdms_file[group_of_interrest]["Hydro1"]

    data = load_fiberscope_data(file_path)
    plt.figure()
    data.signal.plot(x="time", hue="h_index")

    # for idx in data.h_index:
    #     plt.figure()
    #     np.abs(data.stft.sel(h_index=idx)).plot(x="tt", y="ff")

    plt.show()

    print(data)
