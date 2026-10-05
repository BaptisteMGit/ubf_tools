#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2026/09/28 21:54:27
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Useful functions for the acouplane project
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
from get_data.obs_bin.obs_reader_acouplane import convert_raw_data

# ======================================================================================================================
# Data preprocessing
# ======================================================================================================================


def build_obs_datetime_vector(ds_pressure):

    datetime_obs_dict = {}

    for obs_id in ds_pressure.obs_id.values:
        start_datetime_obs = ds_pressure.start_datetime.sel(obs_id=obs_id).values
        signal_duration_obs_s = (
            ds_pressure[f"time_{obs_id}"].values[-1] * 1 / ds_pressure.fs
        )
        end_datetime_obs = ds_pressure.start_datetime.sel(
            obs_id=obs_id
        ).values + timedelta(seconds=signal_duration_obs_s)
        datetime_obs = pd.date_range(
            start=start_datetime_obs,
            end=end_datetime_obs,
            freq=f"{1/ds_pressure.fs}s",
            inclusive="both",
        )

        datetime_obs_dict[obs_id] = datetime_obs

    return datetime_obs_dict


def convert_raw_pressure_to_physical_units(ds_pressure):
    for obs_id in ds_pressure.obs_id.values:
        raw_pressure = ds_pressure[f"pressure_{obs_id}"].values
        pressure = convert_raw_data(
            raw_data=raw_pressure, fullScale=ds_pressure.fullscale
        )
        ds_pressure[f"pressure_{obs_id}"] = (f"time_{obs_id}", pressure)

    return ds_pressure


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
