#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   preprocess_utils.py
@Time    :   2026/01/19 14:26:47
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Utils for data preprocessing
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import sys
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from publication.publication_figure import PubFigure, color
from real_data_analysis.fiberscope_groix.src.data_prepocessing.oceano_utils import (
    SBE39_reader,
)

# Defaults
# SBE39_OBS_folder = r"D:\DATA_CAMPAGNES\DATA_Groix\SBE39_OBS"
SBE39_OBS_FOLDER = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\fiberscope_groix_oct_2025\SBE39_OBS"


# ======================================================================================================================
# SBE DATA
# ======================================================================================================================


def interpolate_sbe_series(
    time_obs_series,
    raw_temperature_obs_series,
    raw_immersion_obs_series,
    time_step="30s",
):

    # Extract data for each sensor
    time_obs1, time_obs2, time_obs3 = time_obs_series
    raw_temperature_obs1, raw_temperature_obs2, raw_temperature_obs3 = (
        raw_temperature_obs_series
    )
    raw_immersion_obs1, raw_immersion_obs2, raw_immersion_obs3 = (
        raw_immersion_obs_series
    )

    # Get time limits
    t_start = np.max([time_obs1.min(), time_obs2.min(), time_obs3.min()])
    t_end = np.min([time_obs1.max(), time_obs2.max(), time_obs3.max()])
    # Round to closest
    t_start = t_start.ceil(time_step)
    t_end = t_end.floor(time_step)

    # Set common time vector
    time_interp = pd.date_range(t_start, t_end, freq=time_step)
    # Convert to linear time
    linear_time_interp = (time_interp - t_start) / np.timedelta64(
        1, "s"
    )  # Linear time from start

    linear_time_obs1 = (time_obs1 - t_start) / np.timedelta64(1, "s")
    linear_time_obs2 = (time_obs2 - t_start) / np.timedelta64(1, "s")
    linear_time_obs3 = (time_obs3 - t_start) / np.timedelta64(1, "s")

    # Interpolate
    temperature_obs1 = np.interp(
        linear_time_interp, linear_time_obs1, raw_temperature_obs1
    )
    temperature_obs2 = np.interp(
        linear_time_interp, linear_time_obs2, raw_temperature_obs2
    )
    temperature_obs3 = np.interp(
        linear_time_interp, linear_time_obs3, raw_temperature_obs3
    )
    immersion_obs1 = np.interp(linear_time_interp, linear_time_obs1, raw_immersion_obs1)
    immersion_obs2 = np.interp(linear_time_interp, linear_time_obs2, raw_immersion_obs2)
    immersion_obs3 = np.interp(linear_time_interp, linear_time_obs3, raw_immersion_obs3)

    return (
        temperature_obs1,
        temperature_obs2,
        temperature_obs3,
        immersion_obs1,
        immersion_obs2,
        immersion_obs3,
        time_interp,
    )


def build_SBE_dataset(sbe_data_folder=SBE39_OBS_FOLDER, interpolation_time_step="30s"):

    # Load SBE39 data
    SBE39_OBS6 = "\SBE39-945.asc"
    SBE39_OBS5 = "\SBE39-1362.asc"
    SBE39_OBS4 = "\SBE39-1123.asc"
    SBE4 = SBE39_reader(SBE39_OBS4, sbe_data_folder)
    SBE5 = SBE39_reader(SBE39_OBS5, sbe_data_folder)
    SBE6 = SBE39_reader(SBE39_OBS6, sbe_data_folder)

    datetime_SBE4 = pd.to_datetime(SBE4.time, format="%Y-%m-%d %H:%M:%S.%f")
    datetime_SBE5 = pd.to_datetime(SBE5.time, format="%Y-%m-%d %H:%M:%S.%f")
    datetime_SBE6 = pd.to_datetime(SBE6.time, format="%Y-%m-%d %H:%M:%S.%f")

    # sélection des périodes immergées
    ind4 = np.where((np.array(SBE4.depth) > 20))[0]
    ind5 = np.where((np.array(SBE5.depth) > 20))[0]
    ind6 = np.where((np.array(SBE6.depth) > 29))[0]

    time_obs1 = datetime_SBE4[ind4]
    time_obs2 = datetime_SBE5[ind5]
    time_obs3 = datetime_SBE6[ind6]
    raw_temperature_obs1 = np.array(SBE4.Ture)[ind4]
    raw_temperature_obs2 = np.array(SBE5.Ture)[ind5]
    raw_temperature_obs3 = np.array(SBE6.Ture)[ind6]
    raw_immersion_obs1 = np.array(SBE4.depth)[ind4]
    raw_immersion_obs2 = np.array(SBE5.depth)[ind5]
    raw_immersion_obs3 = np.array(SBE6.depth)[ind6]

    # Interpolation sur le même vecteur de temps
    (
        temperature_obs1,
        temperature_obs2,
        temperature_obs3,
        immersion_obs1,
        immersion_obs2,
        immersion_obs3,
        time_interp,
    ) = interpolate_sbe_series(
        (time_obs1, time_obs2, time_obs3),
        (raw_temperature_obs1, raw_temperature_obs2, raw_temperature_obs3),
        (raw_immersion_obs1, raw_immersion_obs2, raw_immersion_obs3),
        time_step=interpolation_time_step,
    )

    # Build xarray dataset
    # Correspondance nomenclatures OBS : 1, 2, 3 = 4, 5, 6
    ds_SBE = xr.Dataset(
        data_vars=dict(
            raw_temperature_obs1=(["time_obs1"], raw_temperature_obs1),
            raw_temperature_obs2=(["time_obs2"], raw_temperature_obs2),
            raw_temperature_obs3=(["time_obs3"], raw_temperature_obs3),
            raw_immersion_obs1=(["time_obs1"], raw_immersion_obs1),
            raw_immersion_obs2=(["time_obs2"], raw_immersion_obs2),
            raw_immersion_obs3=(["time_obs3"], raw_immersion_obs3),
            temperature_obs1=(["time"], temperature_obs1),
            temperature_obs2=(["time"], temperature_obs2),
            temperature_obs3=(["time"], temperature_obs3),
            immersion_obs1=(["time"], immersion_obs1),
            immersion_obs2=(["time"], immersion_obs2),
            immersion_obs3=(["time"], immersion_obs3),
        ),
        coords=dict(
            time=time_interp,
            time_obs1=datetime_SBE4[ind4],
            time_obs2=datetime_SBE5[ind5],
            time_obs3=datetime_SBE6[ind6],
        ),
        attrs=dict(
            description="Données SBE39 OBS. Les données ont été sélectionnées pour ne garder que les périodes où les capteurs sont immergés.",
            source="Dossier SBE39_OBS (données SHOM)",
            traitemement="Fonction SBE39_reader adaptée du script Read_oceano.py fourni par le SHOM. Les données (non raw) sont interpolées linéairement sur un vecteur de temps commun.",
            notation="Les OBS sont renommés OBS1, OBS2 et OBS3 conformément à la nomenclature utilisée dans les documents de préparation de campagne. La correspondance est la suivante : OBS1 = SBE39-1123, OBS2 = SBE39-1362, OBS3 = SBE39-945",
            interpolation_time_step=interpolation_time_step,
        ),
    )

    # Add attributes to variables
    ds_SBE.temperature_obs1.attrs["units"] = "°C"
    ds_SBE.temperature_obs2.attrs["units"] = "°C"
    ds_SBE.temperature_obs3.attrs["units"] = "°C"
    ds_SBE.temperature_obs1.attrs["long_name"] = "Température"
    ds_SBE.temperature_obs2.attrs["long_name"] = "Température"
    ds_SBE.temperature_obs3.attrs["long_name"] = "Température"
    ds_SBE.raw_temperature_obs1.attrs["units"] = "°C"
    ds_SBE.raw_temperature_obs2.attrs["units"] = "°C"
    ds_SBE.raw_temperature_obs3.attrs["units"] = "°C"
    ds_SBE.raw_temperature_obs1.attrs["long_name"] = "Température"
    ds_SBE.raw_temperature_obs2.attrs["long_name"] = "Température"
    ds_SBE.raw_temperature_obs3.attrs["long_name"] = "Température"

    ds_SBE.immersion_obs1.attrs["units"] = "m"
    ds_SBE.immersion_obs2.attrs["units"] = "m"
    ds_SBE.immersion_obs3.attrs["units"] = "m"
    ds_SBE.immersion_obs1.attrs["long_name"] = "Immersion"
    ds_SBE.immersion_obs2.attrs["long_name"] = "Immersion"
    ds_SBE.immersion_obs3.attrs["long_name"] = "Immersion"
    ds_SBE.raw_immersion_obs1.attrs["units"] = "m"
    ds_SBE.raw_immersion_obs2.attrs["units"] = "m"
    ds_SBE.raw_immersion_obs3.attrs["units"] = "m"
    ds_SBE.raw_immersion_obs1.attrs["long_name"] = "Immersion"
    ds_SBE.raw_immersion_obs2.attrs["long_name"] = "Immersion"
    ds_SBE.raw_immersion_obs3.attrs["long_name"] = "Immersion"

    ds_SBE.time.attrs["long_name"] = "Temps UTC"
    ds_SBE.time_obs1.attrs["long_name"] = "Temps UTC"
    ds_SBE.time_obs2.attrs["long_name"] = "Temps UTC"
    ds_SBE.time_obs3.attrs["long_name"] = "Temps UTC"

    # Overwriten by xarray automatically while saving to netcdf
    ds_SBE.time.attrs["timezone"] = "UTC"
    ds_SBE.time_obs1.attrs["timezone"] = "UTC"
    ds_SBE.time_obs2.attrs["timezone"] = "UTC"
    ds_SBE.time_obs3.attrs["timezone"] = "UTC"

    return ds_SBE


def plot_SBE(ds_SBE):

    fig, axs = plt.subplots(2, 1, sharex=True)
    ds_SBE.raw_immersion_obs1.plot.scatter(
        ax=axs[0], label="Raw OBS1", marker="x", s=1, color=color(0)
    )
    ds_SBE.raw_immersion_obs2.plot.scatter(
        ax=axs[0], label="Raw OBS2", marker="x", s=1, color=color(1)
    )
    ds_SBE.raw_immersion_obs3.plot.scatter(
        ax=axs[0], label="Raw OBS3", marker="x", s=1, color=color(2)
    )
    ds_SBE.immersion_obs1.plot(ax=axs[0], label="OBS1", color=color(3))
    ds_SBE.immersion_obs2.plot(ax=axs[0], label="OBS2", color=color(4))
    ds_SBE.immersion_obs3.plot(ax=axs[0], label="OBS3", color=color(5))
    axs[0].set_xlabel("")

    ds_SBE.raw_temperature_obs1.plot.scatter(
        ax=axs[1], label="Raw OBS1", marker="x", s=1, color=color(0)
    )
    ds_SBE.raw_temperature_obs2.plot.scatter(
        ax=axs[1], label="Raw OBS2", marker="x", s=1, color=color(1)
    )
    ds_SBE.raw_temperature_obs3.plot.scatter(
        ax=axs[1], label="Raw OBS3", marker="x", s=1, color=color(2)
    )
    ds_SBE.temperature_obs1.plot(ax=axs[1], label="OBS1", color=color(3))
    ds_SBE.temperature_obs2.plot(ax=axs[1], label="OBS2", color=color(4))
    ds_SBE.temperature_obs3.plot(ax=axs[1], label="OBS3", color=color(5))

    axs[0].legend(ncols=2)
    axs[1].legend(ncols=2)

    t_zoom = slice(
        datetime.datetime(2025, 10, 15, 11, 30, 0),
        datetime.datetime(2025, 10, 15, 12, 30, 0),
    )
    fig, axs = plt.subplots(2, 1, sharex=True)

    ds_SBE.raw_immersion_obs1.sel(time_obs1=t_zoom).plot.scatter(
        ax=axs[0], label="Raw OBS1", marker="x", s=1, color=color(0)
    )
    ds_SBE.raw_immersion_obs2.sel(time_obs2=t_zoom).plot.scatter(
        ax=axs[0], label="Raw OBS2", marker="x", s=1, color=color(1)
    )
    ds_SBE.raw_immersion_obs3.sel(time_obs3=t_zoom).plot.scatter(
        ax=axs[0], label="Raw OBS3", marker="x", s=1, color=color(2)
    )
    ds_SBE.sel(time=t_zoom).immersion_obs1.plot(ax=axs[0], label="OBS1", color=color(3))
    ds_SBE.sel(time=t_zoom).immersion_obs2.plot(ax=axs[0], label="OBS2", color=color(4))
    ds_SBE.sel(time=t_zoom).immersion_obs3.plot(ax=axs[0], label="OBS3", color=color(5))
    axs[0].set_xlabel("")

    ds_SBE.raw_temperature_obs1.sel(time_obs1=t_zoom).plot.scatter(
        ax=axs[1], label="Raw OBS1", marker="x", s=1, color=color(0)
    )
    ds_SBE.raw_temperature_obs2.sel(time_obs2=t_zoom).plot.scatter(
        ax=axs[1], label="Raw OBS2", marker="x", s=1, color=color(1)
    )
    ds_SBE.raw_temperature_obs3.sel(time_obs3=t_zoom).plot.scatter(
        ax=axs[1], label="Raw OBS3", marker="x", s=1, color=color(2)
    )
    ds_SBE.sel(time=t_zoom).temperature_obs1.plot(
        ax=axs[1], label="OBS1", color=color(3)
    )
    ds_SBE.sel(time=t_zoom).temperature_obs2.plot(
        ax=axs[1], label="OBS2", color=color(4)
    )
    ds_SBE.sel(time=t_zoom).temperature_obs3.plot(
        ax=axs[1], label="OBS3", color=color(5)
    )

    axs[0].legend(ncols=2)
    axs[1].legend(ncols=2)
