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
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from publication.publication_figure import PubFigure, color
from real_data_analysis.fiberscope_groix.src.data_prepocessing.oceano_utils import (
    SBE39_reader,
)
from misc import progression_bar
from source.utils.utils_gps import gpx_to_csv, interpolate_gps
from source.utils.utils_geo import geodetic_to_ecef, ecef_to_enu


# ======================================================================================================================
# DEFAULTS
# ======================================================================================================================
PROJECT_ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"

# SBE39_OBS_folder = r"D:\DATA_CAMPAGNES\DATA_Groix\SBE39_OBS"
SBE39_OBS_FOLDER = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\fiberscope_groix_oct_2025\SBE39_OBS"

ROOT_GROIX_DATA = os.path.join(PROJECT_ROOT, "data", "fiberscope_groix_oct_2025")
ROOT_GROIX_AIS = os.path.join(ROOT_GROIX_DATA, "ais")
ROOT_GROIX_GPS = os.path.join(ROOT_GROIX_DATA, "gps")
ROOT_GROIX_METADATA = os.path.join(ROOT_GROIX_DATA, "metadata")
ROOT_BATHY_DATA = os.path.join(PROJECT_ROOT, "data", "bathy")
# bathy_fpath = r"/home/program/ubf_tools/data/bathy/mmdpm/PVA_RR48/GEBCO_2021_lon_64.44_67.44_lat_-29.08_-26.08.nc"

AIS_SPATIONAV_FPATH = os.path.join(ROOT_GROIX_AIS, "SPATIONAV_AIS_001.csv")
# ais_spationav_fpath = os.path.join(root_groix_ais, "SPATIONAV_AIS_001.csv")

MMSI_JULES = 226916000

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


# ======================================================================================================================
# GPS / AIS DATA
# ======================================================================================================================


def load_apriori_pos_wgs84(root_groix_metadata=ROOT_GROIX_METADATA):
    apriori_pos_wgs84_fname_before_campaign = (
        "pos_deg.csv"  # Theorical positions (as planned before the campaign)
    )
    apriori_pos_wgs84_before_campaign = pd.read_csv(
        os.path.join(root_groix_metadata, apriori_pos_wgs84_fname_before_campaign),
        index_col=0,
    )
    apriori_pos_wgs84_before_campaign["h"] = 0.0  # Dummmy initialisation as float

    apriori_pos_wgs84_fname = (
        "pos_deg_relevee.csv"  # Position measured during the campaign
    )
    apriori_pos_wgs84 = pd.read_csv(
        os.path.join(root_groix_metadata, apriori_pos_wgs84_fname),
        index_col=0,
    )
    apriori_pos_wgs84["h"] = 0.0  # Dummmy initialisation as float

    return apriori_pos_wgs84_before_campaign, apriori_pos_wgs84


def get_obs_immersion(ds_SBE, verbose=False):
    immersion_obs1 = ds_SBE.immersion_obs1.mean().values
    immersion_obs2 = ds_SBE.immersion_obs2.mean().values
    immersion_obs3 = ds_SBE.immersion_obs3.mean().values

    if verbose:
        print(
            f"z_obs1 = {immersion_obs1} m, z_obs2 = {immersion_obs2} m, z_obs3 = {immersion_obs3} m"
        )

    return immersion_obs1, immersion_obs2, immersion_obs3


def set_apriori_pos_h(apriori_pos_wgs84, N_geoid_undulation, ds_SBE, verbose=False):

    # Get immersion from pressure series
    immersion_obs1, immersion_obs2, immersion_obs3 = get_obs_immersion(
        ds_SBE=ds_SBE, verbose=verbose
    )

    # Update heights accordingly
    apriori_pos_wgs84.loc["obs1", "h"] = N_geoid_undulation - immersion_obs1
    apriori_pos_wgs84.loc["obs2", "h"] = N_geoid_undulation - immersion_obs2
    apriori_pos_wgs84.loc["obs3", "h"] = N_geoid_undulation - immersion_obs3
    apriori_pos_wgs84.loc["t1", "h"] = N_geoid_undulation  # T1 position at sea level
    apriori_pos_wgs84.loc["t2", "h"] = N_geoid_undulation  # T2 position at sea level
    apriori_pos_wgs84.loc["t3", "h"] = N_geoid_undulation  # T3 position at sea level
    apriori_pos_wgs84.loc["t4", "h"] = N_geoid_undulation  # T4 position at sea level
    apriori_pos_wgs84.loc["t5", "h"] = N_geoid_undulation  # T5 position at sea level


def build_GPS_dataset(
    local_frame_origin,
    N_geoid_undulation,
    t_start,
    t_end,
    ds_SBE,
    interpolation_time_step="10s",
    root_groix_gps=ROOT_GROIX_GPS,
    root_groix_metadata=ROOT_GROIX_METADATA,
):
    # Extract GPS data
    root_gps_not_parsed = r"C:\Users\baptiste.menetrier\Desktop\ressource\XP_Fiberscope_Groix_092025\Jules\gps"
    extract_gps_data(root_gps_not_parsed, root_groix_gps)

    # Load GPS data
    gps = load_gps(root_groix_gps)

    # Interpolate GPS data to regular time intervals
    gps_interp = interpolate_gps(df_gps=gps, time_step=interpolation_time_step)

    # Sea level elevation above geoid = MSL
    zeta = (
        ds_SBE.immersion_obs1 - ds_SBE.immersion_obs1.mean().values
    )  # Approximate elevation above MSL
    zeta = zeta.sel(time=gps_interp["datetime"].values, method="nearest")

    # Derive GPS position in local ENU coordinates
    e_gps, n_gps, u_gps = transform_pos_wgs84_ecef(
        pos=gps_interp,
        local_frame_origin=local_frame_origin,
        N_geoid_undulation=N_geoid_undulation,
        zeta=zeta,
        sensor_h_above_sea_level=5,
    )

    # Format GPS data to match common time vector
    gps_interp_lon, gps_interp_lat, e_gps, n_gps, u_gps = format_gps_data(
        gps_interp=gps_interp,
        e_gps=e_gps,
        n_gps=n_gps,
        u_gps=u_gps,
        t_start=t_start,
        t_end=t_end,
        time_step=interpolation_time_step,
    )
    common_time = pd.date_range(start=t_start, end=t_end, freq=interpolation_time_step)

    # Derive GPS speed
    dt = (common_time[1] - common_time[0]).total_seconds()
    v_e_gps, v_n_gps, v_u_gps = get_speed(dt=dt, e=e_gps, n=n_gps, u=u_gps, axis=0)

    # Get apriori pos of interest
    apriori_pos_wgs84_before_campaign, apriori_pos_wgs84 = load_apriori_pos_wgs84(
        root_groix_metadata=root_groix_metadata
    )
    set_apriori_pos_h(
        apriori_pos_wgs84=apriori_pos_wgs84,
        N_geoid_undulation=N_geoid_undulation,
        ds_SBE=ds_SBE,
        verbose=False,
    )
    set_apriori_pos_h(
        apriori_pos_wgs84=apriori_pos_wgs84_before_campaign,
        N_geoid_undulation=N_geoid_undulation,
        ds_SBE=ds_SBE,
        verbose=False,
    )
    apriori_pos_enu = transform_apriori_pos_wgs84_to_ecef(
        local_frame_origin, apriori_pos_wgs84
    )
    apriori_pos_enu_before_campaign = transform_apriori_pos_wgs84_to_ecef(
        local_frame_origin, apriori_pos_wgs84_before_campaign
    )

    # Finaly build GPS dataset
    ds_gps = gps_dataset(
        gps=gps,
        gps_interp_lon=gps_interp_lon,
        gps_interp_lat=gps_interp_lat,
        e_gps=e_gps,
        n_gps=n_gps,
        u_gps=u_gps,
        v_e_gps=v_e_gps,
        v_n_gps=v_n_gps,
        v_u_gps=v_u_gps,
        common_time=common_time,
        local_frame_origin=local_frame_origin,
        apriori_pos_wgs84=apriori_pos_wgs84,
        apriori_pos_wgs84_before_campaign=apriori_pos_wgs84_before_campaign,
        apriori_pos_enu=apriori_pos_enu,
        apriori_pos_enu_before_campaign=apriori_pos_enu_before_campaign,
        time_step=interpolation_time_step,
    )

    return ds_gps


def extract_gps_data(root_gps_not_parsed, root_groix_gps):
    # Get all files within subfolders : folder = 13_10_25; files 13_10_25.txt
    for root, dirs, files in os.walk(root_gps_not_parsed):
        for file in files:
            if file.endswith(".gpx"):
                fpath_in = os.path.join(root, file)
                fname = os.path.splitext(file)[0]
                root_output = os.path.join(root_groix_gps, os.path.basename(root))
                fpath_out = os.path.join(root_output, f"gps_pos_{fname}.csv")
                if not os.path.exists(root_output):
                    os.makedirs(root_output)
                gpx_to_csv(gpx_file=fpath_in, csv_file=fpath_out)


def load_gps(root_groix_gps):
    # Define associated dataframe
    gps = pd.DataFrame()

    for root, dirs, files in os.walk(root_groix_gps):
        for file in files:
            if file.endswith(".csv"):
                fpath_in = os.path.join(root, file)
                df_gps = pd.read_csv(fpath_in, sep=",")

                # Drop elevation column
                if "ele" in df_gps.columns:
                    df_gps = df_gps.drop(columns=["ele"])

                # Convert time column to datetime
                df_gps["datetime"] = pd.to_datetime(df_gps["time"])

                # Drop original time column
                df_gps = df_gps.drop(columns=["time"])

                # Store in dictionary
                date = df_gps["datetime"].dt.date.unique()[0].strftime("%Y-%m-%d")
                # date = os.path.basename(root)
                # gps_pos[date] = df_gps
                gps = pd.concat([gps, df_gps], ignore_index=True)

    return gps


def transform_apriori_pos_wgs84_to_ecef(local_frame_origin, apriori_pos_wgs84):
    # WGS84 -> ECEF
    lat = np.deg2rad(apriori_pos_wgs84.lat)
    lon = np.deg2rad(apriori_pos_wgs84.lon)
    X, Y, Z = geodetic_to_ecef(lat, lon, apriori_pos_wgs84.h)
    # ECEF -> ENU
    lat0, lon0, h0 = (
        np.deg2rad(local_frame_origin["lat"]),
        np.deg2rad(local_frame_origin["lon"]),
        local_frame_origin["h"],
    )
    e, n, u = ecef_to_enu(X, Y, Z, lat0, lon0, h0)

    apriori_pos_enu = pd.DataFrame(
        np.stack((e, n, u)).T, index=apriori_pos_wgs84.index, columns=["e", "n", "u"]
    )

    return apriori_pos_enu


def transform_pos_wgs84_ecef(
    pos, local_frame_origin, N_geoid_undulation, zeta=0, sensor_h_above_sea_level=5
):
    lon = np.deg2rad(pos.lon)
    lat = np.deg2rad(pos.lat)
    # GPS h above WGS84 ellipsoid
    h = N_geoid_undulation + zeta + sensor_h_above_sea_level
    # WGS84 -> ECEF
    x, y, z = geodetic_to_ecef(lat, lon, h)
    # ECEF -> ENU
    lat0, lon0, h0 = (
        np.deg2rad(local_frame_origin["lat"]),
        np.deg2rad(local_frame_origin["lon"]),
        local_frame_origin["h"],
    )
    e, n, u = ecef_to_enu(x, y, z, lat0, lon0, h0)

    return e, n, u


def format_gps_data(gps_interp, e_gps, n_gps, u_gps, t_start, t_end, time_step):
    # Pad lon lat with NaNs to match common_time length
    n_pad_start = (gps_interp["datetime"].iloc[0] - t_start) // pd.Timedelta(time_step)
    n_pad_end = (t_end - gps_interp["datetime"].iloc[-1]) // pd.Timedelta(time_step)
    gps_interp_lon = np.pad(
        gps_interp.lon,
        ((n_pad_start, n_pad_end)),
        mode="constant",
        constant_values=np.nan,
    )
    gps_interp_lat = np.pad(
        gps_interp.lat,
        ((n_pad_start, n_pad_end)),
        mode="constant",
        constant_values=np.nan,
    )
    # Also pad e and n with NaNs to match common_time length
    e_gps = np.pad(
        e_gps,
        (n_pad_start, n_pad_end),
        mode="constant",
        constant_values=np.nan,
    )
    n_gps = np.pad(
        n_gps,
        (n_pad_start, n_pad_end),
        mode="constant",
        constant_values=np.nan,
    )
    u_gps = np.pad(
        u_gps,
        (n_pad_start, n_pad_end),
        mode="constant",
        constant_values=np.nan,
    )

    return gps_interp_lon, gps_interp_lat, e_gps, n_gps, u_gps


def get_speed(e, n, u, dt, axis=0, scheme="gradient"):
    if scheme == "gradient":
        v_e = np.gradient(e, dt, axis=axis)
        v_n = np.gradient(n, dt, axis=axis)
        v_u = np.gradient(u, dt, axis=axis)

    elif scheme == "forward":
        v_e = np.diff(e, axis=axis, append=np.nan) / dt
        v_n = np.diff(n, axis=axis, append=np.nan) / dt
        v_u = np.diff(u, axis=axis, append=np.nan) / dt

    return v_e, v_n, v_u


def gps_dataset(
    gps,
    gps_interp_lon,
    gps_interp_lat,
    e_gps,
    n_gps,
    u_gps,
    v_e_gps,
    v_n_gps,
    v_u_gps,
    common_time,
    local_frame_origin,
    apriori_pos_wgs84,
    apriori_pos_wgs84_before_campaign,
    apriori_pos_enu,
    apriori_pos_enu_before_campaign,
    time_step,
):
    # On transforme tous les datataframes en xarray pour plus de simpliciter
    ds_gps = xr.Dataset(
        data_vars=dict(
            raw_lon=(["raw_time"], gps.lon),
            raw_lat=(["raw_time"], gps.lat),
            lon=(["time"], gps_interp_lon),
            lat=(["time"], gps_interp_lat),
            e=(["time"], e_gps),
            n=(["time"], n_gps),
            u=(["time"], u_gps),
            v_e=(["time"], v_e_gps),
            v_n=(["time"], v_n_gps),
            v_u=(["time"], v_u_gps),
        ),
        coords=dict(
            time=common_time.tz_localize(None),
            raw_time=gps.datetime,
        ),
        attrs=dict(
            description="Positions GPS interpolées",
            interpolation_time_step=time_step,
            campagne="Fiberscope Groix Oct 2025",
            geodesic_frame="WGS84",
            local_frame="ENU",
            local_frame_origin=local_frame_origin["id"],
            local_frame_origin_wgs84_lon=local_frame_origin["lon"],
            local_frame_origin_wgs84_lat=local_frame_origin["lat"],
            local_frame_origin_wgs84_h=local_frame_origin["h"],
        ),
    )

    # Add position of interest coords

    # Post campaign apriori positions
    for pos_id in apriori_pos_wgs84.index:
        for coord in ["lon", "lat", "h"]:
            ds_gps.attrs[f"{pos_id}_{coord}_apriori"] = apriori_pos_wgs84.loc[
                pos_id, coord
            ]
    for pos_id in apriori_pos_enu.index:
        for coord in ["e", "n", "u"]:
            ds_gps.attrs[f"{pos_id}_{coord}_apriori"] = apriori_pos_enu.loc[
                pos_id, coord
            ]

    # Pre-campaign apriori positions
    for pos_id in apriori_pos_wgs84_before_campaign.index:
        for coord in ["lon", "lat", "h"]:
            ds_gps.attrs[f"{pos_id}_{coord}_apriori_target"] = (
                apriori_pos_wgs84_before_campaign.loc[pos_id, coord]
            )
    for pos_id in apriori_pos_enu_before_campaign.index:
        for coord in ["e", "n", "u"]:
            ds_gps.attrs[f"{pos_id}_{coord}_apriori_target"] = (
                apriori_pos_enu_before_campaign.loc[pos_id, coord]
            )

    # Add attributes to variables
    ds_gps.raw_lon.attrs["units"] = "°"
    ds_gps.raw_lat.attrs["units"] = "°"
    ds_gps.lon.attrs["units"] = "°"
    ds_gps.lat.attrs["units"] = "°"
    ds_gps.e.attrs["units"] = "m"
    ds_gps.n.attrs["units"] = "m"
    ds_gps.u.attrs["units"] = "m"
    ds_gps.v_e.attrs["units"] = r"m~s$^{-1}$"
    ds_gps.v_n.attrs["units"] = r"m~s$^{-1}$"
    ds_gps.v_u.attrs["units"] = r"m~s$^{-1}$"

    ds_gps.raw_lon.attrs["long_name"] = "Longitude"
    ds_gps.raw_lat.attrs["long_name"] = "Latitude"
    ds_gps.lon.attrs["long_name"] = "Longitude"
    ds_gps.lat.attrs["long_name"] = "Latitude"
    ds_gps.e.attrs["long_name"] = "E"
    ds_gps.n.attrs["long_name"] = "N"
    ds_gps.u.attrs["long_name"] = "U"
    ds_gps.v_e.attrs["long_name"] = "V_E"
    ds_gps.v_n.attrs["long_name"] = "V_N"
    ds_gps.v_u.attrs["long_name"] = "V_U"

    ds_gps.time.attrs["timezone"] = "UTC"
    ds_gps.raw_time.attrs["timezone"] = "UTC"

    return ds_gps


def build_AIS_dataset(
    local_frame_origin,
    N_geoid_undulation,
    t_start,
    t_end,
    ds_SBE,
    interpolation_time_step="10s",
    ais_spationav_fpath=AIS_SPATIONAV_FPATH,
    root_groix_metadata=ROOT_GROIX_METADATA,
):
    # Load AIS data
    ais = load_ais(ais_spationav_fpath=ais_spationav_fpath)

    # Interpolate GPS data to regular time intervals
    ais_interp, mmsi = interpolate_ais(
        ais, time_step=interpolation_time_step, subset_idx=None
    )

    # Sea level elevation above geoid = MSL
    zeta = (
        ds_SBE.immersion_obs1 - ds_SBE.immersion_obs1.mean().values
    )  # Approximate elevation above MSL
    zeta = zeta.sel(time=ais_interp["datetime"].values, method="nearest")

    # Derive GPS position in local ENU coordinates
    e_ais, n_ais, u_ais = transform_pos_wgs84_ecef(
        pos=ais_interp,
        local_frame_origin=local_frame_origin,
        N_geoid_undulation=N_geoid_undulation,
        zeta=zeta,
        sensor_h_above_sea_level=5,
    )

    # Format GPS data to match common time vector
    ais_lon_mat, ais_lat_mat, ais_e_mat, ais_n_mat, ais_u_mat, common_time = (
        format_ais_data(
            ais_interp=ais_interp,
            e_ais=e_ais,
            n_ais=n_ais,
            u_ais=u_ais,
            mmsi=mmsi,
            t_start=t_start,
            t_end=t_end,
            time_step=interpolation_time_step,
        )
    )

    # Derive AIS speed
    dt = (common_time[1] - common_time[0]).total_seconds()
    v_e_ais_mat, v_n_ais_mat, v_u_ais_mat = get_speed(
        dt=dt, e=ais_e_mat, n=ais_n_mat, u=ais_u_mat, axis=1
    )

    # Get apriori pos of interest
    apriori_pos_wgs84_before_campaign, apriori_pos_wgs84 = load_apriori_pos_wgs84(
        root_groix_metadata=root_groix_metadata
    )
    set_apriori_pos_h(
        apriori_pos_wgs84=apriori_pos_wgs84,
        N_geoid_undulation=N_geoid_undulation,
        ds_SBE=ds_SBE,
        verbose=False,
    )
    set_apriori_pos_h(
        apriori_pos_wgs84=apriori_pos_wgs84_before_campaign,
        N_geoid_undulation=N_geoid_undulation,
        ds_SBE=ds_SBE,
        verbose=False,
    )
    apriori_pos_enu = transform_apriori_pos_wgs84_to_ecef(
        local_frame_origin, apriori_pos_wgs84
    )
    apriori_pos_enu_before_campaign = transform_apriori_pos_wgs84_to_ecef(
        local_frame_origin, apriori_pos_wgs84_before_campaign
    )

    # Finaly build GPS dataset
    ds_ais = ais_dataset(
        ais=ais,
        ais_lon_mat=ais_lon_mat,
        ais_lat_mat=ais_lat_mat,
        ais_e_mat=ais_e_mat,
        ais_n_mat=ais_n_mat,
        ais_u_mat=ais_u_mat,
        ais_v_e_mat=v_e_ais_mat,
        ais_v_n_mat=v_n_ais_mat,
        ais_v_u_mat=v_u_ais_mat,
        common_time=common_time,
        mmsi=mmsi,
        local_frame_origin=local_frame_origin,
        apriori_pos_wgs84=apriori_pos_wgs84,
        apriori_pos_wgs84_before_campaign=apriori_pos_wgs84_before_campaign,
        apriori_pos_enu=apriori_pos_enu,
        apriori_pos_enu_before_campaign=apriori_pos_enu_before_campaign,
        time_step=interpolation_time_step,
    )

    return ds_ais


def load_ais(ais_spationav_fpath=AIS_SPATIONAV_FPATH):
    # Chargement des données AIS détramées par le SHOM
    columns = ["mmsi", "lon", "lat", "datetime"]
    ais = pd.read_csv(
        ais_spationav_fpath, sep="\t", names=columns, header=None, usecols=[1, 2, 3, 4]
    )

    # Convert datetime column to datetime objects
    ais["datetime"] = pd.to_datetime(
        ais["datetime"], format="%Y-%m-%d %H:%M:%S", utc=True
    )

    # Remove AIS lines with unique positions (at least two points to define a trajectory)
    ais = ais.groupby("mmsi").filter(lambda x: len(x) > 1)

    # # MMSI du Jules
    # mmsi_jules = 226916000
    # df_gps_jules = ais.loc[ais["mmsi"] == mmsi_jules]

    return ais


def interpolate_ais(ais, time_step="10s", subset_idx=None, mmsi_to_include=MMSI_JULES):
    mmsi = ais.mmsi.unique()
    if subset_idx is not None:
        mmsi = mmsi[subset_idx]
        # Ensure mmsi_jules is in mmsi in case we preprocess a subset of all the mmsis
        if mmsi_to_include not in mmsi:
            mmsi = np.append(mmsi, mmsi_to_include)

    ais_interp = pd.DataFrame()

    # Test progress bar
    index0 = 0
    indexf = mmsi.size - 1
    prev_progress = 0

    for i, mmsi_i in enumerate(mmsi):
        prev_progress = progression_bar(i, index0, indexf, prev_progress)

        ais_mmsi = ais.loc[ais["mmsi"] == mmsi_i]
        ais_mmsi_interp = interpolate_gps(df_gps=ais_mmsi, time_step=time_step)

        # Add mmsi column
        ais_mmsi_interp["mmsi"] = mmsi_i

        # Concatenate to ais_interp
        ais_interp = pd.concat([ais_interp, ais_mmsi_interp], ignore_index=True)

    return ais_interp, mmsi


def format_ais_data(ais_interp, e_ais, n_ais, u_ais, mmsi, t_start, t_end, time_step):
    # t_start = ais_interp["datetime"].min()
    # t_end = ais_interp["datetime"].max()
    # print(t_start, t_end)

    # Define common time vector
    common_time = pd.date_range(start=t_start, end=t_end, freq=time_step)

    # ais_mat = []
    ais_lon_mat = []
    ais_lat_mat = []
    # ais_enu_mat = []
    ais_e_mat = []
    ais_n_mat = []
    ais_u_mat = []

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
        # Pad e and n with NaNs to match common_time length
        e_ais_mmsi = e_ais[ais_mmsi.index]
        n_ais_mmsi = n_ais[ais_mmsi.index]
        u_ais_mmsi = u_ais[ais_mmsi.index]
        e_ais_mmsi = np.pad(
            e_ais_mmsi,
            (n_pad_start, n_pad_end),
            mode="constant",
            constant_values=np.nan,
        )
        n_ais_mmsi = np.pad(
            n_ais_mmsi,
            (n_pad_start, n_pad_end),
            mode="constant",
            constant_values=np.nan,
        )
        u_ais_mmsi = np.pad(
            u_ais_mmsi,
            (n_pad_start, n_pad_end),
            mode="constant",
            constant_values=np.nan,
        )

        # Add to array
        ais_lon_mat.append(ais_mmsi_arr[:, 0])
        ais_lat_mat.append(ais_mmsi_arr[:, 1])
        ais_e_mat.append(e_ais_mmsi)
        ais_n_mat.append(n_ais_mmsi)
        ais_u_mat.append(u_ais_mmsi)

        # ais_mat.append(ais_mmsi_arr)

    # ais_mat = np.array(ais_mat)
    ais_lon_mat = np.array(ais_lon_mat)
    ais_lat_mat = np.array(ais_lat_mat)
    ais_e_mat = np.array(ais_e_mat)
    ais_n_mat = np.array(ais_n_mat)
    ais_u_mat = np.array(ais_u_mat)

    return ais_lon_mat, ais_lat_mat, ais_e_mat, ais_n_mat, ais_u_mat, common_time


def ais_dataset(
    ais,
    ais_lon_mat,
    ais_lat_mat,
    ais_e_mat,
    ais_n_mat,
    ais_u_mat,
    ais_v_e_mat,
    ais_v_n_mat,
    ais_v_u_mat,
    common_time,
    mmsi,
    local_frame_origin,
    apriori_pos_wgs84,
    apriori_pos_wgs84_before_campaign,
    apriori_pos_enu,
    apriori_pos_enu_before_campaign,
    time_step,
    mmsi_jules=MMSI_JULES,
):
    raw_ais_jules = ais.loc[ais["mmsi"] == mmsi_jules]
    raw_ais_jules_arr = raw_ais_jules.to_numpy(dtype=np.float32)[
        :, 1:3
    ]  # Exclude time and mmsi column

    ds_ais = xr.Dataset(
        data_vars=dict(
            raw_lon_jules=(["raw_time"], raw_ais_jules_arr[:, 0]),
            raw_lat_jules=(["raw_time"], raw_ais_jules_arr[:, 1]),
            lon=(["mmsi", "time"], ais_lon_mat),
            lat=(["mmsi", "time"], ais_lat_mat),
            e=(["mmsi", "time"], ais_e_mat),
            n=(["mmsi", "time"], ais_n_mat),
            u=(["mmsi", "time"], ais_u_mat),
            v_e=(["mmsi", "time"], ais_v_e_mat),
            v_n=(["mmsi", "time"], ais_v_n_mat),
            v_u=(["mmsi", "time"], ais_v_u_mat),
        ),
        coords=dict(
            raw_time=raw_ais_jules.datetime,  # Remove timezone info for xarray compatibility
            time=common_time.tz_localize(
                None
            ),  # Remove timezone info for xarray compatibility
            # mmsi=ais_interp.mmsi.unique(),
            mmsi=mmsi,
        ),
        attrs=dict(
            description="Positions AIS interpolées",
            interpolation_time_step=time_step,
            campagne="Fiberscope Groix Oct 2025",
            geodesic_frame="WGS84",
            local_frame="ENU",
            local_frame_origin=local_frame_origin["id"],
            local_frame_origin_wgs84_lon=local_frame_origin["lon"],
            local_frame_origin_wgs84_lat=local_frame_origin["lat"],
            local_frame_origin_wgs84_h=local_frame_origin["h"],
            mmsi_jules=mmsi_jules,
        ),
    )

    # Add position of interest coords
    # Post compaign apriori positions
    for pos_id in apriori_pos_wgs84.index:
        for coord in ["lon", "lat", "h"]:
            ds_ais.attrs[f"{pos_id}_{coord}_apriori"] = apriori_pos_wgs84.loc[
                pos_id, coord
            ]
    for pos_id in apriori_pos_enu.index:
        for coord in ["e", "n", "u"]:
            ds_ais.attrs[f"{pos_id}_{coord}_apriori"] = apriori_pos_enu.loc[
                pos_id, coord
            ]

    # Pre-campaign apriori positions
    for pos_id in apriori_pos_wgs84_before_campaign.index:
        for coord in ["lon", "lat", "h"]:
            ds_ais.attrs[f"{pos_id}_{coord}_apriori_target"] = (
                apriori_pos_wgs84_before_campaign.loc[pos_id, coord]
            )
    for pos_id in apriori_pos_enu_before_campaign.index:
        for coord in ["e", "n", "u"]:
            ds_ais.attrs[f"{pos_id}_{coord}_apriori_target"] = (
                apriori_pos_enu_before_campaign.loc[pos_id, coord]
            )

    # Add attributes to variables
    ds_ais.raw_lon_jules.attrs["units"] = "°"
    ds_ais.raw_lat_jules.attrs["units"] = "°"
    ds_ais.lon.attrs["units"] = "°"
    ds_ais.lat.attrs["units"] = "°"
    ds_ais.e.attrs["units"] = "m"
    ds_ais.n.attrs["units"] = "m"
    ds_ais.u.attrs["units"] = "m"
    ds_ais.v_e.attrs["units"] = r"m~s$^{-1}$"
    ds_ais.v_n.attrs["units"] = r"m~s$^{-1}$"
    ds_ais.v_u.attrs["units"] = r"m~s$^{-1}$"

    ds_ais.mmsi.attrs["units"] = ""
    ds_ais.time.attrs["timezone"] = "UTC"
    ds_ais.raw_time.attrs["timezone"] = "UTC"

    ds_ais.raw_lon_jules.attrs["long_name"] = "Longitude"
    ds_ais.raw_lat_jules.attrs["long_name"] = "Latitude"
    ds_ais.lon.attrs["long_name"] = "Longitude"
    ds_ais.lat.attrs["long_name"] = "Latitude"
    ds_ais.e.attrs["long_name"] = "E"
    ds_ais.n.attrs["long_name"] = "N"
    ds_ais.u.attrs["long_name"] = "U"
    ds_ais.v_e.attrs["long_name"] = "V_E"
    ds_ais.v_n.attrs["long_name"] = "V_N"
    ds_ais.v_u.attrs["long_name"] = "V_U"

    return ds_ais


# ======================================================================================================================
# Bathy DATA
# ======================================================================================================================


def build_BATHY_dataset(local_frame_origin, root_bathy_data=ROOT_BATHY_DATA):

    # Load bathy data
    bathy = load_bathy(
        local_frame_origin=local_frame_origin, root_bathy_data=root_bathy_data
    )

    # Convert to ecef
    e_bathy, n_bathy, _ = transform_pos_wgs84_ecef(
        pos=bathy,
        local_frame_origin=local_frame_origin,
        N_geoid_undulation=0,
        zeta=0,
        sensor_h_above_sea_level=0,
    )

    # Finally build dataset
    ds_bathy = bathy_dataset(
        bathy=bathy,
        e_bathy=e_bathy,
        n_bathy=n_bathy,
        local_frame_origin=local_frame_origin,
    )

    return ds_bathy


def bathy_dataset(
    bathy,
    e_bathy,
    n_bathy,
    local_frame_origin,
):
    ds_bathy = xr.Dataset(
        data_vars=dict(
            elevation=(["lat", "lon"], bathy.elevation.values),
            elevation_enu=(["n", "e"], bathy.elevation.values),
        ),
        coords=dict(
            lon=bathy.lon.values,
            lat=bathy.lat.values,
            e=e_bathy,
            n=n_bathy,
        ),
        attrs=dict(
            description="Bathymetry data from GEBCO 2021",
            geodesic_frame="WGS84",
            local_frame="ENU",
            local_frame_origin=local_frame_origin["id"],
            local_frame_origin_wgs84_lon=local_frame_origin["lon"],
            local_frame_origin_wgs84_lat=local_frame_origin["lat"],
            local_frame_origin_wgs84_h=local_frame_origin["h"],
        ),
    )

    # Add attributes to variables
    ds_bathy.elevation.attrs["units"] = "m"
    ds_bathy.elevation_enu.attrs["units"] = "m"
    ds_bathy.lon.attrs["units"] = "°"
    ds_bathy.lat.attrs["units"] = "°"
    ds_bathy.e.attrs["units"] = "m"
    ds_bathy.n.attrs["units"] = "m"

    ds_bathy.elevation.attrs["long_name"] = "Elevation (WGS84)"
    ds_bathy.elevation_enu.attrs["long_name"] = "Elevation (ENU)"
    ds_bathy.lon.attrs["long_name"] = "Longitude"
    ds_bathy.lat.attrs["long_name"] = "Latitude"
    ds_bathy.e.attrs["long_name"] = "E"
    ds_bathy.n.attrs["long_name"] = "N"

    return ds_bathy


def load_bathy(
    local_frame_origin, root_bathy_data=ROOT_BATHY_DATA, dlat_box=0.25, dlon_box=0.25
):
    # input_data_root = os.path.join(project_root, "data")
    bathy_fpath = os.path.join(root_bathy_data, "GEBCO_2021_sub_ice_topo.nc")

    # Load bathy data
    ds_bathy = xr.open_dataset(bathy_fpath)

    # Slice data to get the area of interest
    lat0 = local_frame_origin["lat"]
    lon0 = local_frame_origin["lon"]
    ds_bathy = ds_bathy.sel(
        lat=slice(
            lat0 - dlat_box,
            lat0 + dlat_box,
        ),
        lon=slice(
            lon0 - dlon_box,
            lon0 + dlon_box,
        ),
    )

    return ds_bathy


def plot_bathy(ds_bathy, contour_levels=[0]):

    ds_bathy.elevation.plot()
    # Add contours
    plt.contour(
        ds_bathy.lon,
        ds_bathy.lat,
        ds_bathy.elevation,
        levels=contour_levels,
        colors="black",
    )


if __name__ == "__main__":
    pass
