#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   preprocess_utils.py
@Time    :   2026/03/27 11:22:00
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

# from real_data_analysis.fiberscope_groix.src.data_prepocessing.oceano_utils import (
#     SBE39_reader,
#     SBE37_reader,
#     rbr_reader,
# )
from misc import progression_bar
from source.utils.utils_gps import gpx_to_csv, interpolate_gps
from source.utils.utils_geo import geodetic_to_ecef, ecef_to_enu


# ======================================================================================================================
# DEFAULTS
# ======================================================================================================================
PROJECT_ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"

ROOT_RHUMRUM_DATA = os.path.join(PROJECT_ROOT, "data", "rhumrum")
ROOT_RHUMRUM_AIS = os.path.join(ROOT_RHUMRUM_DATA, "ais")
ROOT_RHUMRUM_METADATA = os.path.join(ROOT_RHUMRUM_DATA, "metadata")
ROOT_BATHY_DATA = os.path.join(PROJECT_ROOT, "data", "bathy")
# bathy_fpath = r"/home/program/ubf_tools/data/bathy/mmdpm/PVA_RR48/GEBCO_2021_lon_64.44_67.44_lat_-29.08_-26.08.nc"

AIS_PARSED_DATA_FPATH = os.path.join(
    ROOT_RHUMRUM_AIS, "extract-ais-pos-for-zone-ecole-navale-by-month-201305.csv"
)
# ais_spationav_fpath = os.path.join(root_groix_ais, "SPATIONAV_AIS_001.csv")


# ======================================================================================================================
# GPS / AIS DATA
# ======================================================================================================================


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


def load_obs_pos_wgs84(root_metadata: str = ROOT_RHUMRUM_METADATA):
    obs_pos_wgs84_fname = "rhum_rum_obs_pos.csv"

    obs_pos_wgs84 = pd.read_csv(
        os.path.join(root_metadata, obs_pos_wgs84_fname),
        index_col=0,
        delimiter=",",
    )

    swir_rcv_ids = ["RR41", "RR43", "RR44", "RR47"]
    obs_pos_wgs84 = obs_pos_wgs84.loc[obs_pos_wgs84.index.isin(swir_rcv_ids)]

    obs_pos_wgs84 = obs_pos_wgs84.rename(columns={"elev": "h"})

    return obs_pos_wgs84


# def transform_pos_wgs84_to_enu(
#     local_frame_origin,
#     pos_wgs84,
#     N_geoid_undulation,
#     zeta=0,
#     sensor_h_above_sea_level=0,
# ):
#     # WGS84 -> ECEF
#     lat = np.deg2rad(pos_wgs84.lat)
#     lon = np.deg2rad(pos_wgs84.lon)
#     X, Y, Z = geodetic_to_ecef(lat, lon, pos_wgs84.h)
#     # ECEF -> ENU
#     lat0, lon0, h0 = (
#         np.deg2rad(local_frame_origin["lat"]),
#         np.deg2rad(local_frame_origin["lon"]),
#         local_frame_origin["h"],
#     )
#     e, n, u = ecef_to_enu(X, Y, Z, lat0, lon0, h0)

#     pos_enu = pd.DataFrame(
#         np.stack((e, n, u)).T, index=pos_wgs84.index, columns=["e", "n", "u"]
#     )

#     return pos_enu


def transform_pos_wgs84_enu(
    pos,
    local_frame_origin,
):
    lon = np.deg2rad(pos.lon)
    lat = np.deg2rad(pos.lat)
    h = pos.h

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


def build_AIS_dataset(
    local_frame_origin,
    # t_start,
    # t_end,
    interpolation_time_step="10s",
    ais_fpath=AIS_PARSED_DATA_FPATH,
    root_metadata=ROOT_RHUMRUM_METADATA,
):
    # Load AIS data
    ais = load_ais(ais_fpath=ais_fpath)

    # ais = ais.iloc[-5000:]

    # Interpolate GPS data to regular time intervals
    ais_interp, mmsi = interpolate_ais(
        ais, time_step=interpolation_time_step, subset_idx=None
    )

    # # Sea level elevation above geoid = MSL
    # zeta = (
    #     ds_SBE.immersion_obs1 - ds_SBE.immersion_obs1.mean().values
    # )  # Approximate elevation above MSL
    # zeta = zeta.sel(time=ais_interp["datetime"].values, method="nearest")

    # Default h
    ais_interp["h"] = np.zeros_like(ais_interp.lon.values)

    # Derive AIS position in local ENU coordinates
    e_ais, n_ais, u_ais = transform_pos_wgs84_enu(
        pos=ais_interp,
        local_frame_origin=local_frame_origin,
    )

    # Format AIS data to match common time vector
    ais_lon_mat, ais_lat_mat, ais_e_mat, ais_n_mat, ais_u_mat, common_time = (
        format_ais_data(
            ais_interp=ais_interp,
            e_ais=e_ais,
            n_ais=n_ais,
            u_ais=u_ais,
            mmsi=mmsi,
            # t_start=t_start,
            # t_end=t_end,
            time_step=interpolation_time_step,
        )
    )

    # Derive AIS speed
    dt = (common_time[1] - common_time[0]).total_seconds()
    v_e_ais_mat, v_n_ais_mat, v_u_ais_mat = get_speed(
        dt=dt, e=ais_e_mat, n=ais_n_mat, u=ais_u_mat, axis=1
    )

    # # Get apriori pos of interest
    obs_pos_wgs84 = load_obs_pos_wgs84(root_metadata=root_metadata)
    obs_pos_enu = transform_pos_wgs84_enu(
        pos=obs_pos_wgs84, local_frame_origin=local_frame_origin
    )

    # Convert to Dataframe
    obs_pos_enu = pd.DataFrame(
        np.stack((obs_pos_enu[0], obs_pos_enu[1], obs_pos_enu[2])).T,
        index=obs_pos_wgs84.index,
        columns=["e", "n", "u"],
    )

    # set_apriori_pos_h(
    #     apriori_pos_wgs84=apriori_pos_wgs84,
    #     N_geoid_undulation=N_geoid_undulation,
    #     ds_SBE=ds_SBE,
    #     verbose=False,
    # )
    # set_apriori_pos_h(
    #     apriori_pos_wgs84=apriori_pos_wgs84_before_campaign,
    #     N_geoid_undulation=N_geoid_undulation,
    #     ds_SBE=ds_SBE,
    #     verbose=False,
    # )
    # apriori_pos_enu = transform_apriori_pos_wgs84_to_ecef(
    #     local_frame_origin, apriori_pos_wgs84
    # )
    # apriori_pos_enu_before_campaign = transform_apriori_pos_wgs84_to_ecef(
    #     local_frame_origin, apriori_pos_wgs84_before_campaign
    # )

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
        obs_pos_wgs84=obs_pos_wgs84,
        obs_pos_enu=obs_pos_enu,
        time_step=interpolation_time_step,
    )

    return ds_ais


def load_ais(ais_fpath=AIS_PARSED_DATA_FPATH):
    # Chargement des données AIS
    # columns = ["mmsi", "lon", "lat", "datetime"]
    ais = pd.read_csv(
        ais_fpath,
        sep=";",
        header=0,
        usecols=[0, 2, 3, 4, 5, 16],
    )

    datetime_str = ais["locDate"] + " " + ais["locTime"]
    ais["datetime_str"] = datetime_str
    # Convert datetime column to datetime objects
    # ais["datetime"] = pd.to_datetime(datetime_str, format="%d/%m/%Y %H:%M:%S", utc=True)

    props_dtype = {
        "mmsi": int,
        "datetime_str": str,
        "lon": np.float32,
        "lat": np.float32,
        "shipName": np.float32,
    }

    # Convert each property to the correct dtype
    usefull_data = {}
    for prop in props_dtype.keys():
        usefull_data[prop] = pd.Series(ais[prop], dtype=props_dtype.get(prop, object))

    df_ais = pd.DataFrame(usefull_data)

    df_ais["datetime_str"] = pd.to_datetime(
        df_ais["datetime_str"], format="%d/%m/%Y %H:%M:%S", utc=True
    )
    df_ais.rename(columns={"datetime_str": "datetime"}, inplace=True)

    # Remove AIS lines with unique positions (at least two points to define a trajectory)
    df_ais = df_ais.groupby("mmsi").filter(lambda x: len(x) > 1)

    return df_ais


def interpolate_ais(ais, time_step="10s", subset_idx=None, mmsi_to_include=None):
    mmsi = ais.mmsi.unique()
    if subset_idx is not None:
        mmsi = mmsi[subset_idx]
        # Ensure mmsi_jules is in mmsi in case we preprocess a subset of all the mmsis
        if mmsi_to_include not in mmsi:
            mmsi = np.append(mmsi, mmsi_to_include)

    ais_interp = pd.DataFrame()

    # Set progress bar
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


def format_ais_data(ais_interp, e_ais, n_ais, u_ais, mmsi, time_step):
    t_start = ais_interp["datetime"].min()
    t_end = ais_interp["datetime"].max()
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
    obs_pos_wgs84,
    obs_pos_enu,
    time_step,
):
    # raw_ais_jules = ais.loc[ais["mmsi"] == mmsi_jules]
    # raw_ais_jules_arr = raw_ais_jules.to_numpy(dtype=np.float32)[
    #     :, 1:3
    # ]  # Exclude time and mmsi column

    ds_ais = xr.Dataset(
        data_vars=dict(
            # raw_lon_jules=(["raw_time"], raw_ais_jules_arr[:, 0]),
            # raw_lat_jules=(["raw_time"], raw_ais_jules_arr[:, 1]),
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
            # raw_time=raw_ais_jules.datetime,  # Remove timezone info for xarray compatibility
            time=common_time.tz_localize(
                None
            ),  # Remove timezone info for xarray compatibility
            # mmsi=ais_interp.mmsi.unique(),
            mmsi=mmsi,
        ),
        attrs=dict(
            description="Positions AIS interpolées",
            interpolation_time_step=time_step,
            campagne="RHUM SWIR",
            geodesic_frame="WGS84",
            local_frame="ENU",
            local_frame_origin=local_frame_origin["id"],
            local_frame_origin_wgs84_lon=local_frame_origin["lon"],
            local_frame_origin_wgs84_lat=local_frame_origin["lat"],
            local_frame_origin_wgs84_h=local_frame_origin["h"],
            # mmsi_jules=mmsi_jules,
        ),
    )

    # Add position of interest coords
    # Post compaign apriori positions
    for pos_id in obs_pos_wgs84.index:
        for coord in ["lon", "lat", "h"]:
            ds_ais.attrs[f"{pos_id}_{coord}_apriori"] = obs_pos_wgs84.loc[pos_id, coord]
    for pos_id in obs_pos_enu.index:
        for coord in ["e", "n", "u"]:
            ds_ais.attrs[f"{pos_id}_{coord}_apriori"] = obs_pos_enu.loc[pos_id, coord]

    # Add attributes to variables
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


def build_BATHY_dataset(
    local_frame_origin, root_bathy_data=ROOT_BATHY_DATA, dlat_box=0.25, dlon_box=0.25
):

    # Load bathy data
    bathy = load_bathy(
        local_frame_origin=local_frame_origin,
        root_bathy_data=root_bathy_data,
        dlat_box=dlat_box,
        dlon_box=dlon_box,
    )
    # Default h
    bathy["h"] = np.zeros_like(bathy.lon.values)

    # Convert to ecef
    e_bathy, n_bathy, _ = transform_pos_wgs84_enu(
        local_frame_origin=local_frame_origin,
        pos=bathy,
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
    load_obs_pos_wgs84(ROOT_RHUMRUM_METADATA)

    # ds_bathy = build_BATHY_dataset(local_frame_origin, root_bathy_data=root_bathy_data)
    # plot_bathy(ds_bathy, contour_levels=[0])
