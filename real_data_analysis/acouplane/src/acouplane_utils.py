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

# ======================================================================================================================
# Data preprocessing
# ======================================================================================================================


# def ais_dataset(
#     ais,
#     ais_lon_mat,
#     ais_lat_mat,
#     ais_e_mat,
#     ais_n_mat,
#     ais_u_mat,
#     ais_v_e_mat,
#     ais_v_n_mat,
#     ais_v_u_mat,
#     common_time,
#     mmsi,
#     local_frame_origin,
#     apriori_pos_wgs84,
#     apriori_pos_wgs84_before_campaign,
#     apriori_pos_enu,
#     apriori_pos_enu_before_campaign,
#     time_step,
#     mmsi_jules=MMSI_JULES,
# ):
#     raw_ais_jules = ais.loc[ais["mmsi"] == mmsi_jules]
#     raw_ais_jules_arr = raw_ais_jules.to_numpy(dtype=np.float32)[
#         :, 1:3
#     ]  # Exclude time and mmsi column

#     ds_ais = xr.Dataset(
#         data_vars=dict(
#             raw_lon_jules=(["raw_time"], raw_ais_jules_arr[:, 0]),
#             raw_lat_jules=(["raw_time"], raw_ais_jules_arr[:, 1]),
#             lon=(["mmsi", "time"], ais_lon_mat),
#             lat=(["mmsi", "time"], ais_lat_mat),
#             e=(["mmsi", "time"], ais_e_mat),
#             n=(["mmsi", "time"], ais_n_mat),
#             u=(["mmsi", "time"], ais_u_mat),
#             v_e=(["mmsi", "time"], ais_v_e_mat),
#             v_n=(["mmsi", "time"], ais_v_n_mat),
#             v_u=(["mmsi", "time"], ais_v_u_mat),
#         ),
#         coords=dict(
#             raw_time=raw_ais_jules.datetime,  # Remove timezone info for xarray compatibility
#             time=common_time.tz_localize(
#                 None
#             ),  # Remove timezone info for xarray compatibility
#             # mmsi=ais_interp.mmsi.unique(),
#             mmsi=mmsi,
#         ),
#         attrs=dict(
#             description="Positions AIS interpolées",
#             interpolation_time_step=time_step,
#             campagne="Fiberscope Groix Oct 2025",
#             geodesic_frame="WGS84",
#             local_frame="ENU",
#             local_frame_origin=local_frame_origin["id"],
#             local_frame_origin_wgs84_lon=local_frame_origin["lon"],
#             local_frame_origin_wgs84_lat=local_frame_origin["lat"],
#             local_frame_origin_wgs84_h=local_frame_origin["h"],
#             mmsi_jules=mmsi_jules,
#         ),
#     )

#     # Add position of interest coords
#     # Post compaign apriori positions
#     for pos_id in apriori_pos_wgs84.index:
#         for coord in ["lon", "lat", "h"]:
#             ds_ais.attrs[f"{pos_id}_{coord}_apriori"] = apriori_pos_wgs84.loc[
#                 pos_id, coord
#             ]
#     for pos_id in apriori_pos_enu.index:
#         for coord in ["e", "n", "u"]:
#             ds_ais.attrs[f"{pos_id}_{coord}_apriori"] = apriori_pos_enu.loc[
#                 pos_id, coord
#             ]

#     # Pre-campaign apriori positions
#     for pos_id in apriori_pos_wgs84_before_campaign.index:
#         for coord in ["lon", "lat", "h"]:
#             ds_ais.attrs[f"{pos_id}_{coord}_apriori_target"] = (
#                 apriori_pos_wgs84_before_campaign.loc[pos_id, coord]
#             )
#     for pos_id in apriori_pos_enu_before_campaign.index:
#         for coord in ["e", "n", "u"]:
#             ds_ais.attrs[f"{pos_id}_{coord}_apriori_target"] = (
#                 apriori_pos_enu_before_campaign.loc[pos_id, coord]
#             )

#     # Add attributes to variables
#     ds_ais.raw_lon_jules.attrs["units"] = "°"
#     ds_ais.raw_lat_jules.attrs["units"] = "°"
#     ds_ais.lon.attrs["units"] = "°"
#     ds_ais.lat.attrs["units"] = "°"
#     ds_ais.e.attrs["units"] = "m"
#     ds_ais.n.attrs["units"] = "m"
#     ds_ais.u.attrs["units"] = "m"
#     ds_ais.v_e.attrs["units"] = r"m~s$^{-1}$"
#     ds_ais.v_n.attrs["units"] = r"m~s$^{-1}$"
#     ds_ais.v_u.attrs["units"] = r"m~s$^{-1}$"

#     ds_ais.mmsi.attrs["units"] = ""
#     ds_ais.time.attrs["timezone"] = "UTC"
#     ds_ais.raw_time.attrs["timezone"] = "UTC"

#     ds_ais.raw_lon_jules.attrs["long_name"] = "Longitude"
#     ds_ais.raw_lat_jules.attrs["long_name"] = "Latitude"
#     ds_ais.lon.attrs["long_name"] = "Longitude"
#     ds_ais.lat.attrs["long_name"] = "Latitude"
#     ds_ais.e.attrs["long_name"] = "E"
#     ds_ais.n.attrs["long_name"] = "N"
#     ds_ais.u.attrs["long_name"] = "U"
#     ds_ais.v_e.attrs["long_name"] = "V_E"
#     ds_ais.v_n.attrs["long_name"] = "V_N"
#     ds_ais.v_u.attrs["long_name"] = "V_U"

#     return ds_ais


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
