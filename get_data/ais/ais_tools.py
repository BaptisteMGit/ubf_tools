#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   extract_ais_area.py
@Time    :   2024/09/05 16:14:20
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Script to extract ais data from csv file for a specific area (SWIR) 
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pyproj import Geod, Proj, transform
from publication.PublicationFigure import PubFigure
from illustration.RHUMRUM.SWIR.bathy_obs import plot_swir_bathy, plot_swir_obs


wgs84 = Proj(proj="latlong", datum="WGS84")  # Système géodésique WGS84
ecef = Proj(proj="geocent", datum="WGS84")  # Système ECEF


def extract_ais_area(file_path, lon_min, lon_max, lat_min, lat_max):
    """
    Extract ais data from a csv file for a specific area (SWIR)
    Args:
        file_path (str): path to the csv file
        lon_min (float): minimum longitude of the area
        lon_max (float): maximum longitude of the area
        lat_min (float): minimum latitude of the area
        lat_max (float): maximum latitude of the area
    Returns:
        df (pd.DataFrame): ais data for the specific area
    """

    # Load the data
    df = pd.read_csv(
        file_path,
        sep=";",
        dtype={
            "locDate": "string",
            "locTime": "string",
        },
    )
    # Filter the data
    df = df[
        (df["lon"] >= lon_min)
        & (df["lon"] <= lon_max)
        & (df["lat"] >= lat_min)
        & (df["lat"] <= lat_max)
    ]

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df


def interpolate_trajectories(ais_df, time_step="5min"):
    """
    Interpolate each ship (mmsi) trajectory with a regular time step
    Args:
        ais_df (pd.DataFrame): ais data

    Returns:
        ais_df (pd.DataFrame): ais data with interpolated trajectories
    """

    # Create time column with datetime objects
    time = ais_df["locDate"] + " " + ais_df["locTime"]
    ais_df["time"] = pd.to_datetime(time, format="%d/%m/%Y %H:%M:%S")

    # Define the geodetic object
    geod = Geod(ellps="WGS84")

    df_interp = pd.DataFrame()

    # Iterate over each ship
    for mmsi in ais_df["mmsi"].unique():
        df_tmp = ais_df[ais_df["mmsi"] == mmsi]
        # Ensure the data is sorted by time
        df_tmp = df_tmp.sort_values("time")

        # Create a empty pandas series to store the interpolated data
        high_res_time = pd.DataFrame()
        high_res_pos = pd.DataFrame()

        # Iterate over successive positions
        for i in range(1, len(df_tmp)):
            # Compute a time vector with a regular time step (5 minutes)
            t1 = df_tmp["time"].iloc[i - 1]
            t2 = df_tmp["time"].iloc[i]
            time_interp = pd.date_range(t1, t2, freq=time_step)

            # Convert time_interp to a pandas df
            time_interp = pd.DataFrame({"time": time_interp})

            # Add time_interp to high_res_time
            if high_res_time.empty:
                high_res_time = time_interp
            else:
                high_res_time = pd.concat([high_res_time, time_interp])

            # Compute distance
            lon1 = df_tmp["lon"].iloc[i - 1]
            lat1 = df_tmp["lat"].iloc[i - 1]
            lon2 = df_tmp["lon"].iloc[i]
            lat2 = df_tmp["lat"].iloc[i]

            profile_coords = geod.inv_intermediate(
                lon1, lat1, lon2, lat2, npts=time_interp.size
            )

            # Add profile_coords to high_res_pos
            lon = profile_coords.lons
            lat = profile_coords.lats
            data = {"lon": lon, "lat": lat}
            new_data = pd.DataFrame(data)
            if high_res_pos.empty:
                high_res_pos = new_data
            else:
                high_res_pos = pd.concat([high_res_pos, new_data])

        # Concat high_res_time and high_res_pos
        high_res_time.reset_index(drop=True, inplace=True)
        high_res_pos.reset_index(drop=True, inplace=True)
        df_interp_mmsi = pd.concat([high_res_time, high_res_pos], axis=1)

        mmsi_id = pd.Series([mmsi] * len(df_interp_mmsi))
        shipName = pd.Series([df_tmp["shipName"].values[0]] * len(df_interp_mmsi))
        df_interp_mmsi["mmsi"] = mmsi_id
        df_interp_mmsi["shipName"] = shipName

        if df_interp.empty:
            df_interp = df_interp_mmsi
        else:
            df_interp = pd.concat([df_interp, df_interp_mmsi])

    # Reset index
    df_interp.reset_index(drop=True, inplace=True)

    return df_interp


def intersection_time_2ships(df_2ships):
    """
    Derive intersection time between two ships

    Args:
        df_2ships (pd.DataFrame): ais data with only two ships

    Returns:
        intersection_time (pd.Series): intersection time
    """

    df_ship1 = df_2ships[df_2ships["mmsi"] == df_2ships["mmsi"].unique()[0]]
    df_ship2 = df_2ships[df_2ships["mmsi"] == df_2ships["mmsi"].unique()[1]]

    lon0, lat0 = df_2ships["lon"].median(), df_2ships["lat"].median()
    proj_enu = Proj(proj="aeqd", datum="WGS84", lat_0=lat0, lon_0=lon0)  # ENU

    intersection_wgs84 = []
    intersection_time_ship1 = []
    intersection_time_ship2 = []

    # Iterate over successive trajectory segments
    for i in range(1, len(df_ship1)):
        p1 = (df_ship1["x_enu"].iloc[i - 1], df_ship1["y_enu"].iloc[i - 1])
        p2 = (df_ship1["x_enu"].iloc[i], df_ship1["y_enu"].iloc[i])

        for j in range(1, len(df_ship2)):
            q1 = (df_ship2["x_enu"].iloc[j - 1], df_ship2["y_enu"].iloc[j - 1])
            q2 = (df_ship2["x_enu"].iloc[j], df_ship2["y_enu"].iloc[j])

            intersection = segment_intersection(p1, p2, q1, q2)

            if intersection is not None:
                # Compute the intersection time for ship 1
                t11 = df_ship1["time"].iloc[i - 1]
                t12 = df_ship1["time"].iloc[i]

                V1 = (
                    np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                    / (t12 - t11).seconds
                )
                t_inter_1 = t11 + pd.Timedelta(
                    np.sqrt(
                        (intersection[0] - p1[0]) ** 2 + (intersection[1] - p1[1]) ** 2
                    )
                    / V1,
                    unit="s",
                )
                intersection_time_ship1.append(t_inter_1)

                # Compute the intersection time for ship 2
                t21 = df_ship2["time"].iloc[j - 1]
                t22 = df_ship2["time"].iloc[j]

                V2 = (
                    np.sqrt((q2[0] - q1[0]) ** 2 + (q2[1] - q1[1]) ** 2)
                    / (t22 - t21).seconds
                )
                t_inter_2 = t21 + pd.Timedelta(
                    np.sqrt(
                        (intersection[0] - q1[0]) ** 2 + (intersection[1] - q1[1]) ** 2
                    )
                    / V2,
                    unit="s",
                )
                intersection_time_ship2.append(t_inter_2)

                # Save intersections points in WGS84
                x_inter, y_inter = intersection
                lon_inter, lat_inter = transform(proj_enu, wgs84, x_inter, y_inter)
                intersection_wgs84.append((lon_inter, lat_inter))

    return intersection_wgs84, intersection_time_ship1, intersection_time_ship2


def project_to_enu(df):
    """
    Project trajectories to ENU coordinates

    Args:
        df (pd.DataFrame): ais data

    Returns:
        df_2ships_enu (pd.DataFrame): ais data ENU coordinates
    """

    # Initialisation du système ENU avec un point d'origine (lon0, lat0)
    lon0, lat0 = df["lon"].median(), df["lat"].median()  # Barycentre
    proj_enu = Proj(proj="aeqd", datum="WGS84", lat_0=lat0, lon_0=lon0)  # ENU

    x_enu, y_enu = transform(wgs84, proj_enu, df["lon"], df["lat"])

    df["x_enu"] = x_enu
    df["y_enu"] = y_enu

    return


def cross_product_2d(v1, v2):
    """Calcul du produit vectoriel de deux vecteurs 2D."""
    return v1[0] * v2[1] - v1[1] * v2[0]


def segment_orientation(p, q, r):
    """Calcule l'orientation du triplet (p, q, r)."""
    return cross_product_2d((q[0] - p[0], q[1] - p[1]), (r[0] - p[0], r[1] - p[1]))


def is_point_on_segment(p, q, r):
    """Vérifie si le point r est sur le segment défini par (p, q)."""
    return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[
        1
    ] <= max(p[1], q[1])


def compute_intersection(p1, p2, q1, q2):
    """Calcule le point d'intersection entre les droites p1p2 et q1q2, si possible."""
    # Paramètres de l'équation des droites sous forme ax + by = c
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = a1 * p1[0] + b1 * p1[1]

    a2 = q2[1] - q1[1]
    b2 = q1[0] - q2[0]
    c2 = a2 * q1[0] + b2 * q1[1]

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        return None  # Les droites sont parallèles ou colinéaires

    # Calcul des coordonnées du point d'intersection
    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant

    return (x, y)


def segment_intersection(p1, p2, q1, q2):
    """
    Derive the intersection point between two segments if it exists

    Args:
        p1 (tuple): first point of the first segment
        p2 (tuple): second point of the first segment
        q1 (tuple): first point of the second segment
        q2 (tuple): second point of the second segment

    Returns:
        intersection_point (tuple): intersection point

    Examples:
        # Exemple de points ENU des navires (E, N)
        navire1_segment = [(1, 1), (4, 4)]  # Segment du navire 1
        navire2_segment = [(1, 8), (2, 2)]  # Segment du navire 2

        # Vérification d'intersection
        croisement = segment_intersection(
            navire1_segment[0], navire1_segment[1], navire2_segment[0], navire2_segment[1]
        )
        plt.plot(
            [navire1_segment[0][0], navire1_segment[1][0]],
            [navire1_segment[0][1], navire1_segment[1][1]],
            "o-",
        )
        plt.plot(
            [navire2_segment[0][0], navire2_segment[1][0]],
            [navire2_segment[0][1], navire2_segment[1][1]],
            "o-",
        )
        if croisement is not None:
            plt.plot(croisement[0], croisement[1], "o")
        plt.show()

    """

    intersection_point = compute_intersection(p1, p2, q1, q2)

    if intersection_point is not None:
        if is_point_on_segment(p1, p2, intersection_point) and is_point_on_segment(
            q1, q2, intersection_point
        ):
            return intersection_point
        else:
            return None
    else:
        return None


def plot_traj_over_bathy(
    df_ais, rcv_info, lon_min, lon_max, lat_min, lat_max, intersection_data=None
):
    ds_bathy = plot_swir_bathy(contour=False)
    plot_swir_obs(ds_bathy, rcv_info["id"], col=None)
    PubFigure(legend_fontsize=7)

    if intersection_data is not None:
        intersection_wgs84 = intersection_data["intersection_wgs84"]
        intersection_time = intersection_data["intersection_time"]

    for mmsi in df_ais["mmsi"].unique():
        df_tmp = df_ais[df_ais["mmsi"] == mmsi]

        if intersection_data is not None:
            plt.plot(
                df_tmp["lon"],
                df_tmp["lat"],
                "-",
                label=df_tmp["shipName"].values[0],
                linewidth=3,
            )
            ncol_lgd = 3
            lgd_fsize = 10
        else:
            plt.plot(
                df_tmp["lon"],
                df_tmp["lat"],
                color="k",
                linestyle="--",
                linewidth=3,
            )

            # Add a marker at a random position along the trajectory to identify the ship with its name
            idx = np.random.randint(0, len(df_tmp))
            plt.scatter(
                df_tmp["lon"].iloc[idx],
                df_tmp["lat"].iloc[idx],
                s=100,
                marker="^",
                # label=df_tmp["shipName"].values[0],
                label=df_tmp["mmsi"].values[0],
                zorder=2,
            )
            ncol_lgd = len(df_ais["mmsi"].unique()) // 4
            lgd_fsize = 8

    if intersection_data is not None:
        # Add intersection point
        if len(intersection_wgs84) > 0:
            lon_inter, lat_inter = zip(*intersection_wgs84)
            plt.scatter(
                lon_inter,
                lat_inter,
                s=150,
                marker="x",
                color="r",
                zorder=2,
                label=f"Intersection at {r'$t_{ship1}$'} = {intersection_time[mmsi].strftime('%Y-%m-%d %H:%M')}",
            )

    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncols=ncol_lgd,
        mode="expand",
        borderaxespad=0.0,
        fontsize=lgd_fsize,
    )

    plt.ylim(lat_min, lat_max)
    plt.xlim(lon_min, lon_max)


def compute_distance_ship_rcv(ais_data, rcv_info):
    geod = Geod(ellps="WGS84")
    distance = {}
    for mmsi in ais_data["mmsi"].unique():
        distance[mmsi] = {}
        df_tmp = ais_data[ais_data["mmsi"] == mmsi]
        for i, rcv_id in enumerate(rcv_info["id"]):
            distance[mmsi][rcv_id] = {}
            lon = df_tmp["lon"]
            lat = df_tmp["lat"]
            npos = len(lon)
            # Derive distance
            _, _, ranges = geod.inv(
                lons1=[rcv_info["lons"][i]] * npos,
                lats1=[rcv_info["lats"][i]] * npos,
                lons2=lon,
                lats2=lat,
                return_back_azimuth=True,
            )
            distance[mmsi][rcv_id] = ranges

    return distance


def get_cpa(ais_data, distance, rcv_info):
    """
    Derive the closest point of approach between the ship and the receiver

    Args:
        ais_data (pd.DataFrame): ais data
        distance (dict): distance between the ship and the receiver (derived from compute_distance_ship_rcv)
        rcv_info (dict): receiver information

    Returns:
        cpa (dict): closest point of approach

    """

    cpa = {}
    available_mmsi = ais_data["mmsi"].unique()

    for j, mmsi in enumerate(available_mmsi):
        cpa[mmsi] = {}
        for i, rcv_id in enumerate(rcv_info["id"]):
            cpa_idx = np.argmin(distance[mmsi][rcv_id])
            lon = ais_data[ais_data["mmsi"] == mmsi]["lon"].iloc[cpa_idx]
            lat = ais_data[ais_data["mmsi"] == mmsi]["lat"].iloc[cpa_idx]
            time = ais_data[ais_data["mmsi"] == mmsi]["time"].iloc[cpa_idx]
            r = distance[mmsi][rcv_id][cpa_idx]
            cpa[mmsi][rcv_id] = {"lon": lon, "lat": lat, "time": time, "r_cpa": r}

    return cpa


def plot_distance_ship_rcv(ais_data, distance, cpa, rcv_info):
    """
    Plot the distance between the ship and the receiver

    Args:
        ais_data (pd.DataFrame): ais data
        distance (dict): distance between the ship and the receiver (derived from compute_distance_ship_rcv)
        cpa (dict): closest point of approach
        rcv_info (dict): receiver information

    Returns:
        None
    """

    available_mmsi = ais_data["mmsi"].unique()

    fig, ax = plt.subplots(len(available_mmsi), 1, sharex=False)
    for i, rcv_id in enumerate(rcv_info["id"]):
        for j, mmsi in enumerate(available_mmsi):
            df_mmsi = ais_data[ais_data["mmsi"] == mmsi]

            ax[j].plot(
                df_mmsi["time"], distance[mmsi][rcv_id], label=f"Receiver {rcv_id}"
            )

            # Plot the CPA
            cpa_time = cpa[mmsi][rcv_id]["time"]
            cpa_r = cpa[mmsi][rcv_id]["r_cpa"]
            ax[j].scatter(cpa_time, cpa_r, color="r")
            # Add annotation with the CPA time (annotation is centered on the CPA point)
            ax[j].annotate(
                f"CPA at {cpa_time}",
                xy=(cpa_time, cpa_r),
                xytext=(cpa_time, cpa_r + 10000),
                arrowprops=dict(facecolor="black", arrowstyle="->"),
                ha="center",
            )

            ship_name = df_mmsi["shipName"].values[0]
            ax[j].set_title(f"{ship_name}")
            # ax[j].set_ylabel("Distance [m]")
            ax[j].legend(loc="upper right", fontsize=12)
            # ax[j].set_xlabel("Time [s]")

            ax[j].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax[j].xaxis.set_major_locator(mdates.HourLocator(interval=1))

            # Get the last date in the time data
            end_date = df_mmsi["time"].iloc[0].strftime("%Y-%m-%d")

            # Add the date as text at the end of the x-axis
            ax[j].text(
                0.99,
                0.05,
                f"Date: {end_date}",
                transform=ax[j].transAxes,
                ha="right",
                fontsize=12,
            )
            ax[j].tick_params(axis="x", rotation=0)

    fig.supxlabel("Time [h]")
    fig.supylabel("Distance [m]")
    # plt.tight_layout()


def get_available_time_window(ais_data, intersection_time):

    available_mmsi = ais_data["mmsi"].unique()

    # WAV data
    # Define the time window according to the ais data available
    ais_data_duration = {}
    for mmsi in available_mmsi:
        ais_data_tmp = ais_data[ais_data["mmsi"] == mmsi]
        t_start = ais_data_tmp["time"].min()
        t_end = ais_data_tmp["time"].max()

        t_intersec_to_start = intersection_time[mmsi] - t_start
        t_end_to_intersec = t_end - intersection_time[mmsi]
        ais_data_duration[mmsi] = max(t_intersec_to_start, t_end_to_intersec)

    # Define time window half length
    min_duration = min(ais_data_duration.values())
    # Convert min_duration to minutes
    min_duration_minutes = np.floor(min_duration.total_seconds() / 60)
    # Define half window duration
    t_half_window = pd.Timedelta(f"{min_duration_minutes//2}min")
    # print(f"Half window duration: {t_half_window}")

    mmsi1 = available_mmsi[0]
    mmsi2 = available_mmsi[1]
    t_start_1 = intersection_time[mmsi1] - t_half_window
    t_start_2 = intersection_time[mmsi2] - t_half_window
    t_end_1 = intersection_time[mmsi1] + t_half_window
    t_end_2 = intersection_time[mmsi2] + t_half_window
    duration_seconds = 2 * t_half_window.total_seconds()

    start_times = {mmsi1: t_start_1, mmsi2: t_start_2}
    end_times = {mmsi1: t_end_1, mmsi2: t_end_2}

    return duration_seconds, start_times, end_times


def restrict_ais_data_to_time_window(ais_data, start_times, end_times):

    mmsi1 = list(start_times.keys())[0]
    mmsi2 = list(start_times.keys())[1]

    # Restrict ais data to the time window
    ais_data_1 = ais_data[ais_data["mmsi"] == mmsi1]
    ais_data_2 = ais_data[ais_data["mmsi"] == mmsi2]

    ais_data_restricted = pd.concat(
        [
            ais_data_1[
                (ais_data_1["time"] >= start_times[mmsi1])
                & (ais_data_1["time"] <= end_times[mmsi1])
            ],
            ais_data_2[
                (ais_data_2["time"] >= start_times[mmsi2])
                & (ais_data_2["time"] <= end_times[mmsi2])
            ],
        ]
    )

    ais_data_restricted.reset_index(drop=True, inplace=True)

    return ais_data_restricted


def get_maxmin_tdoa(ais_data, wav_data, tdoa, verbose=False):
    min_tdoa = {}
    max_tdoa = {}
    for i, rcv_couple in enumerate(wav_data.keys()):
        min_tdoa[rcv_couple] = {}
        max_tdoa[rcv_couple] = {}
        for mmsi in wav_data[rcv_couple].keys():
            min_tdoa[rcv_couple][mmsi] = {}
            max_tdoa[rcv_couple][mmsi] = {}

            tdoa_mmsi = tdoa[rcv_couple][
                mmsi
            ]  # Delay between receiver for the given ship

            idx_tdoa_min = np.argmin(np.abs(tdoa_mmsi))
            idx_tdoa_max = np.argmax(np.abs(tdoa_mmsi))

            min_tdoa[rcv_couple][mmsi]["idx"] = idx_tdoa_min
            min_tdoa[rcv_couple][mmsi]["tdoa"] = tdoa_mmsi[idx_tdoa_min]
            max_tdoa[rcv_couple][mmsi]["idx"] = idx_tdoa_max
            max_tdoa[rcv_couple][mmsi]["tdoa"] = tdoa_mmsi[idx_tdoa_max]

            time_min_tdoa = ais_data[ais_data["mmsi"] == mmsi]["time"].values[
                idx_tdoa_min
            ]
            time_max_tdoa = ais_data[ais_data["mmsi"] == mmsi]["time"].values[
                idx_tdoa_max
            ]

            if verbose:
                print(
                    f"Time corresponding to minimum tdoa for ship {mmsi}: {time_min_tdoa}"
                )
                print(
                    f"Time corresponding to maximum tdoa for ship {mmsi}: {time_max_tdoa}"
                )

            # Convert time min tdoa to second from begining
            t0 = ais_data[ais_data["mmsi"] == mmsi]["time"].values[0]
            delta_from_start_min_tdoa = (
                (time_min_tdoa - t0).astype("timedelta64[s]").astype(float)
            )
            delta_from_start_max_tdoa = (
                (time_max_tdoa - t0).astype("timedelta64[s]").astype(float)
            )

            # Derive corresponding time bin in crosscorr_data
            tt = wav_data[rcv_couple][mmsi]["tt"]
            idx_tt_min = np.argmin(np.abs(tt - delta_from_start_min_tdoa))
            idx_tt_max = np.argmin(np.abs(tt - delta_from_start_max_tdoa))
            min_tdoa[rcv_couple][mmsi]["idx_tt"] = idx_tt_min
            max_tdoa[rcv_couple][mmsi]["idx_tt"] = idx_tt_max
            min_tdoa[rcv_couple][mmsi]["tt"] = delta_from_start_min_tdoa
            max_tdoa[rcv_couple][mmsi]["tt"] = delta_from_start_max_tdoa

    return min_tdoa, max_tdoa


if __name__ == "__main__":

    root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ais\extract-ais-pos-for-zone-ecole-navale-by-month-201305.csv"
    fname = "extract-ais-pos-for-zone-ecole-navale-by-month-201305.csv"
    fpath = os.path.join(root, fname)

    ## SWIR obs locations (lon, lat, depth)
    # RR41,65.3344,-27.7330,-5430
    # RR42,65.4376,-27.6192,-4771
    # RR43,65.5826,-27.5338,-4264
    # RR44,65.7481,-27.5324,-4548.00
    # RR45,65.6019,-27.6581,-2822.00
    # RR46,65.5835,-27.7909,-3640
    # RR47,65.7553,-27.6958,-4582
    # RR48,65.9430,-27.5792,-4830.00

    lon_min = 65
    lon_max = 66
    lat_min = -28
    lat_max = -27

    df = extract_ais_area(fpath, lon_min, lon_max, lat_min, lat_max)
    # print(df.head())

    # Remove ships with less than 2 points
    df = df.groupby("mmsi").filter(lambda x: len(x) > 1)

    # Pick one MMSI among the list and plot trajectory
    # mmsi = df["mmsi"][0]
    # df = df[df["mmsi"] == mmsi]

    # plt.figure()
    # plt.plot(df["lon"], df["lat"], "o-")
    # plt.show()

    # Plot all trajectories
    # plt.figure()
    # for mmsi in df["mmsi"].unique():
    #     df_tmp = df[df["mmsi"] == mmsi]
    #     plt.plot(df_tmp["lon"], df_tmp["lat"], "o-", label=df_tmp["shipName"].values[0])
    # plt.legend()

    # Interpolate trajectories
    from publication.PublicationFigure import PubFigure

    # PubFigure(legend_fontsize=5)

    df_interp = interpolate_trajectories(df)

    # Plot all trajectories
    plt.figure()

    rcv_id = ["RR41", "RR44", "RR45", "RR47"]
    ds_bathy = plot_swir_bathy(contour=False)
    plot_swir_obs(ds_bathy, rcv_id, col=None)
    PubFigure(legend_fontsize=7)

    for mmsi in df_interp["mmsi"].unique():
        df_tmp = df_interp[df_interp["mmsi"] == mmsi]
        plt.plot(
            df_tmp["lon"],
            df_tmp["lat"],
            # label=df_tmp["shipName"].values[0],
            color="k",
            linestyle="--",
            linewidth=3,
        )

        # Add a marker at a random position along the trajectory to identify the ship with its name
        idx = np.random.randint(0, len(df_tmp))
        plt.scatter(
            df_tmp["lon"].iloc[idx],
            df_tmp["lat"].iloc[idx],
            s=100,
            marker="^",
            # label=df_tmp["shipName"].values[0],
            label=df_tmp["mmsi"].values[0],
            zorder=2,
        )

        # plt.text(
        #     df_tmp["lon"].iloc[idx],
        #     df_tmp["lat"].iloc[idx],
        #     df_tmp["shipName"].values[0],
        #     fontsize=8,
        # )

    # plt.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.05))
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncols=7,
        mode="expand",
        borderaxespad=0.0,
    )

    plt.ylim(lat_min, lat_max)
    plt.xlim(lon_min, lon_max)

    # Select given ships
    df_select = df_interp[df_interp["mmsi"].isin([416004485, 373759000])]

    project_to_enu(df=df_select)

    intersection_wgs84, intersection_time_s1, intersection_time_s2 = (
        intersection_time_2ships(df_2ships=df_select)
    )

    print("Intersection time ship 1: ", intersection_time_s1)
    print("Intersection time ship 2: ", intersection_time_s2)
    print("Intersection position: ", intersection_wgs84)
    # print(df_select)

    # Plot selected ships
    plt.figure()

    ds_bathy = plot_swir_bathy(contour=False)
    plot_swir_obs(ds_bathy, rcv_id, col=None)
    PubFigure(legend_fontsize=20)

    for mmsi in df_select["mmsi"].unique():
        df_tmp = df_select[df_select["mmsi"] == mmsi]
        plt.plot(
            df_tmp["lon"],
            df_tmp["lat"],
            "-",
            label=df_tmp["shipName"].values[0],
            linewidth=3,
        )

    # Add intersection point
    if len(intersection_wgs84) > 0:
        lon_inter, lat_inter = zip(*intersection_wgs84)
        plt.scatter(
            lon_inter,
            lat_inter,
            s=150,
            marker="x",
            color="r",
            zorder=2,
            label=f"Intersection at {r'$t_{ship1}$'} = {intersection_time_s1[0]}",
        )

    plt.legend(
        loc="upper right",
    )

    plt.ylim(lat_min, lat_max)
    plt.xlim(lon_min, lon_max)

    plt.show()
