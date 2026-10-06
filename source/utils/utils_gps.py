#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils_gpx.py
@Time    :   2025/12/12 15:17:18
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import gpxpy
import csv
import pandas as pd
from pyproj import Geod


def gpx_to_csv(gpx_file, csv_file="output.csv", verbose=False):

    # Charger le contenu GPX
    with open(gpx_file, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    rows = []

    # Parcours des traces
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                rows.append(
                    {
                        "time": point.time.isoformat() if point.time else "",
                        "lat": point.latitude,
                        "lon": point.longitude,
                        "ele": point.elevation if point.elevation is not None else "",
                    }
                )

    # Écriture CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "lat", "lon", "ele"])
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"✔ Conversion GPX → CSV terminée.")
        print(f"✔ {len(rows)} points GPS exportés dans {csv_file}.")


def interpolate_gps(df_gps, time_step="10s"):
    """
    Interpolate gps data to have regular time steps
    Args:
        df_gps (pd.DataFrame): gps data
        time_step (str): time step for interpolation (e.g., '10s' for 10 seconds)

    Returns:
        gps_df (pd.DataFrame): gps data with interpolated trajectories
    """

    # Define the geodetic object
    geod = Geod(ellps="WGS84")

    df_interp = pd.DataFrame()

    # Ensure the data is sorted by time
    df_gps = df_gps.sort_values("datetime")

    # Create a empty pandas series to store the interpolated data
    high_res_pos = pd.DataFrame()

    # Compute datetime interpolation points from first to last instant with time_step
    t_start = df_gps["datetime"].iloc[0]
    t_start = t_start.ceil(
        time_step
    )  # Round t_start to the nearest upper time_step so that different trajectories align
    t_end = df_gps["datetime"].iloc[-1]
    time_interp = pd.date_range(t_start, t_end, freq=time_step)
    # Convert time_interp to a pandas df
    high_res_time = pd.DataFrame({"datetime": time_interp})

    # Iterate over successive positions
    for i in range(1, len(df_gps)):
        # Get interpolations points which lies between two successive gps points
        t1 = df_gps["datetime"].iloc[i - 1]
        t2 = df_gps["datetime"].iloc[i]
        if i == len(df_gps) - 1:
            time_interp = high_res_time[
                (high_res_time["datetime"] >= t1) & (high_res_time["datetime"] <= t2)
            ]
        else:
            time_interp = high_res_time[
                (high_res_time["datetime"] >= t1) & (high_res_time["datetime"] < t2)
            ]

        if time_interp.empty:
            continue

        # print(f"Interpolating between {t1} and {t2} with {len(time_interp)} points.")
        # Interpolate positions between gps points 1 and 2
        lon1 = df_gps["lon"].iloc[i - 1]
        lat1 = df_gps["lat"].iloc[i - 1]
        lon2 = df_gps["lon"].iloc[i]
        lat2 = df_gps["lat"].iloc[i]

        profile_coords = geod.inv_intermediate(
            lon1, lat1, lon2, lat2, npts=time_interp.size, return_back_azimuth=False
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
    df_interp = pd.concat([high_res_time, high_res_pos], axis=1)

    # Reset index
    df_interp.reset_index(drop=True, inplace=True)

    return df_interp


if __name__ == "__main__":

    import os

    root_input = r"C:\Users\baptiste.menetrier\Desktop\ressource\XP_Fiberscope_Groix_092025\Jules\Trajectoires\14_10_25"
    fname = "trace_14_10_25"
    fpath_in = os.path.join(root_input, fname + ".gpx")

    root_output = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\fiberscope_groix_oct_2025\gps"
    fpath_out = os.path.join(root_output, f"gps_pos_{fname}.csv")

    gpx_to_csv(fpath_in, fpath_out)
