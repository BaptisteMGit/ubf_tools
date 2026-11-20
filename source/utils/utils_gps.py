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
import numpy as np
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


def _empty_gps_result(t):
    """Empty output with the right columns and the datetime dtype of `t`."""
    return pd.DataFrame(
        {
            "datetime": t.iloc[:0].reset_index(drop=True),
            "lon": np.empty(0),
            "lat": np.empty(0),
        }
    )


def interpolate_gps(df_gps, time_step="10s"):
    """
    Interpolate gps data on a regular time grid (vectorised version).

    Each grid instant is placed on the geodesic joining the two surrounding
    measurements, at the fraction (t - t1) / (t2 - t1) of the segment, i.e.
    assuming constant speed between two successive positions.
    No extrapolation: the grid goes from ceil(first time) to last time.

    Invalid rows are discarded before interpolation:
      - missing datetime / lon / lat
      - lon outside [-180, 180] or lat outside [-90, 90] (this removes the AIS
        "not available" codes lon=181, lat=91)
      - duplicated timestamps (the last measurement is kept)
    An empty DataFrame (same columns) is returned when fewer than 2 valid
    measurements remain, or when no grid instant falls in the time span.

    Args:
        df_gps (pd.DataFrame): gps data with columns "datetime", "lon", "lat"
        time_step (str): time step for interpolation (e.g., '10s')

    Returns:
        pd.DataFrame: columns "datetime", "lon", "lat" on the regular grid
    """
    geod = Geod(ellps="WGS84")

    # --- Cleaning -----------------------------------------------------------
    df = df_gps.dropna(subset=["datetime"])
    valid = df["lon"].between(-180, 180) & df["lat"].between(-90, 90)  # False if NaN
    df = df[valid]
    df = df.sort_values("datetime", kind="stable")
    df = df.drop_duplicates(subset="datetime", keep="last")

    t = df["datetime"]
    if len(df) < 2:
        return _empty_gps_result(t)

    # --- Regular grid, aligned on multiples of time_step (ceil) so that
    # different trajectories share the same instants -------------------------
    grid = pd.date_range(t.iloc[0].ceil(time_step), t.iloc[-1], freq=time_step)
    if len(grid) == 0:
        return _empty_gps_result(t)

    # Times in seconds relative to the first measurement (tz-safe, float64 is
    # more than precise enough for relative times)
    t0 = t.iloc[0]
    t_sec = (t - t0).dt.total_seconds().to_numpy()
    g_sec = (grid - t0).total_seconds().to_numpy()

    # Index i of the segment [t_i, t_{i+1}) containing each grid instant.
    # The last grid instant may be exactly t_end: clip it into the last segment.
    idx = np.searchsorted(t_sec, g_sec, side="right") - 1
    idx = np.clip(idx, 0, len(t_sec) - 2)

    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()

    # Fraction of the segment elapsed at each grid instant
    # (t_{i+1} > t_i is guaranteed: timestamps are unique and sorted)
    t1, t2 = t_sec[idx], t_sec[idx + 1]
    frac = (g_sec - t1) / (t2 - t1)

    # Geodesic (azimuth, distance) of each segment, then move along it
    az12, _, dist = geod.inv(lon[idx], lat[idx], lon[idx + 1], lat[idx + 1])
    lon_i, lat_i, _ = geod.fwd(lon[idx], lat[idx], az12, frac * dist)

    return pd.DataFrame({"datetime": grid, "lon": lon_i, "lat": lat_i})


# def interpolate_gps(df_gps, time_step="10s"):
#     """
#     Interpolate gps data to have regular time steps
#     Args:
#         df_gps (pd.DataFrame): gps data
#         time_step (str): time step for interpolation (e.g., '10s' for 10 seconds)

#     Returns:
#         gps_df (pd.DataFrame): gps data with interpolated trajectories
#     """

#     # Define the geodetic object
#     geod = Geod(ellps="WGS84")

#     df_interp = pd.DataFrame()

#     # Ensure the data is sorted by time
#     df_gps = df_gps.sort_values("datetime")

#     # Create a empty pandas series to store the interpolated data
#     high_res_pos = pd.DataFrame()

#     # Compute datetime interpolation points from first to last instant with time_step
#     t_start = df_gps["datetime"].iloc[0]
#     t_start = t_start.ceil(
#         time_step
#     )  # Round t_start to the nearest upper time_step so that different trajectories align
#     t_end = df_gps["datetime"].iloc[-1]
#     time_interp = pd.date_range(t_start, t_end, freq=time_step)
#     # Convert time_interp to a pandas df
#     high_res_time = pd.DataFrame({"datetime": time_interp})

#     # Iterate over successive positions
#     for i in range(1, len(df_gps)):
#         # Get interpolations points which lies between two successive gps points
#         t1 = df_gps["datetime"].iloc[i - 1]
#         t2 = df_gps["datetime"].iloc[i]
#         if i == len(df_gps) - 1:
#             time_interp = high_res_time[
#                 (high_res_time["datetime"] >= t1) & (high_res_time["datetime"] <= t2)
#             ]
#             include_last_point = True
#         else:
#             time_interp = high_res_time[
#                 (high_res_time["datetime"] >= t1) & (high_res_time["datetime"] < t2)
#             ]
#             include_last_point = False

#         if time_interp.empty:
#             continue

#         # print(f"Interpolating between {t1} and {t2} with {len(time_interp)} points.")
#         # Interpolate positions between gps points 1 and 2
#         lon1 = df_gps["lon"].iloc[i - 1]
#         lat1 = df_gps["lat"].iloc[i - 1]
#         lon2 = df_gps["lon"].iloc[i]
#         lat2 = df_gps["lat"].iloc[i]

#         # NOTE : Must include first point and last point (if needed) --> Fix 12/11/2026
#         # profile_coords = geod.inv_intermediate(
#         #     lon1, lat1, lon2, lat2, npts=time_interp.size, return_back_azimuth=False
#         # )
#         # Fix 13/11/2026 to avoid duplicated points at begining / end
#         if include_last_point:
#             terminus_idx = 0  # geod.inv_intermediate convention terminus_idx=0 means include_last = True
#         else:
#             terminus_idx = 1

#         profile_coords = geod.inv_intermediate(
#             lon1,
#             lat1,
#             lon2,
#             lat2,
#             npts=time_interp.size,
#             return_back_azimuth=False,
#             initial_idx=0,
#             terminus_idx=terminus_idx,
#         )

#         # Add profile_coords to high_res_pos
#         lon = profile_coords.lons
#         lat = profile_coords.lats
#         data = {"lon": lon, "lat": lat}
#         new_data = pd.DataFrame(data)
#         if high_res_pos.empty:
#             high_res_pos = new_data
#         else:
#             high_res_pos = pd.concat([high_res_pos, new_data])

#     # Concat high_res_time and high_res_pos
#     high_res_time.reset_index(drop=True, inplace=True)
#     high_res_pos.reset_index(drop=True, inplace=True)
#     df_interp = pd.concat([high_res_time, high_res_pos], axis=1)

#     # Reset index
#     df_interp.reset_index(drop=True, inplace=True)

#     return df_interp


if __name__ == "__main__":

    import os

    root_input = r"C:\Users\baptiste.menetrier\Desktop\ressource\XP_Fiberscope_Groix_092025\Jules\Trajectoires\14_10_25"
    fname = "trace_14_10_25"
    fpath_in = os.path.join(root_input, fname + ".gpx")

    root_output = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\fiberscope_groix_oct_2025\gps"
    fpath_out = os.path.join(root_output, f"gps_pos_{fname}.csv")

    gpx_to_csv(fpath_in, fpath_out)
