import numpy as np
import pandas as pd
from pyproj import Geod


def _empty_result(t):
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
        return _empty_result(t)

    # --- Regular grid, aligned on multiples of time_step (ceil) so that
    # different trajectories share the same instants -------------------------
    grid = pd.date_range(t.iloc[0].ceil(time_step), t.iloc[-1], freq=time_step)
    if len(grid) == 0:
        return _empty_result(t)

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
