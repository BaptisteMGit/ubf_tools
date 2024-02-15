#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   bathy_profile_extraction.py
@Time    :   2024/02/13 15:52:45
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================

import os
import time
import numpy as np
import xarray as xr
from pyproj import Geod
from scipy.interpolate import RegularGridInterpolator


# ======================================================================================================================
# Data extraction
# ======================================================================================================================


def get_coords_along_profile(
    start_lat,
    start_lon,
    range_resolution,
    max_range_m=None,
    azimuth=None,
    stop_lat=None,
    stop_lon=None,
):
    """
    Get coordinates along a given profile.
    If azimuth is given, the porfile is defined by initial point and azimuth.
    If stop_lat and stop_lon are given, the profile is defined by initial and final points.

    :param start_lat:
    :param stop_lat:
    :param start_lon:
    :param stop_lon:
    :param range_resolution:
    :return:
    """

    # Define the geodetic object
    geod = Geod(ellps="WGS84")

    if azimuth is not None and max_range_m is not None:
        # Number of points in the profile
        n_points = int(max_range_m / range_resolution) + 1

    elif stop_lat is not None and stop_lon is not None:
        # Get distance and azimuth between the two points
        azimuth, azimuth2, distance = geod.inv(
            start_lon, start_lat, stop_lon, stop_lat, return_back_azimuth=True
        )

        # Number of points in the profile
        n_points = int(distance / range_resolution) + 1

        return profile_coords, r

    else:
        raise ValueError(
            "You must provide either an azimuth and a maximum range or a final point coordinates."
        )

    # List of longitude/latitude pairs describing npts equally spaced intermediate points along the geodesic
    # between the initial and terminus points
    profile_coords = geod.fwd_intermediate(
        start_lon,
        start_lat,
        azimuth,
        npts=n_points,
        del_s=range_resolution,
        return_back_azimuth=False,
    )
    range_along_profile = np.arange(0, n_points * range_resolution, range_resolution)

    return profile_coords, range_along_profile


def extract_bathy_profile(
    xr_bathy,
    start_lat,
    start_lon,
    range_resolution,
    max_range_m=None,
    azimuth=None,
    stop_lat=None,
    stop_lon=None,
):
    """
    Extract oblique profile (both longitude and latitude varying).

    :param xr_hycom:
    :param start_lat:
    :param stop_lat:
    :param start_lon:
    :param stop_lon:
    :param range_resolution:
    :return:
    """

    # Get points coordinates along profile
    profile_coords, range_along_profile = get_coords_along_profile(
        start_lat=start_lat,
        start_lon=start_lon,
        stop_lat=stop_lat,
        stop_lon=stop_lon,
        max_range_m=max_range_m,
        azimuth=azimuth,
        range_resolution=range_resolution,
    )

    # Interpolate the bathymetry data along the profile
    interp_func = RegularGridInterpolator(
        (xr_bathy.lat.data, xr_bathy.lon.data), xr_bathy.bathymetry.data
    )
    bathymetry_profile = interp_func((profile_coords.lats, profile_coords.lons))

    return range_along_profile, bathymetry_profile


if __name__ == "__main__":
    pass
