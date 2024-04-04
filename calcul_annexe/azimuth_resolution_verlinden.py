#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   azimuth_resolution_verlinden.py
@Time    :   2024/04/04 10:51:24
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================

import os
import numpy as np
import matplotlib.pyplot as plt


def ellipsoid_arc_length_meridian(lat, e):
    """
    Calcule la longueur d'un arc d'ellipsoïde dans la direction d'un méridien à partir des premiers termes du développement en série de la longueur pour une latitude lat.

    Arguments :
    - lat : Latitude en radians.
    - e

    Returns :
    - La longueur de l'arc de l'ellipsoïde dans la direction du méridien.
    """
    e_squared = e**2
    # Définition des coefficients de la série
    b0 = (
        -175 / 16384 * e_squared**4
        - 5 / 256 * e_squared**3
        - 3 / 64 * e_squared**2
        - 1 / 4 * e_squared
        + 1
    )
    b1 = (
        -105 / 4096 * e_squared**4
        - 45 / 1024 * e_squared**3
        - 3 / 32 * e_squared**2
        - 3 / 8 * e_squared
    )
    b2 = (
        +525 / 16384 * e_squared**4 + 45 / 1024 * e_squared**3 + 15 / 256 * e_squared**2
    )
    b3 = -175 / 12288 * e_squared**4 - 35 / 3072 * e_squared**3
    b4 = 315 / 131072 * e_squared**4

    # Calcul de la longueur de l'arc
    L = a * (
        b0 * lat
        + b1 * np.sin(2 * lat)
        + b2 * np.sin(4 * lat)
        + b3 * np.sin(6 * lat)
        + b4 * np.sin(8 * lat)
    )

    return L


if __name__ == "__main__":
    lon, lat = 65.6019, -27.6581
    lat_rad = np.radians(lat)  # Latitude en radians
    lon_rad = np.radians(lon)  # Longitude en radians

    grid_size = 15 / 3600 * np.pi / 180  # 15" (secondes d'arc)
    lat_0 = lat_rad - grid_size
    lat_1 = lat_rad + grid_size
    lon_0 = lon_rad - grid_size
    lon_1 = lon_rad + grid_size
    a = 6378137.0  # Grand demi-axe de l'ellipsoïde (mètres)
    b = 6356752.314245  # Petit demi-axe de l'ellipsoïde (mètres)

    e = np.sqrt(1 - (b**2 / a**2))
    v = np.sqrt(1 - e**2 * np.sin(lat_rad) ** 2)
    N = a / v

    L = ellipsoid_arc_length_meridian(lat_rad, e)
    print(L)

    L1 = ellipsoid_arc_length_meridian(lat_0, e)
    print("L1 = ", L1)
    L2 = ellipsoid_arc_length_meridian(lat_1, e)
    print("L2 = ", L2)
    dlat = L2 - L1
    print("dlat = ", dlat)

    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    _, _, dlat = geod.inv(
        lons1=lon,
        lats1=np.degrees(lat_0),
        lons2=lon,
        lats2=np.degrees(lat_1),
    )

    print("dlat = ", dlat)

    _, _, dlon = geod.inv(
        lons1=np.degrees(lon_0),
        lats1=lat,
        lons2=np.degrees(lon_1),
        lats2=lat,
    )

    print("dlon = ", dlon)
