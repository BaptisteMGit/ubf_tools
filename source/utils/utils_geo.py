#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils_geo.py
@Time    :   2025/12/10 14:14:33
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
from pyproj import Geod


# ---------------------------------------------------------------------------
# 1) Conversion WGS84 (lat, lon, h) -> ECEF (X, Y, Z)
# ---------------------------------------------------------------------------
def geodetic_to_ecef(lat, lon, h):
    """
    lat, lon : en radians (float ou array)
    h       : en mètres (float ou array)
    Retourne (X, Y, Z) vectorisés.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    h = np.asarray(h)

    # WGS84
    wgs84_geod = Geod(ellps="WGS84")
    a = wgs84_geod.a  # demi-grand axe (tableau p.80)
    e2 = wgs84_geod.es  # excentricité^2 (p.25)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = a / np.sqrt(1 - e2 * sin_lat**2)

    X = (N + h) * cos_lat * cos_lon
    Y = (N + h) * cos_lat * sin_lon
    Z = (N * (1 - e2) + h) * sin_lat

    return X, Y, Z


# ---------------------------------------------------------------------------
# 2) Conversion ECEF -> ENU local
# ---------------------------------------------------------------------------


def ecef_to_enu(X, Y, Z, lat0, lon0, h0):
    """
    X,Y,Z : ECEF (float ou array)
    lat0,lon0 : origine en radians
    h0 : hauteur origine
    Retourne (e,n,u) vectorisés.
    """

    # Origine locale en ECEF
    X0, Y0, Z0 = geodetic_to_ecef(lat0, lon0, h0)

    # Passer en array
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    # Vecteurs translatés
    dX = X - X0
    dY = Y - Y0
    dZ = Z - Z0

    sin_lat = np.sin(lat0)
    cos_lat = np.cos(lat0)
    sin_lon = np.sin(lon0)
    cos_lon = np.cos(lon0)

    # Matrice ECEF->ENU
    R = np.array(
        [
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )

    # Appliquer rotation à N points : R @ [dX, dY, dZ] pour tous
    d = np.vstack((dX, dY, dZ))  # shape (3, N)
    enu = R @ d  # shape (3, N)

    return enu[0], enu[1], enu[2]


def enu_to_ecef(e, n, u, lat0, lon0, h0):
    """
    e,n,u : ENU (float ou array)
    Retourne X,Y,Z vectorisés.
    """

    X0, Y0, Z0 = geodetic_to_ecef(lat0, lon0, h0)

    e = np.asarray(e)
    n = np.asarray(n)
    u = np.asarray(u)

    sin_lat = np.sin(lat0)
    cos_lat = np.cos(lat0)
    sin_lon = np.sin(lon0)
    cos_lon = np.cos(lon0)

    # Matrice ENU->ECEF = R^T
    R_T = np.array(
        [
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
            [0, cos_lat, sin_lat],
        ]
    )

    enu = np.vstack((e, n, u))  # shape (3, N)
    ecef = R_T @ enu  # shape (3, N)

    X = ecef[0] + X0
    Y = ecef[1] + Y0
    Z = ecef[2] + Z0

    return X, Y, Z


def ecef_to_geodetic(X, Y, Z):
    """
    X,Y,Z : float ou array
    Retourne (lat, lon en radians, h)
    """

    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    # WGS84
    wgs84_geod = Geod(ellps="WGS84")
    a = wgs84_geod.a  # demi-grand axe (tableau p.80)
    e2 = wgs84_geod.es  # excentricité^2 (p.25)

    lon = np.arctan2(Y, X)
    p = np.sqrt(X**2 + Y**2)

    lat = np.arctan2(Z, p * (1 - e2))  # init

    # Boucles vectorisées
    for _ in range(5):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(Z, p * (1 - e2 * (N / (N + h))))

    # Hauteur finale
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e2 * sin_lat**2)
    h = p / np.cos(lat) - N

    return lat, lon, h


# ---------------------------------------------------------------------------
# Exemple d'utilisation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Point qui va être converti
    lat = np.radians(48.8566)
    lon = np.radians(2.3522)
    h = 35

    # Point d'origine ENU
    lat0 = np.radians(48.85)
    lon0 = np.radians(2.35)
    h0 = 35

    # 1) WGS84 -> ECEF
    X, Y, Z = geodetic_to_ecef(lat, lon, h)
    print(X, Y, Z)

    # 2) ECEF -> ENU
    e, n, u = ecef_to_enu(X, Y, Z, lat0, lon0, h0)
    print(f"ENU", e, n, u)

    # 3) ENU -> ECEF
    X2, Y2, Z2 = enu_to_ecef(e, n, u, lat0, lon0, h0)
    print(f"ECEF", X2, Y2, Z2)

    # 4) ECEF -> WGS84
    lat2, lon2, h2 = ecef_to_geodetic(X2, Y2, Z2)
    print(f"Geod", lat2, lon2, h2)

    print("Diff lat (rad):", lat - lat2)
    print("Diff lon (rad):", lon - lon2)
    print("Diff h (m):", h - h2)
