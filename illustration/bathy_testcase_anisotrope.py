#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   bathy_testcase_anisotrope.py
@Time    :   2024/03/11 16:38:42
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


def bathy_sin_slope():

    alpha = 50
    min_depth = 150
    max_range = 50
    range_periodicity = 6

    # Define bathymetry
    fr = 1 / range_periodicity
    dr = 1 / (20 * fr)
    r = np.arange(0, max_range + dr, dr)
    theta = np.arange(0, 360, 0.01)

    x = np.linspace(-max_range, max_range, 100)
    y = np.linspace(-max_range, max_range, 100)
    X, Y = np.meshgrid(x, y)
    THETA = np.arctan2(Y, X)
    R = np.sqrt(X**2 + Y**2)
    H = min_depth + alpha * (
        1 + np.cos(2 * np.pi * R / range_periodicity * np.cos(THETA))
    )

    plt.figure(figsize=(16, 8))
    plt.contourf(X, Y, H, cmap="viridis")
    plt.colorbar()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Bathymetry")
    # plt.show()

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, H, cmap="terrain")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    # Set z limits
    ax.set_zlim(0, np.max(H))
    # Revert z axis
    ax.invert_zaxis()
    plt.title("Bathymetry")
    plt.show()

    # h = min_depth + alpha * (
    #     1 + np.cos(2 * np.pi * r / range_periodicity * np.cos(theta * np.pi / 180))
    # )


if __name__ == "__main__":
    bathy_sin_slope()
