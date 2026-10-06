#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   mfp_processor_and_grid_resolution.py
@Time    :   2026/09/23 15:57:54
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
from publication.publication_figure import PubFigure, set_subfigures_abc_labels

PubFigure(label_fontsize=26, ticks_fontsize=24)

# ======================================================================================================================
# Function
# ======================================================================================================================


# Assume cost function is a 2D Gaussian
def cost_fct(xx, yy, x0, y0, sigma_x, sigma_y):
    J = 1 - np.exp(
        -((xx - x0) ** 2 / (2 * sigma_x**2) + (yy - y0) ** 2 / (2 * sigma_y**2))
    )
    return J


# Sparse grid made of ship trajs
def generate_ship_track(x, y):
    # x coords
    L_x = x.max() - x.min()
    xs = np.random.uniform(x.min() - L_x, x.max() - L_x / 2)
    xe = np.random.uniform(x.min() + L_x / 2, x.max() + L_x)
    L_y = y.max() - y.min()
    ys = np.random.uniform(y.min() - L_y, y.max() - L_y / 2)
    ye = np.random.uniform(y.min() + L_y / 2, y.max() + L_y)

    a = (ye - ys) / (xe - xs)
    b = ye - a * xe

    x_track = np.linspace(xs, xe, 50)
    y_track = a * x_track + b

    y_track_in_area_idx = np.logical_and(y_track >= y.min(), y_track <= y.max())
    x_track_in_area_idx = np.logical_and(x_track >= x.min(), x_track <= x.max())
    idx_in_area = np.logical_and(x_track_in_area_idx, y_track_in_area_idx)

    return x_track[idx_in_area], y_track[idx_in_area]


def generate_ship_tracks(x, y, ntracks=5):
    x_tracks, y_tracks = [], []
    for itrack in range(ntracks):
        x_track, y_track = generate_ship_track(x, y)
        x_tracks.append(x_track)
        y_tracks.append(y_track)
    return x_tracks, y_tracks
    # return np.array(x_tracks), np.array(y_tracks)


# ======================================================================================================================
# Params
# ======================================================================================================================

cmap = "Reds_r"
grid_color = "k"

### Very dense grid ####
dx = 1
x = np.arange(-500, 500, dx)
y = np.arange(-500, 500, dx)
xx, yy = np.meshgrid(x, y)

print(x[1] - x[0])

### Regular grid sampling ####
dx_lib = 50
dy_lib = 50
x_lib = np.arange(-0.5 * 1e3, 0.5 * 1e3, dx_lib) + dx_lib / 2
y_lib = np.arange(-0.5 * 1e3, 0.5 * 1e3, dy_lib) + dy_lib / 2
xx_lib, yy_lib = np.meshgrid(x_lib, y_lib)

# Source position
x0 = 225
y0 = -300


def generate_fig(sigma_x, sigma_y):

    # Cost function on dense grid
    Jxy = cost_fct(xx, yy, x0, y0, sigma_x=sigma_x, sigma_y=sigma_y)
    # Cost function evaluated at regular lib pos
    Jxy_lib = cost_fct(xx_lib, yy_lib, x0, y0, sigma_x=sigma_x, sigma_y=sigma_y)
    # Cost function evaluated at sparse lib pos
    Jxy_lib_sparse = cost_fct(
        x_lib_sparse, y_lib_sparse, x0, y0, sigma_x=sigma_x, sigma_y=sigma_y
    )

    # Plot figures
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    im = axs[0].pcolormesh(x, y, Jxy, cmap=cmap, vmin=0, vmax=1)

    # Grid
    axs[0].scatter(xx_lib, yy_lib, color=grid_color, marker="x", alpha=0.2)

    axs[1].scatter(
        xx_lib.flatten(),
        yy_lib.flatten(),
        c=Jxy_lib.flatten(),
        cmap=cmap,
        vmin=0,
        vmax=1,
        s=50,
    )

    axs[1].scatter(
        xx_lib.flatten(), yy_lib.flatten(), color=grid_color, marker="x", alpha=0.2
    )

    # fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    im = axs[2].pcolormesh(x, y, Jxy, cmap=cmap, vmin=0, vmax=1)
    axs[2].scatter(x0, y0, marker="x", color="r", s=150)

    # axs[1].pcolormesh(x_lib, y_lib, Jxy_lib, cmap=cmap, vmin=0, vmax=1)
    axs[3].scatter(
        x_lib_sparse, y_lib_sparse, c=Jxy_lib_sparse, cmap=cmap, vmin=0, vmax=1, s=50
    )

    fig.supxlabel("x [m]")
    fig.supylabel("y [m]")

    plt.colorbar(im, ax=axs, label=r"J(x, y)")
    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )

    for itrack in range(len(x_tracks)):
        axs[2].plot(
            x_tracks[itrack],
            y_tracks[itrack],
            color=grid_color,
            linestyle="--",
            marker="x",
            alpha=0.2,
        )

    axs[3].scatter(x_lib_sparse, y_lib_sparse, color=grid_color, marker="x", alpha=0.2)

    for ax in axs:
        ax.scatter(x0, y0, marker="x", color="r", s=150)


# Compare wide lobe cost function to narrow lobe
generate_tracks = False

#### Ship track sampling ####
import os
import pandas as pd

root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration\mfp\mfp_measured\ship_tracks"
if generate_tracks:
    x_tracks, y_tracks = generate_ship_tracks(x, y, ntracks=20)
    x_lib_sparse = np.concat(x_tracks)
    y_lib_sparse = np.concat(y_tracks)

    for i in range(len(x_tracks)):
        tracks_dict = {"x_ship": x_tracks[i], "y_ship": y_tracks[i]}
        df = pd.DataFrame(tracks_dict)
        fpath = os.path.join(root, f"ship_track_{i}.csv")
        df.to_csv(fpath)

else:
    import glob

    ship_track_files = sorted(glob.glob(os.path.join(root, f"ship_track_*.csv")))
    # ship_track_files = sorted(glob.glob(os.path.join(root, f"ship_track_*")))

    x_tracks = []
    y_tracks = []
    for ship_file in ship_track_files:
        df = pd.read_csv(ship_file)
        x_tracks.append(df.x_ship.values)
        y_tracks.append(df.y_ship.values)

    x_lib_sparse = np.concat(x_tracks)
    y_lib_sparse = np.concat(y_tracks)

generate_fig(sigma_x=100, sigma_y=100)
generate_fig(sigma_x=25, sigma_y=25)


plt.show()
