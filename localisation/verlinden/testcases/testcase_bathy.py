import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from publication.PublicationFigure import PubFigure

ENV_ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\testcase_working_directory"

pfig = PubFigure()


def bathy_sin_slope(
    testcase_name="testcase1",
    min_waveguide_depth=150,
    max_range=50,
    theta=94,
    range_periodicity=6,
):
    # Define bathymetry
    fr = 1 / range_periodicity
    dr = 1 / (20 * fr)
    r = np.arange(0, max_range + dr, dr)

    alpha = 50
    h = min_waveguide_depth - alpha * (
        -1
        + np.sin(
            2 * np.pi * r * np.cos(theta * np.pi / 180) / range_periodicity - np.pi / 2
        )
    )

    # Save bathymetry
    env_dir = os.path.join(ENV_ROOT, testcase_name)
    pd.DataFrame({"r": np.round(r, 3), "h": np.round(h, 3)}).to_csv(
        os.path.join(env_dir, "bathy.csv"), index=False, header=False
    )

    max_depth = min_waveguide_depth + 2 * alpha
    plt.figure(figsize=(16, 8))
    plt.plot(r, h, color="k", linewidth=2, marker="o", markersize=2)
    plt.ylim([0, max_depth])
    plt.fill_between(r, h, max_depth, color="lightgrey")
    plt.gca().invert_yaxis()
    plt.xlabel("Range (km)", fontsize=pfig.label_fontsize)
    plt.ylabel("Depth (m)", fontsize=pfig.label_fontsize)
    pfig.apply_ticks_fontsize()
    plt.grid()
    plt.savefig(os.path.join(env_dir, "bathy.png"))


def bathy_seamount(
    testcase_name="testcase1",
    min_waveguide_depth=150,
    max_range=50,
    max_depth=250,
    seamount_width=6,
):
    # Define bathymetry
    fr = 1 / seamount_width
    dr = 1 / (20 * fr)
    r = np.arange(0, max_range + dr, dr)

    r_seamount = r.max() / 2
    r0 = r_seamount - seamount_width / 2
    r1 = r_seamount + seamount_width / 2

    h_seamount = min_waveguide_depth
    h = np.ones(r.size) * max_depth

    alpha = (h_seamount - max_depth) / (r_seamount - r0)
    upslope = alpha * (r - r0) + max_depth
    downslope = -alpha * (r - r_seamount) + h_seamount

    idx_r_before = (r0 < r) * (r <= r_seamount)
    h[idx_r_before] = upslope[idx_r_before]
    idx_r_after = (r_seamount <= r) * (r < r1)
    h[idx_r_after] = downslope[idx_r_after]

    # Save bathymetry
    env_dir = os.path.join(ENV_ROOT, testcase_name)
    pd.DataFrame({"r": np.round(r, 3), "h": np.round(h, 3)}).to_csv(
        os.path.join(env_dir, "bathy.csv"), index=False, header=False
    )

    plt.figure(figsize=(16, 8))
    plt.plot(r, h, color="k", linewidth=2, marker="o", markersize=2)
    plt.ylim([0, max_depth])
    plt.fill_between(r, h, max_depth, color="lightgrey")
    plt.gca().invert_yaxis()
    plt.xlabel("Range (km)", fontsize=pfig.label_fontsize)
    plt.ylabel("Depth (m)", fontsize=pfig.label_fontsize)
    pfig.apply_ticks_fontsize()
    plt.grid()
    plt.savefig(os.path.join(env_dir, "bathy.png"))
