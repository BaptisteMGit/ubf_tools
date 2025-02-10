#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   process_ctds.py
@Time    :   2025/01/23 16:44:47
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
import pandas as pd
import matplotlib.pyplot as plt

from publication.PublicationFigure import PubFigure

pfig = PubFigure()

# ======================================================================================================================
# Constants
# ======================================================================================================================
CTD_FOLDER = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\SwellEx96\ctds"


# ======================================================================================================================
# Functions
# ======================================================================================================================
def load_ctds():
    """Load CTD data from the SwellEx96 campaign.

    Returns:
    --------
    ctds: dict
        Dictionary containing the CTD data.
    """
    ctds = {}
    columns = ["Depth_m", "Temperature_C", "Salinity_PSU", "SoundSpeed_m_s", "Sigma_t"]

    # Store mean values
    mean_values = pd.DataFrame(columns=columns, dtype=np.float64)

    # List .prn files in folder
    ctd_files = [f for f in os.listdir(CTD_FOLDER) if f.endswith(".prn")]
    ctds["ctds_files"] = [ctd_name.split(".")[0] for ctd_name in ctd_files]
    for i_ctd, ctd_file in enumerate(ctd_files):
        ctd_name = ctd_file.split(".")[0]
        ctds[ctd_name] = pd.read_csv(
            os.path.join(CTD_FOLDER, ctd_file),
            sep=",",
            dtype=np.float64,
            names=columns,
        )

        # Add one column to count files contributing to the mean values
        ctds[ctd_name]["nb_files"] = 1

        # Store mean and std values
        if i_ctd == 0:
            mean_values = ctds[ctd_name].copy()
        else:
            mean_values = mean_values.add(ctds[ctd_name], fill_value=0)

    # Save nb_files series separately
    nb_files = mean_values["nb_files"].copy()

    # Compute mean values
    mean_values = mean_values.div(mean_values["nb_files"], axis=0)

    # Compute std values
    std_values = pd.DataFrame(columns=columns, dtype=np.float64)
    std_values["Depth_m"] = mean_values["Depth_m"]
    for ctd_name in ctds["ctds_files"]:
        ctd = ctds[ctd_name]
        # Compute std values for each column exept the depth and the nb_files
        for column in columns[1:-1]:
            eps = ctd[column].sub(mean_values[column], fill_value=0)
            # Replace unexistant values by 0
            ctd_max_depth = ctd["Depth_m"].max()
            eps[std_values["Depth_m"] > ctd_max_depth] = 0
            std_values[column] = std_values[column].add(eps**2, fill_value=0)

    std_values = std_values.div(nb_files, axis=0)

    # Add mean and std values to ctds
    ctds["mean_values"] = mean_values
    ctds["std_values"] = std_values

    return ctds


def plot_ctds(ctds):
    """Plot CTD data."""

    plt.figure()

    for ctd_name in ctds["ctds_files"]:
        ctd = ctds[ctd_name]
        plt.plot(ctd["SoundSpeed_m_s"], ctd["Depth_m"], alpha=0.2, color="b")

    # Plot mean values
    mean_values = ctds["mean_values"]
    plt.plot(
        mean_values["SoundSpeed_m_s"],
        mean_values["Depth_m"],
        color="k",
        label="Mean values",
    )

    # Add the +/- std values
    a = 2
    std_values = ctds["std_values"]
    plt.fill_betweenx(
        mean_values["Depth_m"],
        mean_values["SoundSpeed_m_s"] - a * np.sqrt(std_values["SoundSpeed_m_s"]),
        mean_values["SoundSpeed_m_s"] + a * np.sqrt(std_values["SoundSpeed_m_s"]),
        color="k",
        alpha=0.1,
    )

    # Customize plot
    axis = plt.gca()
    axis.invert_yaxis()
    axis.xaxis.tick_top()  # Move ticks to top
    axis.xaxis.set_label_position("top")  # Move label to top
    axis.spines["top"].set_position(("axes", 1.0))  # Ensure x-axis is at the top
    axis.spines["top"].set_visible(True)

    # Keep left spine (y-axis) normal but move bottom and right spines
    axis.spines["left"].set_position(("axes", 0))
    axis.spines["bottom"].set_visible(False)  # Hide bottom spine
    axis.spines["right"].set_visible(False)  # Hide right spine

    # Add arrowheads at the end of the axes
    axis.plot(
        1, 0, ">k", transform=axis.get_yaxis_transform(), clip_on=False
    )  # Right arrow for x-axis
    # axis.plot(
    #     0, 1, "^k", transform=axis.get_xaxis_transform(), clip_on=False
    # )  # Upward arrow for x-axis (ticks on top)
    axis.plot(
        0,
        max(mean_values["Depth_m"]),
        "vk",
        transform=axis.get_yaxis_transform(),
        clip_on=False,
    )  # Downward arrow for y-axis at bottom

    plt.xlabel("Sound speed [m/s]")
    plt.ylabel("Depth [m]")

    # plt.xlim([1465, 1535])  # Adjust based on your data range
    plt.ylim([max(mean_values["Depth_m"]), 0])  # Ensure depth is downward
    plt.grid()


def format_ctd_files():
    """Format CTD data : replace space by commas to make it readable by pandas."""

    # List .prn files in folder
    ctd_files = [f for f in os.listdir(CTD_FOLDER) if f.endswith(".prn")]

    for ctd_file in ctd_files:
        with open(os.path.join(CTD_FOLDER, ctd_file), "r") as f:
            lines = f.readlines()
        with open(os.path.join(CTD_FOLDER, ctd_file), "w") as f:
            for line in lines:
                formated_line = line.replace("         ", ",").replace("      ", ", ")
                f.write(formated_line)


def write_mean_ssp_profile(ctds):
    """Write mean sound speed profile in a file."""
    mean_ssp = ctds["mean_values"][["Depth_m", "SoundSpeed_m_s"]]
    mean_ssp.to_csv(os.path.join(CTD_FOLDER, "mean_ssp_profile.csv"), index=False)

    # Subsample the mean ssp profile
    subsample = 5
    mean_ssp_subsample = mean_ssp.iloc[::subsample]
    # Round values to 1 decimal
    mean_ssp_subsample = mean_ssp_subsample.round(1)
    mean_ssp_subsample.to_csv(
        os.path.join(CTD_FOLDER, "mean_ssp_profile_subsample.csv"), index=False, sep=" "
    )


if __name__ == "__main__":
    # format_ctd_files()
    ctds = load_ctds()

    # for ctd_name in ctds["ctds_files"]:
    #     ctd = ctds[ctd_name]
    #     print(ctd["SoundSpeed_m_s"])

    root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\real_data\SwellEx96"
    plot_ctds(ctds)
    fname = "swellex96_ssp"
    fpath = os.path.join(root, fname)
    plt.savefig(fpath + ".pdf", dpi=300, format="pdf")
    plt.savefig(fpath + ".png", dpi=300, format="png")

    # plt.show()
    write_mean_ssp_profile(ctds)
