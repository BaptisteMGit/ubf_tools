#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   oceano_utils.py
@Time    :   2025/12/16 15:51:08
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Code adapté du code fourni par le SHOM Read_oceano.py
"""


# ======================================================================================================================
# Import
# ======================================================================================================================

import pandas as pd
from datetime import datetime
import matplotlib

# matplotlib.use("TkAgg", force=True)
from matplotlib import pyplot as plt
import numpy as np
import os

# # Set the default color cycle
# matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
#     color=["r", "k", "c", "b", "grey"]
# )


def rbr_reader(rbr_filepath):

    # Read file
    with open(rbr_filepath, "r") as f:
        lines = f.readlines()

    # Header lines contain //
    header_lines = [line for line in lines if line.startswith("//")]
    header_end_line_idx = len(header_lines)

    # Save to csv for simplicity
    header = lines[header_end_line_idx]
    data_lines = lines[header_end_line_idx + 1 :]
    csv_fpath = rbr_filepath.replace(".txt", "_data.csv")
    data_lines = [header] + data_lines

    # Get rid of the 3 first columns
    data_lines = [",".join(l.split("\t")[3:]) for l in data_lines]
    data_lines = [l[:-2] + "\n" for l in data_lines]  # Remove trailing newline
    with open(csv_fpath, "w") as csv_file:
        csv_file.writelines(data_lines)

    rbr = pd.read_csv(csv_fpath, sep=",", header=0)

    return rbr


class SSP_class:
    "profil de célérité mesuré en un point [lon,lat] et échantillonné sur un ensemble d'immersions :depth"

    def segmente_data(self, verbose=False):
        diff = np.array(self.depth[4::]) - np.array(self.depth[:-4])
        flag = np.zeros(np.shape(self.depth))
        num_sonde = 1
        for ind in range(4, len(diff)):
            if flag[ind - 1] == 0:
                if (
                    (diff[ind - 4] < -0.2)
                    * (diff[ind - 3] < -0.2)
                    * (diff[ind - 2] < -0.2)
                    * (diff[ind - 1] < -0.2)
                    * (diff[ind - 0] < -0.2)
                ):
                    flag[ind] = -1 * num_sonde
            elif flag[ind - 1] == -1 * num_sonde:
                if (diff[ind] > -0.1) * (diff[ind + 1] > -0.1):
                    flag[ind] = 1 * num_sonde
                else:
                    flag[ind] = -1 * num_sonde
            elif flag[ind - 1] == 1 * num_sonde:
                if diff[ind] < 0:
                    flag[ind] = 0
                    num_sonde += 1
                else:
                    flag[ind] = 1 * num_sonde
        SSP_num = []
        for num in range(num_sonde):
            ind = np.where((flag == -num))[0]
            if verbose:
                print(ind)
            SSP_down = np.array(self.depth)[ind]
            if len(np.where((SSP_down <= -10))[0]) == 0:
                flag[ind] = 0
                ind = np.where((flag == num))[0]
                flag[ind] = 0
            else:
                flag2 = np.zeros(np.shape(flag))
                ind = np.where((flag == -num))[0]
                flag2[ind] = -1
                ind = np.where((flag == num))[0]
                flag2[ind] = 1
                SSP_num.append(flag2)
        self.ssp_num = SSP_num
        # fig,ax = plt.subplots(2,1,sharex=True)
        # ax[0].plot(diff)
        # ax[0].plot(self.depth)
        # ax[1].plot(self.ssp)
        # ax[0].plot(flag)
        # ax[0].plot(np.zeros(np.shape(self.depth)))
        # plt.show()


def SBE39_reader(file, folder, verbose=False):
    SourProf = SSP_class()
    f = open(folder + file)
    lines = f.readlines()
    f.close()
    d = []
    t = []
    Ture = []
    for j in range(len(lines)):
        # if lines[j] == 'Depth (m) - Temperature (°C) - Sound Velocity (m/s)\n':
        # if lines[j] == 'Depth(m) - Temperature(°C) - Conductivity(mS / cm) - Salinity(ppt) - Sound Velocity(m / s) - Density(kg / m³)\n':
        if lines[j] == "*END*\n":
            id_ = j
            print(lines[j])
    for j in range(id_ + 4, len(lines)):
        if verbose:
            print(lines[j])
        if len(lines[j].split(",")) == 4:
            time_fb = lines[j].split(",")[2] + " " + lines[j].split(",")[3][:-1]
            if verbose:
                print(time_fb)
            d.append(float(lines[j].split(",")[1]))
            dt = datetime.strptime(time_fb, " %d %b %Y  %H:%M:%S")
            t.append(dt.strftime("%Y-%m-%d %H:%M:%S.%f"))
            Ture.append(float(lines[j].split(",")[0]))
    SourProf.depth = d
    SourProf.time = t
    SourProf.Ture = Ture
    return SourProf


def SBE37_reader(cnv_filepath, verbose=False):
    ssp = SSP_class()

    with open(cnv_filepath, "r") as f:
        lines = f.readlines()

    # f = open(filepath)
    # lines = f.readlines()
    # f.close()

    # cel = []
    # d = []
    # t = []
    # Ture = []
    # sigma = []

    header_end_line_idx = np.where(np.array(lines) == "*END*\n")[0][0]
    if verbose:
        print(f"First data line index: {header_end_line_idx + 1}")
        print(f"First data line content: {lines[header_end_line_idx + 1]}")

    # for j in range(len(lines)):
    #     # if lines[j] == 'Depth (m) - Temperature (°C) - Sound Velocity (m/s)\n':
    #     # if lines[j] == 'Depth(m) - Temperature(°C) - Conductivity(mS / cm) - Salinity(ppt) - Sound Velocity(m / s) - Density(kg / m³)\n':
    #     if lines[j] == "*END*\n":
    #         id_ = j
    #         print(lines[j])

    col_names = []
    col_name_line_idx = [
        i for i, line in enumerate(lines) if line.startswith("# name 0")
    ][0]
    while lines[col_name_line_idx].startswith("# name"):
        col_name_line = lines[col_name_line_idx]
        col_name = col_name_line.split("=")[1].strip()
        col_name = col_name.split(":")[0].strip()
        col_names.append(col_name)
        col_name_line_idx += 1

    # Save data content to csv for simplicity
    header = " ".join(col_names)
    csv_fpath = cnv_filepath.replace(".cnv", "_data.csv")
    data_lines = lines[header_end_line_idx + 1 :]
    data_lines = [header + "\n"] + data_lines
    with open(csv_fpath, "w") as csv_file:
        csv_file.writelines(data_lines)

    # Now read the data using pandas for easier processing
    data = pd.read_csv(csv_fpath, sep="\s+", header=0)

    # Associate values to the SSP class attributes
    ssp.ssp = data["svDM"].values
    ssp.depth = -1 * data["depSM"].values
    ssp.lat = []
    ssp.lon = []
    ssp.time = (
        data["timeJ"].values - data["timeJ"].values[0]
    )  # Relative time from start
    ssp.Ture = data["tv290C"].values
    ssp.Sigma = data["cond0S/m"].values

    # for j in range(header_end_line_idx + 1, len(lines)):
    #     print(lines[j])
    #     print(lines[j].split(";"))

    #     d.append(-1 * float(lines[j].split(";")[8]))
    #     cel.append(float(lines[j].split(";")[7]))
    #     t.append(
    #         float(lines[j].split(";")[0])
    #         - float(lines[header_end_line_idx + 1].split(";")[0])
    #     )
    #     Ture.append(float(lines[j].split(";")[6]))
    #     sigma.append(float(lines[j].split(";")[4]))
    # SSP.ssp = cel
    # SSP.depth = d
    # SSP.lat = []
    # SSP.lon = []
    # SSP.time = t
    # SSP.Ture = Ture
    # SSP.Sigma = sigma
    # return SSP

    return ssp


if __name__ == "__main__":
    pass
