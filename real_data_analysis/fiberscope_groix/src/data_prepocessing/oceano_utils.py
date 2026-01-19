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


def rbr_reader(file, folder):
    rbr = pd.read_csv(folder + file, sep=",")  # , engine="python")
    return rbr


class SSP_class:
    "profil de célérité mesuré en un point [lon,lat] et échantillonné sur un ensemble d'immersions :depth"

    def segmente_data(self):
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


if __name__ == "__main__":
    pass
