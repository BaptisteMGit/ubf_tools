#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   ssp_profiles.py
@Time    :   2025/05/12 11:21:40
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
import source.global_constants as g

from scipy.io import loadmat

# ======================================================================================================================
# Class to handle SSP profiles
# ======================================================================================================================


class SSPProfile:
    """
    Class to handle SSP profiles
    """

    def __init__(self, name: str = "", z: np.ndarray = None, c: np.ndarray = None):
        """
        Constructor
        :param name: Name of the profile
        :param z: Depth (m)
        :param c: Sound speed (m/s)
        """
        self.name = name
        self.z = z
        self.c = c

    def plot(self):
        """
        Plot the SSP profile
        """
        plt.plot(self.c, self.z)
        # Reverse y_axis
        plt.gca().invert_yaxis()
        plt.xlabel("Sound speed (m/s)")
        plt.ylabel("Depth (m)")
        plt.title(f"SSP profile {self.name}")

    def set_munk_profile(self, zmin, zmax, z_channel, nz=1000):
        """
        Set the Munk profile
        H. Munk, ``Sound channel in an exponentially stratified ocean with applications to SOFAR,'' J. Acoust. Soc. Am. 55, 220--226 (1974).
        :param zmin: Minimum depth (m)
        :param zmax: Maximum depth (m)
        :param z_channel: Depth of the channel (m)
        """
        # Assert zmin < z_channel < zmax
        if not (zmin < z_channel < zmax):
            raise ValueError("z_channel must be between zmin and zmax")

        eps = 0.00737
        # Define z vector
        self.z = np.linspace(zmin, zmax, nz)
        # Scaled depth
        b = 1300
        z_bar = 2 * (self.z - z_channel) / b
        # Munk profile
        self.c = g.c0 * (1 + eps * (z_bar - 1 + np.exp(-z_bar)))

        # Set name
        self.name = "Munk profile"

    def set_rhumrum_ssp(self, zmin, zmax, nz=None):

        # Make sure data path fits the current os 
        fpath = r"data\ssp\mmdpm\PVA_RR48\mmdpm_test_PVA_RR48_ssp.mat"
        fpath_parts = os.path.normpath(fpath).split("\\")
        rr48_ssp_path = os.path.join(g.project_root, *fpath_parts)

        # Load .mat file
        ssp = loadmat(rr48_ssp_path)["ssp"]
        zp = ssp["z"][0][0][:, 0]
        cp = ssp["c"][0][0][:, 0]

        if nz is None:
            nz = len(zp)

        # Interpolate on a regular vector
        z = np.linspace(zmin, zmax, nz)
        c = np.interp(z, zp, cp)

        self.z = z
        self.c = c


if __name__ == "__main__":
    ssp = SSPProfile()

    ssp.set_rhumrum_ssp(0, 5000, nz=1000)
    ssp.plot()
    plt.show()
