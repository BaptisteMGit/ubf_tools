#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   zhang_utils.py
@Time    :   2025/01/27 11:57:36
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Misc functions for Zhang et al 2023 testcase 
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr
import scipy.signal as sp
import matplotlib.pyplot as plt

from cst import RHO_W, C0
from signals.signals import lfm_chirp
from misc import cast_matrix_to_target_shape, mult_along_axis
from publication.PublicationFigure import PubFigure


from skimage import measure  # Import for contour detection
from sklearn.cluster import KMeans
from sklearn import preprocessing

PubFigure(ticks_fontsize=22)

ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\zhang_et_al_testcase"
ROOT_DATA = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\zhang_et_al_testcase\data"
ROOT_IMG = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\zhang_et_al_2023"

# Minimum value to replace 0 before converting metrics to dB scale
MIN_VAL_LOG = 1e-5

# =====================================================================================================================
# Functions
# ======================================================================================================================


def params():
    # General parameters of the test case following Zhang et al. 2023
    depth = 150

    # Receivers are located on a hexagone at the same depth
    z_rcv = depth - 2  # 148 m according to Fig 3 of Zhang et al. 2023
    a_hex = 250  # Hexagone side length
    ri_hex = np.sqrt(3) * a_hex / 2  # Hexagone internal radius
    # Coordinates relative to the center of the hexagone
    x_rcv = np.array([-a_hex / 2, a_hex / 2, a_hex, a_hex / 2, -a_hex / 2, -a_hex])
    y_rcv = np.array([-ri_hex, -ri_hex, 0, ri_hex, ri_hex, 0])

    # Translate the orginin of the frame to the origin used in Zhang et al. 2023
    x_origin_zhang = 0
    y_origin_zhang = ri_hex
    x_rcv += x_origin_zhang
    y_rcv += y_origin_zhang

    receivers = {"x": x_rcv, "y": y_rcv, "z": z_rcv}

    # Source
    # Estimated from figure 4 of Zhang et al. 2023
    x_src_s1 = 3990
    y_src_s1 = 6790

    # x_src_s1 = 4500
    # y_src_s1 = 7400
    # Switch to the frame used in Zhang et al. 2023
    # x_src_0 = np.array(x_rcv[0] + x_src_s1)  # OS = OS1 + S1S
    # y_src_0 = np.array(y_rcv[0] + y_src_s1)  # OS = OS1 + S1S

    # Position defined in the frame used in Zhang et al. 2023
    x_src_0 = x_src_s1
    y_src_0 = y_src_s1

    # Derive range from each receiver
    r_src_rcv = np.sqrt((x_src_0 - x_rcv) ** 2 + (y_src_0 - y_rcv) ** 2)
    z_src = 5
    source = {
        "x": np.round(x_src_0, 0),
        "y": np.round(y_src_0, 0),
        "z": z_src,
        "r": r_src_rcv,
    }

    # Detection area
    l_detection_area = 1e3  # Length of the detection area
    d_rcv1_bott_left_corner = (
        8 * 1e3
    )  # Range from rcv_1 to the left bottom corner of the detection area

    # Derive grid pixels range from the origin
    bearing_degree = 60  # Bearing from the receiver 1 to the left bottom corner of the detection area
    x_bott_left_corner = (
        d_rcv1_bott_left_corner * np.cos(np.radians(bearing_degree)) + x_rcv[0]
    )
    y_bott_left_corner = (
        d_rcv1_bott_left_corner * np.sin(np.radians(bearing_degree)) + y_rcv[0]
    )

    # # Assume 8km refers to the center of the detection area
    # x_offset = -l_detection_area / 2
    # y_offset = -l_detection_area / 2
    # x_bott_left_corner += x_offset
    # y_bott_left_corner += y_offset

    # Manual def (ugly)
    x_bott_left_corner = 3500
    y_bott_left_corner = 6400

    # Grid
    dx = 20
    dy = 20
    x_detection_area = np.arange(
        x_bott_left_corner, x_bott_left_corner + l_detection_area, dx
    )
    y_detection_area = np.arange(
        y_bott_left_corner, y_bott_left_corner + l_detection_area, dy
    )
    x_grid, y_grid = np.meshgrid(x_detection_area, y_detection_area)
    r_grid = np.zeros((len(x_rcv), len(x_grid), len(y_grid)))
    for i in range(len(x_rcv)):
        r_grid[i] = np.sqrt((x_grid - x_rcv[i]) ** 2 + (y_grid - y_rcv[i]) ** 2)

    grid = {
        "x": x_grid,
        "y": y_grid,
        "r": r_grid,
        "rmax": np.ceil(np.max(r_grid) * 1e-3) * 1e3,
        "dx": dx,
        "dy": dy,
    }

    # Frequencies
    f0 = 100
    f1 = 500
    nf = 801
    freqs = np.linspace(f0, f1, nf)
    frequency = {"f0": f0, "f1": f1, "nf": nf, "freqs": freqs}

    # Bottom
    bott_hs_properties = {
        "rho": 2.5 * RHO_W * 1e-3,  # Density (g/cm^3)
        # "c_p": 1500,  # P-wave celerity (m/s)
        "c_p": 4650,  # P-wave celerity (m/s)
        "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
        "a_p": 0.01,  # Compression wave attenuation (dB/wavelength)
        "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
        "z": 275,
    }

    return depth, receivers, source, grid, frequency, bott_hs_properties


def library_src_spectrum():
    """Library source signal spectrum : Zhang et al. 2023 -> LFM 100 - 500 Hz"""
    # Library source is defined as a LFM 100-500 Hz
    library_props = {
        "f0": 100,
        "f1": 500,
        "fs": 2000,
        "T": 10,
        "phi": 0,
    }

    # s, t = lfm_chirp(
    #     library_props["f0"],
    #     library_props["f1"],
    #     library_props["fs"],
    #     library_props["T"],
    # )

    # pad_time_s = 1
    # npad = pad_time_s * library_props["fs"]
    t = np.arange(0, library_props["T"], 1 / library_props["fs"])
    s = sp.chirp(
        # t[npad:-npad],
        t,
        library_props["f0"],
        library_props["T"],
        library_props["f1"],
        method="linear",
        # method="hyperbolic",
    )
    # Pad with 0 before and after
    # s = np.pad(s, (npad, npad))
    # Apply window
    # s *= sp.windows.hann(len(s))

    S_f_library = np.fft.rfft(s)
    f_library = np.fft.rfftfreq(len(s), 1 / library_props["fs"])

    # Plot signal and spectrum
    plt.figure()
    plt.plot(t, s)
    plt.xlabel(r"$\textrm{Time [s]}$")
    plt.ylabel(r"$s(t)$")
    fpath = os.path.join(ROOT_IMG, f"library_source_signal.png")
    plt.savefig(fpath)

    plt.figure()
    plt.plot(f_library, np.abs(S_f_library))
    plt.xlabel(r"$\textrm{Frequency [Hz]}$")
    plt.ylabel(r"$|S(f)|$")
    fpath = os.path.join(ROOT_IMG, f"library_source_spectrum.png")
    plt.savefig(fpath)
    plt.close("all")
    # plt.show()

    # Interp library source spectrum at desired frequencies
    # S_f_library = np.interp(f, f_library, S_f_library)

    # Keep frequencies between 50 and 550 Hz
    idx_in_band = (f_library >= 50) & (f_library <= 550)

    # Right frequencies in band in a
    f_in_band = f_library[idx_in_band]
    txt = " ".join([f"{f:.2f}" for f in f_in_band])
    fpath = os.path.join(ROOT, "tmp", "f_in_band.txt")
    with open(fpath, "w") as f:
        f.write(txt)
    # np.savetxt(fpath, f_in_band, fmt="%.2f")

    # f_library = f_library[idx]
    # S_f_library = S_f_library[idx]

    return library_props, S_f_library, f_library, idx_in_band


def event_src_spectrum(f):
    """Event source signal spectrum : Zhang et al. 2023 -> Gaussian noise"""

    event_props = {}
    # Event source is defined as a Gaussian noise
    S_f_event = np.ones_like(f)

    return event_props, S_f_event


def find_mainlobe(ds_fa):

    mainlobe_contours = {}

    for dist in ["d_gcc", "d_rtf"]:

        amb_surf = ds_fa[dist]

        ### 1) Double k-means approach ###
        # 1.1) K-means clustering on the ambiguity surface level

        # Reshape ambiguity surface to 1D array
        amb_surf_1d = amb_surf.values.flatten()

        # Apply K-means clustering
        n_clusters = 7
        x_coord, y_coord = np.meshgrid(ds_fa.x.values, ds_fa.y.values)
        X = np.vstack(
            [x_coord.flatten(), y_coord.flatten(), amb_surf_1d]
        )  # 3 Columns x, y, S(x, y)
        X_norm = preprocessing.normalize(X).T
        X_norm[:, 0:2] *= 2  # Increase the weight of the spatial coordinates
        # X_norm = X.T
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        kmeans.fit(X_norm)

        # 1.2) Segmentation
        # Reshape labels to 2D array
        labels = kmeans.labels_.reshape(amb_surf.shape)

        # 1.3) Plot
        f, ax = plt.subplots(1, 1, figsize=(5, 5))

        # Define a discrete colormap with n_clusters colors
        cmap = plt.get_cmap("jet", n_clusters)
        im = ax.pcolormesh(
            ds_fa["x"].values * 1e-3,
            ds_fa["y"].values * 1e-3,
            labels.T,
            cmap=cmap,
        )

        # Add colorbar with n_clusters ticks
        cbar = plt.colorbar(
            im, ax=ax, label=r"$\textrm{Class}$", ticks=range(n_clusters)[::2]
        )
        ax.set_title(f"Full array")
        ax.set_xlabel(r"$x \textrm{[km]}$")
        ax.set_ylabel(r"$y \, \textrm{[km]}$")
        ax.set_xticks([3.500, 4.000, 4.500])
        ax.set_yticks([6.400, 6.900, 7.400])

        # Save figure
        fpath = os.path.join(ROOT_IMG, f"loc_zhang2023_fig5_segmentation_{dist}.png")
        plt.savefig(fpath, dpi=300, bbox_inches="tight")

        # 1.4) Select the class corresponding to the estimated position defined by the maximum of the ambiguity surface
        x_idx, y_idx = np.unravel_index(np.argmax(amb_surf.values), amb_surf.shape)
        x_src_hat = amb_surf.x[x_idx]
        y_src_hat = amb_surf.y[y_idx]
        src_hat_class = labels[x_idx, y_idx]

        # Find contours of src_hat_class and select the contour corresponding to the estimated position
        contours = measure.find_contours(labels == src_hat_class, level=0.5)
        for contour in contours:
            # Check if src_hat is within the contour
            idx_x_min = np.min(contour[:, 0].astype(int))
            idx_x_max = np.max(contour[:, 0].astype(int))
            idx_y_min = np.min(contour[:, 1].astype(int))
            idx_y_max = np.max(contour[:, 1].astype(int))
            if (idx_x_min <= x_idx <= idx_x_max) and (idx_y_min <= y_idx <= idx_y_max):
                break

        mainlobe_contours[dist] = contour

        # 1.5) Plot ambiguity surface and highligh pixels falling into the src_hat_class
        f, ax = plt.subplots(1, 1, figsize=(5, 5))

        # Define a discrete colormap with n_clusters colors
        im = ax.pcolormesh(
            ds_fa["x"].values * 1e-3,
            ds_fa["y"].values * 1e-3,
            amb_surf.values.T,
            cmap="jet",
            vmin=-10,
            vmax=0,
        )

        # # Highligh mainlobe pixels
        ax.plot(
            ds_fa["x"].values[contour[:, 0].astype(int)] * 1e-3,
            ds_fa["y"].values[contour[:, 1].astype(int)] * 1e-3,
            color="k",
            linewidth=2,
        )
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label=r"$\textrm{[dB]}$")
        ax.scatter(
            x_src_hat * 1e-3,
            y_src_hat * 1e-3,
            facecolors="none",
            edgecolors="k",
            label="Estimated source position",
            s=20,
            linewidths=3,
        )

        ax.set_title(f"Full array")
        ax.set_xlabel(r"$x \textrm{[km]}$")
        ax.set_ylabel(r"$y \, \textrm{[km]}$")
        ax.set_xticks([3.500, 4.000, 4.500])
        ax.set_yticks([6.400, 6.900, 7.400])

        # Save figure
        fpath = os.path.join(
            ROOT_IMG, f"loc_zhang2023_fig5_segmentation_highlight_{dist}.png"
        )
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close("all")

    return mainlobe_contours


if __name__ == "__main__":
    pass
