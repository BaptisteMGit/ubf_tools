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
import pandas as pd
import scipy.signal as sp
import matplotlib.pyplot as plt

from cst import RHO_W, C0
from skimage import measure  # Import for contour detection
from matplotlib.path import Path
from sklearn import preprocessing
from sklearn.cluster import KMeans
from itertools import combinations
from scipy.spatial import ConvexHull
from signals.signals import colored_noise
from publication.PublicationFigure import PubFigure
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_params import *


PubFigure(ticks_fontsize=22)


def params(debug=False, antenna_type="zhang"):
    # General parameters of the test case following Zhang et al. 2023
    depth = 150

    # Receivers are located on a hexagone at the same depth
    z_rcv = depth - 2  # 148 m according to Fig 3 of Zhang et al. 2023

    if antenna_type == "zhang":
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

    elif antenna_type == "random":
        # Define a maximum radius
        rmax = 1.5e3  # 1km radius
        nr = 6  # Same number of receivers as zhang
        x_rcv, y_rcv = load_random_antenna(rmax, nr)

    receivers = {"x": x_rcv, "y": y_rcv, "z": z_rcv}

    # Source
    # Estimated from figure 4 of Zhang et al. 2023
    # x_src_s1 = 3990
    # y_src_s1 = 6790

    # To fall exactly into a grid pixel
    x_src_s1 = 3900
    y_src_s1 = 6800

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

    if debug:
        # Test with non square area
        l_detection_area_x = 0.4e3  # Length of the detection area along x axis
        l_detection_area_y = 0.5e3  # Length of the detection area along y axis

        # Manual def (ugly)
        x_bott_left_corner = 3800
        y_bott_left_corner = 6700
        # d_rcv1_bott_left_corner = 7.5 * 1e3

    # Usual case
    else:
        # # Manual def (ugly)
        l_detection_area = 1e3  # Length of the detection area
        l_detection_area_x = l_detection_area_y = l_detection_area
        x_bott_left_corner = 3500
        y_bott_left_corner = 6400

    ### Original definition of the area location implies distance and angle from the antenna ###
    # l_detection_area_x = l_detection_area_y = l_detection_area
    # d_rcv1_bott_left_corner = (
    #     8 * 1e3
    # )  # Range from rcv_1 to the left bottom corner of the detection area

    # Derive grid pixels range from the origin
    # bearing_degree = 60  # Bearing from the receiver 1 to the left bottom corner of the detection area
    # x_bott_left_corner = (
    #     d_rcv1_bott_left_corner * np.cos(np.radians(bearing_degree)) + x_rcv[0]
    # )
    # y_bott_left_corner = (
    #     d_rcv1_bott_left_corner * np.sin(np.radians(bearing_degree)) + y_rcv[0]
    # )

    # # Assume 8km refers to the center of the detection area
    # x_offset = -l_detection_area / 2
    # y_offset = -l_detection_area / 2
    # x_bott_left_corner += x_offset
    # y_bott_left_corner += y_offset
    ### End of the area definition ###

    # Grid
    dx = 20
    dy = 20
    x_detection_area = np.arange(
        x_bott_left_corner, x_bott_left_corner + l_detection_area_x + dx, dx
    )
    y_detection_area = np.arange(
        y_bott_left_corner, y_bott_left_corner + l_detection_area_y + dy, dy
    )
    x_grid, y_grid = np.meshgrid(x_detection_area, y_detection_area)
    r_grid = np.zeros((len(x_rcv),) + x_grid.shape)
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


def generate_random_antenna(rmax, nr):

    theta = np.random.rand(nr) * 2 * np.pi
    theta = np.sort(theta)  # Order for easier interpretation
    rho = np.random.rand(nr) * rmax
    x_rcv = rho * np.cos(theta)
    y_rcv = rho * np.sin(theta)

    txt = np.c_[x_rcv, y_rcv]
    fpath = os.path.join(ROOT_DATA, f"random_antenna_rmax{rmax}m_nr{nr}.txt")
    np.savetxt(fpath, txt, fmt="%.2f")


def load_random_antenna(rmax, nr):
    fpath = os.path.join(ROOT_DATA, f"random_antenna_rmax{rmax}m_nr{nr}.txt")

    if os.path.exists(fpath):
        data = np.loadtxt(fpath, dtype=float)
        x_rcv, y_rcv = data[:, 0], data[:, 1]

    else:
        generate_random_antenna(rmax, nr)
        data = np.loadtxt(fpath, dtype=float)
        x_rcv, y_rcv = data[0, :], data[1, :]

    return x_rcv, y_rcv


def library_src_spectrum(f0=100, f1=500, fs=2000):
    """Library source signal spectrum : Zhang et al. 2023 -> LFM 100 - 500 Hz"""
    # Library source is defined as a LFM 100-500 Hz
    library_props = {
        "f0": f0,
        "f1": f1,
        "fs": fs,
        "T": 10,
        "phi": 0,
    }

    t = np.arange(0, library_props["T"], 1 / library_props["fs"])
    s = sp.chirp(
        t,
        library_props["f0"],
        library_props["T"],
        library_props["f1"],
        method="linear",
    )

    # Normalise signal to get unit variance
    s /= np.std(s)

    # # Add library signal power to the library props to calibrate the event signal
    # library_props["sig_var"] = np.var(s)
    # library_props["nt"] = len(s)

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


def generate_event_src_signal(T, fs):
    """
    Generate event source signal : Zhang et al. 2023 -> Gaussian noise

    The event signal is generated once and saved to a txt file to be used in the simulation
    """
    # Create white noise signal
    t, s_e = colored_noise(T, fs, "white")

    # Normalize to unit variance
    s_e /= np.std(s_e)

    # Save t, s to txt
    txt = np.c_[t, s_e]
    fpath = os.path.join(ROOT_DATA, "event_src_signal.txt")
    np.savetxt(fpath, txt, fmt="%.6f")

    nt = len(t)
    S_f_event = np.fft.rfft(s_e)
    f_event = np.fft.rfftfreq(nt, 1 / fs)

    # Save f, S to txt
    txt = np.c_[f_event, np.real(S_f_event), np.imag(S_f_event)]
    fpath = os.path.join(ROOT_DATA, "event_src_spectrum.txt")
    np.savetxt(fpath, txt, fmt="%.6f")


def event_src_spectrum(T, fs, stype="wn"):
    """
    Event source signal spectrum : Zhang et al. 2023 -> Gaussian noise
    The variance of the signal is set to the variance of the library signal (assuming both signal have same power)

    The event signal is generated once and saved to a txt file to be used in the simulation

    Parameters
    ----------
    T : float
        Duration of the signal
    fs : int
        Sampling frequency
    stype : str
        Type of event source signal
        - wn : White noise
        - lfm : Linear Frequency Modulation

    Returns
    -------
    event_props : dict
        Event source properties
    S_f_event : np.ndarray
        Event source signal spectrum
    f_event : np.ndarray
        Frequency vector

    Examples
    --------
    event_props, S_f_event, f_event = event_src_spectrum(T=10, fs=2000, stype="wn")

    """

    # generate_event_src_signal(T, fs)
    # # Event source is defined as a Gaussian noise

    # # The easiest, yet not elegant way to calibrate the event signal is to set the variance of the event signal to the variance of the library signal in the time domain
    # t = np.arange(0, T, 1 / fs)
    # nt = len(t)
    # s_e = np.random.normal(0, 1, nt)  # Unit var signal

    if stype == "wn":
        event_props = {}
        # Load event signal
        fpath = os.path.join(ROOT_DATA, "event_src_signal.txt")
        if not os.path.exists(fpath):
            generate_event_src_signal(T, fs)

        t, s_e = np.loadtxt(fpath, unpack=True)

        # Load event spectrum
        fpath = os.path.join(ROOT_DATA, "event_src_spectrum.txt")
        f_event, S_f_event_real, S_f_event_imag = np.loadtxt(fpath, unpack=True)
        S_f_event = S_f_event_real + 1j * S_f_event_imag
        stype_name = "Gaussian white noise"

    elif stype == "lfm":
        event_props, S_f_event, f_event, _ = library_src_spectrum(fs=fs)

        # Derive t, s_e by inverse fourier transform
        s_e = np.fft.irfft(S_f_event)
        t = np.arange(0, event_props["T"], 1 / event_props["fs"])
        stype_name = "LFM Chirp"

    # nt = len(t)
    # S_f_event = np.fft.rfft(s_e)
    # f_event = np.fft.rfftfreq(nt, 1 / fs)

    plt.figure()
    plt.plot(t, s_e)
    plt.xlabel(r"$\textrm{Time [s]}$")
    plt.ylabel(r"$s(t)$")
    plt.title(f"Event source signal = {stype_name}")
    fpath = os.path.join(ROOT_IMG, f"event_source_signal.png")
    plt.savefig(fpath)

    plt.figure()
    plt.plot(f_event, np.abs(S_f_event))
    plt.xlabel(r"$\textrm{Frequency [Hz]}$")
    plt.ylabel(r"$|S(f)|$")
    plt.title(f"Event source signal = {stype_name}")
    fpath = os.path.join(ROOT_IMG, f"event_source_spectrum.png")
    plt.savefig(fpath)
    # plt.close("all")

    # Compute and plot DSP
    dsp = sp.welch(s_e, fs, nperseg=1024, noverlap=512, nfft=1024)
    plt.figure()
    plt.plot(dsp[0], 10 * np.log10(dsp[1]))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [dB/Hz]")
    plt.title(f"Event source signal = {stype_name}")
    # plt.yscale("log")
    plt.savefig(os.path.join(ROOT_IMG, "event_source_dsp.png"))
    # S_f_event = np.ones_like(f)

    # Plot noise signal derived from inverse FFT
    s_e_ifft = np.fft.irfft(S_f_event)
    plt.figure()
    plt.plot(t, s_e_ifft)
    plt.xlabel("Time [s]")
    plt.ylabel("s(t)")
    plt.title(f"Event source signal = {stype_name}")
    plt.savefig(os.path.join(ROOT_IMG, "event_source_signal_ifft.png"))
    plt.close("all")

    return event_props, S_f_event, f_event


def find_mainlobe(ds_fa):

    mainlobe_contours = {}

    for dist in ["d_gcc", "d_rtf"]:

        amb_surf = ds_fa[dist]

        ### 1) K-means approach ###

        # Reshape ambiguity surface to 1D array
        amb_surf_1d = amb_surf.values.flatten()

        # Apply K-means clustering
        n_clusters = 7
        x_coord, y_coord = np.meshgrid(ds_fa.x.values, ds_fa.y.values)
        X = np.vstack(
            [x_coord.flatten(), y_coord.flatten(), amb_surf_1d]
        )  # 3 Columns x, y, S(x, y)
        # X = amb_surf_1d.reshape(-1, 1)
        X_norm = preprocessing.normalize(X).T
        X_norm[:, 0:2] *= 1  # Increase the weight of the spatial coordinates
        # X_norm = X.T
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        kmeans.fit(X_norm)

        # # K means using only the ambiguity surface
        # n_clusters = 7
        # X = amb_surf_1d.reshape(-1, 1)
        # X_norm = preprocessing.normalize(X)
        # kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        # kmeans.fit(X_norm)

        # 1.2) Segmentation
        # Reshape labels to 2D array
        labels = kmeans.labels_.reshape(amb_surf.shape)

        # 1.3) Plot
        f, ax = plt.subplots(1, 1, figsize=(5, 5))

        # Define a discrete colormap with n_clusters colors
        cmap = plt.get_cmap("jet", n_clusters)
        # im = ax.pcolormesh(
        #     ds_fa["x"].values * 1e-3,
        #     ds_fa["y"].values * 1e-3,
        #     labels.T,
        #     cmap=cmap,
        # )

        # # Add colorbar with n_clusters ticks
        # cbar = plt.colorbar(
        #     im, ax=ax, label=r"$\textrm{Class}$", ticks=range(n_clusters)[::2]
        # )
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
        # im = ax.pcolormesh(
        #     ds_fa["x"].values * 1e-3,
        #     ds_fa["y"].values * 1e-3,
        #     amb_surf.values.T,
        #     cmap="jet",
        #     vmin=-10,
        #     vmax=0,
        # )

        # # # Highligh mainlobe pixels
        # ax.plot(
        #     ds_fa["x"].values[contour[:, 0].astype(int)] * 1e-3,
        #     ds_fa["y"].values[contour[:, 1].astype(int)] * 1e-3,
        #     color="k",
        #     linewidth=2,
        # )
        # # Add colorbar
        # cbar = plt.colorbar(im, ax=ax, label=r"$\textrm{[dB]}$")
        # ax.scatter(
        #     x_src_hat * 1e-3,
        #     y_src_hat * 1e-3,
        #     facecolors="none",
        #     edgecolors="k",
        #     label="Estimated source position",
        #     s=20,
        #     linewidths=3,
        # )

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


def estimate_msr(ds_fa, plot=False, root_img=None, verbose=False):
    # Derive mainlobe to side lobe ratio

    if plot:
        f, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        cmap = "jet"
        vmin = -10
        vmax = 0

        xticks_pos_km = [3.5, 4.0, 4.5]
        yticks_pos_km = [6.4, 6.9, 7.4]
        xticks_pos_m = [xt * 1e3 for xt in xticks_pos_km]
        yticks_pos_m = [yt * 1e3 for yt in yticks_pos_km]
        # xticks_label_km = [xt for xt in xticks_pos_km]
        # yticks_label_km = [yt for yt in yticks_pos_km]
        # xticks_label_km = [f"${xt:.1f}$" for xt in xticks_pos_km]
        # yticks_label_km = [f"${yt:.1f}$" for yt in yticks_pos_km]

    # Find mainlobe contours
    mainlobe_contours = find_mainlobe(ds_fa)
    msr = {}
    pos_hat = {}
    for i, dist in enumerate(["d_gcc", "d_rtf"]):
        if plot:
            ax = axs[i]

        amb_surf = ds_fa[dist]

        # Source pos
        x_idx, y_idx = np.unravel_index(np.argmax(amb_surf.values), amb_surf.shape)
        x_src_hat = amb_surf.x[x_idx]
        y_src_hat = amb_surf.y[y_idx]
        pos_hat[dist] = {"x": x_src_hat.values, "y": y_src_hat.values}

        mainlobe_mask = np.zeros_like(amb_surf.values, dtype=bool)

        contour = mainlobe_contours[dist]

        # Convert contour indices to integers
        contour_x_idx = np.round(contour[:, 0]).astype(int)
        contour_y_idx = np.round(contour[:, 1]).astype(int)

        # Ensure indices stay within valid bounds
        contour_x_idx = np.clip(contour_x_idx, 0, ds_fa["x"].size - 1)
        contour_y_idx = np.clip(contour_y_idx, 0, ds_fa["y"].size - 1)

        contour_points = np.c_[
            ds_fa["x"].values[contour_x_idx], ds_fa["y"].values[contour_y_idx]
        ]

        # Step 3: Compute convex hull
        try:
            hull = ConvexHull(contour_points)
            hull_points = contour_points[hull.vertices]  # Get convex hull vertices

            # Step 4: Convert convex hull to a polygon
            poly_path = Path(hull_points)

            # # Convert contour indices to actual x and y values
            # poly_path = Path(
            #     np.c_[ds_fa["x"].values[contour_x_idx], ds_fa["y"].values[contour_y_idx]]
            # )

            # Step 3: Create a grid of coordinates
            X, Y = np.meshgrid(ds_fa["x"].values, ds_fa["y"].values, indexing="ij")

            # Step 4: Flatten the grid and check which points are inside the polygon
            points = np.c_[X.ravel(), Y.ravel()]  # Flatten grid coordinates
            inside = poly_path.contains_points(points)

            # Step 5: Reshape the result into the original grid shape and update the mask
            mainlobe_mask |= inside.reshape(
                X.shape
            )  # Use logical OR to combine multiple contours
            # mainlobe_mask = mainlobe_mask.T
            plot_hull = True

        except:
            # Handle case where it impossible to define a contour
            mainlobe_mask[x_idx, y_idx] = 1  # Mainlobe = single pixel
            plot_hull = False

        # Compute mainlobe to side lobe ratio
        main_lobe = np.max(amb_surf.values)
        side_lobe = np.max(amb_surf.values[~mainlobe_mask])
        # print(f"Mainlobe lvl = {main_lobe:.2f} dB")
        # print(f"Side lobe lvl = {side_lobe:.2f} dB")
        msr[dist] = -(main_lobe - side_lobe)  # MSR = mainlobe_dB - side_lobe_dB

        if verbose:
            print(f"MSR {dist} : {msr[dist]:.2f} dB")

        if plot:
            # Plot ambiguity surface without mainlobe pixels
            amb_surf_without_mainlobe = amb_surf.copy(deep=True)
            amb_surf_without_mainlobe = amb_surf_without_mainlobe.values
            amb_surf_without_mainlobe[mainlobe_mask] = np.nan

            im = ax.pcolormesh(
                ds_fa["x"].values,
                ds_fa["y"].values,
                amb_surf_without_mainlobe.T,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            ax.plot(
                ds_fa["x"].values[contour[:, 0].astype(int)],
                ds_fa["y"].values[contour[:, 1].astype(int)],
                color="k",
                linewidth=2,
                # label="Mainlobe Boundary" if i == 0 else None,
            )

            # Add convex hull to the plot
            if plot_hull:
                hull_points = np.vstack([hull_points, hull_points[0]])

                ax.plot(
                    hull_points[:, 0],
                    hull_points[:, 1],
                    "r-",
                    linewidth=2,
                    label="Mainlobe Convex Hull",
                )

            ax.scatter(
                x_src_hat,
                y_src_hat,
                facecolors="none",
                edgecolors="k",
                label="Estimated source position",
                s=20,
                linewidths=3,
            )

            ax.set_title(f"Full array")
            ax.set_xlabel(r"$x \textrm{[m]}$")
            if i == 0:
                ax.set_ylabel(r"$y \, \textrm{[m]}$")
            else:
                ax.set_ylabel("")

            # # Set xticks
            ax.set_xticks(xticks_pos_m)
            ax.set_yticks(yticks_pos_m)
            # ax.set_xticklabels(xticks_label_km, fontsize=22)
            # ax.set_yticklabels(yticks_label_km, fontsize=22)

            # ax.set_xticks(
            #     [3.500, 4.000, 4.500],
            # )
            # ax.set_xticklabels([3.500, 4.000, 4.500], fontsize=22)
            # ax.set_yticks([6.400, 6.900, 7.400])
            # ax.set_yticklabels([6.400, 6.900, 7.400], fontsize=22)

    if plot:
        # Save figure
        fpath = os.path.join(root_img, "loc_zhang2023_fig5_nomainlobe.png")
        plt.savefig(fpath, dpi=300)
        plt.close("all")

    return msr, pos_hat


def get_rcv_couples(idx_receivers):
    """
    Get all possible receiver couples
    """
    # rcv_couples = []
    # for i in idx_receivers:
    #     for j in idx_receivers:
    #         if j > i:
    #             rcv_couples.append([i, j])
    # rcv_couples = np.array(rcv_couples)

    rcv_couples = np.array(list(combinations(idx_receivers, 2)))
    rcv_couples = np.atleast_2d(rcv_couples)  # In case only two receivers

    return rcv_couples


def get_subarrays(nr_fullarray, nr_subarray):
    """Find all sub array containing nr_subarray within the fullarray"""

    fa_idx = np.arange(nr_fullarray)
    # Find all combinations of nr_subarray elements (elements order does not matter)
    subarrays = np.array(list(combinations(fa_idx, nr_subarray)))
    return subarrays


def get_array_label(rcv_idx):
    array_label = "_".join([f"s{i+1}" for i in rcv_idx])
    return array_label


def build_subarrays_args(subarrays_list):
    subarrays_args = {
        index: {
            "idx_rcv": subarrays_list[index],
            "array_label": get_array_label(subarrays_list[index]),
            "msr_filepath": None,
            "dr_pos_filepath": None,
        }
        for index in range(len(subarrays_list))
    }
    return subarrays_args


def init_msr_file(folder, run_mode, subarrays_args):

    root_msr = os.path.join(ROOT_DATA, folder, "msr")
    if not os.path.exists(root_msr):
        os.makedirs(root_msr)

    for index, sa_item in subarrays_args.items():

        msr_txt_filepath = os.path.join(
            root_msr, f"msr_snr_{sa_item['array_label']}.txt"
        )

        if (
            not os.path.exists(msr_txt_filepath) or run_mode == "w"
        ):  # To avoid writting over existing file
            header_line = "snr i_mc d_gcc d_rtf\n"
            with open(msr_txt_filepath, "w") as f:
                f.write(header_line)

        sa_item["msr_filepath"] = msr_txt_filepath


def init_dr_file(folder, run_mode, subarrays_args):

    root_dr = os.path.join(ROOT_DATA, folder, "dr_pos")
    if not os.path.exists(root_dr):
        os.makedirs(root_dr)

    for index, sa_item in subarrays_args.items():

        dr_txt_filepath = os.path.join(
            root_dr, f"dr_pos_snr_{sa_item['array_label']}.txt"
        )

        if (
            not os.path.exists(dr_txt_filepath) or run_mode == "w"
        ):  # To avoid writting over existing file
            header_line = "snr i_mc dr_gcc dr_rtf\n"
            with open(dr_txt_filepath, "w") as f:
                f.write(header_line)

        sa_item["dr_pos_filepath"] = dr_txt_filepath


def load_msr_rmse_res_subarrays(subarrays_list, snrs, dx, dy):
    folder = f"from_signal_dx{dx}m_dy{dy}m"
    subarrays_args = build_subarrays_args(subarrays_list)

    init_dr_file(folder, run_mode="a", subarrays_args=subarrays_args)
    init_msr_file(folder, run_mode="a", subarrays_args=subarrays_args)

    msr_mu = []
    msr_sig = []
    dr_mu = []
    dr_sig = []
    rmse_ = []

    for sa_idx, sa_item in subarrays_args.items():
        msr_txt_filepath = sa_item["msr_filepath"]
        dr_txt_filepath = sa_item["dr_pos_filepath"]

        # Load msr and position error results
        msr = pd.read_csv(msr_txt_filepath, sep=" ")
        dr = pd.read_csv(dr_txt_filepath, sep=" ")

        # Keep only snrs of interest
        msr = msr[msr["snr"].isin(snrs)]
        dr = dr[dr["snr"].isin(snrs)]

        # Compute mean and std of msr for each snr
        msr_mean = msr.groupby("snr").mean()
        msr_std = msr.groupby("snr").std()
        msr_mu.append(msr_mean)
        msr_sig.append(msr_std)

        # Compute mean and std of position error for each snr
        dr_mean = dr.groupby("snr").mean()
        dr_std = dr.groupby("snr").std()
        dr_mu.append(dr_mean)
        dr_sig.append(dr_std)

        dr["d_gcc"] = dr["dr_gcc"] ** 2
        dr["d_rtf"] = dr["dr_rtf"] ** 2
        mse = dr.groupby("snr").mean()
        rmse = np.sqrt(mse)
        rmse_.append(rmse)

    return msr_mu, msr_sig, dr_mu, dr_sig, rmse_


if __name__ == "__main__":
    # params(debug=False, antenna_type="random")
    # get_subarrays(nr_fullarray=6, nr_subarray=3)
    get_rcv_couples(idx_receivers=np.arange(6))
