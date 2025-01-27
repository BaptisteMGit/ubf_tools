#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   zhang_waveguide.py
@Time    :   2025/01/15 13:55:32
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Test case following Zhang et al. 2023
Zhang, T., Zhou, D., Cheng, L., & Xu, W. (2023). Correlation-based passive localization: Linear system modeling and sparsity-aware optimization. 
The Journal of the Acoustical Society of America, 154(1), 295–306. https://doi.org/10.1121/10.0020154
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
from propa.kraken_toolbox.run_kraken import readshd, run_kraken_exec, run_field_exec
from propa.rtf.rtf_utils import D_hermitian_angle_fast, normalize_metric_contrast

from skimage import measure  # Import for contour detection
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.spatial import ConvexHull
from matplotlib.path import Path


PubFigure()
ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\zhang_et_al_testcase"
ROOT_DATA = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\zhang_et_al_testcase\data"
ROOT_IMG = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\zhang_et_al_2023"

# Minimum value to replace 0 before converting metrics to dB scale
MIN_VAL_LOG = 1e-5


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


def save_simulation_netcdf():

    # Load params
    depth, receivers, source, grid, frequency, _ = params()

    # Frequency
    f = frequency["freqs"]

    # Read shd from previously run kraken
    working_dir = os.path.join(ROOT, "tmp")
    os.chdir(working_dir)
    shdfile = r"testcase_zhang2023.shd"

    _, _, _, _, read_freq, _, field_pos, pressure_field = readshd(
        filename=shdfile, freq=f
    )

    tf = np.squeeze(pressure_field, axis=(1, 2, 3))  # (nf, nr)

    # Define xarray dataset to store results
    tf_zhang = xr.Dataset(
        data_vars=dict(
            tf_real=(
                ["f", "r"],
                np.real(tf),
            ),
            tf_imag=(["f", "r"], np.imag(tf)),
        ),
        coords={
            "f": f,
            "r": field_pos["r"]["r"],
        },
    )

    # Save waveguide transfert functions as netcdf
    fpath = os.path.join(ROOT_DATA, "tf_zhang.nc")
    tf_zhang.to_netcdf(fpath)
    tf_zhang.close()


def build_signal():
    """Derive received signal(library / event)"""

    # Load params
    depth, receivers, source, grid, frequency, _ = params()

    # Load gridded dataset
    dx = np.diff(grid["x"][0, :])[0]
    dy = np.diff(grid["y"][:, 0])[0]
    fname = f"tf_zhang_grid_dx{dx}m_dy{dy}m.nc"
    fpath = os.path.join(ROOT_DATA, fname)
    ds = xr.open_dataset(fpath)

    # Load library spectrum
    f = ds.f.values
    # library_props, S_f_library = library_src_spectrum(f)
    library_props, S_f_library, f_library, idx_in_band = library_src_spectrum()

    # Load event spectrum
    _, S_f_event = event_src_spectrum(f)

    df = ds.df
    shape_grid = (
        ds.sizes["idx_rcv"],
        ds.sizes["f"],
        ds.sizes["x"],
        ds.sizes["y"],
    )

    # Derive delay for each receiver
    delay_rcv = []
    for i_rcv in range(len(receivers["x"])):
        r_grid = grid["r"][i_rcv].flatten()
        tau_rcv = r_grid / C0
        tau_rcv = tau_rcv.reshape((ds.sizes["x"], ds.sizes["y"]))
        delay_rcv.append(tau_rcv)

    delay_rcv = np.array(delay_rcv)

    # Add delay to dataset
    ds["delay_rcv"] = (["idx_rcv", "x", "y"], delay_rcv)

    # Same delay is applied to each receiver : the receiver with the minimum delay is taken as the time reference
    # (we are only interested on relative time difference)
    tau = ds.delay_rcv.min(dim="idx_rcv")
    # Cast tau to grid shape
    tau = cast_matrix_to_target_shape(tau, ds.tf_real.shape[1:])

    y_t_grid = []
    for i_rcv in range(len(receivers["x"])):

        tf_grid = ds.tf_real.sel(idx_rcv=i_rcv) + 1j * ds.tf_imag.sel(idx_rcv=i_rcv)

        # Derive received spectrum (Y = SH)
        k0 = 2 * np.pi * ds.f / C0
        norm_factor = np.exp(1j * k0) / (4 * np.pi)

        y_f = mult_along_axis(tf_grid, S_f_library * norm_factor, axis=0)

        # Derive delay factor to take into account the propagation time
        tau_vec = mult_along_axis(tau, ds.f, axis=0)
        delay_f = np.exp(1j * 2 * np.pi * tau_vec)

        # Apply delay
        y_f *= delay_f

        # FFT inv to get signal
        y_t = np.fft.irfft(y_f, axis=0)

        y_t_grid.append(y_t)

        # plt.figure()
        # plt.plot(y_t[:, 0, 0])
        # plt.savefig(f"test_{i_rcv}.png")
        # plt.show()

    y_t_grid = np.array(y_t_grid)

    # Add to dataset
    t = np.arange(0, library_props["T"], 1 / library_props["fs"])
    ds.coords["t"] = t
    ds["s_l"] = (["idx_rcv", "t", "x", "y"], y_t_grid)

    # Drop vars to reduce size
    ds = ds.drop_vars(["tf_real", "tf_imag", "f"])

    # Save dataset
    fname = f"zhang_library_dx{dx}m_dy{dy}m.nc"
    fpath = os.path.join(ROOT_DATA, fname)
    ds.to_netcdf(fpath)


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


def build_tf_dataset():

    library_props, S_f_library, freq, idx_in_band = library_src_spectrum()

    f = freq[idx_in_band]

    # For too long frequencies vector field fails to compute -> we will iterate over frequency subband to compute the transfert function
    n_subband = 900
    i_subband = 1
    f0 = f[0]
    f1 = f[n_subband]

    # Load tf to get range vector information
    fpath = os.path.join(ROOT_DATA, "tf_zhang.nc")
    ds = xr.open_dataset(fpath)
    nr = ds.sizes["r"]

    nf = len(f)
    h_grid = np.zeros((nf, nr), dtype=complex)

    fname = "testcase_zhang2023"
    working_dir = os.path.join(ROOT, "tmp")
    env_file = os.path.join(working_dir, f"{fname}.env")

    # Read env file
    with open(env_file, "r") as file:
        lines = file.readlines()

    while f0 < f[-1]:

        # Frequency subband
        f_kraken = f[(f < f1) & (f >= f0)]
        # print(i_subband, f0, f1, len(f_kraken))
        pad_before = np.sum(f < f0)
        pad_after = np.sum(f >= f1)

        # Modify number of frequencies
        nb_freq = f"{len(f_kraken)}                                                     ! Number of frequencies\n"
        lines[-2] = nb_freq
        # Replace frequencies in the env file
        new_freq_line = " ".join([f"{fi:.2f}" for fi in f_kraken])
        new_freq_line += "    ! Frequencies (Hz)"
        lines[-1] = new_freq_line

        # Write new env file
        with open(env_file, "w") as file:
            file.writelines(lines)

        # Run kraken and field
        os.chdir(working_dir)
        run_kraken_exec(fname)
        run_field_exec(fname)

        # Read shd from previously run kraken
        shdfile = f"{fname}.shd"

        _, _, _, _, read_freq, _, field_pos, pressure_field = readshd(
            filename=shdfile, freq=f_kraken
        )
        tf_subband = np.squeeze(pressure_field, axis=(1, 2, 3))  # (nf, nr)

        h_grid += np.pad(tf_subband, ((pad_before, pad_after), (0, 0)))

        # Update frequency subband
        i_subband += 1
        f0 = f1
        f1 = f[min(n_subband * i_subband, len(f) - 1)]

    # return f, r, z, h_grid

    # Pad h_grid with 0 for frequencies outside the 50 - 550 Hz band
    pad_before = np.sum(freq < 50)
    pad_after = np.sum(freq > 550)
    h_grid = np.pad(h_grid, ((pad_before, pad_after), (0, 0)))

    # Build xarray dataset
    library_zhang = xr.Dataset(
        data_vars=dict(
            tf_real=(
                ["f", "r"],
                np.real(h_grid),
            ),
            tf_imag=(["f", "r"], np.imag(h_grid)),
        ),
        coords={
            "f": freq,
            "r": field_pos["r"]["r"],
        },
    )

    # Save as netcdf
    fpath = os.path.join(ROOT_DATA, "tf_zhang_dataset.nc")
    library_zhang.to_netcdf(fpath)


def grid_dataset():

    # Load dataset
    fpath = os.path.join(ROOT_DATA, "tf_zhang_dataset.nc")
    ds = xr.open_dataset(fpath)

    # Load param
    depth, receivers, source, grid, frequency, _ = params()

    # Create new dataset
    ds_grid = xr.Dataset(
        coords=dict(
            f=ds.f.values,
            x=grid["x"][0, :],
            y=grid["y"][:, 0],
            idx_rcv=range(len(receivers["x"])),
        ),
        attrs=dict(
            df=ds.f.diff("f").values[0],
            dx=np.diff(grid["x"][0, :])[0],
            dy=np.diff(grid["y"][:, 0])[0],
            testcase="zhang_et_al_2023",
        ),
    )

    # Grid tf to the desired resolution
    # Preprocess tf to decrease the number of point for further interpolation
    r_grid_all_rcv = np.array(
        [grid["r"][i_rcv].flatten() for i_rcv in range(len(receivers["x"]))]
    )
    r_grid_all_rcv_unique = np.unique(np.round(r_grid_all_rcv.flatten(), 0))

    tf_vect = ds.tf_real.sel(
        r=r_grid_all_rcv_unique, method="nearest"
    ) + 1j * ds.tf_imag.sel(r=r_grid_all_rcv_unique, method="nearest")

    tf_grid_ds = xr.Dataset(
        coords=dict(
            f=ds.f.values,
            r=r_grid_all_rcv_unique,
        ),
        data_vars=dict(tf=(["f", "r"], tf_vect.values)),
    )

    gridded_tf = []
    grid_shape = (ds_grid.sizes["f"], ds_grid.sizes["x"], ds_grid.sizes["y"])
    for i_rcv in range(len(receivers["x"])):
        r_grid = grid["r"][i_rcv].flatten()
        tf_ircv = tf_grid_ds.tf.sel(r=r_grid, method="nearest")

        tf_grid = tf_ircv.values.reshape(grid_shape)
        gridded_tf.append(tf_grid)

    gridded_tf = np.array(gridded_tf)
    # Add to dataset
    grid_coords = ["idx_rcv", "f", "x", "y"]
    ds_grid["tf_real"] = (grid_coords, np.real(gridded_tf))
    ds_grid["tf_imag"] = (grid_coords, np.imag(gridded_tf))

    # Save dataset
    fname = f"tf_zhang_grid_dx{ds_grid.dx}m_dy{ds_grid.dy}m.nc"
    fpath = os.path.join(ROOT_DATA, fname)
    ds_grid.to_netcdf(fpath)
    ds_grid.close()


def event_src_spectrum(f):
    """Event source signal spectrum : Zhang et al. 2023 -> Gaussian noise"""

    event_props = {}
    # Event source is defined as a Gaussian noise
    S_f_event = np.ones_like(f)

    return event_props, S_f_event


def build_features():

    # Load tf dataset
    fpath = os.path.join(ROOT_DATA, "tf_zhang.nc")
    ds = xr.open_dataset(fpath)

    # Load params
    depth, receivers, source, grid, frequency, _ = params()

    # Load library source spectrum
    _, S_f_library = library_src_spectrum(ds.f.values)

    # Load event source spectrum
    _, S_f_event = event_src_spectrum(ds.f.values)

    # # Add x and y coordinates to dataset
    # ds.coords["x"] = grid["x"][0, :]
    # ds.coords["y"] = grid["y"][:, 0]

    rtf_src = []  # RFT vector at the source position
    rtf_grid = []  # RTF vector evaluated at each grid pixel
    gcc_src = []  # GCC vector evaluated at the source position
    gcc_grid = []  # GCC-SCOT vector evaluated at each grid pixel

    for i_ref in range(len(receivers["x"])):
        # i_ref = 0
        r_ref = grid["r"][i_ref].flatten()
        tf_ref = ds.tf_real.sel(r=r_ref, method="nearest") + 1j * ds.tf_imag.sel(
            r=r_ref, method="nearest"
        )

        r_src_ref = source["r"][i_ref]
        tf_src_ref = ds.tf_real.sel(
            r=r_src_ref, method="nearest"
        ) + 1j * ds.tf_imag.sel(r=r_src_ref, method="nearest")

        # Received spectrum -> reference receiver
        y_ref = mult_along_axis(tf_ref.values, S_f_library, axis=0)
        y_ref_src = mult_along_axis(tf_src_ref.values, S_f_event, axis=0)

        # Power spectral density at each grid pixel associated to the reference receiver -> library
        # Sxx_ref = tf_ref * np.conj(tf_ref)
        Sxx_ref = y_ref * np.conj(y_ref)
        # Power spectral density at the source position associated to the reference receiver -> event
        # Sxx_src_ref = tf_src_ref * np.conj(tf_src_ref)
        Sxx_src_ref = y_ref_src * np.conj(y_ref_src)

        for i_rcv in range(len(receivers["x"])):

            ## Kraken RTF ##
            # Grid
            r_i = grid["r"][i_rcv].flatten()
            tf_i = ds.tf_real.sel(r=r_i, method="nearest") + 1j * ds.tf_imag.sel(
                r=r_i, method="nearest"
            )
            rtf_i = tf_i.values / tf_ref.values
            rtf_i = rtf_i.reshape((ds.sizes["f"], ds.sizes["x"], ds.sizes["y"]))
            rtf_grid.append(rtf_i)

            # Source
            r_src_i = r_src_ref = source["r"][i_rcv]
            tf_src_i = ds.tf_real.sel(
                r=r_src_i, method="nearest"
            ) + 1j * ds.tf_imag.sel(r=r_src_i, method="nearest")
            rtf_src_i = tf_src_i.values / tf_src_ref.values
            rtf_src.append(rtf_src_i)

            # ## Covariance substraction RTF ##
            # # Grid
            # r_i = grid["r"][i_rcv].flatten()
            # tf_i = ds.tf_real.sel(r=r_i, method="nearest") + 1j * ds.tf_imag.sel(
            #     r=r_i, method="nearest"
            # )
            # rtf_i = tf_i.values / tf_ref.values
            # rtf_i = rtf_i.reshape((ds.sizes["f"], ds.sizes["x"], ds.sizes["y"]))
            # rtf_grid.append(rtf_i)

            # # Source
            # r_src_i = r_src_ref = source["r"][i_rcv]
            # tf_src_i = ds.tf_real.sel(
            #     r=r_src_i, method="nearest"
            # ) + 1j * ds.tf_imag.sel(r=r_src_i, method="nearest")
            # rtf_src_i = tf_src_i.values / tf_src_ref.values
            # rtf_src.append(rtf_src_i)

            ## GCC SCOT ##

            ## Grid -> library ##
            # Add the signal spectrum information
            y_i = mult_along_axis(tf_i.values, S_f_library, axis=0)

            # Power spectral density at each grid point associated to the receiver i
            # Syy = tf_i.values * np.conj(tf_i.values)  # Simpler formulation considering S(f) = 1

            Syy = y_i * np.conj(y_i)

            # Cross power spectral density between the reference receiver and receiver i
            # Sxy = tf_ref.values * np.conj(tf_i.values)
            Sxy = y_ref * np.conj(y_i)

            # Compute weights for GCC-SCOT
            w = 1 / np.abs(np.sqrt(Sxx_ref * Syy))
            # Apply GCC-SCOT
            gcc_grid_i = w * Sxy
            gcc_grid_i = gcc_grid_i.reshape(
                (ds.sizes["f"], ds.sizes["x"], ds.sizes["y"])
            )
            gcc_grid.append(gcc_grid_i)

            ## Event source -> event ##
            y_src_i = mult_along_axis(tf_src_i.values, S_f_event, axis=0)

            # Power spectral density at the source position associated to the receiver i
            # Syy_src = tf_src_i.values * np.conj(tf_src_i.values)
            Syy_src = y_src_i * np.conj(y_src_i)

            # Cross power spectral density between reference receiver and receiver i at source position$
            # Sxy_src = tf_src_ref.values * np.conj(tf_src_i.values)
            Sxy_src = y_ref_src * np.conj(y_src_i)

            # Compute weights for GCC-SCOT
            w_src = 1 / np.abs(np.sqrt(Sxx_src_ref * Syy_src))
            # Apply GCC-SCOT
            gcc_src_i = w_src * Sxy_src
            gcc_src.append(gcc_src_i)

    # Add coords
    ds.coords["idx_rcv"] = range(len(receivers["x"]))
    ds.coords["idx_rcv_ref"] = range(len(receivers["x"]))

    shape_src = (ds.sizes["idx_rcv_ref"], ds.sizes["idx_rcv"], ds.sizes["f"])
    shape_grid = (
        ds.sizes["idx_rcv_ref"],
        ds.sizes["idx_rcv"],
        ds.sizes["f"],
        ds.sizes["x"],
        ds.sizes["y"],
    )

    # RTF
    rtf_src = np.array(rtf_src).reshape(shape_src)
    rtf_src = np.moveaxis(rtf_src, 1, -1)  # (idx_rcv_ref, f, idx_rcv)
    rtf_grid = np.array(rtf_grid).reshape(shape_grid)
    rtf_grid = np.moveaxis(rtf_grid, 1, -1)  # (idx_rcv_ref, f, x, y, idx_rcv)

    # GCC SCOT (idx_rcv_ref, f, x, y, idx_rcv)
    gcc_src = np.array(gcc_src).reshape(shape_src)  # (idx_rcv_ref, f, idx_rcv)
    gcc_src = np.moveaxis(gcc_src, 1, -1)
    gcc_grid = np.array(gcc_grid).reshape(shape_grid)
    gcc_grid = np.moveaxis(gcc_grid, 1, -1)  # (idx_rcv_ref, f, x, y, idx_rcv)

    # Add rft to dataset
    ds["rtf_real"] = (
        ["idx_rcv_ref", "f", "x", "y", "idx_rcv"],
        rtf_grid.real,
    )
    ds["rtf_imag"] = (
        ["idx_rcv_ref", "f", "x", "y", "idx_rcv"],
        rtf_grid.imag,
    )
    ds["rtf_src_real"] = (["idx_rcv_ref", "f", "idx_rcv"], rtf_src.real)
    ds["rtf_src_imag"] = (["idx_rcv_ref", "f", "idx_rcv"], rtf_src.imag)

    # Add gcc-scot to dataset
    ds["gcc_real"] = (
        ["idx_rcv_ref", "f", "x", "y", "idx_rcv"],
        gcc_grid.real,
    )
    ds["gcc_imag"] = (
        ["idx_rcv_ref", "f", "x", "y", "idx_rcv"],
        gcc_grid.imag,
    )
    ds["gcc_src_real"] = (["idx_rcv_ref", "f", "idx_rcv"], gcc_src.real)
    ds["gcc_src_imag"] = (["idx_rcv_ref", "f", "idx_rcv"], gcc_src.imag)

    # Save updated dataset
    fpath = os.path.join(ROOT_DATA, f"rtf_zhang_dx{grid['dx']}m_dy{grid['dy']}m.nc")
    ds.to_netcdf(fpath)
    ds.close()


def plot_study_zhang2023():
    # Load params
    depth, receivers, source, grid, frequency, _ = params()

    # Define plot args for ambiguity surfaces
    plot_args_theta = {
        "dist": "hermitian_angle",
        "root_img": ROOT_IMG,
        "testcase": "zhang_et_al_2023",
        "dist_label": r"$\theta \, \textrm{[°]}$",
        "vmax": 50,
        "vmin": 0,
    }

    plot_args_d_rtf = {
        "dist": "normalized_metric",
        "root_img": ROOT_IMG,
        "testcase": "zhang_et_al_2023",
        # "dist_label": r"$d_{rtf}$",
        "dist_label": r"$\textrm{[dB]}$",
        # "vmax": 1,
        # "vmin": 0,
        # dB scale
        "vmax": 0,
        "vmin": -10,
    }

    plot_args_gcc = {
        "dist": "gcc_scot",
        "root_img": ROOT_IMG,
        "testcase": "zhang_et_al_2023",
        # "dist_label": r"$d_{gcc}$",
        "dist_label": r"$\textrm{[dB]}$",
        # "vmax": 1,
        # "vmin": 0,
        # dB scale
        "vmax": 0,
        "vmin": -10,
    }

    ###### Two sensor pairs ######
    # Select receivers to build the sub-array
    rcv_couples = np.array([[0, 2], [1, 4], [3, 5]])  # s1s3, s2s5, s4s6
    for rcv_cpl in rcv_couples:
        fpath = os.path.join(
            ROOT_DATA,
            f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc",
        )
        ds_cpl = xr.open_dataset(fpath)

        # Update sub array args
        plot_args_theta["sub_array"] = rcv_cpl
        plot_args_d_rtf["sub_array"] = rcv_cpl
        plot_args_gcc["sub_array"] = rcv_cpl

        # Theta
        plot_ambiguity_surface(
            amb_surf=ds_cpl.theta_rtf,
            source=source,
            plot_args=plot_args_theta,
            loc_arg="min",
        )
        plt.close("all")

        # d_rtf
        plot_ambiguity_surface(
            amb_surf=ds_cpl.d_rtf,
            source=source,
            plot_args=plot_args_d_rtf,
            loc_arg="max",
        )
        plt.close("all")

        # d_gcc
        plot_ambiguity_surface(
            amb_surf=ds_cpl.d_gcc, source=source, plot_args=plot_args_gcc, loc_arg="max"
        )
        plt.close("all")

    ###### Full array ######
    fpath = os.path.join(
        ROOT_DATA,
        f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_fullarray.nc",
    )
    ds_fa = xr.open_dataset(fpath)

    # Update sub array args
    plot_args_theta["sub_array"] = None
    plot_args_d_rtf["sub_array"] = None
    plot_args_gcc["sub_array"] = None

    # Theta
    plot_ambiguity_surface(
        amb_surf=ds_fa.theta_rtf,
        source=source,
        plot_args=plot_args_theta,
        loc_arg="min",
    )
    plt.close("all")

    # d_rtf
    plot_ambiguity_surface(
        amb_surf=ds_fa.d_rtf, source=source, plot_args=plot_args_d_rtf, loc_arg="max"
    )
    plt.close("all")

    # d_gcc
    plot_ambiguity_surface(
        amb_surf=ds_fa.d_gcc, source=source, plot_args=plot_args_gcc, loc_arg="max"
    )
    plt.close("all")

    ###### Figure 4 : Subplot in Zhang et al 2023 ######
    cmap = "jet"
    # vmax = 1
    # vmin = 0

    # dB scale
    vmax = 0
    vmin = -10

    x_src = source["x"]
    y_src = source["y"]
    true_pos_label = (
        r"$X_{src} = ( "
        + f"{x_src:.0f}\,"
        + r"\textrm{m},\,"
        + f"{y_src:.0f}\,"
        + r"\textrm{m})$"
    )

    f, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)

    for i_cpl, rcv_cpl in enumerate(rcv_couples):
        fpath = os.path.join(
            ROOT_DATA,
            f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc",
        )
        ds_cpl = xr.open_dataset(fpath)

        # Plot d_gcc and d_rtf
        for i, dist in enumerate(["d_gcc", "d_rtf"]):
            ax = axs[i, i_cpl]
            amb_surf = ds_cpl[dist]
            # # Estimated source position defined as one of the extremum of the ambiguity surface
            # x_idx, y_idx = np.unravel_index(np.argmax(amb_surf.values), amb_surf.shape)
            # x_src_hat = amb_surf.x[x_idx]
            # y_src_hat = amb_surf.y[y_idx]

            # amb_surf.plot(
            #     x="x",
            #     y="y",
            #     ax=ax,
            #     cmap=cmap,
            #     vmin=vmin,
            #     vmax=vmax,
            #     extend="neither",
            #     cbar_kwargs={"label": ""},
            # )
            im = ax.pcolormesh(
                ds_cpl["x"].values * 1e-3,
                ds_cpl["y"].values * 1e-3,
                amb_surf.values.T,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                # extend="neither",
                # cbar_kwargs={"label": ""},
            )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, label=r"$\textrm{[dB]}$")

            ax.scatter(
                x_src * 1e-3,
                y_src * 1e-3,
                facecolors="none",
                edgecolors="k",
                label=true_pos_label,
                s=200,
                linewidths=3,
            )

            ax.set_title(
                r"$s_{" + str(rcv_cpl[0] + 1) + "} - s_{" + str(rcv_cpl[1] + 1) + r"}$"
            )
            if i == 1:
                ax.set_xlabel(r"$x \, \textrm{[km]}$")
            else:
                ax.set_xlabel("")
            if i_cpl == 0:
                ax.set_ylabel(r"$y \, \textrm{[km]}$")
            else:
                ax.set_ylabel("")

            # # Set xticks
            ax.set_xticks([3.500, 4.000, 4.500])
            ax.set_yticks([6.400, 6.900, 7.400])

    # Save figure
    fpath = os.path.join(ROOT_IMG, "loc_zhang2023_fig4.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close("all")

    ###### Figure 5 : Subplot in Zhang et al 2023 ######
    f, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    vmin = -10  # dB
    vmax = 0  # dB

    # Plot d_gcc and d_rtf
    for i, dist in enumerate(["d_gcc", "d_rtf"]):
        ax = axs[i]
        amb_surf = ds_fa[dist]

        im = ax.pcolormesh(
            ds_fa["x"].values * 1e-3,
            ds_fa["y"].values * 1e-3,
            amb_surf.values.T,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label=r"$\textrm{[dB]}$")

        ax.scatter(
            x_src * 1e-3,
            y_src * 1e-3,
            facecolors="none",
            edgecolors="k",
            label=true_pos_label,
            s=200,
            linewidths=3,
        )

        ax.set_title(f"Full array")
        ax.set_xlabel(r"$x \textrm{[km]}$")
        if i == 0:
            ax.set_ylabel(r"$y \, \textrm{[km]}$")
        else:
            ax.set_ylabel("")

        # # Set xticks
        ax.set_xticks([3.500, 4.000, 4.500])
        ax.set_yticks([6.400, 6.900, 7.400])

    # Save figure
    fpath = os.path.join(ROOT_IMG, "loc_zhang2023_fig5.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close("all")

    ###### Figure 5 distribution ######
    f, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    percentile_threshold = 0.995
    bins = ds_fa["d_gcc"].size // 10

    # Plot d_gcc and d_rtf
    mainlobe_th = {}
    for i, dist in enumerate(["d_gcc", "d_rtf"]):
        ax = axs[i]
        amb_surf = ds_fa[dist]

        amb_surf.plot.hist(ax=ax, bins=bins, alpha=0.5, color="b")

        # Vertical line representing the percentile threshold
        percentile = np.percentile(amb_surf.values, percentile_threshold * 100)
        mainlobe_th[dist] = percentile
        ax.axvline(
            percentile,
            color="r",
            linestyle="--",
            label=f"{percentile_threshold*100:.0f}th percentile",
        )

        ax.set_title(f"Full array")
        ax.set_xlim(-20, 0)
        ax.set_xlabel(r"$\textrm{[dB]}$")

    # Save figure
    fpath = os.path.join(ROOT_IMG, "loc_zhang2023_fig5_dist.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close("all")

    ###### Figure 5 showing pixels selected as the mainlobe ######
    f, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Find mainlobe contours
    mainlobe_contours = find_mainlobe(ds_fa)

    # Plot d_gcc and d_rtf
    for i, dist in enumerate(["d_gcc", "d_rtf"]):
        ax = axs[i]
        amb_surf = ds_fa[dist]

        im = ax.pcolormesh(
            ds_fa["x"].values * 1e-3,
            ds_fa["y"].values * 1e-3,
            amb_surf.values.T,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        contour = mainlobe_contours[dist]
        ax.plot(
            ds_fa["x"].values[contour[:, 0].astype(int)] * 1e-3,
            ds_fa["y"].values[contour[:, 1].astype(int)] * 1e-3,
            color="k",
            linewidth=2,
            # label="Mainlobe Boundary" if i == 0 else None,
        )

        ax.set_title(f"Full array")
        ax.set_xlabel(r"$x \textrm{[km]}$")
        if i == 0:
            ax.set_ylabel(r"$y \, \textrm{[km]}$")
        else:
            ax.set_ylabel("")

        # # Set xticks
        ax.set_xticks([3.500, 4.000, 4.500])
        ax.set_yticks([6.400, 6.900, 7.400])

    # Save figure
    fpath = os.path.join(ROOT_IMG, "loc_zhang2023_fig5_mainlobe.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close("all")

    # TODO : move the msr part to a dedicated function once the rtf estimation block is ok
    # Derive mainlobe to side lobe ratio

    f, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    msr = {}
    for i, dist in enumerate(["d_gcc", "d_rtf"]):
        mainlobe_mask = np.zeros_like(amb_surf.values, dtype=bool)

        ax = axs[i]
        amb_surf = ds_fa[dist]

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

        # Plot ambiguity surface without mainlobe pixels
        amb_surf_without_mainlobe = amb_surf.copy()
        amb_surf_without_mainlobe.values[mainlobe_mask] = np.nan

        im = ax.pcolormesh(
            ds_fa["x"].values * 1e-3,
            ds_fa["y"].values * 1e-3,
            amb_surf_without_mainlobe.values.T,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.plot(
            ds_fa["x"].values[contour[:, 0].astype(int)] * 1e-3,
            ds_fa["y"].values[contour[:, 1].astype(int)] * 1e-3,
            color="k",
            linewidth=2,
            # label="Mainlobe Boundary" if i == 0 else None,
        )

        # Add convex hull to the plot
        hull_points = np.vstack([hull_points, hull_points[0]])

        ax.plot(
            hull_points[:, 0] * 1e-3,
            hull_points[:, 1] * 1e-3,
            "r-",
            linewidth=2,
            label="Mainlobe Convex Hull",
        )

        # Source pos
        x_idx, y_idx = np.unravel_index(np.argmax(amb_surf.values), amb_surf.shape)
        x_src_hat = amb_surf.x[x_idx]
        y_src_hat = amb_surf.y[y_idx]
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
        if i == 0:
            ax.set_ylabel(r"$y \, \textrm{[km]}$")
        else:
            ax.set_ylabel("")

        # # Set xticks
        ax.set_xticks([3.500, 4.000, 4.500])
        ax.set_yticks([6.400, 6.900, 7.400])

        # Compute mainlobe to side lobe ratio
        msr[dist] = np.max(
            amb_surf.values[~mainlobe_mask]
        )  # MSR = mainlobe_dB - side_lobe_dB (mainlobe_dB = max(ambsurf) = 0dB)

        print(f"MSR {dist} : {msr[dist]:.2f} dB")

    # Save figure
    fpath = os.path.join(ROOT_IMG, "loc_zhang2023_fig5_nomainlobe.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close("all")


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

    return mainlobe_contours


def main_lobe_segmentation_study():

    # Load params
    depth, receivers, source, grid, frequency, _ = params()

    # Load full array dataset
    fpath = os.path.join(
        ROOT_DATA,
        f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_fullarray.nc",
    )
    ds_fa = xr.open_dataset(fpath)

    dist = "d_gcc"
    amb_surf = ds_fa[dist]

    ### 1) Double k-means approach ###
    # 1.1) K-means clustering on the ambiguity surface level

    # Reshape ambiguity surface to 1D array
    amb_surf_1d = amb_surf.values.flatten()

    # Apply K-means clustering
    n_clusters = 8
    x_coord, y_coord = np.meshgrid(ds_fa.x.values, ds_fa.y.values)
    X = np.vstack(
        [x_coord.flatten(), y_coord.flatten(), amb_surf_1d]
    )  # 3 Columns x, y, S(x, y)
    X_norm = preprocessing.normalize(X).T
    X_norm[:, 0:2] *= 2  # Increase the weight of the spatial coordinates
    # X_norm = X.T
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    kmeans.fit(X_norm)

    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
    #     amb_surf_1d.reshape(-1, 1)
    # )

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
        # amb_surf.values.T,
        labels.T,
        cmap=cmap,
        # cmap="jet",
        # vmin=-10,
        # vmax=0,
    )

    # Add colorbar with n_clusters ticks
    cbar = plt.colorbar(im, ax=ax, label=r"$\textrm{Class}$", ticks=range(n_clusters))
    # cbar = plt.colorbar(im, ax=ax, label=r"$\textrm{Class}$")
    ax.set_title(f"Full array")
    ax.set_xlabel(r"$x \textrm{[km]}$")
    ax.set_ylabel(r"$y \, \textrm{[km]}$")
    ax.set_xticks([3.500, 4.000, 4.500])
    ax.set_yticks([6.400, 6.900, 7.400])

    # # Plot segmentation
    # for i in range(n_clusters):
    #     mask = labels == i
    #     ax.contour(mask, levels=[0.5])

    # Save figure
    fpath = os.path.join(ROOT_IMG, "loc_zhang2023_fig5_segmentation.png")
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

    # mask = labels == src_hat_class
    # ax.contour(
    #     ds_fa["x"].values * 1e-3,
    #     ds_fa["y"].values * 1e-3,
    #     mask.T,
    #     levels=[0.5],
    #     colors="k",
    # )

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
    fpath = os.path.join(ROOT_IMG, "loc_zhang2023_fig5_segmentation_highlight.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")


def process_localisation_zhang2023(nf=10):
    # Load params
    depth, receivers, source, grid, frequency, _ = params()

    # Load rtf data
    fpath = os.path.join(ROOT_DATA, f"rtf_zhang_dx{grid['dx']}m_dy{grid['dy']}m.nc")
    ds = xr.open_dataset(fpath)

    # Compute distance between the RTF vector associated with the source and the RTF vector at each grid pixel
    # Match field processing #
    dist_func = D_hermitian_angle_fast
    dist_kwargs = {
        "ax_rcv": 3,
        "unit": "deg",
        "apply_mean": True,
    }

    # Select a few frequencies
    # nf = 10
    df = np.diff(ds.f.values)[0]
    f_loc = np.random.choice(ds.f.values, nf)
    ds = ds.sel(f=f_loc)

    theta_max = 90

    d_gcc_fullarray = []
    ###### Two sensor pairs ######
    # Select receivers to build the sub-array
    rcv_couples = np.array([[0, 2], [1, 4], [3, 5]])  # s1s3, s2s5, s4s6
    for rcv_cpl in rcv_couples:
        i_ref = rcv_cpl[0]

        ## RTF ##
        # Extract data corresponding to the two-sensor pair rcv_cpl
        ds_cpl_rtf = ds.sel(idx_rcv_ref=i_ref, idx_rcv=rcv_cpl)

        rtf_grid = ds_cpl_rtf.rtf_real.values + 1j * ds_cpl_rtf.rtf_imag.values
        rtf_src = ds_cpl_rtf.rtf_src_real.values + 1j * ds_cpl_rtf.rtf_src_imag.values

        theta = dist_func(rtf_src, rtf_grid, **dist_kwargs)

        # Add theta to dataset
        ds_cpl_rtf["theta"] = (["x", "y"], theta.T)
        # # Convert theta to a metric between -1 and 1
        # theta_inv = (
        #     theta_max - ds_cpl_rtf.theta
        # )  # So that the source position is the maximum value
        # d_rtf = (theta_inv - theta_max / 2) / (theta_max / 2)  # To lie between -1 and 1

        # Normalize
        d_rtf = normalize_metric_contrast(-ds_cpl_rtf.theta)

        # Convert to dB
        d_rtf = d_rtf.values
        d_rtf[d_rtf == 0] = MIN_VAL_LOG
        d_rtf = 10 * np.log10(d_rtf)
        ds_cpl_rtf["d_rtf"] = (["x", "y"], d_rtf)

        ## GCC ##
        ds_cpl_gcc = ds.sel(idx_rcv_ref=rcv_cpl[0], idx_rcv=rcv_cpl[1])

        gcc_grid = ds_cpl_gcc.gcc_real.values + 1j * ds_cpl_gcc.gcc_imag.values
        gcc_src = ds_cpl_gcc.gcc_src_real.values + 1j * ds_cpl_gcc.gcc_src_imag.values

        # Cast gcc_src to the same shape as gcc_grid
        gcc_src = cast_matrix_to_target_shape(gcc_src, gcc_grid.shape)

        # Build cross corr (Equation (8) in Zhang et al. 2023)
        # d_gcc = np.sum(gcc_grid * np.conj(gcc_src) * df, axis=0)
        d_gcc = np.abs(np.sum(gcc_grid * np.conj(gcc_src) * df, axis=0))
        # d_gcc = d_gcc / np.max(d_gcc)

        # Normalize
        d_gcc = normalize_metric_contrast(d_gcc)

        # Convert to dB
        d_gcc = d_gcc
        d_gcc[d_gcc == 0] = MIN_VAL_LOG
        d_gcc = 10 * np.log10(d_gcc)  # Convert to dB

        # Add d to dataset
        ds_cpl_gcc["d_gcc"] = (["x", "y"], d_gcc.T)

        # Store d_gcc for full array incoherent processing
        d_gcc_fullarray.append(d_gcc)

        # Build dataset to be saved as netcdf
        ds_cpl = xr.Dataset(
            data_vars=dict(
                theta_rtf=(["x", "y"], ds_cpl_rtf.theta.values),
                d_rtf=(["x", "y"], ds_cpl_rtf.d_rtf.values),
                d_gcc=(["x", "y"], ds_cpl_gcc.d_gcc.values),
            ),
            coords={
                "x": ds.x.values,
                "y": ds.y.values,
            },
        )

        # Save dataset
        fpath = os.path.join(
            ROOT_DATA,
            f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc",
        )
        ds_cpl.to_netcdf(fpath)

    ###### Full array ######

    ## RTF ##
    i_ref = 0
    # Extract data corresponding to the two-sensor pair rcv_cpl
    ds_cpl_rtf = ds.sel(idx_rcv_ref=i_ref)

    rtf_grid = ds_cpl_rtf.rtf_real.values + 1j * ds_cpl_rtf.rtf_imag.values
    rtf_src = ds_cpl_rtf.rtf_src_real.values + 1j * ds_cpl_rtf.rtf_src_imag.values

    theta = dist_func(rtf_src, rtf_grid, **dist_kwargs)

    # Add theta to dataset
    ds_cpl_rtf["theta"] = (["x", "y"], theta.T)
    # # Convert theta to a metric between -1 and 1
    # theta_inv = (
    #     theta_max - ds_cpl_rtf.theta
    # )  # So that the source position is the maximum value
    # d_rtf = (theta_inv - theta_max / 2) / (theta_max / 2)  # To lie between -1 and 1

    d_rtf = normalize_metric_contrast(-ds_cpl_rtf.theta)  # q in [0, 1]

    #  Replace 0 by 1e-5 to avoid log(0) in dB conversion
    d_rtf = d_rtf.values
    d_rtf[d_rtf == 0] = MIN_VAL_LOG
    d_rtf = 10 * np.log10(d_rtf)  # Convert to dB
    ds_cpl_rtf["d_rtf"] = (["x", "y"], d_rtf)

    ## GCC ##
    d_gcc_fullarray = np.array(d_gcc_fullarray)
    # Convert back to linear scale before computing the mean
    d_gcc_fullarray = 10 ** (d_gcc_fullarray / 10)
    d_gcc_fullarray = np.mean(d_gcc_fullarray, axis=0)

    # Convert to dB
    d_gcc_fullarray[d_gcc_fullarray == 0] = MIN_VAL_LOG
    d_gcc_fullarray = 10 * np.log10(d_gcc_fullarray)

    # Build dataset to be saved as netcdf
    ds_cpl_fullarray = xr.Dataset(
        data_vars=dict(
            theta_rtf=(["x", "y"], ds_cpl_rtf.theta.values),
            d_rtf=(["x", "y"], ds_cpl_rtf.d_rtf.values),
            d_gcc=(["x", "y"], d_gcc_fullarray.T),
        ),
        coords={
            "x": ds.x.values,
            "y": ds.y.values,
        },
    )

    # Save dataset
    fpath = os.path.join(
        ROOT_DATA,
        f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_fullarray.nc",
    )
    ds_cpl_fullarray.to_netcdf(fpath)


def localise(sub_array=None):

    # Load params
    depth, receivers, source, grid, frequency, _ = params()

    # Load rtf data
    # fpath = os.path.join(ROOT_DATA, "rtf_zhang.nc")
    fpath = os.path.join(ROOT_DATA, f"rtf_zhang_dx{grid['dx']}m_dy{grid['dy']}m.nc")
    ds = xr.open_dataset(fpath)

    # Compute distance between the RTF vector associated with the source and the RTF vector at each grid pixel
    # Match field processing #
    dist_func = D_hermitian_angle_fast
    dist_kwargs = {
        "ax_rcv": 3,
        "unit": "deg",
        "apply_mean": True,
    }

    # Select a few frequencies
    nf = 10
    df = np.diff(ds.f.values)[0]
    f_loc = np.random.choice(ds.f.values, nf)
    ds = ds.sel(f=f_loc)

    # Select a few receivers
    # idx_rcv = [0, 1, 4, 5]
    if sub_array is not None:
        ds = ds.sel(idx_rcv=sub_array)

    ## RTF ##
    # Compute distance bewteen the estimated RTF and RTF at each grid point
    rtf_src = ds.rtf_src_real.values + 1j * ds.rtf_src_imag.values
    rtf_grid = ds.rtf_real.values + 1j * ds.rtf_imag.values

    theta = dist_func(rtf_src, rtf_grid, **dist_kwargs)

    # Add theta to dataset
    ds["theta"] = (["x", "y"], theta.T)
    amb_surf = ds.theta

    # Plot ambiguity surfaces
    plot_args = {
        "dist": "hermitian_angle",
        "vmax_percentile": 5,
        "root_img": ROOT_IMG,
        "testcase": "zhang_et_al_2023",
        "dist_label": r"$\theta \, \textrm{[°]}$",
        "vmax": 50,
        "vmin": 0,
        "sub_array": sub_array,
    }

    plot_ambiguity_surface(
        amb_surf=amb_surf, source=source, plot_args=plot_args, loc_arg="min"
    )

    # Convert theta to a metric between -1 and 1
    theta_max = 90
    theta_inv = theta_max - amb_surf  # So that the source position is the maximum value
    d = (theta_inv - theta_max / 2) / (theta_max / 2)  # To lie between -1 and 1
    # q = (np.max(amb_surf) - amb_surf) / (np.max(amb_surf) - np.min(amb_surf))

    # Add d to dataset
    # ds["d"] = (["x", "y"], d)
    # amb_surf = ds.theta

    plot_args["dist"] = "normalized_metric"
    plot_args["dist_label"] = r"$d_{rtf}$"
    plot_args["vmax"] = 1
    plot_args["vmin"] = -0.2
    plot_ambiguity_surface(
        amb_surf=d, source=source, plot_args=plot_args, loc_arg="max"
    )

    ## GCC-SCOT ##
    gcc_src = ds.gcc_src_real.values + 1j * ds.gcc_src_imag.values
    gcc_grid = ds.gcc_real.values + 1j * ds.gcc_imag.values

    # Cast gcc_src to the same shape as gcc_grid
    gcc_src = cast_matrix_to_target_shape(gcc_src, gcc_grid.shape)

    # Build cross corr (Equation (8) in Zhang et al. 2023)
    d = np.abs(np.sum(gcc_grid * np.conj(gcc_src) * df, axis=0))
    d = d / np.max(d)

    plot_args["dist"] = "gcc_scot"
    plot_args["dist_label"] = r"$d$"
    plot_args["vmax"] = 1
    plot_args["vmin"] = -0.2
    for i in range(len(receivers["x"])):
        plot_args["sub_array"] = [0, i]
        amb_surf_i = d[..., i].T
        amb_surf_da = xr.DataArray(
            amb_surf_i,
            coords={"x": ds.x.values, "y": ds.y.values},
            dims=["x", "y"],
        )

        plot_ambiguity_surface(
            amb_surf=amb_surf_da, source=source, plot_args=plot_args, loc_arg="max"
        )


def plot_ambiguity_surface(amb_surf, source, plot_args, loc_arg):

    dist = plot_args["dist"]
    testcase = plot_args["testcase"]
    root_img = plot_args["root_img"]
    dist_label = plot_args["dist_label"]
    vmax = plot_args["vmax"]
    vmin = plot_args["vmin"]
    sub_array = plot_args["sub_array"]

    # Source position
    x_src = source["x"]
    y_src = source["y"]
    print("True source position: ", x_src, y_src)

    # Estimated source position defined as one of the extremum of the ambiguity surface
    if loc_arg == "max":
        x_idx, y_idx = np.unravel_index(np.argmax(amb_surf.values), amb_surf.shape)
        cmap = "jet"
    elif loc_arg == "min":
        x_idx, y_idx = np.unravel_index(np.argmin(amb_surf.values), amb_surf.shape)
        cmap = "jet_r"

    x_src_hat = amb_surf.x[x_idx]
    y_src_hat = amb_surf.y[y_idx]
    print(
        "Estimated source position: ",
        np.round(x_src_hat.values, 1),
        np.round(y_src_hat.values, 1),
    )

    plt.figure(figsize=(14, 12))
    amb_surf.plot(
        x="x",
        y="y",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        # aspect="equal",
        extend="neither",
        # robust=True,
        cbar_kwargs={"label": dist_label},
        # cbar_kwargs={"label": r"$\textrm{[dB]}$"},
    )
    # amb_surf.plot.contourf(
    #     x="x",
    #     y="y",
    #     cmap=cmap,
    #     vmin=vmin,
    #     vmax=vmax,
    #     extend="neither",
    #     levels=20,
    #     # robust=True,
    #     cbar_kwargs={"label": dist_label},
    # )

    true_pos_label = (
        r"$X_{src} = ( "
        + f"{x_src:.0f}\,"
        + r"\textrm{m},\,"
        + f"{y_src:.0f}\,"
        + r"\textrm{m})$"
    )
    # estimated_pos_label = (
    #     r"$\hat{X}_{src} = ( "
    #     + f"{x_src_hat:.0f}\,"
    #     + r"\textrm{m},\,"
    #     + f"{y_src_hat:.0f}\,"
    #     + r"\textrm{m})$"
    # )
    # estimated_pos_label = r"$\hat{X}_{src}" + f" = ({r_src_hat:.2f}, {z_src_hat:.2f})$"
    # plt.scatter(
    #     x_src_hat, y_src_hat, color="w", marker="o", label=estimated_pos_label, s=100
    # )  # Estimated source position
    plt.scatter(
        x_src,
        y_src,
        facecolors="none",
        edgecolors="k",
        label=true_pos_label,
        s=200,
        linewidths=3,
    )  # True source position

    # # Add receiver positions
    _, receivers, _, grid, _, _ = params()
    # x_rcv = np.concatenate([receivers["x"], [receivers["x"][0]]])
    # y_rcv = np.concatenate([receivers["y"], [receivers["y"][0]]])
    # plt.plot(
    #     x_rcv,
    #     y_rcv,
    #     color="k",
    #     marker="o",
    #     linestyle="--",
    #     markersize=7,
    #     # label=[f"$s_{i}$" for i in range(len(receivers["x"]))],
    # )

    # txt_offset = 100
    # sgn_y = [-1, -1, 0, 0, 0, 0]
    # sgn_x = [0, 0, 1.5, 1.5, -1.5, -1.5]
    # for i, txt in enumerate([f"$s_{i+1}$" for i in range(len(receivers["x"]))]):
    #     plt.annotate(
    #         txt,
    #         (receivers["x"][i], receivers["y"][i]),
    #         # (receivers["x"][i] + sgn_x[i] * 50, receivers["y"][i] + sgn_y[i] * 50),
    #         fontsize=16,
    #     )
    #     # plt.text(
    #     #     receivers["x"][i] + sgn_x[i] * txt_offset,
    #     #     receivers["y"][i] + sgn_y[i] * txt_offset,
    #     #     txt,
    #     #     fontsize=16,
    #     # )

    # plt.xlim([grid["x"][0, 0], grid["x"][0, -1]])
    # plt.ylim([grid["y"][0, 0], grid["y"][-1, 0]])
    plt.axis("equal")
    plt.xlabel(r"$x \, \textrm{[m]}$")
    plt.ylabel(r"$y \, \textrm{[m]}$")
    plt.legend()

    # Save figure
    path = os.path.join(root_img)
    if not os.path.exists(path):
        os.makedirs(path)

    sa_lab = (
        "" if sub_array is None else "_" + "_".join([f"s{sa+1}" for sa in sub_array])
    )
    fname = f"{testcase}_ambiguity_surface_{dist}{sa_lab}.png"
    fpath = os.path.join(path, fname)
    plt.savefig(fpath)


if __name__ == "__main__":
    # params()
    # save_simulation_netcdf()
    # build_tf_dataset()
    # grid_dataset()
    # build_signal()
    # build_features()

    # sub_arrays = [
    #     None,
    #     [0, 1, 4, 5],
    #     [0, 2, 3, 5],
    #     [0, 1],
    #     [0, 2],
    #     [0, 3],
    #     [0, 4],
    #     [0, 5],
    # ]
    # sub_arrays = [[0, 1, 4, 5], [0, 2, 3, 5]]

    # for sub_array in sub_arrays:
    # localise(sub_array=sub_array)

    # process_localisation_zhang2023(nf=10)
    # plot_study_zhang2023()

    # main_lobe_segmentation_study()


## Left overs ##


# tc_varin = {
#     "freq": f,
#     "src_depth": z_src,
#     "max_range_m": rmax,
#     "mode_theory": "adiabatic",
#     "flp_n_rcv_z": nz,
#     "flp_rcv_z_min": z_min,
#     "flp_rcv_z_max": z_max,
#     "min_depth": depth,
#     "max_depth": depth,
#     "dr_flp": dr,
#     "nb_modes": 200,
#     "bottom_boundary_condition": "acousto_elastic",
#     "nmedia": 4,
#     "phase_speed_limits": [200, 20000],
#     "bott_hs_properties": bott_hs_properties,
# }
# tc = TestCase1_0(mode="prod", testcase_varin=tc_varin)
# title = "Zhang et al. 2023 test case"
# tc.title = title
# tc.env_dir = os.path.join(ROOT, "tmp")
# tc.update(tc_varin)

# rho = np.array([1.0, 1.75, 2.01, 2.5]) * RHO_W * 1e-3
# c_p = np.array([1500, 1600, 1900, 4650])  # P-wave celerity (m/s)
# a_p = np.array([0, 0.2, 0.06, 0.01])  # Compression wave attenuation (dB/wavelength)
# z = np.array([0, 150, 175, 275])

# medium = KrakenMedium(
#     ssp_interpolation_method="C_linear", z_ssp=z, c_p=c_p, a_p=a_p, rho=rho
# )
# tc.medium = medium
# # Write flp and env files
# tc.write_kraken_files()
