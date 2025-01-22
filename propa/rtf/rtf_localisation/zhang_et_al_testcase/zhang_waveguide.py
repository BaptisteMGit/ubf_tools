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
import matplotlib.pyplot as plt

from cst import RHO_W
from signals.signals import lfm_chirp
from misc import cast_matrix_to_target_shape, mult_along_axis
from publication.PublicationFigure import PubFigure
from propa.kraken_toolbox.run_kraken import readshd
from propa.rtf.rtf_utils import D_hermitian_angle_fast

PubFigure()
ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\zhang_et_al_testcase"
ROOT_DATA = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\zhang_et_al_testcase\data"
ROOT_IMG = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\zhang_et_al_2023"


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


def build_features():

    # Load tf dataset
    fpath = os.path.join(ROOT_DATA, "tf_zhang.nc")
    ds = xr.open_dataset(fpath)

    # Load params
    depth, receivers, source, grid, frequency, _ = params()

    # Library source is defined as a LFM 100-500 Hz
    library_src = {
        "f0": 100,
        "f1": 500,
        "fs": 5000,
        "T": 100,
        "phi": 0,
    }

    s, t = lfm_chirp(
        library_src["f0"],
        library_src["f1"],
        library_src["fs"],
        library_src["T"],
        library_src["phi"],
    )

    # TODO : remove
    import scipy.signal as sp

    s = sp.chirp(
        t, library_src["f0"], library_src["T"], library_src["f1"], method="hyperbolic"
    )

    S_f_library = np.fft.rfft(s)
    f_library = np.fft.rfftfreq(len(s), 1 / library_src["fs"])
    # Interp library source spectrum at desired frequencies
    S_f_library = np.interp(ds.f.values, f_library, S_f_library)

    # Event source is defined as a Gaussian noise, ie S_f_event = 1
    S_f_event = np.ones_like(S_f_library)

    # Add x and y coordinates to dataset
    ds.coords["x"] = grid["x"][0, :]
    ds.coords["y"] = grid["y"][:, 0]

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
            gcc_grid_i = gcc_grid_i.values.reshape(
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
        "dist_label": r"$d_{rtf}$",
        "vmax": 1,
        "vmin": -0.2,
    }

    plot_args_gcc = {
        "dist": "gcc_scot",
        "root_img": ROOT_IMG,
        "testcase": "zhang_et_al_2023",
        "dist_label": r"$d_{gcc}$",
        "vmax": 1,
        "vmin": -0.2,
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
        # d_rtf
        plot_ambiguity_surface(
            amb_surf=ds_cpl.d_rtf,
            source=source,
            plot_args=plot_args_d_rtf,
            loc_arg="max",
        )
        # d_gcc
        plot_ambiguity_surface(
            amb_surf=ds_cpl.d_gcc, source=source, plot_args=plot_args_gcc, loc_arg="max"
        )

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
    # d_rtf
    plot_ambiguity_surface(
        amb_surf=ds_fa.d_rtf, source=source, plot_args=plot_args_d_rtf, loc_arg="max"
    )
    # d_gcc
    plot_ambiguity_surface(
        amb_surf=ds_fa.d_gcc, source=source, plot_args=plot_args_gcc, loc_arg="max"
    )

    ###### Figure 4 : Subplot in Zhang et al 2023 ######
    cmap = "jet"
    vmax = 1
    vmin = -0.2

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
            cbar = plt.colorbar(im, ax=ax)

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

    ###### Figure 5 : Subplot in Zhang et al 2023 ######
    f, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

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
        cbar = plt.colorbar(im, ax=ax)

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
        # Convert theta to a metric between -1 and 1
        theta_inv = (
            theta_max - ds_cpl_rtf.theta
        )  # So that the source position is the maximum value
        d_rtf = (theta_inv - theta_max / 2) / (theta_max / 2)  # To lie between -1 and 1
        ds_cpl_rtf["d_rtf"] = (["x", "y"], d_rtf.values)

        ## GCC ##
        ds_cpl_gcc = ds.sel(idx_rcv_ref=rcv_cpl[0], idx_rcv=rcv_cpl[1])

        gcc_grid = ds_cpl_gcc.gcc_real.values + 1j * ds_cpl_gcc.gcc_imag.values
        gcc_src = ds_cpl_gcc.gcc_src_real.values + 1j * ds_cpl_gcc.gcc_src_imag.values

        # Cast gcc_src to the same shape as gcc_grid
        gcc_src = cast_matrix_to_target_shape(gcc_src, gcc_grid.shape)

        # Build cross corr (Equation (8) in Zhang et al. 2023)
        d_gcc = np.abs(np.sum(gcc_grid * np.conj(gcc_src) * df, axis=0))
        d_gcc = d_gcc / np.max(d_gcc)

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
    # Convert theta to a metric between -1 and 1
    theta_inv = (
        theta_max - ds_cpl_rtf.theta
    )  # So that the source position is the maximum value
    d_rtf = (theta_inv - theta_max / 2) / (theta_max / 2)  # To lie between -1 and 1
    ds_cpl_rtf["d_rtf"] = (["x", "y"], d_rtf.values)

    ## GCC ##
    d_gcc_fullarray = np.array(d_gcc_fullarray)
    d_gcc_fullarray = np.mean(d_gcc_fullarray, axis=0)

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
    build_features()
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

    process_localisation_zhang2023(nf=10)
    plot_study_zhang2023()


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
