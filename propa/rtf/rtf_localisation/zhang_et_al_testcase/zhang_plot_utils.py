#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   zhang_plot_utils.py
@Time    :   2025/01/27 12:01:55
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Useful functions to plot results for Zhang et al 2023
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

from misc import compute_hyperbola
from publication.PublicationFigure import PubFigure
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import (
    params,
    get_array_label,
    get_rcv_couples,
    estimate_msr,
    find_mainlobe,
)
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_params import (
    ROOT_IMG,
    ROOT_DATA,
)

# ======================================================================================================================
# Functions
# ======================================================================================================================


def plot_study_zhang2023(
    folder,
    data_fname=None,
    debug=False,
    antenna_type="zhang",
    rcv_in_fullarray=np.arange(6),
    plot_args={},
):
    # Load params
    _, _, source, grid, _, _ = params(antenna_type=antenna_type)

    # Extract plot flags
    plot_array = plot_args.get("plot_array", False)
    plot_single_cpl_surf = plot_args.get("plot_single_cpl_surf", False)
    plot_fullarray_surf = plot_args.get("plot_fullarray_surf", False)
    plot_cpl_surf_comparison = plot_args.get("plot_cpl_surf_comparison", False)
    plot_fullarray_surf_comparison = plot_args.get(
        "plot_fullarray_surf_comparison", False
    )
    plot_surf_dist_comparison = plot_args.get("plot_surf_dist_comparison", False)
    plot_mainlobe_contour = plot_args.get("plot_mainlobe_contour", False)
    plot_msr_estimation = plot_args.get("plot_msr_estimation", False)

    # Define folder to store images
    root_img = os.path.join(ROOT_IMG, folder)
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    # Define folder to store data
    root_data = os.path.join(ROOT_DATA, folder)
    # if not os.path.exists(root_data):
    #     os.makedirs(root_data)

    # Load fullarray data
    array_label = get_array_label(rcv_in_fullarray)
    if data_fname is None:
        data_fname_fa = (
            f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_fullarray_{array_label}.nc"
        )
    else:
        data_fname_fa = f"{data_fname}_fullarray_{array_label}.nc"

    fpath = os.path.join(
        root_data,
        # f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_fullarray.nc",
        data_fname_fa,
    )
    ds_fa = xr.open_dataset(fpath)

    vmin_dB = np.round(np.max([ds_fa[dist].median() for dist in ["d_gcc", "d_rtf"]]), 0)

    # Define plot args for ambiguity surfaces
    plot_args_theta = {
        "dist": "theta_rtf",
        "root_img": root_img,
        "testcase": "zhang_et_al_2023",
        "dist_label": r"$\theta$" + " [°]",
        "vmax": 50,
        "vmin": 0,
        "add_hyperbola": True,
    }

    plot_args_d_rtf = {
        "dist": "q_rtf",
        "root_img": root_img,
        "testcase": "zhang_et_al_2023",
        # "dist_label": r"$d_{rtf}$",
        "dist_label": "[dB]",
        # "vmax": 1,
        # "vmin": 0,
        # dB scale
        "vmax": 0,
        "vmin": vmin_dB,
        "add_hyperbola": True,
    }

    plot_args_gcc = {
        "dist": "q_dcf",
        "root_img": root_img,
        "testcase": "zhang_et_al_2023",
        # "dist_label": r"$d_{gcc}$",
        "dist_label": "[dB]",
        # "vmax": 1,
        # "vmin": 0,
        # dB scale
        "vmax": 0,
        "vmin": vmin_dB,
        "add_hyperbola": True,
    }

    if plot_array:
        # Plot antenna geometry and research area
        plot_antenna_and_search_area(
            root_img=root_img,
            debug=debug,
            antenna_type=antenna_type,
            rcv_in_fullarray=list(ds_fa.idx_rcv),
        )

    ###### Two sensor pairs ######
    rcv_couples = get_rcv_couples(ds_fa.idx_rcv)

    if plot_single_cpl_surf:
        cpl_foldername = "ambiguity_surface_receivers_pair"
        # Select receivers to build the sub-array
        # rcv_couples = np.array([[0, 2], [1, 4], [3, 5]])  # s1s3, s2s5, s4s6

        for rcv_cpl in rcv_couples:

            # Load data
            if data_fname is None:
                data_fname_cpl = f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc"
            else:
                data_fname_cpl = f"{data_fname}_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc"

            fpath = os.path.join(
                root_data,
                # f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc",
                data_fname_cpl,
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
                folder_name=cpl_foldername,
            )

            # d_rtf
            plot_ambiguity_surface(
                amb_surf=ds_cpl.d_rtf,
                source=source,
                plot_args=plot_args_d_rtf,
                loc_arg="max",
                folder_name=cpl_foldername,
            )

            # d_gcc
            plot_ambiguity_surface(
                amb_surf=ds_cpl.d_gcc,
                source=source,
                plot_args=plot_args_gcc,
                loc_arg="max",
                folder_name=cpl_foldername,
            )

    ###### Full array ######
    fa_foldername = "ambiguity_surface_fullarray"
    if plot_fullarray_surf:
        # Update sub array args
        plot_args_theta["sub_array"] = ds_fa.attrs["idx_rcv"]
        plot_args_d_rtf["sub_array"] = ds_fa.attrs["idx_rcv"]
        plot_args_gcc["sub_array"] = ds_fa.attrs["idx_rcv"]
        plot_args_theta["add_circle"] = True
        plot_args_d_rtf["add_circle"] = True
        plot_args_gcc["add_circle"] = True
        plot_args_theta["add_hyperbola"] = True
        plot_args_d_rtf["add_hyperbola"] = True
        plot_args_gcc["add_hyperbola"] = True

        # hyperbola_cpls = [[0, 2], [0, 4], [1, 3], [3, 5]]
        hyperbola_cpls = [[2, 4], [3, 5]]
        plot_args_gcc["hyperbola_cpls"] = hyperbola_cpls

        # Theta
        plot_ambiguity_surface(
            amb_surf=ds_fa.theta_rtf,
            source=source,
            plot_args=plot_args_theta,
            loc_arg="min",
            folder_name=fa_foldername,
        )

        # d_rtf
        plot_ambiguity_surface(
            amb_surf=ds_fa.d_rtf,
            source=source,
            plot_args=plot_args_d_rtf,
            loc_arg="max",
            folder_name=fa_foldername,
        )

        # d_gcc
        plot_ambiguity_surface(
            amb_surf=ds_fa.d_gcc,
            source=source,
            plot_args=plot_args_gcc,
            loc_arg="max",
            folder_name=fa_foldername,
        )

    # Define plot args for ambiguity surfaces
    xticks_pos_km = [3.6, 4.0, 4.4]
    yticks_pos_km = [6.5, 6.9, 7.3]
    xticks_pos_m = [xt * 1e3 for xt in xticks_pos_km]
    yticks_pos_m = [yt * 1e3 for yt in yticks_pos_km]

    cmap = "jet"
    # vmax = 1
    # vmin = 0

    # dB scale
    vmax = 0
    vmin = vmin_dB

    x_src = source["x"]
    y_src = source["y"]

    ###### Figure 4 : Subplot in Zhang et al 2023 ######
    if len(ds_fa.idx_rcv) == 6 and not np.isnan(ds_fa.snr):
        rcv_couples = np.array([[0, 2], [1, 4], [3, 5]])  # s1s3, s2s5, s4s6
    else:  # Full simu case
        rcv_couples = get_rcv_couples(ds_fa.idx_rcv)

    if plot_cpl_surf_comparison:
        plot_subarrays_ambiguity_surfaces(
            root_img,
            rcv_couples,
            data_fname,
            root_data,
            grid,
            x_src,
            y_src,
            vmin,
            vmax,
            xticks_pos_m,
            yticks_pos_m,
            cmap=cmap,
        )

    ###### Figure 5 : Subplot in Zhang et al 2023 ######
    if plot_fullarray_surf_comparison:
        plot_fullarray_ambiguity_surfaces(
            ds_fa,
            root_img,
            x_src,
            y_src,
            vmin,
            vmax,
            xticks_pos_m,
            yticks_pos_m,
            cmap=cmap,
        )

    ###### Figure 5 distribution ######
    if plot_surf_dist_comparison:
        plot_ambiguity_surface_distribution(ds_fa, root_img)

    ###### Figure 5 showing pixels selected as the mainlobe ######
    if plot_mainlobe_contour:
        plot_ambiguity_surface_mainlobe_contour(
            ds_fa, root_img, vmin, vmax, xticks_pos_m, yticks_pos_m, cmap=cmap
        )

    estimate_msr(ds_fa=ds_fa, plot=plot_msr_estimation, root_img=root_img, verbose=True)


def plot_antenna_and_search_area(
    root_img, debug=False, antenna_type="zhang", rcv_in_fullarray=[]
):
    _, receivers, source, grid, _, _ = params(debug=debug, antenna_type=antenna_type)

    root_arrays = os.path.join(root_img, "arrays")
    if not os.path.exists(root_arrays):
        os.makedirs(root_arrays)

    area_square_x = [
        grid["x"].min(),
        grid["x"].min(),
        grid["x"].max(),
        grid["x"].max(),
        grid["x"].min(),
    ]

    area_square_y = [
        grid["y"].min(),
        grid["y"].max(),
        grid["y"].max(),
        grid["y"].min(),
        grid["y"].min(),
    ]
    rcv_x = np.append(receivers["x"], receivers["x"][0])
    rcv_y = np.append(receivers["y"], receivers["y"][0])

    x_src, y_src = source["x"], source["y"]
    true_pos_label = (
        r"$X_{src} = ( "
        + f"{x_src:.0f}\,"
        + r"\textrm{m},\,"
        + f"{y_src:.0f}\,"
        + r"\textrm{m})$"
    )
    Lx = grid["x"].max() - grid["x"].min()
    Ly = grid["y"].max() - grid["y"].min()
    area_label = (
        r"$\mathcal{A} \,("
        + f"L_x = {Lx:.0f}\,"
        + r"\textrm{m},\,"
        + f"L_y = {Ly:.0f}\,"
        + r"\textrm{m})$"
    )

    plt.figure()
    plt.plot(
        rcv_x,
        rcv_y,
        color="k",
        linestyle="--",
        marker="o",
        markersize=10,
        label="Antenna",
    )

    label_offset_pts = (7, 7)  # Shift right and up in display units

    for i in range(len(receivers["x"])):

        plt.annotate(
            f"$s_{i+1}$",
            xy=(receivers["x"][i], receivers["y"][i]),
            xycoords="data",
            xytext=label_offset_pts,
            textcoords="offset points",
            fontsize=18,
        )

    if rcv_in_fullarray:
        # Color selected antenna
        rcv_x_fa = np.append(
            receivers["x"][rcv_in_fullarray], receivers["x"][rcv_in_fullarray][0]
        )
        rcv_y_fa = np.append(
            receivers["y"][rcv_in_fullarray], receivers["y"][rcv_in_fullarray][0]
        )
        plt.plot(
            rcv_x_fa,
            rcv_y_fa,
            color="r",
            marker="o",
            markersize=10,
            linestyle="--",
        )

    plt.plot(area_square_x, area_square_y, color="r", label=area_label)
    plt.scatter(x_src, y_src, color="k", label=true_pos_label, marker="2", s=250)
    plt.legend()
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    array_label = get_array_label(rcv_in_fullarray)
    fpath = os.path.join(root_arrays, f"antenna_search_area_{array_label}.png")
    plt.savefig(fpath, dpi=300)


def plot_subarrays_ambiguity_surfaces(
    root_img,
    rcv_couples,
    data_fname,
    root_data,
    grid,
    x_src,
    y_src,
    vmin,
    vmax,
    xticks_pos_m,
    yticks_pos_m,
    cmap="jet",
):

    true_pos_label = (
        r"$X_{src} = ( "
        + f"{x_src:.0f}\,"
        + r"\textrm{m},\,"
        + f"{y_src:.0f}\,"
        + r"\textrm{m})$"
    )

    f, axs = plt.subplots(
        2, rcv_couples.shape[0], figsize=(38, 20), sharex=True, sharey=True
    )
    if rcv_couples.shape[0] == 1:
        axs = np.atleast_2d(axs).T  # Ensure axs has necessary shape

    all_rcv_idx = []
    for i_cpl, rcv_cpl in enumerate(rcv_couples):

        # Load data
        if data_fname is None:
            data_fname_cpl = f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc"
        else:
            data_fname_cpl = f"{data_fname}_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc"

        fpath = os.path.join(
            root_data,
            data_fname_cpl,
        )
        ds_cpl = xr.open_dataset(fpath)

        # Store all rcvs
        all_rcv_idx += list(ds_cpl.idx_rcv)

        if i_cpl == axs.shape[1] - 1:
            cbar_kwargs = {"label": r"$\textrm{[dB]}$"}
            add_colorbar = True
        else:
            cbar_kwargs = {}
            add_colorbar = False

        # Plot d_gcc and d_rtf
        for i, dist in enumerate(["d_gcc", "d_rtf"]):
            ax = axs[i, i_cpl]
            amb_surf = ds_cpl[dist]
            # # Estimated source position defined as one of the extremum of the ambiguity surface
            # x_idx, y_idx = np.unravel_index(np.argmax(amb_surf.values), amb_surf.shape)
            # x_src_hat = amb_surf.x[x_idx]
            # y_src_hat = amb_surf.y[y_idx]

            im = amb_surf.plot(
                x="x",
                y="y",
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extend="neither",
                cbar_kwargs=cbar_kwargs,
                add_colorbar=add_colorbar,
            )
            # im = ax.pcolormesh(
            #     ds_cpl["x"].values * 1e-3,
            #     ds_cpl["y"].values * 1e-3,
            #     amb_surf.values,
            #     cmap=cmap,
            #     vmin=vmin,
            #     vmax=vmax,
            #     # extend="neither",
            #     # cbar_kwargs={"label": ""},
            # )

            # Add colorbar
            # cbar = plt.colorbar(im, ax=ax, label=r"$\textrm{[dB]}$")

            ax.scatter(
                x_src,
                y_src,
                color="k",
                # facecolors="none",
                # edgecolors="k",
                label=true_pos_label,
                marker="2",
                s=250,
                linewidths=3,
            )

            ax.set_title(
                r"$s_{" + str(rcv_cpl[0] + 1) + "} - s_{" + str(rcv_cpl[1] + 1) + r"}$"
            )
            if i == 1:
                ax.set_xlabel(r"$x$" + " [m]")
            else:
                ax.set_xlabel("")
            if i_cpl == 0:
                ax.set_ylabel(r"$y$" + " [m]")
            else:
                ax.set_ylabel("")

            # # Set xticks
            # ax.set_xticks([3500, 4000, 4500])
            # ax.set_yticks([6400, 6900, 7400])
            ax.set_xticks(xticks_pos_m)
            ax.set_yticks(yticks_pos_m)
            # ax.set_xticklabels(xticks_label_km, fontsize=22)
            # ax.set_yticklabels(yticks_label_km, fontsize=22)

    # Sup title with SNR
    all_rcv_idx = np.unique(all_rcv_idx)
    rcv_str = "$" + ", \,".join([f"s_{id+1}" for id in all_rcv_idx]) + "$"
    plt.suptitle(f"SNR = {ds_cpl.snr} dB, Receivers = ({rcv_str})")

    # Save figure
    root_subarrays_comparison = os.path.join(root_img, "subarrays_comparison")
    if not os.path.exists(root_subarrays_comparison):
        os.makedirs(root_subarrays_comparison)

    rcv_lab = "_".join([f"s{id+1}" for id in all_rcv_idx])
    fpath = os.path.join(
        root_subarrays_comparison,
        f"loc_zhang2023_fig4_snr{ds_cpl.snr}dB_rcvs_{rcv_lab}.png",
    )
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close("all")


def plot_fullarray_ambiguity_surfaces(
    ds_fa,
    root_img,
    x_src,
    y_src,
    vmin,
    vmax,
    xticks_pos_m,
    yticks_pos_m,
    cmap="jet",
):

    true_pos_label = (
        r"$X_{src} = ( "
        + f"{x_src:.0f}\,"
        + r"\textrm{m},\,"
        + f"{y_src:.0f}\,"
        + r"\textrm{m})$"
    )

    titles = {"d_gcc": "DCF", "d_rtf": "RTF"}

    f, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Plot d_gcc and d_rtf
    for i, dist in enumerate(["d_gcc", "d_rtf"]):
        ax = axs[i]
        amb_surf = ds_fa[dist]

        im = amb_surf.plot(
            x="x",
            y="y",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            extend="neither",
            cbar_kwargs={"label": "q [dB]"},
            # robust=True,
            # cbar_kwargs={"label": dist_label},
        )

        # Add colorbar
        ax.scatter(
            x_src,
            y_src,
            color="k",
            label=true_pos_label,
            marker="2",
            s=200,
            linewidths=2,
        )

        ax.set_title(titles[dist])
        ax.set_xlabel(r"$x$" + " [m]")
        if i == 0:
            ax.set_ylabel(r"$y$" + " [m]")
        else:
            ax.set_ylabel("")

        # Set xticks
        ax.set_xticks(xticks_pos_m)
        ax.set_yticks(yticks_pos_m)
        # ax.set_xticklabels(xticks_label_km, fontsize=22)
        # ax.set_yticklabels(yticks_label_km, fontsize=22)

    root_fullarray_comparison = os.path.join(root_img, "fullarray_comparison")
    if not os.path.exists(root_fullarray_comparison):
        os.makedirs(root_fullarray_comparison)

    rcv_lab = "_".join([f"s{id+1}" for id in ds_fa.idx_rcv])
    fpath = os.path.join(
        root_fullarray_comparison,
        f"loc_zhang2023_fig5_snr{ds_fa.snr}dB_rcvs_{rcv_lab}.png",
    )
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close("all")


def plot_fullarray_ambiguity_surfaces_publi(
    ds_fa,
    root_img,
    x_src,
    y_src,
    vmin,
    vmax,
    xticks_pos_m,
    yticks_pos_m,
    cmap="jet",
):

    true_pos_label = (
        r"$X_{src} = ( "
        + f"{x_src:.0f}\,"
        + r"\textrm{m},\,"
        + f"{y_src:.0f}\,"
        + r"\textrm{m})$"
    )

    # Plot d_gcc and d_rtf
    pfig = PubFigure(label_fontsize=40, ticks_fontsize=40, labelpad=25)

    for i, dist in enumerate(["d_gcc", "d_rtf"]):

        f, ax = plt.subplots(1, 1, figsize=(16, 12), sharey=True)
        amb_surf = ds_fa[dist]

        im = amb_surf.plot(
            x="x",
            y="y",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            extend="neither",
            cbar_kwargs={"label": "q [dB]"},
            rasterized=True,
        )

        # Add colorbar
        ax.scatter(
            x_src,
            y_src,
            color="k",
            label=true_pos_label,
            marker="o",
            facecolors="none",
            s=900,
            linewidths=5,
        )

        ax.set_xlabel(r"$x \,\textrm{[m]}$")
        ax.set_ylabel(r"$y \, \textrm{[m]}$")

        # Set xticks
        # ax.set_xticks(xticks_pos_m)
        # ax.set_yticks(yticks_pos_m)

        # Save figure
        fpath = os.path.join(root_img, f"amb_surf_{dist}")
        plt.savefig(f"{fpath}.eps", dpi=300)
        plt.savefig(f"{fpath}.png", dpi=300)


def plot_ambiguity_surface_mainlobe_contour(
    ds_fa, root_img, vmin, vmax, xticks_pos_m, yticks_pos_m, cmap="jet"
):
    # Find mainlobe contours
    mainlobe_contours = find_mainlobe(ds_fa)

    f, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    # Plot d_gcc and d_rtf
    for i, dist in enumerate(["d_gcc", "d_rtf"]):
        ax = axs[i]
        amb_surf = ds_fa[dist]

        amb_surf.plot(
            x="x",
            y="y",
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extend="neither",
            cbar_kwargs={"label": "[dB]"},
        )

        # im = ax.pcolormesh(
        #     ds_fa["x"].values,
        #     ds_fa["y"].values,
        #     amb_surf.values.T,
        #     cmap=cmap,
        #     vmin=vmin,
        #     vmax=vmax,
        # )

        # Add colorbar
        # cbar = plt.colorbar(im, ax=ax, label=r"$\textrm{[dB]}$")

        # contour = mainlobe_contours[dist]
        # ax.plot(
        #     ds_fa["x"].values[contour[:, 0].astype(int)],
        #     ds_fa["y"].values[contour[:, 1].astype(int)],
        #     color="k",
        #     linewidth=2,
        #     # label="Mainlobe Boundary" if i == 0 else None,
        # )

        ax.set_title(r"$\textrm{Full array}$")
        ax.set_xlabel(r"$x$" + " [m]")
        if i == 0:
            ax.set_ylabel(r"$y$" + " [m]")
        else:
            ax.set_ylabel("")

        # Set xticks
        ax.set_xticks(xticks_pos_m)
        ax.set_yticks(yticks_pos_m)
        # ax.set_xticklabels(xticks_label_km, fontsize=22)
        # ax.set_yticklabels(yticks_label_km, fontsize=22)

    # Save figure
    root_mainlobe = os.path.join(root_img, "mainlobe")
    if not os.path.exists(root_mainlobe):
        os.makedirs(root_mainlobe)

    fpath = os.path.join(root_mainlobe, "loc_zhang2023_fig5_mainlobe.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close("all")


def plot_ambiguity_surface_distribution(ds_fa, root_img):
    """
    Plot the distribution of the ambiguity surfaces for d_gcc and d_rtf
    """
    f, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    percentile_threshold = 0.995
    bins = {"d_gcc": ds_fa["d_gcc"].size // 10, "d_rtf": ds_fa["d_rtf"].size // 10}

    # Plot d_gcc and d_rtf
    mainlobe_th = {}
    for i, dist in enumerate(["d_gcc", "d_rtf"]):
        ax = axs[i]
        amb_surf = ds_fa[dist]

        amb_surf.plot.hist(ax=ax, bins=bins[dist], alpha=0.5, color="b")

        # Vertical line representing the percentile threshold
        percentile = np.percentile(amb_surf.values, percentile_threshold * 100)
        mainlobe_th[dist] = percentile
        ax.axvline(
            percentile,
            color="r",
            linestyle="--",
            label=f"{percentile_threshold*100:.0f}th percentile",
        )

        ax.set_title("Full array")
        ax.set_xlim(-20, 0)
        ax.set_xlabel("[dB]")

    # Save figure
    root_dist = os.path.join(root_img, "ambiguity_surf_distribution")
    if not os.path.exists(root_dist):
        os.makedirs(root_dist)

    fpath = os.path.join(root_dist, "loc_zhang2023_fig5_dist.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close("all")


def plot_ambiguity_surface(
    amb_surf, source, plot_args, loc_arg, antenna_type="zhang", folder_name=""
):

    dist = plot_args["dist"]
    testcase = plot_args["testcase"]
    root_img = plot_args["root_img"]
    dist_label = plot_args["dist_label"]
    vmax = plot_args["vmax"]
    vmin = plot_args["vmin"]
    sub_array = plot_args["sub_array"]

    # To plot the hyperbola corresponding to TDOA
    add_hyperbola = plot_args.get("add_hyperbola", False)
    hyperbola_cpls = plot_args.get("hyperbola_cpls", None)
    # To plot the circle centered on the center of the antenna array and passing through the source
    add_circle = plot_args.get("add_circle", False)

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
    # plt.scatter(
    #     x_src,
    #     y_src,
    #     facecolors="none",
    #     edgecolors="k",
    #     label=true_pos_label,
    #     s=200,
    #     linewidths=3,
    # )  # True source position

    plt.scatter(
        x_src,
        y_src,
        color="k",
        # facecolors="none",
        # edgecolors="k",
        label=true_pos_label,
        marker="2",
        s=400,
        linewidths=4,
    )

    # # Add receiver positions
    _, receivers, _, grid, _, _ = params(antenna_type=antenna_type)
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

    # Add hyperbola if required
    if add_hyperbola:
        # print("Add hyperbola")
        src_pos = (x_src, y_src)

        if hyperbola_cpls is None:
            # Compute hyperbola for each pair of receivers
            # default_cpls = [[0, 2], [1, 4], [3, 5]]
            hyperbola_cpls = get_rcv_couples(np.arange(len(receivers["x"])))

        for i, sa in enumerate(hyperbola_cpls):
            receiver1 = (receivers["x"][sa[0]], receivers["y"][sa[0]])
            receiver2 = (receivers["x"][sa[1]], receivers["y"][sa[1]])
            (right_branch, left_branch) = compute_hyperbola(
                receiver1, receiver2, src_pos, num_points=1000, tmax=10
            )

            # Plot both branches
            plt.plot(right_branch[0], right_branch[1], "k", linestyle="--", zorder=15)
            plt.plot(left_branch[0], left_branch[1], "k", linestyle="--", zorder=16)

        # else:
        #     receiver1 = (
        #         receivers["x"][hyperbola_cpls[0]],
        #         receivers["y"][hyperbola_cpls[0]],
        #     )
        #     receiver2 = (
        #         receivers["x"][hyperbola_cpls[1]],
        #         receivers["y"][hyperbola_cpls[1]],
        #     )
        #     (right_branch, left_branch) = compute_hyperbola(
        #         receiver1, receiver2, src_pos, tmax=5
        #     )

        #     # Plot both branches
        #     plt.plot(right_branch[0], right_branch[1], "k", linestyle="--", zorder=10)
        #     plt.plot(left_branch[0], left_branch[1], "k", linestyle="--", zorder=10)

    # Add circle if required
    if add_circle:
        barycentre_x = np.mean(receivers["x"])
        barycentre_y = np.mean(receivers["y"])
        radius = np.sqrt((barycentre_x - x_src) ** 2 + (barycentre_y - y_src) ** 2)

        circle = plt.Circle(
            (barycentre_x, barycentre_y),
            radius,
            color="k",
            fill=False,
            linestyle="--",
            linewidth=2,
            label=r"$\mathcal{C}((\hat{x_r}, \hat{y_r}), r_{s})$",
        )
        plt.gca().add_artist(circle)

    plt.xlim([grid["x"][0, 0], grid["x"][0, -1]])
    plt.ylim([grid["y"][0, 0], grid["y"][-1, 0]])
    # plt.ylim([grid["x"][0, 0], grid["x"][0, -1]])
    # plt.xlim([grid["y"][0, 0], grid["y"][-1, 0]])

    # plt.axis("equal")
    sub_array = amb_surf.idx_rcv
    rcv_str = "$" + ", \,".join([f"s_{id+1}" for id in sub_array]) + "$"
    plt.title(f"SNR = {amb_surf.attrs['snr']} dB, Receivers = ({rcv_str})")
    plt.xlabel(r"$x$" + " [m]")
    plt.ylabel(r"$y$" + " [m]")
    plt.legend()

    # Save figure
    root_amb_surf = os.path.join(root_img, folder_name)
    if not os.path.exists(root_amb_surf):
        os.makedirs(root_amb_surf)

    sa_lab = (
        "" if sub_array is None else "_" + "_".join([f"s{sa+1}" for sa in sub_array])
    )
    fname = f"{testcase}_ambiguity_surface_{dist}{sa_lab}.png"
    fpath = os.path.join(root_amb_surf, fname)
    plt.savefig(fpath)
    plt.close("all")


def check_signal_noise(ds_sig_noise):
    """
    Plot library signal at source position and event signal as well as associated noise signals to check that the dataset is built as required.
    """
    s_l = ds_sig_noise.s_l.sel(x=ds_sig_noise.xs, y=ds_sig_noise.ys, method="nearest")
    x_l = ds_sig_noise.x_l.sel(x=ds_sig_noise.xs, y=ds_sig_noise.ys, method="nearest")
    n_l = ds_sig_noise.n_l.sel(x=ds_sig_noise.xs, y=ds_sig_noise.ys, method="nearest")
    s_e = ds_sig_noise.s_e
    x_e = ds_sig_noise.x_e
    n_e = ds_sig_noise.n_e

    img_check_path = os.path.join(ds_sig_noise.root_img, "check")
    if not os.path.exists(img_check_path):
        os.makedirs(img_check_path)

    for i_rcv in ds_sig_noise.idx_rcv.values:

        f, axs = plt.subplots(3, 2, figsize=(20, 12), sharex=True, sharey=True)

        # First column -> library
        s_l.sel(idx_rcv=i_rcv).plot(ax=axs[0, 0])
        axs[0, 0].set_title("$z(t)$")

        n_l.sel(idx_rcv=i_rcv).plot(ax=axs[1, 0])
        axs[1, 0].set_title("$v(t)$")

        x_l.sel(idx_rcv=i_rcv).plot(ax=axs[2, 0])
        axs[2, 0].set_title("$x(t) = z(t) + v(t)$")

        # Second column -> event
        s_e.sel(idx_rcv=i_rcv).plot(ax=axs[0, 1])
        axs[0, 1].set_title("$z(t)$")

        n_e.sel(idx_rcv=i_rcv).plot(ax=axs[1, 1])
        axs[1, 1].set_title("$v(t)$")

        x_e.sel(idx_rcv=i_rcv).plot(ax=axs[2, 1])
        axs[2, 1].set_title("$x(t) = z(t) + v(t)$")

        # Remove xlabel for row 0 and 1
        for irow in [0, 1]:
            for icol in [0, 1]:
                axs[irow, icol].set_xlabel("")

        plt.suptitle(f"SNR = {ds_sig_noise.snr} dB")
        fpath = os.path.join(img_check_path, f"sig_noise_ircv{i_rcv}.png")
        plt.savefig(fpath)

    plt.close("all")


def check_signal_noise_stft(ds_sig_noise):
    """
    Plot library signal stft at source position and event signal stft as well as associated noise signals stfts to check that the dataset is built as required.
    """
    s_l = ds_sig_noise.s_l.sel(x=ds_sig_noise.xs, y=ds_sig_noise.ys, method="nearest")
    x_l = ds_sig_noise.x_l.sel(x=ds_sig_noise.xs, y=ds_sig_noise.ys, method="nearest")
    n_l = ds_sig_noise.n_l.sel(x=ds_sig_noise.xs, y=ds_sig_noise.ys, method="nearest")
    s_e = ds_sig_noise.s_e
    x_e = ds_sig_noise.x_e
    n_e = ds_sig_noise.n_e

    # Set stft params
    fs = 1 / ds_sig_noise.t.diff("t").values[0]
    nperseg = 2**8
    noverlap = nperseg // 2
    # Derive stfts
    ff, tt, s_l_stft = sp.stft(
        s_l.values, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1
    )
    _, _, x_l_stft = sp.stft(
        x_l.values, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1
    )
    _, _, n_l_stft = sp.stft(
        n_l.values, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1
    )
    _, _, s_e_stft = sp.stft(
        s_e.values, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1
    )
    _, _, x_e_stft = sp.stft(
        x_e.values, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1
    )
    _, _, n_e_stft = sp.stft(
        n_e.values, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1
    )

    # Normalize all stfts
    s_l_stft = s_l_stft / np.max(np.abs(s_l_stft))
    x_l_stft = x_l_stft / np.max(np.abs(x_l_stft))
    n_l_stft = n_l_stft / np.max(np.abs(n_l_stft))
    s_e_stft = s_e_stft / np.max(np.abs(s_e_stft))
    x_e_stft = x_e_stft / np.max(np.abs(x_e_stft))
    n_e_stft = n_e_stft / np.max(np.abs(n_e_stft))

    # Store stfts in xarray for plot facilities
    stft_ds = xr.Dataset(
        {
            "s_l_stft": (["idx_rcv", "f", "t"], 20 * np.log10(np.abs(s_l_stft))),
            "x_l_stft": (["idx_rcv", "f", "t"], 20 * np.log10(np.abs(x_l_stft))),
            "n_l_stft": (["idx_rcv", "f", "t"], 20 * np.log10(np.abs(n_l_stft))),
            "s_e_stft": (["idx_rcv", "f", "t"], 20 * np.log10(np.abs(s_e_stft))),
            "x_e_stft": (["idx_rcv", "f", "t"], 20 * np.log10(np.abs(x_e_stft))),
            "n_e_stft": (["idx_rcv", "f", "t"], 20 * np.log10(np.abs(n_e_stft))),
        },
        coords={"idx_rcv": ds_sig_noise.idx_rcv.values, "f": ff, "t": tt},
    )

    img_check_path = os.path.join(ds_sig_noise.root_img, "check")
    if not os.path.exists(img_check_path):
        os.makedirs(img_check_path)

    cmap = "jet"
    vmin = -40
    vmax = 0

    for i_rcv in ds_sig_noise.idx_rcv.values:

        f, axs = plt.subplots(3, 2, figsize=(20, 12), sharex=True, sharey=True)

        # First column -> library
        # s_l.sel(idx_rcv=i_rcv).plot(ax=axs[0, 0])
        stft_ds.s_l_stft.sel(idx_rcv=i_rcv).plot(
            ax=axs[0, 0], cmap=cmap, vmin=vmin, vmax=vmax
        )
        axs[0, 0].set_title("$z(t)$")

        # n_l.sel(idx_rcv=i_rcv).plot(ax=axs[1, 0])
        stft_ds.n_l_stft.sel(idx_rcv=i_rcv).plot(
            ax=axs[1, 0], cmap=cmap, vmin=vmin, vmax=vmax
        )
        axs[1, 0].set_title("$v(t)$")

        # x_l.sel(idx_rcv=i_rcv).plot(ax=axs[2, 0])
        stft_ds.x_l_stft.sel(idx_rcv=i_rcv).plot(
            ax=axs[2, 0], cmap=cmap, vmin=vmin, vmax=vmax
        )
        axs[2, 0].set_title("$x(t) = z(t) + v(t)$")

        # Second column -> event
        # s_e.sel(idx_rcv=i_rcv).plot(ax=axs[0, 1])
        stft_ds.s_e_stft.sel(idx_rcv=i_rcv).plot(
            ax=axs[0, 1], cmap=cmap, vmin=vmin, vmax=vmax
        )
        axs[0, 1].set_title("$z(t)$")

        # n_e.sel(idx_rcv=i_rcv).plot(ax=axs[1, 1])
        stft_ds.n_e_stft.sel(idx_rcv=i_rcv).plot(
            ax=axs[1, 1], cmap=cmap, vmin=vmin, vmax=vmax
        )
        axs[1, 1].set_title("$v(t)$")

        # x_e.sel(idx_rcv=i_rcv).plot(ax=axs[2, 1])
        stft_ds.x_e_stft.sel(idx_rcv=i_rcv).plot(
            ax=axs[2, 1], cmap=cmap, vmin=vmin, vmax=vmax
        )
        axs[2, 1].set_title("$x(t) = z(t) + v(t)$")

        # Remove xlabel for row 0 and 1
        for irow in [0, 1]:
            for icol in [0, 1]:
                axs[irow, icol].set_xlabel("")

        plt.suptitle(f"SNR = {ds_sig_noise.snr} dB")
        fpath = os.path.join(img_check_path, f"stft_sig_noise_ircv{i_rcv}.png")
        plt.savefig(fpath)

    plt.close("all")


def check_rtf_features(ds_rtf_cs, folder, antenna_type="zhang"):

    # Define folder to store images
    root_img = os.path.join(ROOT_IMG, folder, "check_rtf")
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    # Load dataset with KRAKEN TF to derive reference RTF
    _, _, source, grid, frequency, _ = params(antenna_type=antenna_type)

    # Load gridded dataset
    fname = f"tf_zhang_grid_dx{grid['dx']}m_dy{grid['dy']}m.nc"
    fpath = os.path.join(ROOT_DATA, fname)
    ds_tf = xr.open_dataset(fpath)
    # Build complex tf
    tf = ds_tf.tf_real + 1j * ds_tf.tf_imag
    # Extract tf between fmin and fmax from ds_rtf_cs
    tf = tf.sel(f=slice(ds_rtf_cs.f_rtf.min(), ds_rtf_cs.f_rtf.max()))

    # Define reference receiver to use
    i_rcv_ref = 0
    ds_rtf_cs = ds_rtf_cs.sel(idx_rcv_ref=i_rcv_ref)
    rtf_cs = ds_rtf_cs.rtf_real + 1j * ds_rtf_cs.rtf_imag

    # Define tf_ref
    tf_ref = tf.sel(idx_rcv=i_rcv_ref)

    # List position where we want to compare estimated RTF to ref RTF (KRAKEN)
    # Source position + the 4 corners of the grid + one position inside the grid
    x_check = [
        source["x"],
        ds_tf.x.min().values,
        ds_tf.x.min().values,
        ds_tf.x.max().values,
        ds_tf.x.max().values,
        ds_tf.x.values[int(ds_tf.sizes["x"] * 2 / 3)],
    ]

    y_check = [
        source["y"],
        ds_tf.y.min().values,
        ds_tf.y.max().values,
        ds_tf.y.max().values,
        ds_tf.y.min().values,
        ds_tf.y.values[int(ds_tf.sizes["y"] * 1 / 3)],
    ]

    # Iterate over receivers
    for i_rcv in tf.idx_rcv.values:

        # Build "true" RTF
        rtf_true = tf.sel(idx_rcv=i_rcv) / tf_ref

        # Iterate over positions to check
        for i_check in range(len(x_check)):
            x_i = x_check[i_check]
            y_i = y_check[i_check]

            # Extract data at required position
            rtf_cs_pos = rtf_cs.sel(idx_rcv=i_rcv).sel(x=x_i, y=y_i, method="nearest")
            rtf_true_pos = rtf_true.sel(x=x_i, y=y_i, method="nearest")

            abs_cs = np.abs(rtf_cs_pos)
            abs_true = np.abs(rtf_true_pos)
            # Compare rtf_true to estimated rtf

            plt.figure()
            abs_true.plot(
                label=r"$\Pi_{" + str(i_rcv) + r"}^{(Kraken)}$",
                linestyle="-",
                color="k",
                linewidth=1.5,
            )
            abs_cs.plot(
                # x="f",
                linestyle="-",
                label=r"$\Pi_{" + str(i_rcv) + r"}^{(CS)}$",
                color="r",
                marker="o",
                linewidth=0.2,
                markersize=3,
            )
            plt.legend()
            plt.yscale("log")
            plt.xlabel(r"$f$" + " [Hz]")
            plt.ylabel(r"$|\Pi(f)|$")

            # Save figure
            fname = f"check_rtf_rcv{i_rcv}_x{x_i}_y{y_i}.png"
            fpath = os.path.join(root_img, fname)
            plt.savefig(fpath)
            plt.close("all")

    ds_tf.close()


def check_gcc_features(ds_gcc, folder):

    # Define folder to store images
    root_img = os.path.join(ROOT_IMG, folder, "check_gcc")
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    # Load dataset with KRAKEN TF to derive reference RTF
    _, _, source, _, _, _ = params()

    # Define reference receiver to use
    i_rcv_ref = 0
    gcc_ref = ds_gcc.sel(idx_rcv_ref=i_rcv_ref)
    gcc_library = gcc_ref.gcc_real + 1j * gcc_ref.gcc_imag
    gcc_event = gcc_ref.gcc_event_real + 1j * gcc_ref.gcc_event_imag

    # List position where we want to compare estimated RTF to ref RTF (KRAKEN)
    # Source position + the 4 corners of the grid + one random position inside the grid
    x_check = [
        source["x"],
        ds_gcc.x.min().values,
        ds_gcc.x.min().values,
        ds_gcc.x.max().values,
        ds_gcc.x.max().values,
        ds_gcc.x.values[int(ds_gcc.sizes["x"] * 2 / 3)],
    ]

    y_check = [
        source["y"],
        ds_gcc.y.min().values,
        ds_gcc.y.max().values,
        ds_gcc.y.max().values,
        ds_gcc.y.min().values,
        ds_gcc.y.values[int(ds_gcc.sizes["y"] * 1 / 3)],
    ]

    # Iterate over receivers
    for i_rcv in ds_gcc.idx_rcv.values:

        gcc_l = gcc_library.sel(idx_rcv=i_rcv)
        gcc_e = gcc_event.sel(idx_rcv=i_rcv)

        # Iterate over positions to check
        for i_check in range(len(x_check)):
            x_i = x_check[i_check]
            y_i = y_check[i_check]

            # Extract data at required position
            gcc_l_pos = gcc_l.sel(x=x_i, y=y_i, method="nearest")

            # Due to SCOT weights the module is = 1, relevent information is only contained in the phase of the gcc
            phi_gcc_l = np.unwrap(np.angle(gcc_l_pos))
            phi_gcc_e = np.unwrap(np.angle(gcc_e))

            # Compare library and event gcc
            plt.figure()
            plt.plot(
                ds_gcc.f_gcc,
                phi_gcc_l,
                label=r"$GCC_{" + str(i_rcv) + r"}^{(l)}$",
                linestyle="-",
                color="k",
                linewidth=1.5,
            )

            plt.plot(
                ds_gcc.f_gcc,
                phi_gcc_e,
                linestyle="-",
                label=r"$GCC_{" + str(i_rcv) + r"}^{(e)}$",
                color="r",
                marker="o",
                linewidth=0.2,
                markersize=3,
            )

            plt.legend()
            plt.xlabel(r"$f$" + " [Hz]")
            plt.ylabel(r"$\phi(GCC(f))$")

            # Save figure
            fname = f"check_gcc_rcv{i_rcv}_x{x_i}_y{y_i}.png"
            fpath = os.path.join(root_img, fname)
            plt.savefig(fpath)
            plt.close("all")

    ds_gcc.close()


def study_msr_vs_snr(subarrays_args):
    """Plot metrics (MSR, RMSE) vs SNR for both GCC and RTF"""

    folder = "from_signal_dx20m_dy20m"
    root_img = os.path.join(ROOT_IMG, folder, "perf_vs_snr")
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    for sa_idx, sa_item in subarrays_args.items():
        msr_txt_filepath = sa_item["msr_filepath"]
        dr_txt_filepath = sa_item["dr_pos_filepath"]

        # Load msr results
        # msr_txt_filepath = os.path.join(ROOT_DATA, folder, "msr_snr.txt")
        msr = pd.read_csv(msr_txt_filepath, sep=" ")

        # Compute mean and std of msr for each snr
        msr_mean = msr.groupby("snr").mean()
        msr_std = msr.groupby("snr").std()

        # Plot results
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.errorbar(
            msr_mean.index,
            msr_mean["d_gcc"],
            yerr=msr_std["d_gcc"],
            fmt="o-",
            label="DCF",
        )
        ax.errorbar(
            msr_mean.index,
            msr_mean["d_rtf"],
            yerr=msr_std["d_rtf"],
            fmt="o-",
            label="RTF",
        )
        ax.set_xlabel("NR [dB]")
        ax.set_ylabel("SR [dB]")
        ax.legend()
        ax.grid()
        # plt.show()
        rcv_str = "$" + ", \,".join([f"s_{id+1}" for id in sa_item["idx_rcv"]]) + "$"
        plt.suptitle(f"Receivers = ({rcv_str})")

        fpath = os.path.join(root_img, f"msr_snr_{sa_item['array_label']}.png")
        plt.savefig(fpath)
        plt.close("all")

        # Load position error results
        # dr_txt_filepath = os.path.join(ROOT_DATA, folder, "dr_pos_snr.txt")
        dr = pd.read_csv(dr_txt_filepath, sep=" ")

        # Compute mean and std of position error for each snr
        dr_mean = dr.groupby("snr").mean()
        dr_std = dr.groupby("snr").std()

        # Plot results
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.errorbar(
            dr_mean.index,
            dr_mean["dr_gcc"],
            yerr=dr_std["dr_gcc"],
            fmt="o-",
            label="DCF",
        )
        ax.errorbar(
            dr_mean.index,
            dr_mean["dr_rtf"],
            yerr=dr_std["dr_rtf"],
            fmt="o-",
            label="RTF",
        )
        ax.set_xlabel("SNR [dB]")
        ax.set_ylabel(r"$\Delta_r$" + " [m]")
        ax.legend()
        ax.grid()

        rcv_str = "$" + ", \,".join([f"s_{id+1}" for id in sa_item["idx_rcv"]]) + "$"
        plt.suptitle(f"Receivers = ({rcv_str})")

        fpath = os.path.join(root_img, f"dr_pos_snr_{sa_item['array_label']}.png")
        plt.savefig(fpath)

        dr["dr_gcc"] = dr["dr_gcc"] ** 2
        dr["dr_rtf"] = dr["dr_rtf"] ** 2
        mse = dr.groupby("snr").mean()
        rmse = np.sqrt(mse)

        # Plot results
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(rmse.index, rmse["dr_gcc"], "o-", label="DCF")
        ax.plot(rmse.index, rmse["dr_rtf"], "o-", label="RTF")
        ax.set_xlabel("SNR [dB]")
        ax.set_ylabel("RMSE [m]")
        ax.legend()
        ax.grid()

        rcv_str = "$" + ", \,".join([f"s_{id+1}" for id in sa_item["idx_rcv"]]) + "$"
        plt.suptitle(f"Receivers = ({rcv_str})")

        fpath = os.path.join(root_img, f"rmse_snr_{sa_item['array_label']}.png")
        plt.savefig(fpath)
        plt.close("all")

    # plt.show()


if __name__ == "__main__":
    pass
