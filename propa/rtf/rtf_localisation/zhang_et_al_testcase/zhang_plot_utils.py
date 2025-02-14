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
import scipy.signal as sp
import matplotlib.pyplot as plt

from misc import compute_hyperbola
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import *

# ======================================================================================================================
# Functions
# ======================================================================================================================


def plot_study_zhang2023(folder, data_fname=None):
    # Load params
    _, _, source, grid, _, _ = params()

    # Define folder to store images
    root_img = os.path.join(ROOT_IMG, folder)
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    # Define folder to store data
    root_data = os.path.join(ROOT_DATA, folder)
    if not os.path.exists(root_data):
        os.makedirs(root_data)

    # Load fullarray data
    if data_fname is None:
        data_fname_fa = f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_fullarray.nc"
    else:
        data_fname_fa = f"{data_fname}_fullarray.nc"
    fpath = os.path.join(
        root_data,
        # f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_fullarray.nc",
        data_fname_fa,
    )
    ds_fa = xr.open_dataset(fpath)

    vmin_dB = np.round(np.max([ds_fa[dist].median() for dist in ["d_gcc", "d_rtf"]]), 0)

    # Define plot args for ambiguity surfaces
    plot_args_theta = {
        "dist": "hermitian_angle",
        "root_img": root_img,
        "testcase": "zhang_et_al_2023",
        "dist_label": r"$\theta \, \textrm{[°]}$",
        "vmax": 50,
        "vmin": 0,
        "add_hyperbola": True,
    }

    plot_args_d_rtf = {
        "dist": "normalized_metric",
        "root_img": root_img,
        "testcase": "zhang_et_al_2023",
        # "dist_label": r"$d_{rtf}$",
        "dist_label": r"$\textrm{[dB]}$",
        # "vmax": 1,
        # "vmin": 0,
        # dB scale
        "vmax": 0,
        "vmin": vmin_dB,
        "add_hyperbola": True,
    }

    plot_args_gcc = {
        "dist": "gcc_scot",
        "root_img": root_img,
        "testcase": "zhang_et_al_2023",
        # "dist_label": r"$d_{gcc}$",
        "dist_label": r"$\textrm{[dB]}$",
        # "vmax": 1,
        # "vmin": 0,
        # dB scale
        "vmax": 0,
        "vmin": vmin_dB,
        "add_hyperbola": True,
    }

    ###### Two sensor pairs ######
    # Select receivers to build the sub-array
    # rcv_couples = np.array([[0, 2], [1, 4], [3, 5]])  # s1s3, s2s5, s4s6
    rcv_couples = get_rcv_couples(ds_fa.idx_rcv)

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

    # Update sub array args
    plot_args_theta["sub_array"] = None
    plot_args_d_rtf["sub_array"] = None
    plot_args_gcc["sub_array"] = None
    plot_args_theta["add_circle"] = True
    plot_args_d_rtf["add_circle"] = True
    plot_args_gcc["add_circle"] = True

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
    true_pos_label = (
        r"$X_{src} = ( "
        + f"{x_src:.0f}\,"
        + r"\textrm{m},\,"
        + f"{y_src:.0f}\,"
        + r"\textrm{m})$"
    )

    f, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)

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
                cbar_kwargs={"label": r"$\textrm{[dB]}$"},
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
                ax.set_xlabel(r"$x \, \textrm{[m]}$")
            else:
                ax.set_xlabel("")
            if i_cpl == 0:
                ax.set_ylabel(r"$y \, \textrm{[m]}$")
            else:
                ax.set_ylabel("")

            # # Set xticks
            # ax.set_xticks([3500, 4000, 4500])
            # ax.set_yticks([6400, 6900, 7400])
            ax.set_xticks(xticks_pos_m)
            ax.set_yticks(yticks_pos_m)
            # ax.set_xticklabels(xticks_label_km, fontsize=22)
            # ax.set_yticklabels(yticks_label_km, fontsize=22)

    # Save figure
    fpath = os.path.join(root_img, "loc_zhang2023_fig4.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close("all")

    ###### Figure 5 : Subplot in Zhang et al 2023 ######
    f, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Plot d_gcc and d_rtf
    for i, dist in enumerate(["d_gcc", "d_rtf"]):
        ax = axs[i]
        amb_surf = ds_fa[dist]

        # im = ax.pcolormesh(
        #     ds_fa["x"].values * 1e-3,
        #     ds_fa["y"].values * 1e-3,
        #     amb_surf.values,
        #     cmap=cmap,
        #     vmin=vmin,
        #     vmax=vmax,
        # )
        im = amb_surf.plot(
            x="x",
            y="y",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            extend="neither",
            cbar_kwargs={"label": r"$\textrm{[dB]}$"},
            # robust=True,
            # cbar_kwargs={"label": dist_label},
        )

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
            s=200,
            linewidths=2,
        )

        ax.set_title(r"$\textrm{Full array}$")
        ax.set_xlabel(r"$x \textrm{[m]}$")
        if i == 0:
            ax.set_ylabel(r"$y \, \textrm{[m]}$")
        else:
            ax.set_ylabel("")

        # # Set xticks
        # ax.set_xticks([3500, 4000, 4500])
        # ax.set_yticks([6400, 6900, 7400])
        # # Set xticks
        ax.set_xticks(xticks_pos_m)
        ax.set_yticks(yticks_pos_m)
        # ax.set_xticklabels(xticks_label_km, fontsize=22)
        # ax.set_yticklabels(yticks_label_km, fontsize=22)

    # Save figure
    fpath = os.path.join(root_img, "loc_zhang2023_fig5.png")
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

        ax.set_title(r"$\textrm{Full array}$")
        ax.set_xlim(-20, 0)
        ax.set_xlabel(r"$\textrm{[dB]}$")

    # Save figure
    fpath = os.path.join(root_img, "loc_zhang2023_fig5_dist.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close("all")

    ###### Figure 5 showing pixels selected as the mainlobe ######

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
            cbar_kwargs={"label": r"$\textrm{[dB]}$"},
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
        ax.set_xlabel(r"$x \textrm{[m]}$")
        if i == 0:
            ax.set_ylabel(r"$y \, \textrm{[m]}$")
        else:
            ax.set_ylabel("")

        # Set xticks
        ax.set_xticks(xticks_pos_m)
        ax.set_yticks(yticks_pos_m)
        # ax.set_xticklabels(xticks_label_km, fontsize=22)
        # ax.set_yticklabels(yticks_label_km, fontsize=22)

    # Save figure
    fpath = os.path.join(root_img, "loc_zhang2023_fig5_mainlobe.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close("all")

    estimate_msr(ds_fa=ds_fa, plot=False, root_img=root_img, verbose=True)


def plot_ambiguity_surface(amb_surf, source, plot_args, loc_arg):

    dist = plot_args["dist"]
    testcase = plot_args["testcase"]
    root_img = plot_args["root_img"]
    dist_label = plot_args["dist_label"]
    vmax = plot_args["vmax"]
    vmin = plot_args["vmin"]
    sub_array = plot_args["sub_array"]

    # To plot the hyperbola corresponding to TDOA
    add_hyperbola = plot_args.get("add_hyperbola", False)
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

    # Add hyperbola if required
    if add_hyperbola:
        # print("Add hyperbola")
        src_pos = (x_src, y_src)

        if sub_array is None:
            # Compute hyperbola for each pair of receivers
            for i, sa in enumerate([[0, 2], [1, 4], [3, 5]]):
                receiver1 = (receivers["x"][sa[0]], receivers["y"][sa[0]])
                receiver2 = (receivers["x"][sa[1]], receivers["y"][sa[1]])
                (right_branch, left_branch) = compute_hyperbola(
                    receiver1, receiver2, src_pos
                )

                # Plot both branches
                plt.plot(
                    right_branch[0], right_branch[1], "k", linestyle="--", zorder=10
                )
                plt.plot(left_branch[0], left_branch[1], "k", linestyle="--", zorder=10)

        else:
            receiver1 = (receivers["x"][sub_array[0]], receivers["y"][sub_array[0]])
            receiver2 = (receivers["x"][sub_array[1]], receivers["y"][sub_array[1]])
            (right_branch, left_branch) = compute_hyperbola(
                receiver1, receiver2, src_pos, tmax=5
            )

            # Plot both branches
            plt.plot(right_branch[0], right_branch[1], "k", linestyle="--", zorder=10)
            plt.plot(left_branch[0], left_branch[1], "k", linestyle="--", zorder=10)

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
    plt.close("all")


def check_signal_noise(ds_sig_noise):
    """
    Plot library signal at source position and event signal aswell as associated noise signals to check that the dataset is built as required.
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


def check_rtf_features(ds_rtf_cs, folder):

    # Define folder to store images
    root_img = os.path.join(ROOT_IMG, folder, "check_rtf")
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    # Load dataset with KRAKEN TF to derive reference RTF
    _, _, source, grid, frequency, _ = params()

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
            plt.xlabel(r"$f \, \textrm{[Hz]}$")
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
    _, _, source, grid, frequency, _ = params()

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
            plt.xlabel(r"$f \, \textrm{[Hz]}$")
            plt.ylabel(r"$\phi(GCC(f))$")

            # Save figure
            fname = f"check_gcc_rcv{i_rcv}_x{x_i}_y{y_i}.png"
            fpath = os.path.join(root_img, fname)
            plt.savefig(fpath)
            plt.close("all")

    ds_gcc.close()


if __name__ == "__main__":
    pass
