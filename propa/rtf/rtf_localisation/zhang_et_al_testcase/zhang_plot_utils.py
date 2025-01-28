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

from matplotlib.path import Path
from scipy.spatial import ConvexHull
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import *

# ======================================================================================================================
# Functions
# ======================================================================================================================


def plot_study_zhang2023(folder):
    # Load params
    depth, receivers, source, grid, frequency, _ = params()

    # Define folder to store images
    root_img = os.path.join(ROOT_IMG, folder)
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    # Define folder to store data
    root_data = os.path.join(ROOT_DATA, folder)
    if not os.path.exists(root_data):
        os.makedirs(root_data)

    # Define plot args for ambiguity surfaces
    plot_args_theta = {
        "dist": "hermitian_angle",
        "root_img": root_img,
        "testcase": "zhang_et_al_2023",
        "dist_label": r"$\theta \, \textrm{[°]}$",
        "vmax": 50,
        "vmin": 0,
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
        "vmin": -10,
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
        "vmin": -10,
    }

    ###### Two sensor pairs ######
    # Select receivers to build the sub-array
    rcv_couples = np.array([[0, 2], [1, 4], [3, 5]])  # s1s3, s2s5, s4s6
    for rcv_cpl in rcv_couples:
        fpath = os.path.join(
            root_data,
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
        root_data,
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
            root_data,
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
                ax.set_xlabel(r"$x \, \textrm{[m]}$")
            else:
                ax.set_xlabel("")
            if i_cpl == 0:
                ax.set_ylabel(r"$y \, \textrm{[m]}$")
            else:
                ax.set_ylabel("")

            # # Set xticks
            ax.set_xticks([3500, 4000, 4500])
            ax.set_yticks([6400, 6900, 7400])

    # Save figure
    fpath = os.path.join(root_img, "loc_zhang2023_fig4.png")
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
            x_src * 1e-3,
            y_src * 1e-3,
            facecolors="none",
            edgecolors="k",
            label=true_pos_label,
            s=200,
            linewidths=3,
        )

        ax.set_title(f"Full array")
        ax.set_xlabel(r"$x \textrm{[m]}$")
        if i == 0:
            ax.set_ylabel(r"$y \, \textrm{[m]}$")
        else:
            ax.set_ylabel("")

        # # Set xticks
        ax.set_xticks([3500, 4000, 4500])
        ax.set_yticks([6400, 6900, 7400])

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

        ax.set_title(f"Full array")
        ax.set_xlim(-20, 0)
        ax.set_xlabel(r"$\textrm{[dB]}$")

    # Save figure
    fpath = os.path.join(root_img, "loc_zhang2023_fig5_dist.png")
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
            amb_surf.values,
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
    fpath = os.path.join(root_img, "loc_zhang2023_fig5_mainlobe.png")
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
        amb_surf_without_mainlobe = amb_surf_without_mainlobe.values
        amb_surf_without_mainlobe[mainlobe_mask] = np.nan

        im = ax.pcolormesh(
            ds_fa["x"].values * 1e-3,
            ds_fa["y"].values * 1e-3,
            amb_surf_without_mainlobe.T,
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
    fpath = os.path.join(root_img, "loc_zhang2023_fig5_nomainlobe.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close("all")


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
    pass
