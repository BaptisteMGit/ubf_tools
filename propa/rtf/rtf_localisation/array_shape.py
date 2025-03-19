#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   array_shape.py
@Time    :   2025/02/19 11:39:14
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

from publication.PublicationFigure import PubFigure
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import (
    params,
    load_random_antenna,
    get_subarrays,
    load_msr_rmse_res_subarrays,
)
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_params import (
    ROOT_IMG,
    USE_TEX,
)

pfig = PubFigure(use_tex=USE_TEX)


def random_array(rmax, nr):

    theta = np.random.rand(nr) * 2 * np.pi
    theta = np.sort(theta)  # Order for easier interpretation
    rho = np.random.rand(nr) * rmax
    x_rcv = rho * np.cos(theta)
    y_rcv = rho * np.sin(theta)

    return x_rcv, y_rcv


def angle_between_vectors(u, v):
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)

    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return theta_rad


def triangle_area(x_rcv, y_rcv):
    # Derive sides of the triangles
    x_rcv = np.append(x_rcv, x_rcv[0])
    y_rcv = np.append(y_rcv, y_rcv[0])

    sides = []
    for i_v in range(len(x_rcv) - 1):
        u_i_j = [
            x_rcv[i_v + 1] - x_rcv[i_v],
            y_rcv[i_v + 1] - y_rcv[i_v],
        ]
        sides.append(np.linalg.norm(u_i_j))

    # Apply Heron's formula to derive the area of the triangle
    a, b, c = sides
    s = np.sum(sides) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))

    return area


def gamma1(x_rcv, y_rcv, xs, ys):
    """
    First metric to evaluate the potential of the array
    """

    # Derive receivers barycentre
    xb = np.mean(x_rcv)
    yb = np.mean(y_rcv)

    # Vector from barycentre to source
    v_bs = [xs - xb, ys - yb]

    u_br = []
    theta_rad = []
    for i in range(len(x_rcv)):
        u_br_i = [x_rcv[i] - xb, y_rcv[i] - yb]
        u_br.append(u_br_i)
        theta_rad.append(angle_between_vectors(u_br_i, v_bs))
    theta_rad = np.array(theta_rad)
    u_br = np.array(u_br)

    u_br_norm = np.linalg.norm(u_br, axis=1)
    # gamma_ = np.sum(np.abs(np.sin(theta_rad)) * u_br_norm)
    gamma_ = np.mean(np.abs(np.sin(theta_rad)) * u_br_norm)

    return gamma_


def gamma2(x_rcv, y_rcv):
    """
    Second metric to evaluate the potential of the array
    """
    nr = len(x_rcv)
    alpha_rad = []
    # Iterate over the vertices
    for i_v in range(nr):
        angle_vects = []
        for j_v in range(nr):
            if i_v != j_v:
                u_i_j = [
                    x_rcv[j_v] - x_rcv[i_v],
                    y_rcv[j_v] - y_rcv[i_v],
                ]
                angle_vects.append(u_i_j)

        alpha_rad.append(angle_between_vectors(angle_vects[0], angle_vects[1]))

    alpha_min = np.min(alpha_rad)

    # Derive array area
    area = triangle_area(x_rcv, y_rcv)

    gamma_ = np.sin(alpha_min * 3 / 2) * area

    return gamma_


def generate_and_eval_arrays(n_arrays, rmax, nr, xs, ys, ncol=5):

    # Generate arrays
    x_arrays = []
    y_arrays = []
    for i in range(n_arrays):
        x_rcv, y_rcv = random_array(rmax, nr)
        # Store vars
        x_arrays.append(x_rcv)
        y_arrays.append(y_rcv)
    x_arrays = np.array(x_arrays)
    y_arrays = np.array(y_arrays)

    gamma = {"1": [], "2": []}
    for i in range(n_arrays):
        x_rcv = x_arrays[i, :]
        y_rcv = y_arrays[i, :]
        gamma["1"].append(gamma1(x_rcv, y_rcv, xs, ys))
        gamma["2"].append(gamma2(x_rcv, y_rcv))

    # Normalise gamma
    gamma["1"] = np.array(gamma["1"]) / np.max(gamma["1"])
    gamma["2"] = np.array(gamma["2"]) / np.max(gamma["2"])
    gamma["12"] = (gamma["1"] + gamma["2"]) / 2

    nrow = int(np.ceil(n_arrays / ncol))

    for gamma_f in ["1", "2", "12"]:
        gamma_arrays = gamma[gamma_f]
        # Sort by gamma values
        x_arrays = x_arrays[np.argsort(gamma_arrays)]
        y_arrays = y_arrays[np.argsort(gamma_arrays)]
        gamma_arrays = np.sort(gamma_arrays)

        # Barycentre
        xb_arrays = np.mean(x_arrays, axis=1)
        yb_arrays = np.mean(y_arrays, axis=1)

        # Plot all arrays shape and associated gamma values
        f, axs = plt.subplots(
            nrows=nrow, ncols=ncol, figsize=(18, 10), sharex=True, sharey=True
        )

        xlim = [-1.1 * rmax, 1.25 * rmax]
        ylim = [-1.5 * rmax, 1.25 * rmax]

        for i in range(n_arrays):
            ax = axs.flatten()[i]
            x_r = np.append(x_arrays[i, :], (x_arrays[i, 0]))
            y_r = np.append(y_arrays[i, :], (y_arrays[i, 0]))
            ax.plot(x_r, y_r, linestyle="--", marker="o", markersize=5, color="k")
            ax.scatter(
                xb_arrays[i], yb_arrays[i], marker="o", color="r", s=5, label=r"$X_b$"
            )
            ax.plot(
                [xb_arrays[i], xs],
                [yb_arrays[i], ys],
                linestyle="--",
                color="r",
            )

            ax.set_title(r"$\gamma_{" + gamma_f + "} = " + f"{gamma_arrays[i]:.3f}$")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        plt.savefig(f"test_{gamma_f}")

    print()


def eval_arrays(subarrays_list, antenna_type, snrs, ncol=3, dx=20, dy=20):

    # Min / max subarray list
    sa_sizes = np.unique([len(sa) for sa in subarrays_list])
    if len(sa_sizes) > 1:
        subarrays_size_info = (
            f"subarrays_with_{np.min(sa_sizes)}_to_{np.max(sa_sizes)}_receivers"
        )
    else:
        subarrays_size_info = f"subarrays_with_{sa_sizes[0]}_receivers"

    folder = "from_signal_dx20m_dy20m"
    root_img = os.path.join(
        ROOT_IMG, folder, "analyse_arrays", antenna_type, subarrays_size_info
    )
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    _, receivers, source, _, _, _ = params(antenna_type=antenna_type)
    xs, ys = source["x"], source["y"]

    # Load receivers
    if antenna_type == "zhang":
        x_rcv = receivers["x"]
        y_rcv = receivers["y"]
        r = np.sqrt(x_rcv**2 + y_rcv**2)
        rmax = np.round(np.max(r), 0)
    elif antenna_type == "random":
        rmax = 1.5e3  # 1.5km radius
        nr = 6  # Same number of receivers as zhang
        x_rcv, y_rcv = load_random_antenna(rmax, nr)

    # Generate arrays
    n_arrays = len(subarrays_list)
    x_arrays = []
    y_arrays = []
    for i_sa, sa in enumerate(subarrays_list):
        # Store vars
        x_arrays.append(x_rcv[sa])
        y_arrays.append(y_rcv[sa])

    gamma = {"1": []}
    for i in range(n_arrays):
        x_rcv = x_arrays[i]
        y_rcv = y_arrays[i]
        gamma["1"].append(gamma1(x_rcv, y_rcv, xs, ys))
        # gamma["2"].append(gamma2(x_rcv, y_rcv))

    # Normalise gamma
    gamma["1"] = np.array(gamma["1"]) / np.max(gamma["1"])
    # gamma["2"] = np.array(gamma["2"]) / np.max(gamma["2"])
    # gamma["12"] = (gamma["1"] + gamma["2"]) / 2

    nrow = int(np.ceil(n_arrays / ncol))

    for gamma_f in ["1"]:
        gamma_arrays = gamma[gamma_f]
        gamma_order = np.argsort(gamma_arrays)
        # Sort by gamma values
        x_arrays = [x_arrays[i] for i in gamma_order]
        y_arrays = [y_arrays[i] for i in gamma_order]
        gamma_arrays = np.sort(gamma_arrays)

        # Barycentre
        xb_arrays = [np.mean(x_ar) for x_ar in x_arrays]
        yb_arrays = [np.mean(y_ar) for y_ar in y_arrays]

        # Plot all arrays shape and associated gamma values
        f, axs = plt.subplots(
            nrows=nrow, ncols=ncol, figsize=(38, 20), sharex=True, sharey=True
        )

        xlim = [-1.1 * rmax, 1.25 * rmax]
        ylim = [-1.5 * rmax, 1.25 * rmax]

        for i in range(n_arrays):
            ax = axs.flatten()[i]
            x_r = np.append(x_arrays[i], (x_arrays[i][0]))
            y_r = np.append(y_arrays[i], (y_arrays[i][0]))
            ax.plot(x_r, y_r, linestyle="--", marker="o", markersize=5, color="k")
            ax.scatter(
                xb_arrays[i], yb_arrays[i], marker="o", color="r", s=5, label=r"$X_b$"
            )
            ax.plot(
                [xb_arrays[i], xs],
                [yb_arrays[i], ys],
                linestyle="--",
                color="r",
            )

            ax.set_title(r"$\gamma_{" + gamma_f + "} = " + f"{gamma_arrays[i]:.3f}$")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        fpath = os.path.join(root_img, "subarrays_gamma1_score.png")
        plt.savefig(fpath, dpi=300)

    # Load msr and rmse results to plot them vs gamma
    # msr_mu, msr_sig, dr_mu, dr_sig, rmse_ = load_msr_rmse_res_subarrays(
    #     subarrays_list, snrs, dx, dy
    # )

    msr, dr, rmse = load_msr_rmse_res_subarrays(subarrays_list, snrs, dx, dy)

    for snr in snrs:
        # Plot RMSE vs nr_in_sa
        rmse_gcc = [rmse[key]["dcf"].loc[snr] for key in list(rmse.keys())]
        rmse_rtf = [rmse[key]["rtf"].loc[snr] for key in list(rmse.keys())]
        # Order by gamma
        rmse_gcc = np.array(rmse_gcc)[gamma_order]
        rmse_rtf = np.array(rmse_rtf)[gamma_order]

        plt.figure(figsize=(8, 6))
        plt.plot(gamma_arrays, rmse_gcc, "o-", label="DCF")
        plt.plot(gamma_arrays, rmse_rtf, "o-", label="RTF")
        plt.xlabel(r"$\gamma_1$")
        plt.ylabel("RMSE [m]")
        plt.title(f"SNR = {snr} dB")
        plt.legend()

        fpath = os.path.join(root_img, f"rmse_vs_gamma1_snr{snr}.png")
        plt.savefig(fpath, dpi=300)
        plt.close("all")

        # Plot DR vs nr_in_sa
        dr_gcc = [dr[key]["dcf_mean"].loc[snr] for key in list(dr.keys())]
        dr_rtf = [dr[key]["rtf_mean"].loc[snr] for key in list(dr.keys())]

        # dr_gcc = [dr["dr_gcc"].loc[snr] for dr in dr_mu]
        # dr_rtf = [dr["dr_rtf"].loc[snr] for dr in dr_mu]
        # Order by gamma
        dr_gcc = np.array(dr_gcc)[gamma_order]
        dr_rtf = np.array(dr_rtf)[gamma_order]

        plt.figure(figsize=(8, 6))
        plt.plot(gamma_arrays, dr_gcc, "o-", label="DCF")
        plt.plot(gamma_arrays, dr_rtf, "o-", label="RTF")
        plt.xlabel(r"$\gamma_1$")
        plt.ylabel("DR [dB]")
        plt.title(f"SNR = {snr} dB")
        plt.legend()

        fpath = os.path.join(root_img, f"dr_vs_gamma1_snr{snr}.png")
        plt.savefig(fpath, dpi=300)

        # Plot MSR vs nr_in_sa
        msr_gcc = [msr[key]["dcf_mean"].loc[snr] for key in list(msr.keys())]
        msr_rtf = [msr[key]["rtf_mean"].loc[snr] for key in list(msr.keys())]

        # msr_gcc = [msr["d_gcc"].loc[snr] for msr in msr_mu]
        # msr_rtf = [msr["d_rtf"].loc[snr] for msr in msr_mu]
        # Order by gamma
        msr_gcc = np.array(msr_gcc)[gamma_order]
        msr_rtf = np.array(msr_rtf)[gamma_order]

        plt.figure(figsize=(8, 6))
        plt.plot(gamma_arrays, msr_gcc, "o-", label="DCF")
        plt.plot(gamma_arrays, msr_rtf, "o-", label="RTF")
        plt.xlabel(r"$\gamma_1$")
        plt.ylabel("MSR [m]")
        plt.title(f"SNR = {snr} dB")
        plt.legend()

        fpath = os.path.join(root_img, f"msr_vs_gamma1_snr{snr}.png")
        plt.savefig(fpath, dpi=300)
        plt.close("all")

        sort_and_plot_by_perf_order(
            rmse_gcc, "gcc", "rmse", x_arrays, y_arrays, xs, ys, rmax, ncol, root_img
        )
        sort_and_plot_by_perf_order(
            rmse_rtf, "rtf", "rmse", x_arrays, y_arrays, xs, ys, rmax, ncol, root_img
        )
        sort_and_plot_by_perf_order(
            msr_gcc, "gcc", "msr", x_arrays, y_arrays, xs, ys, rmax, ncol, root_img
        )
        sort_and_plot_by_perf_order(
            msr_rtf, "rtf", "msr", x_arrays, y_arrays, xs, ys, rmax, ncol, root_img
        )

        # Order arrays by hybrid metric
        hybrid_gcc = rmse_gcc / np.max(rmse_gcc) + (
            1 + msr_gcc / np.max(np.abs(msr_gcc))
        )
        hybrid_rtf = rmse_rtf / np.max(rmse_rtf) + (
            1 + msr_rtf / np.max(np.abs(msr_rtf))
        )
        sort_and_plot_by_perf_order(
            hybrid_gcc,
            "gcc",
            "hybrid",
            x_arrays,
            y_arrays,
            xs,
            ys,
            rmax,
            ncol,
            root_img,
        )
        sort_and_plot_by_perf_order(
            hybrid_rtf,
            "rtf",
            "hybrid",
            x_arrays,
            y_arrays,
            xs,
            ys,
            rmax,
            ncol,
            root_img,
        )


def sort_and_plot_by_perf_order(
    metric_values,
    method_name,
    metric_name,
    x_arrays,
    y_arrays,
    xs,
    ys,
    rmax,
    ncol,
    root_img,
):
    # Order subarrays by rmse and plot all subarrays
    metric_order = np.argsort(metric_values)[::-1]  # Descending order
    # Sort by gamma values
    x_arrays = [x_arrays[i] for i in metric_order]
    y_arrays = [y_arrays[i] for i in metric_order]
    metric_values = metric_values[metric_order]

    # Barycentre
    xb_arrays = [np.mean(x_ar) for x_ar in x_arrays]
    yb_arrays = [np.mean(y_ar) for y_ar in y_arrays]

    # Plot all arrays shape and associated rmse values
    n_arrays = len(x_arrays)
    nrow = int(np.ceil(n_arrays / ncol))
    f, axs = plt.subplots(
        nrows=nrow, ncols=ncol, figsize=(38, 20), sharex=True, sharey=True
    )

    xlim = [-1.1 * rmax, 1.25 * rmax]
    ylim = [-1.5 * rmax, 1.25 * rmax]

    for i in range(n_arrays):
        ax = axs.flatten()[i]
        x_r = np.append(x_arrays[i], (x_arrays[i][0]))
        y_r = np.append(y_arrays[i], (y_arrays[i][0]))
        ax.plot(x_r, y_r, linestyle="--", marker="o", markersize=5, color="k")
        ax.scatter(
            xb_arrays[i], yb_arrays[i], marker="o", color="r", s=5, label=r"$X_b$"
        )
        ax.plot(
            [xb_arrays[i], xs],
            [yb_arrays[i], ys],
            linestyle="--",
            color="r",
        )

        ax.set_title(f"{metric_values[i]:.3f}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fpath = os.path.join(
        root_img, f"subarrays_by_{metric_name}_{method_name}_order.png"
    )
    plt.savefig(fpath, dpi=300)


if __name__ == "__main__":
    # nr = 3
    # n_arrays = 35
    # rmax = 1.5e3  # 1km radius
    # xs, ys = 3900, 6800
    # x_rcv, y_rcv = random_array(rmax, nr)
    # gamma(x_rcv, y_rcv, xs, ys)

    # generate_and_eval_arrays(n_arrays, rmax, nr, xs, ys, ncol=7)

    # subarrays_list = [
    #     [0, 1],
    #     [0, 1, 2],
    #     [0, 1, 2, 3],
    #     [0, 1, 2, 3, 4],
    #     [0, 1, 2, 3, 4, 5],
    # ]

    # eval_random_array(subarrays_list, xs=xs, ys=ys, ncol=5)
    # All possible subarrays with n_rcv receivers
    subarrays_list = []
    # n_rcv = [2, 3, 4, 5, 6]
    n_rcv = [3]
    for i in n_rcv:
        subarrays_list += list(get_subarrays(nr_fullarray=6, nr_subarray=i))
    eval_arrays(subarrays_list, antenna_type="zhang", snrs=[-15], ncol=7)
    print()
