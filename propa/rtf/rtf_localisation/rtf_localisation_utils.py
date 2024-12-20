#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_localisation_utils.py
@Time    :   2024/11/06 14:02:23
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

# ======================================================================================================================
# Functions
# ======================================================================================================================


def plot_ambiguity_surface(amb_surf, r_src, z_src, plot_args):

    snr = plot_args["snr"]
    dist = plot_args["dist"]
    testcase = plot_args["testcase"]
    root_img = plot_args["root_img"]
    rtf_method = plot_args["rtf_method"]
    mfp_method = plot_args["mfp_method"]
    dist_label = plot_args["dist_label"]
    vmax_percentile = plot_args["vmax_percentile"]

    # Estimated source position defined as the minimum of the ambiguity surface
    z_min_cs, r_min_cs = np.unravel_index(np.argmin(amb_surf.values), amb_surf.shape)
    r_src_hat = amb_surf.r[r_min_cs]
    z_src_hat = amb_surf.z[z_min_cs]

    vmin = amb_surf.min()
    vmax = np.percentile(amb_surf, vmax_percentile)

    plt.figure()
    amb_surf.plot(
        x="r",
        y="z",
        cmap="jet_r",
        vmin=vmin,
        vmax=vmax,
        cbar_kwargs={"label": f"{dist_label}"},
    )
    plt.gca().invert_yaxis()
    true_pos_label = (
        r"$X_{src} = ( "
        + f"{r_src*1e-3:.2f}\,"
        + r"\textrm{km},\,"
        + f"{z_src:.2f}\,"
        + r"\textrm{m})$"
    )
    estimated_pos_label = (
        r"$\hat{X}_{src} = ( "
        + f"{r_src_hat*1e-3:.2f}\,"
        + r"\textrm{km},\,"
        + f"{z_src_hat:.2f}\,"
        + r"\textrm{m})$"
    )
    # estimated_pos_label = r"$\hat{X}_{src}" + f" = ({r_src_hat:.2f}, {z_src_hat:.2f})$"
    plt.scatter(
        r_src_hat, z_src_hat, color="w", marker="o", label=estimated_pos_label, s=100
    )  # Estimated source position
    plt.scatter(
        r_src, z_src, color="k", marker="x", label=true_pos_label, s=100
    )  # True source position
    plt.xlabel(r"$r \, \textrm{[m]}$")
    plt.ylabel(r"$z \, \textrm{[m]}$")
    plt.legend()

    # plt.title(title)

    # Save figure
    path = os.path.join(root_img, mfp_method, testcase, f"snr_{snr}dB")
    if not os.path.exists(path):
        os.makedirs(path)

    fname = f"amb_surf_{rtf_method}_{dist}.png"
    fpath = os.path.join(path, fname)
    plt.savefig(fpath)
