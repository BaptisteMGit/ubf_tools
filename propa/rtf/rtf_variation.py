#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_variation.py
@Time    :   2024/09/27 16:30:32
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
from propa.rtf.ideal_waveguide import *

from publication.PublicationFigure import PubFigure

PubFigure(label_fontsize=22, title_fontsize=24, legend_fontsize=16, ticks_fontsize=20)

# ======================================================================================================================
# Functions
# ======================================================================================================================


def PI_var_r(
    depth,
    f,
    r_src,
    z_src,
    x_rcv,
    z_rcv,
    covered_range,
    dr,
    dist="both",
    smooth_tf=False,
    bottom_bc="pressure_release",
):

    # Create folder to store the images
    root, folder = get_folderpath(
        x_rcv,
        r_src,
        z_src,
        bottom_bc=bottom_bc,
    )
    subfolder = f"dr_{dr}"
    root = os.path.join(root, folder, subfolder)
    if not os.path.exists(root):
        os.makedirs(root)

    n_rcv = len(x_rcv)

    # Define the ranges
    r_src_list = np.arange(r_src - covered_range, r_src + covered_range, dr)

    # Ensure that the ref position is included in the list
    r_src_list = np.sort(np.append(r_src_list, r_src))
    # Cast to required shape depending on the number of receivers
    r_src_list_2D = np.tile(r_src_list, (len(x_rcv) - 1, 1)).T
    nb_pos_r = len(r_src_list)
    r_src_rcv_ref = r_src_list_2D - x_rcv[0]
    r_src_rcv = r_src_list_2D - x_rcv[1:]

    # Derive the RTF vector for the reference source position
    r_ref = r_src - x_rcv
    # Derive the reference RTF vector g_ref is of shape (nf, nrcv, 1)
    f, g_ref = g_mat(
        f,
        z_src,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        depth=depth,
        r_rcv_ref=r_ref[0],
        r=r_ref[1:],
        smooth_tf=smooth_tf,
    )

    f, g_r = g_mat(
        f,
        z_src,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        depth=depth,
        r_rcv_ref=r_src_rcv_ref,
        r=r_src_rcv,
        smooth_tf=smooth_tf,
    )

    # Expand g_ref to the same shape as g_r
    tile_shape = tuple([g_r.shape[i] - g_ref.shape[i] + 1 for i in range(g_r.ndim)])
    g_ref_expanded = np.tile(g_ref, tile_shape)

    # # Derive generalised distance combining all receivers
    Df = D_frobenius(g_ref, g_r)

    if dist == "D1" or dist == "both":
        d1 = np.sum(np.abs(g_ref_expanded - g_r), axis=0)
        d1 += 1
        d1 = 10 * np.log10(d1)
    if dist == "D2" or dist == "both":
        d2 = np.sum(np.abs(g_ref_expanded - g_r) ** 2, axis=0)
        d2 += 1
        d2 = 10 * np.log10(d2)

    range_displacement = r_src_list - r_src

    # Plot generalized distance map
    plt.figure()
    plt.plot(
        range_displacement,
        Df,
    )
    plt.ylabel(r"$10log_{10}(1+D)\, \textrm{[dB]}$")
    plt.xlabel(r"$r - r_s \, \textrm{[m]}$")
    plt.title(r"$\mathcal{D}_F(r, z=z_s)$")
    plt.grid()

    # Save
    fpath = os.path.join(
        root,
        f"D_frobenius_r_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}.png",
    )
    plt.savefig(fpath)

    # Iterate over receivers
    d1max = 0
    d2max = 0
    plt.figure()

    for i_rcv in range(n_rcv - 1):
        if dist == "D1" or dist == "both":
            plt.plot(
                range_displacement,
                d1[:, i_rcv, 0],
                label=r"$D_1$" + r"$\,\, (\Pi_{" + f"{i_rcv+1},0" + "})$",
            )
            d1max = np.max(d1)
        if dist == "D2" or dist == "both":
            plt.plot(
                range_displacement,
                d2[:, i_rcv, 0],
                label=r"$D$" + r"$\,\, (\Pi_{" + f"{i_rcv+1},0" + "})$",
            )
            d2max = np.max(d2)

    plt.plot(range_displacement, Df, label=r"$\mathcal{D}_F$", color="k")

    if dist == "both" or n_rcv > 2:
        plt.legend()

    if dist == "both":
        dist_lab = "D1D2"
    else:
        dist_lab = dist

    plt.ylim(0, max(d1max, d2max) + 5)
    plt.xlabel(r"$r - r_s \, \textrm{[m]}$")
    plt.ylabel(r"$10log_{10}(1+D)\, \textrm{[dB]}$")
    plt.grid()

    # Save
    fpath = os.path.join(
        root,
        f"{dist_lab}_r_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}.png",
    )
    plt.savefig(fpath)
    plt.close("all")


def PI_var_z(
    depth,
    f,
    r_src,
    z_src,
    x_rcv,
    z_rcv,
    zmin,
    zmax,
    dz,
    dist="both",
    bottom_bc="pressure_release",
):
    # Create folder to store the images
    root, folder = get_folderpath(x_rcv, r_src, z_src, bottom_bc=bottom_bc)
    subfolder = f"dz_{dz}"
    root = os.path.join(root, folder, subfolder)
    if not os.path.exists(root):
        os.makedirs(root)

    n_rcv = len(x_rcv)

    # Define detphs
    zmin = np.max([1, zmin])  # Avoid negative depths
    z_src_list = np.arange(zmin, zmax, dz)

    # Ensure that the ref position is included in the list
    z_src_list = np.sort(np.append(z_src_list, z_src))
    nb_pos_z = len(z_src_list)
    r_src_rcv_ref = r_src - x_rcv[0]
    r_src_rcv = r_src - x_rcv[1:]

    # Derive the reference RTF vector, g_ref is of shape (nf, nrcv, 1)
    r_ref = r_src - x_rcv

    f, g_ref = g_mat(
        f,
        z_src,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        depth=depth,
        r_rcv_ref=r_ref[0],
        r=r_ref[1:],
    )

    f, g_z = g_mat(
        f,
        z_src_list,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        depth=depth,
        r_rcv_ref=r_src_rcv_ref,
        r=r_src_rcv,
    )

    # Expand g_ref to the same shape as g_z
    tile_shape = tuple([g_z.shape[i] - g_ref.shape[i] + 1 for i in range(g_z.ndim)])
    g_ref_expanded = np.tile(g_ref, tile_shape)

    # # Derive generalised distance combining all receivers
    Df = D_frobenius(g_ref, g_z)

    if dist == "D1" or dist == "both":
        d1 = np.sum(np.abs(g_ref_expanded - g_z), axis=0)
        d1 += 1
        d1 = 10 * np.log10(d1)
    if dist == "D2" or dist == "both":
        d2 = np.sum(np.abs(g_ref_expanded - g_z) ** 2, axis=0)
        d2 += 1
        d2 = 10 * np.log10(d2)

    depth_displacement = z_src_list - z_src

    # Plot generalized distance map
    plt.figure()
    plt.plot(
        depth_displacement,
        Df,
    )
    plt.ylabel(r"$10log_{10}(1+D)\, \textrm{[dB]}$")
    plt.xlabel(r"$z - z_s \, \textrm{[m]}$")
    plt.title(r"$\mathcal{D}_F(r=r_s, z)$")
    plt.grid()

    # Save
    fpath = os.path.join(
        root,
        f"D_frobenius_z_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}.png",
    )
    plt.savefig(fpath)

    d1max = 0
    d2max = 0
    plt.figure()
    # Iterate over receivers
    for i_rcv in range(n_rcv - 1):
        if dist == "D1" or dist == "both":
            plt.plot(
                depth_displacement,
                d1[0, i_rcv, :],
                label=r"$D_1$" + r"$\,\, (\Pi_{" + f"{i_rcv+1},0" + "})$",
            )
            d1max = np.max(d1)
        if dist == "D2" or dist == "both":
            plt.plot(
                depth_displacement,
                d2[0, i_rcv, :],
                label=r"$D$" + r"$\,\, (\Pi_{" + f"{i_rcv+1},0" + "})$",
            )
            d2max = np.max(d2)

    plt.plot(depth_displacement, Df, label=r"$D_{F}$", color="k")
    plt.xlabel(r"$z - z_s \, \textrm{[m]}$")
    plt.ylabel(r"$10log_{10}(1+D)\, \textrm{[dB]}$")
    plt.ylim(0, max(d1max, d2max) + 5)
    plt.grid()

    if dist == "both" or n_rcv > 2:
        plt.legend()

    if dist == "both":
        dist_lab = "D1D2"
    else:
        dist_lab = dist

    # Save
    fpath = os.path.join(
        root,
        f"{dist_lab}_z_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}.png",
    )
    plt.savefig(fpath)
    plt.close("all")


def Pi_var_rz(
    depth,
    f,
    r_src,
    z_src,
    x_rcv,
    z_rcv,
    covered_range,
    dr,
    zmin,
    zmax,
    dz,
    dist="both",
    bottom_bc="pressure_release",
):

    # Create folder to store the images
    root, folder = get_folderpath(x_rcv, r_src, z_src, bottom_bc=bottom_bc)
    subfolder = f"dr_{dr}_dz_{dz}"
    root = os.path.join(root, folder, subfolder)
    if not os.path.exists(root):
        os.makedirs(root)

    n_rcv = len(x_rcv)

    # Create the list of potential source positions
    r_src_list = np.arange(r_src - covered_range, r_src + covered_range, dr)
    zmin = np.max([1, zmin])  # Avoid negative depths
    z_src_list = np.arange(zmin, zmax, dz)

    # Ensure that the ref position is included in the list
    r_src_list = np.sort(np.append(r_src_list, r_src))
    z_src_list = np.sort(np.append(z_src_list, z_src))

    nb_pos_r = len(r_src_list)
    nb_pos_z = len(z_src_list)

    # Cast to required shape depending on the number of receivers
    r_src_list_2D = np.tile(r_src_list, (len(x_rcv) - 1, 1)).T
    r_src_rcv_ref = r_src_list_2D - x_rcv[0]
    r_src_rcv = r_src_list_2D - x_rcv[1:]

    # Derive the reference RTF vector, g_ref is of shape (nf, nr, nrcv, 1)
    r_ref = r_src - x_rcv
    f, g_ref = g_mat(
        f,
        z_src,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        depth=depth,
        r_rcv_ref=r_ref[0],
        r=r_ref[1:],
        bottom_bc=bottom_bc,
    )

    t0 = time()
    f, g_rz = g_mat(
        f,
        z_src_list,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        depth=depth,
        r_rcv_ref=r_src_rcv_ref,
        r=r_src_rcv,
        bottom_bc=bottom_bc,
    )
    print(time() - t0)

    # Expand g_ref to the same shape as g_rz
    tile_shape = tuple([g_rz.shape[i] - g_ref.shape[i] + 1 for i in range(g_rz.ndim)])
    g_ref_expanded = np.tile(g_ref, tile_shape)

    # # Derive generalised distance combining all receivers
    Df = D_frobenius(g_ref, g_rz)

    if dist == "D1" or dist == "both":
        d1 = np.sum(np.abs(g_ref_expanded - g_rz), axis=0)
    if dist == "D2" or dist == "both":
        d2 = np.sum(np.abs(g_ref_expanded - g_rz) ** 2, axis=0)

    range_displacement = r_src_list - r_src
    depth_displacement = z_src_list - z_src

    # Plot generalized distance map
    plt.figure()
    plt.pcolormesh(
        range_displacement,
        depth_displacement,
        Df.T,
        shading="auto",
        cmap="jet_r",
        vmin=0,
        vmax=np.percentile(Df, 45),
    )
    plt.colorbar()
    plt.xlabel(r"$r - r_s \, \textrm{[m]}$")
    plt.ylabel(r"$z - z_s \, \textrm{[m]}$")
    plt.title(r"$\mathcal{D}_F(r, z)$")

    # Save
    fpath = os.path.join(
        root,
        f"D_frobenius_rr_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_zz_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}.png",
    )
    plt.savefig(fpath)

    for i_rcv in range(n_rcv - 1):

        # plot d1 map
        if dist == "D1" or dist == "both":
            plt.figure()
            plt.pcolormesh(
                range_displacement,
                depth_displacement,
                d1[:, i_rcv, :].T,
                shading="auto",
                cmap="jet_r",
                vmin=0,
                # vmax=np.median(d1),
                vmax=np.percentile(d1, 45),
            )
            plt.colorbar()
            plt.xlabel(r"$r - r_s \, \textrm{[m]}$")
            plt.ylabel(r"$z - z_s \, \textrm{[m]}$")
            plt.title(r"$D_1$" + r"$\,\, (\Pi_{" + f"{i_rcv+1},0" + "})$")

            # Save
            fpath = os.path.join(
                root,
                f"D1_rcv_{i_rcv+1}_rr_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_zz_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}.png",
                # f"D1_rcv_{i_rcv+1}_rr_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_dr_{dr}_zz_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}_dz_{dz}.png",
            )
            plt.savefig(fpath)

        # plot d2 map
        if dist == "D2" or dist == "both":
            plt.figure()
            plt.pcolormesh(
                range_displacement,
                depth_displacement,
                d2[:, i_rcv, :].T,
                shading="auto",
                cmap="jet_r",
                vmin=0,
                vmax=np.percentile(d2, 45),
            )
            plt.colorbar()
            plt.xlabel(r"$r - r_s \, \textrm{[m]}$")
            plt.ylabel(r"$z - z_s \, \textrm{[m]}$")
            plt.title(r"$D$" + r"$\,\, (\Pi_{" + f"{i_rcv+1},0" + "})$")

            # Save
            fpath = os.path.join(
                root,
                f"D2_rcv_{i_rcv+1}_rr_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_zz_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}.png",
                # f"D2_rcv_{i_rcv+1}_rr_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_dr_{dr}_zz_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}_dr_{dz}.png",
            )
            plt.savefig(fpath)

        plt.close("all")


def get_folderpath(x_rcv, r_src, z_src, bottom_bc="pressure_release"):
    delta_rcv = x_rcv[1] - x_rcv[0]
    n_rcv = len(x_rcv)
    root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\ideal_waveguide"
    testname = f"nrcv_{n_rcv}_deltarcv_{delta_rcv:.0f}m_rsrc_{r_src*1e-3:.0f}km_zsrc_{z_src:.0f}m"
    folder = os.path.join(bottom_bc, testname)
    return root, folder


def full_test(
    covered_range, dr, zmin, zmax, dz, dist="D2", bottom_bc="pressure_release"
):

    # Load default parameters
    depth, r_src, z_src, z_rcv, n_rcv, delta_rcv, f = default_params()

    # Define the receivers position
    x_rcv = np.array([i * delta_rcv for i in range(n_rcv)])

    # # Variations along the range axis
    PI_var_r(
        depth,
        f,
        r_src,
        z_src,
        x_rcv,
        z_rcv,
        covered_range=covered_range,
        dr=dr,
        dist=dist,
        bottom_bc=bottom_bc,
    )

    # Variations along the depth axis
    PI_var_z(
        depth,
        f,
        r_src,
        z_src,
        x_rcv,
        z_rcv,
        zmin=zmin,
        zmax=zmax,
        dz=dz,
        dist=dist,
        bottom_bc=bottom_bc,
    )

    # Variations along the range and depth axes
    Pi_var_rz(
        depth,
        f,
        r_src,
        z_src,
        x_rcv,
        z_rcv,
        covered_range=covered_range,
        dr=dr,
        zmin=zmin,
        zmax=zmax,
        dz=dz,
        dist=dist,
        bottom_bc=bottom_bc,
    )

    # Transmission loss for a few frequencies

    # Create folder to store the images
    root, folder = get_folderpath(x_rcv, r_src, z_src, bottom_bc=bottom_bc)
    subfolder = f"tl_f"
    root = os.path.join(root, folder, subfolder)
    if not os.path.exists(root):
        os.makedirs(root)

    n_rcv = len(x_rcv)
    delta_rcv = x_rcv[1] - x_rcv[0]
    r_rcv_0 = r_src
    z_rcv_0 = z_rcv
    r_rcv_plot = np.array([r_rcv_0 + delta_rcv * i for i in range(n_rcv)])
    z_rcv_plot = np.array([z_rcv_0] * n_rcv)

    rmax = np.max(r_rcv_plot) + 1e3
    zmax = max(np.max(z_rcv), np.max(z_src), depth)
    r = np.linspace(1, rmax, int(1e3))
    z = np.linspace(0, zmax, 100)

    freq_to_plot = [1, 5, 10, 20, 50]
    f, _, _, p_field = field(freq_to_plot, z_src, r, z, depth)
    for fp in freq_to_plot:
        plot_tl(
            f,
            r,
            z,
            p_field,
            f_plot=fp,
            z_src=z_src,
            r_rcv=r_rcv_plot,
            z_rcv=z_rcv_plot,
            show=False,
        )

        plt.savefig(os.path.join(root, f"tl_f{fp}Hz.png"))

    # Zoom on receivers
    rmin = np.min(r_rcv_plot) - 1000
    rmax = np.max(r_rcv_plot) + 1000
    r = np.linspace(rmin, rmax, int(1e3))
    z = np.linspace(zmax - 20, zmax, 100)

    freq_to_plot = [1, 5, 10, 20, 50]
    f, _, _, p_field = field(freq_to_plot, z_src, r, z, depth)
    for fp in freq_to_plot:
        plot_tl(
            f,
            r,
            z,
            p_field,
            f_plot=fp,
            z_src=None,
            r_rcv=r_rcv_plot,
            z_rcv=z_rcv_plot,
            show=False,
        )

        plt.savefig(os.path.join(root, f"tl_f{fp}Hz_zoomrcv.png"))


def sensibility_ideal_waveguide(
    param="delta_rcv", axis="both", bottom_bc="pressure_release"
):
    """Study the sensibility of the proposed distance metric (D_frobenius) to the source position.
    The source position is varied along the range axis and the depth axis. The distance metric is computed for each source position.
    The sensibility along each axis is evaluated separately by computing the main lobe aperture of D_frobenius.
    Several parameters are studied :
       - The distance between receivers (delta_rcv)
       - The reference source range (r_src)
       - The reference source depth (z_src)
       - The number of receivers (n_rcv)
    """

    # Sensibility to the distance between receivers
    if param == "delta_rcv":
        sensibility_ideal_waveguide_delta_rcv(axis=axis, bottom_bc=bottom_bc)

    # Sensibility to the reference source range
    if param == "r_src":
        sensibility_ideal_waveguide_r_src(axis=axis, bottom_bc=bottom_bc)

    # Sensibility to the reference source depth
    if param == "z_src":
        sensibility_ideal_waveguide_z_src(axis=axis, bottom_bc=bottom_bc)

    # Sensibility to the number of receivers
    if param == "n_rcv":
        sensibility_ideal_waveguide_n_rcv(axis=axis, bottom_bc=bottom_bc)

    # Sensibility to the frequency resolution
    if param == "df":
        sensibility_ideal_waveguide_df(axis=axis, bottom_bc=bottom_bc)


def derive_Pi_ref(r_src, z_src, x_rcv, f, axis="r", bottom_bc="pressure_release"):

    depth, _, _, z_rcv, _, _, _ = default_params()

    if axis == "r":
        # Define the ranges
        dr = 0.1
        covered_range = 30  # Assume the range aperture is smaller than 30 m
        r_src_list = np.arange(r_src - covered_range, r_src + covered_range, dr)
        # Ensure that the ref position is included in the list
        r_src_list = np.sort(np.append(r_src_list, r_src))

        # Cast to required shape depending on the number of receivers
        r_src_list_2D = np.tile(r_src_list, (len(x_rcv) - 1, 1)).T
        nb_pos_r = len(r_src_list)
        r_src_rcv_ref = r_src_list_2D - x_rcv[0]
        r_src_rcv = r_src_list_2D - x_rcv[1:]

        # Derive the RTF vector for the reference source position
        r_ref = r_src - x_rcv
        # Derive the reference RTF vector g_ref is of shape (nf, nrcv, 1)
        f, g_ref = g_mat(
            f,
            z_src,
            z_rcv_ref=z_rcv,
            z_rcv=z_rcv,
            depth=depth,
            r_rcv_ref=r_ref[0],
            r=r_ref[1:],
            bottom_bc=bottom_bc,
        )

        return r_src_list, r_src_rcv_ref, r_src_rcv, depth, f, g_ref

    if axis == "z":

        # Define detphs
        dz = 0.1
        zmin = z_src - 10
        zmin = np.max([1, zmin])  # Avoid negative depths
        zmax = z_src + 20
        z_src_list = np.arange(zmin, zmax, dz)

        # Ensure that the ref position is included in the list
        z_src_list = np.sort(np.append(z_src_list, z_src))
        nb_pos_z = len(z_src_list)
        r_src_rcv_ref = r_src - x_rcv[0]
        r_src_rcv = r_src - x_rcv[1:]

        # Derive the reference RTF vector, g_ref is of shape (nf, nrcv, 1)
        r_ref = r_src - x_rcv

        f, g_ref = g_mat(
            f,
            z_src,
            z_rcv_ref=z_rcv,
            z_rcv=z_rcv,
            depth=depth,
            r_rcv_ref=r_ref[0],
            r=r_ref[1:],
            bottom_bc=bottom_bc,
        )

        return z_src_list, r_src_rcv_ref, r_src_rcv, depth, f, g_ref


def sensibility_ideal_waveguide_delta_rcv(axis="r", bottom_bc="pressure_release"):
    # Define the range of the study
    # delta_rcv = np.logspace(0, 4, 10)
    # delta_max = 3 * 1e3
    # alpha = 1.2
    # delta_rcv = np.array(
    #     [1 + x**alpha for x in np.linspace(0, delta_max ** (1 / alpha), 100)]
    # )
    # delta_rcv = np.round(delta_rcv, 1)
    # delta_rcv = np.array(
    #     [
    #         0.1,
    #         0.2,
    #         0.3,
    #         0.5,
    #         0.6,
    #         0.7,
    #         0.8,
    #         0.9,
    #         1,
    #         1.5,
    #         2,
    #         3,
    #         5,
    #         7,
    #         10,
    #         15,
    #         20,
    #         30,
    #         50,
    #         70,
    #         100,
    #         150,
    #         200,
    #         300,
    #         500,
    #         700,
    #         1000,
    #     ]
    # )
    delta_rcv = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            2,
            2.5,
            3,
            3.5,
            4,
            4.5,
            5,
            6,
            7,
            8,
            9,
            10,
            15,
            20,
            30,
            50,
        ]
    )

    _, r_src, z_src, _, n_rcv, delta_rcv, f = default_params()

    root_img = init_sensibility_path("delta_rcv", bottom_bc=bottom_bc)

    # Define input vars
    input_var = {}
    input_var["r_src"] = r_src
    input_var["z_src"] = z_src
    input_var["n_rcv"] = n_rcv
    input_var["df"] = f[1] - f[0]

    # Define param vars
    param_var = {}
    param_var["name"] = "delta_rcv"
    param_var["unit"] = "m"
    param_var["root_img"] = root_img
    param_var["th_r"] = 3
    param_var["th_z"] = 3
    param_var["values"] = delta_rcv
    param_var["xlabel"] = r"$\delta_{r_{rcv}} \, \textrm{[m]}$"

    apertures_r = []
    apertures_z = []

    # Compute the distance metric for each delta_rcv
    for i_d, delta in enumerate(delta_rcv):

        param_var["idx"] = i_d
        param_var["value"] = delta
        input_var["delta_rcv"] = delta
        study_param_sensibility(
            input_var, param_var, apertures_r, apertures_z, axis=axis
        )

    # Plot the main lobe aperture as a function of delta_rcv
    plot_sensibility(apertures_r, apertures_z, param_var, axis=axis)


def sensibility_ideal_waveguide_r_src(axis="both", bottom_bc="pressure_release"):
    src_range = np.arange(10, 101, 10) * 1e3

    # Define the source depth
    _, r_src, z_src, _, n_rcv, delta_rcv, f = default_params()

    root_img = init_sensibility_path("r_src", bottom_bc=bottom_bc)

    # Define input vars
    input_var = {}
    input_var["z_src"] = z_src
    input_var["n_rcv"] = n_rcv
    input_var["delta_rcv"] = delta_rcv
    input_var["df"] = f[1] - f[0]

    # Define param vars
    param_var = {}
    param_var["name"] = "r_src"
    param_var["unit"] = "m"
    param_var["root_img"] = root_img
    param_var["th_r"] = 3
    param_var["th_z"] = 3
    param_var["values"] = src_range
    param_var["xlabel"] = r"$r_{s} \, \textrm{[m]}$"

    apertures_r = []
    apertures_z = []

    # Compute the distance metric for each delta_rcv
    for i_d, r_src in enumerate(src_range):

        param_var["idx"] = i_d
        param_var["value"] = r_src
        input_var["r_src"] = r_src

        study_param_sensibility(
            input_var, param_var, apertures_r, apertures_z, axis=axis
        )

    # Plot the main lobe aperture as a function of delta_rcv
    plot_sensibility(apertures_r, apertures_z, param_var, axis=axis)


def sensibility_ideal_waveguide_z_src(axis, bottom_bc="pressure_release"):
    src_depth = np.arange(1, 991, 1)

    _, r_src, z_src, _, n_rcv, delta_rcv, f = default_params()

    root_img = init_sensibility_path("z_src", bottom_bc=bottom_bc)

    # Define input vars
    input_var = {}
    input_var["r_src"] = r_src
    input_var["n_rcv"] = n_rcv
    input_var["delta_rcv"] = delta_rcv
    input_var["df"] = f[1] - f[0]

    # Define param vars
    param_var = {}
    param_var["name"] = "z_src"
    param_var["unit"] = "m"
    param_var["root_img"] = root_img
    param_var["th_r"] = 3
    param_var["th_z"] = 3
    param_var["values"] = src_depth
    param_var["xlabel"] = r"$z_{s} \, \textrm{[m]}$"

    apertures_r = []
    apertures_z = []

    # Compute the distance metric for each delta_rcv
    for i_d, z_src in enumerate(src_depth):

        param_var["idx"] = i_d
        param_var["value"] = z_src
        input_var["z_src"] = z_src

        study_param_sensibility(
            input_var, param_var, apertures_r, apertures_z, axis=axis
        )

    # Plot the main lobe aperture as a function of delta_rcv
    plot_sensibility(apertures_r, apertures_z, param_var, axis=axis)


def sensibility_ideal_waveguide_n_rcv(axis, bottom_bc="pressure_release"):
    nb_rcv = np.arange(2, 11, 1)

    _, r_src, z_src, _, n_rcv, delta_rcv, f = default_params()

    root_img = init_sensibility_path("n_rcv", bottom_bc=bottom_bc)

    # Define input vars
    input_var = {}
    input_var["r_src"] = r_src
    input_var["z_src"] = z_src
    input_var["delta_rcv"] = delta_rcv
    input_var["df"] = f[1] - f[0]

    # Define param vars
    param_var = {}
    param_var["name"] = "n_rcv"
    param_var["unit"] = ""
    param_var["root_img"] = root_img
    param_var["th_r"] = 3
    param_var["th_z"] = 3
    param_var["values"] = nb_rcv
    param_var["xlabel"] = r"$n_{rcv}$"

    apertures_r = []
    apertures_z = []

    # Compute the distance metric for each delta_rcv
    for i_d, n_rcv in enumerate(nb_rcv):

        param_var["idx"] = i_d
        param_var["value"] = n_rcv
        input_var["n_rcv"] = n_rcv

        study_param_sensibility(
            input_var, param_var, apertures_r, apertures_z, axis=axis
        )

    # Plot the main lobe aperture as a function of delta_rcv
    plot_sensibility(apertures_r, apertures_z, param_var, axis=axis)


def sensibility_ideal_waveguide_df(axis, bottom_bc="pressure_release"):
    freq_res = np.arange(0.1, 15.1, 0.1)

    _, r_src, z_src, _, n_rcv, delta_rcv, _ = default_params()

    root_img = init_sensibility_path("df", bottom_bc=bottom_bc)

    # Define input vars
    input_var = {}
    input_var["n_rcv"] = n_rcv
    input_var["r_src"] = r_src
    input_var["z_src"] = z_src
    input_var["delta_rcv"] = delta_rcv

    # Define param vars
    param_var = {}
    param_var["name"] = "df"
    param_var["unit"] = "Hz"
    param_var["root_img"] = root_img
    param_var["th_r"] = 3
    param_var["th_z"] = 3
    param_var["values"] = freq_res
    param_var["xlabel"] = r"$\Delta_f \, \textrm{Hz}$"

    apertures_r = []
    apertures_z = []

    # Compute the distance metric for each delta_rcv
    for i_d, df in enumerate(freq_res):

        param_var["idx"] = i_d
        param_var["value"] = df
        input_var["df"] = df

        study_param_sensibility(
            input_var, param_var, apertures_r, apertures_z, axis=axis
        )

    # Plot the main lobe aperture as a function of delta_rcv
    plot_sensibility(apertures_r, apertures_z, param_var, axis=axis)


def D_frobenius(g_ref, g):
    """Derive the generalised distance combining all receivers."""
    # Expand g_ref to the same shape as g_r
    tile_shape = tuple([g.shape[i] - g_ref.shape[i] + 1 for i in range(g.ndim)])
    g_ref_expanded = np.tile(g_ref, tile_shape)

    nb_pos_r = g.shape[1]
    nb_pos_z = g.shape[3]

    Df_shape = (nb_pos_r, nb_pos_z)

    D_frobenius = np.zeros(Df_shape)
    for i_r in range(nb_pos_r):
        for i_z in range(nb_pos_z):
            Gamma = g_ref_expanded[:, i_r, :, i_z] - g[:, i_r, :, i_z]
            D_frobenius[i_r, i_z] = np.linalg.norm(Gamma, ord="fro")

    if nb_pos_z == 1 or nb_pos_r == 1:
        D_frobenius = D_frobenius.flatten()

    return D_frobenius


def init_sensibility_path(param, bottom_bc="pressure_release"):
    # Create folder to store the images
    root = "C:\\Users\\baptiste.menetrier\\Desktop\\devPy\\phd\\img\\illustration\\rtf\\ideal_waveguide"
    folder = bottom_bc
    subfolder = "sensibility"
    subsubfolder = param
    root = os.path.join(root, folder, subfolder, subsubfolder)
    if not os.path.exists(root):
        os.makedirs(root)

    return root


def Df_aperture(Df, x, th_dB=10):
    """Derive main lobe aperture along the x axis.

    Aperture is defined as the width of the 3dB main lobe.
    """

    # Find the main lobe
    idx_main_lobe = np.where(Df <= th_dB)[0]
    i1 = idx_main_lobe[0]
    i2 = idx_main_lobe[-1]
    aperture = np.abs(x[i1] - x[i2])

    return i1, i2, np.round(aperture, 3)


def study_param_sensibility(
    input_var,
    param_var,
    aperture_r,
    aperture_z,
    axis="both",
    bottom_bc="pressure_release",
):
    # Load default params
    depth, _, _, z_rcv, _, _, f = default_params()

    # Load input vars
    delta_rcv = input_var["delta_rcv"]
    r_src = input_var["r_src"]
    z_src = input_var["z_src"]
    n_rcv = input_var["n_rcv"]
    df = input_var["df"]

    f = np.arange(f[0], f[-1], df)

    # Load param vars
    name = param_var["name"]
    idx = param_var["idx"]
    value = param_var["value"]
    unit = param_var["unit"]
    root_img = param_var["root_img"]
    th_r = param_var["th_r"]
    th_z = param_var["th_z"]

    # Define the receivers position
    x_rcv = np.array([i * delta_rcv for i in range(n_rcv)])

    ### Range axis ###
    if axis == "r" or axis == "both":
        # Derive ref RTF
        r_src_list, r_src_rcv_ref, r_src_rcv, _, f, g_ref = derive_Pi_ref(
            r_src, z_src, x_rcv, f, axis="r", bottom_bc=bottom_bc
        )

        # Derive RTF for the set of potential source positions (r_src_rcv)
        f, g = g_mat(
            f,
            z_src,
            z_rcv_ref=z_rcv,
            z_rcv=z_rcv,
            depth=depth,
            r_rcv_ref=r_src_rcv_ref,
            r=r_src_rcv,
            bottom_bc=bottom_bc,
        )

        # Derive the distance
        Df_r = D_frobenius(g_ref, g)
        range_displacement = r_src_list - r_src

        # Derive aperure of the main lobe
        th_r = 3  # threshold
        i1_r, i2_r, ap_r = Df_aperture(Df_r, range_displacement, th_r)
        aperture_r.append(ap_r)

    ### Depth axis ###
    if axis == "z" or axis == "both":
        # Derive ref RTF
        z_src_list, r_src_rcv_ref, r_src_rcv, _, f, g_ref = derive_Pi_ref(
            r_src, z_src, x_rcv, f, axis="z", bottom_bc=bottom_bc
        )

        # Derive RTF for the set of potential source positions (r_src_rcv)
        f, g = g_mat(
            f,
            z_src_list,
            z_rcv_ref=z_rcv,
            z_rcv=z_rcv,
            depth=depth,
            r_rcv_ref=r_src_rcv_ref,
            r=r_src_rcv,
            bottom_bc=bottom_bc,
        )

        # Derive the distance
        Df_z = D_frobenius(g_ref, g)
        depth_displacement = z_src_list - z_src

        # Derive aperure of the main lobe
        th_z = 3  # threshold
        i1_z, i2_z, ap_z = Df_aperture(Df_z, range_displacement, th_z)
        aperture_z.append(ap_z)

    # Plot generalized distance map
    if idx % 5 == 0:
        if axis == "r" or axis == "both":
            plt.figure()
            plt.plot(
                range_displacement,
                Df_r,
            )
            # Add the main lobe aperture
            plt.axvline(x=range_displacement[i1_r], color="r", linestyle="--")
            plt.axvline(x=range_displacement[i2_r], color="r", linestyle="--")
            plt.text(
                range_displacement[i2_r] + 0.15,
                80,
                r"$2 r_{"
                + f"{th_r}"
                + r"\textrm{dB} } = "
                + f"{ap_r}"
                + r"\, \textrm{[m]}$",
                rotation=90,
                color="r",
                fontsize=12,
            )

            plt.ylabel(r"$10log_{10}(1+D)\, \textrm{[dB]}$")
            plt.xlabel(r"$r - r_s \, \textrm{[m]}$")
            plt.title(r"$D_{Frobenius}$")
            plt.ylim(0, 100)
            plt.grid()

            param_lab = f"{value:.3f}{unit}"
            root_r = os.path.join(root_img, "r")
            if not os.path.exists(root_r):
                os.makedirs(root_r)

            plt.savefig(os.path.join(root_r, f"{name}_{param_lab}.png"))
            plt.close("all")

        if axis == "z" or axis == "both":
            plt.figure()
            plt.plot(
                depth_displacement,
                Df_z,
            )
            # Add the main lobe aperture
            plt.axvline(x=depth_displacement[i1_z], color="r", linestyle="--")
            plt.axvline(x=depth_displacement[i2_z], color="r", linestyle="--")

            plt.text(
                depth_displacement[i2_z] + 0.15,
                30,
                r"$2 z_{"
                + f"{th_z}"
                + r"\textrm{dB} } = "
                + f"{ap_z}"
                + r"\, \textrm{[m]}$",
                rotation=90,
                color="r",
                fontsize=12,
            )

            plt.ylabel(r"$10log_{10}(1+D)\, \textrm{[dB]}$")
            plt.xlabel(r"$z - z_s \, \textrm{[m]}$")
            plt.title(r"$D_{Frobenius}$")
            plt.ylim(0, 50)
            plt.grid()

            param_lab = f"{value:.3f}{unit}"
            root_z = os.path.join(root_img, "z")
            if not os.path.exists(root_z):
                os.makedirs(root_z)

            plt.savefig(os.path.join(root_z, f"{name}_{param_lab}.png"))
            plt.close("all")


def plot_sensibility(apertures_r, apertures_z, param_var, axis="both"):

    # Unpack param vars
    root_img = param_var["root_img"]
    values = param_var["values"]
    xlabel = param_var["xlabel"]

    if axis == "r" or axis == "both":
        plt.figure()
        plt.plot(values, apertures_r, marker=".", color="k")
        plt.xlabel(xlabel)
        plt.ylabel(
            r"$2 r_{" + f"{param_var['th_r']}" + r"\textrm{dB} }\, \textrm{[m]}$"
        )
        plt.grid()
        plt.savefig(os.path.join(root_img, f"aperture_r.png"))
        plt.close("all")

    if axis == "z" or axis == "both":
        plt.figure()
        plt.plot(values, apertures_z, marker=".", color="r")
        plt.xlabel(xlabel)
        plt.ylabel(
            r"$2 z_{" + f"{param_var['th_z']}" + r"\textrm{dB} }\, \textrm{[m]}$"
        )
        plt.grid()
        plt.savefig(os.path.join(root_img, f"aperture_z.png"))
        plt.close("all")

    if axis == "both":
        plt.figure()
        plt.plot(
            values,
            apertures_r,
            marker=".",
            color="k",
            label=r"$2 r_{" + f"{param_var['th_r']}" + r"\textrm{dB} }$",
        )
        plt.plot(
            values,
            apertures_z,
            marker=".",
            color="r",
            label=r"$2 z_{" + f"{param_var['th_r']}" + r"\textrm{dB} }$",
        )

        plt.xlabel(xlabel)
        plt.ylabel(
            r"$2 x_{" + f"{param_var['th_r']}" + r"\textrm{dB} }\, \textrm{[m]}$"
        )
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(root_img, f"aperture_r_z.png"))
        plt.close("all")


def default_params():
    # Define receivers and source positions
    depth = 1000
    # Source
    z_src = 5
    r_src = 30 * 1e3
    # Receivers
    z_rcv = depth - 1
    n_rcv = 5
    delta_rcv = 10

    # Frequency range
    fmin = 1
    fmax = 50
    df = 0.1
    f = np.arange(fmin, fmax + df, df)
    # nb_freq = 500
    # f = np.linspace(fmin, fmax, nb_freq)

    return depth, r_src, z_src, z_rcv, n_rcv, delta_rcv, f


if __name__ == "__main__":

    # covered_range = 5
    # dr = 0.1
    # zmin = 3
    # zmax = 7
    # dz = 0.1
    # dist = "D2"

    # # Load default parameters
    # depth, r_src, z_src, z_rcv, n_rcv, delta_rcv, f = default_params()

    # # Define the receivers position
    # x_rcv = np.array([i * delta_rcv for i in range(n_rcv)])

    # # # Variations along the range axis
    # PI_var_r(
    #     depth,
    #     f,
    #     r_src,
    #     z_src,
    #     x_rcv,
    #     z_rcv,
    #     covered_range=covered_range,
    #     dr=dr,
    #     dist=dist,
    #     smooth_tf=True,
    # )

    bottom_bc = "pressure_release"

    # params = ["r_src", "z_src", "n_rcv", "delta_rcv", "df"]
    # # params = ["z_src"]

    # for param in params:
    #     sensibility_ideal_waveguide(param=param, axis="both")

    covered_range = 25
    dr = 0.1
    zmin = 1
    zmax = 25
    dz = 0.1
    full_test(covered_range, dr, zmin, zmax, dz, dist="D2", bottom_bc=bottom_bc)

    # covered_range = 10 * 1e3
    # dr = 10
    # zmin = 1
    # zmax = D - 1
    # dz = 1

    # full_test(covered_range, dr, zmin, zmax, dz, dist="D2")
