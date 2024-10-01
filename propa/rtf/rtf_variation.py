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

PubFigure(label_fontsize=22, title_fontsize=24, legend_fontsize=12, ticks_fontsize=20)

# ======================================================================================================================
# Functions
# ======================================================================================================================


def PI_var_r(
    D,
    f,
    r_src,
    z_src,
    x_rcv,
    covered_range,
    dr,
    dist="both",
):
    n_rcv = len(x_rcv)

    # Define the ranges
    r_src_list = np.arange(r_src - covered_range, r_src + covered_range, dr)

    # Ensure that the ref position is included in the list
    r_src_list = np.sort(np.append(r_src_list, r_src))
    # Cast to required shape depending on the number of receivers
    r_src_list_2D = np.tile(r_src_list, (len(x_rcv) - 1, 1)).T
    # nb_pos_r = len(r_src_list)
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
        D=D,
        r_rcv_ref=r_ref[0],
        r=r_ref[1:],
    )

    f, g_r = g_mat(
        f,
        z_src,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        D=D,
        r_rcv_ref=r_src_rcv_ref,
        r=r_src_rcv,
    )

    # Expand g_ref to the same shape as g_r
    tile_shape = tuple([g_r.shape[i] - g_ref.shape[i] + 1 for i in range(g_r.ndim)])
    g_ref_expanded = np.tile(g_ref, tile_shape)

    if dist == "D1" or dist == "both":
        d1 = np.sum(np.abs(g_ref_expanded - g_r), axis=0)
        d1 += 1
        d1 = 10 * np.log10(d1)
    if dist == "D2" or dist == "both":
        d2 = np.sum(np.abs(g_ref_expanded - g_r) ** 2, axis=0)
        d2 += 1
        d2 = 10 * np.log10(d2)

    range_displacement = r_src_list - r_src

    # Create folder to store the images
    root, folder = get_folderpath(x_rcv, r_src, z_src)
    subfolder = f"dr_{dr}"
    root = os.path.join(root, folder, subfolder)
    if not os.path.exists(root):
        os.makedirs(root)

    plt.figure()
    # Iterate over receivers
    for i_rcv in range(n_rcv - 1):
        if dist == "D1" or dist == "both":
            plt.plot(
                range_displacement,
                d1[:, i_rcv, 0],
                label=r"$D_1$" + r"$\,\, (\Pi_{" + f"{i_rcv+1},0" + "})$",
            )
        if dist == "D2" or dist == "both":
            plt.plot(
                range_displacement,
                d2[:, i_rcv, 0],
                label=r"$D_2$" + r"$\,\, (\Pi_{" + f"{i_rcv+1},0" + "})$",
            )

    if dist == "both" or n_rcv > 2:
        plt.legend()

    if dist == "both":
        dist_lab = "D1D2"
    else:
        dist_lab = dist

    plt.xlabel(r"$r - r_s \, \textrm{[m]}$")
    plt.ylabel(r"$10log_{10}(1+D)\, \textrm{[dB]}$")
    plt.grid()

    # Save
    fpath = os.path.join(
        root,
        f"{dist_lab}_r_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_dr_{dr}.png",
    )
    plt.savefig(fpath)
    plt.close("all")


def PI_var_z(
    D,
    f,
    r_src,
    z_src,
    x_rcv,
    zmin,
    zmax,
    dz,
    dist="both",
):
    n_rcv = len(x_rcv)

    # Define detphs
    zmin = np.max([1, zmin])  # Avoid negative depths
    z_src_list = np.arange(zmin, zmax, dz)

    # Ensure that the ref position is included in the list
    z_src_list = np.sort(np.append(z_src_list, z_src))
    # nb_pos_z = len(z_src_list)
    r_src_rcv_ref = r_src - x_rcv[0]
    r_src_rcv = r_src - x_rcv[1:]

    # Derive the reference RTF vector, g_ref is of shape (nf, nrcv, 1)
    r_ref = r_src - x_rcv

    f, g_ref = g_mat(
        f,
        z_src,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        D=D,
        r_rcv_ref=r_ref[0],
        r=r_ref[1:],
    )

    f, g_z = g_mat(
        f,
        z_src_list,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        D=D,
        r_rcv_ref=r_src_rcv_ref,
        r=r_src_rcv,
    )

    # Expand g_ref to the same shape as g_z
    tile_shape = tuple([g_z.shape[i] - g_ref.shape[i] + 1 for i in range(g_z.ndim)])
    g_ref_expanded = np.tile(g_ref, tile_shape)

    if dist == "D1" or dist == "both":
        d1 = np.sum(np.abs(g_ref_expanded - g_z), axis=0)
        d1 += 1
        d1 = 10 * np.log10(d1)
    if dist == "D2" or dist == "both":
        d2 = np.sum(np.abs(g_ref_expanded - g_z) ** 2, axis=0)
        d2 += 1
        d2 = 10 * np.log10(d2)

    depth_displacement = z_src_list - z_src

    # Create folder to store the images
    root, folder = get_folderpath(x_rcv, r_src, z_src)
    subfolder = f"dz_{dz}"
    root = os.path.join(root, folder, subfolder)
    if not os.path.exists(root):
        os.makedirs(root)

    plt.figure()
    # Iterate over receivers
    for i_rcv in range(n_rcv - 1):
        if dist == "D1" or dist == "both":
            plt.plot(
                depth_displacement,
                d1[0, i_rcv, :],
                label=r"$D_1$" + r"$\,\, (\Pi_{" + f"{i_rcv+1},0" + "})$",
            )
        if dist == "D2" or dist == "both":
            plt.plot(
                depth_displacement,
                d2[0, i_rcv, :],
                label=r"$D_2$" + r"$\,\, (\Pi_{" + f"{i_rcv+1},0" + "})$",
            )

    plt.xlabel(r"$z - z_s \, \textrm{[m]}$")
    plt.ylabel(r"$10log_{10}(1+D)\, \textrm{[dB]}$")
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
    D,
    f,
    r_src,
    z_src,
    x_rcv,
    covered_range,
    dr,
    zmin,
    zmax,
    dz,
    dist="both",
):

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
        D=D,
        r_rcv_ref=r_ref[0],
        r=r_ref[1:],
    )

    t0 = time()
    f, g_rz = g_mat(
        f,
        z_src_list,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        D=D,
        r_rcv_ref=r_src_rcv_ref,
        r=r_src_rcv,
    )
    print(time() - t0)

    # Expand g_ref to the same shape as g_rz
    tile_shape = tuple([g_rz.shape[i] - g_ref.shape[i] + 1 for i in range(g_rz.ndim)])
    g_ref_expanded = np.tile(g_ref, tile_shape)

    if dist == "D1" or dist == "both":
        d1 = np.sum(np.abs(g_ref_expanded - g_rz), axis=0)
    if dist == "D2" or dist == "both":
        d2 = np.sum(np.abs(g_ref_expanded - g_rz) ** 2, axis=0)

    # Derive generalised distance combining all receivers
    # First version = ugly iterative method
    D_frobenius = np.zeros((nb_pos_r, nb_pos_z))
    for i_r in range(g_rz.shape[1]):
        for i_z in range(g_rz.shape[3]):
            Gamma = g_ref_expanded[:, i_r, :, i_z] - g_rz[:, i_r, :, i_z]
            D_frobenius[i_r, i_z] = np.linalg.norm(Gamma, ord="fro")

    range_displacement = r_src_list - r_src
    depth_displacement = z_src_list - z_src

    # Create folder to store the images
    root, folder = get_folderpath(x_rcv, r_src, z_src)
    subfolder = f"dr_{dr}_dz_{dz}"
    root = os.path.join(root, folder, subfolder)
    if not os.path.exists(root):
        os.makedirs(root)

    # Plot generalized distance map
    plt.figure()
    plt.pcolormesh(
        range_displacement,
        depth_displacement,
        D_frobenius.T,
        shading="auto",
        cmap="jet_r",
        vmin=0,
        vmax=np.percentile(D_frobenius, 45),
    )
    plt.colorbar()
    plt.xlabel(r"$r - r_s \, \textrm{[m]}$")
    plt.ylabel(r"$z - z_s \, \textrm{[m]}$")
    plt.title(r"$D_{Frobenius}$")

    # Save
    fpath = os.path.join(
        root,
        f"D_frobenius_rr_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_dr_{dr}_zz_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}_dr_{dz}.png",
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
                f"D1_rcv_{i_rcv+1}_rr_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_dr_{dr}_zz_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}_dr_{dz}.png",
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
            plt.title(r"$D_2$" + r"$\,\, (\Pi_{" + f"{i_rcv+1},0" + "})$")

            # Save
            fpath = os.path.join(
                root,
                f"D2_rcv_{i_rcv+1}_rr_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_zz_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}.png",
                # f"D2_rcv_{i_rcv+1}_rr_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_dr_{dr}_zz_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}_dr_{dz}.png",
            )
            plt.savefig(fpath)

        plt.close("all")


def get_folderpath(x_rcv, r_src, z_src):
    delta_rcv = x_rcv[1] - x_rcv[0]
    root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\ideal_waveguide"
    folder = f"nrcv_{n_rcv}_deltarcv_{delta_rcv:.0f}m_rsrc_{r_src*1e-3:.0f}km_zsrc_{z_src:.0f}m"
    return root, folder


def full_test(f, D, r_src, z_src, x_rcv, covered_range, dr, zmin, zmax, dz, dist="D2"):

    # # Variations along the range axis
    PI_var_r(
        D,
        f,
        r_src,
        z_src,
        x_rcv,
        covered_range=covered_range,
        dr=dr,
        dist=dist,
    )

    # Variations along the depth axis
    PI_var_z(
        D,
        f,
        r_src,
        z_src,
        x_rcv,
        zmin=zmin,
        zmax=zmax,
        dz=dz,
        dist=dist,
    )

    # Variations along the range and depth axes
    Pi_var_rz(
        D,
        f,
        r_src,
        z_src,
        x_rcv,
        covered_range=covered_range,
        dr=dr,
        zmin=zmin,
        zmax=zmax,
        dz=dz,
        dist=dist,
    )

    # Transmission loss for a few frequencies

    # Create folder to store the images
    root, folder = get_folderpath(x_rcv, r_src, z_src)
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
    zmax = np.max(max(z_rcv, z_src)) + 10
    r = np.arange(1, rmax, 1)
    z = np.arange(0, zmax, 1)

    freq_to_plot = [1, 5, 10, 20, 50]
    f, _, _, p_field = field(freq_to_plot, z_src, r, z, D)
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


# 1) Define receivers and source positions
D = 1000
# Source
z_src = 5
r_src = 30 * 1e3
# Receivers
n_rcv = 5
delta_rcv = 1e3
# x_rcv = np.linspace(0, 10, n_rcv)
x_rcv = np.array([i * delta_rcv for i in range(n_rcv)])
# r_src_rcv = x_rcv - r_src
r_src_rcv = r_src - x_rcv
z_rcv = D - 1


# 2) Define frequency range
fmin = 1
fmax = 50
nb_freq = 500
f = np.linspace(fmin, fmax, nb_freq)

# 3) Compute RTF for the reference source position
idx_rcv_ref = 0

f, g_ref = g(
    f, z_src, z_rcv_ref=z_rcv, z_rcv=z_rcv, D=D, r_rcv_ref=r_src_rcv[0], r=r_src_rcv[1]
)

# # 4) Derive RTF for another source position
# r_src_bis = r_src + 50
# r_src_rcv = x_rcv - r_src_bis

# f, g_1_bis = g(
#     f, z_src, z_rcv_ref=z_rcv, z_rcv=z_rcv, D=D, r_rcv_ref=r_src_rcv[0], r=r_src_rcv[1]
# )

# plt.figure()
# plt.plot(f, np.abs(g_ref))
# plt.plot(f, np.abs(g_1_bis))
# plt.xlabel("Frequency (Hz)")
# plt.ylabel(f"$g_ref(f)$")
# plt.title("RTF")
# plt.grid()

# # 5) Compute the difference between the two RTFs
# diff = np.abs(g_ref - g_1_bis)

# plt.figure()
# plt.plot(f, diff)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel(r"$|g_ref - g_1^{bis}|$")
# plt.title("Difference between RTFs")
# plt.grid()

# plt.close("all")

# # 6) Compute the relative difference between the two RTFs
# d1 = np.sum(diff)
# d2 = np.sum(diff**2)
# # print(f"d1 = {d1:.2e}")
# # print(f"d2 = {d2:.2e}")


# 7) Loop over potential source positions to compute d1 and d2 as a function of source range displacement
covered_range = 50
dr = 0.1
zmin = z_src - 10
zmax = z_src + 10
dz = 0.1

full_test(f, D, r_src, z_src, x_rcv, covered_range, dr, zmin, zmax, dz, dist="D2")

covered_range = 5 * 1e3
dr = 10
zmin = z_src - 10
zmax = D
dz = 10

full_test(f, D, r_src, z_src, x_rcv, covered_range, dr, zmin, zmax, dz, dist="D2")


# Test config
# PI_var_r(
#     D,
#     f,
#     r_src,
#     z_src,
#     x_rcv,
#     covered_range=1e2,
#     dr=10,
#     dist="D2",
# )

# PI_var_r(
#     D,
#     f,
#     r_src,
#     z_src,
#     x_rcv,
#     covered_range=1e4,
#     dr=10,
#     dist="D2",
# )

# PI_var_r(
#     D,
#     f,
#     r_src,
#     z_src,
#     x_rcv,
#     covered_range=0.5 * 1e2,
#     dr=0.1,
#     dist="D2",
# )

# PI_var_z(
#     D,
#     f,
#     r_src,
#     z_src,
#     x_rcv,
#     zmin=z_src - 30,
#     zmax=z_src + 30,
#     dz=0.1,
#     dist="D2",
# )

# PI_var_z(
#     D,
#     f,
#     r_src,
#     z_src,
#     x_rcv,
#     zmin=1,
#     zmax=D - 1,
#     dz=0.1,
#     dist="D2",
# )

# 8) Loop over potential source positions to compute d1 and d2 as a function of source range displacement and source depth
# Pi_var_rz(
#     D,
#     f,
#     r_src,
#     z_src,
#     x_rcv,
#     covered_range=20,
#     dr=1,
#     zmin=z_src - 10,
#     zmax=z_src + 10,
#     dz=1,
#     dist="both",
# )


# Pi_var_rz(
#     D,
#     f,
#     r_src,
#     z_src,
#     x_rcv,
#     covered_range=10,
#     dr=0.1,
#     zmin=z_src - 10,
#     zmax=z_src + 10,
#     dz=0.1,
#     dist="both",
# )

# Pi_var_rz(
#     D,
#     f,
#     r_src,
#     z_src,
#     x_rcv,
#     covered_range=1e4,
#     dr=10,
#     zmin=1,
#     zmax=D,
#     dz=10,
#     dist="both",
# )
