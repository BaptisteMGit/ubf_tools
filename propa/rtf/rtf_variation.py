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

PubFigure(label_fontsize=16, title_fontsize=12, legend_fontsize=12, ticks_fontsize=14)

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
    # Define the ranges
    r_src_list = np.arange(r_src - covered_range, r_src + covered_range, dr)

    # Ensure that the ref position is included in the list
    r_src_list = np.sort(np.append(r_src_list, r_src))
    nb_pos_r = len(r_src_list)
    r_src_rcv_ref = r_src_list - x_rcv[0]
    r_src_rcv = r_src_list - x_rcv[1]

    f, g_1_bis_m = g_mat(
        f,
        z_src,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        D=D,
        r_rcv_ref=r_src_rcv_ref,
        r=r_src_rcv,
    )

    g_1_cast = np.tile(g_1, (1, nb_pos_r, 1)).T
    if dist == "D1" or dist == "both":
        d1 = np.sum(np.abs(g_1_cast - g_1_bis_m), axis=0)
        d1 += 1
        d1 = 10 * np.log10(d1)
    if dist == "D2" or dist == "both":
        d2 = np.sum(np.abs(g_1_cast - g_1_bis_m) ** 2, axis=0)
        d2 += 1
        d2 = 10 * np.log10(d2)

    range_displacement = r_src_list - r_src

    plt.figure()
    if dist == "D1" or dist == "both":
        plt.plot(range_displacement, d1, label=r"$D_1$")
    if dist == "D2" or dist == "both":
        plt.plot(range_displacement, d2, label=r"$D_2$")

    if dist == "both":
        plt.legend()
        dist_lab = "D1D2"
    else:
        dist_lab = dist

    plt.xlabel(r"$r - r_s$" + " [m]")
    plt.ylabel(r"$10log_{10}(1+D)$" + " [dB]")
    plt.grid()

    # Save
    root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\ideal_waveguide"
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
    # Define detphs
    z_src_list = np.arange(zmin, zmax, dz)

    # Ensure that the ref position is included in the list
    z_src_list = np.sort(np.append(z_src_list, z_src))
    nb_pos_z = len(z_src_list)
    r_src_rcv_ref = r_src - x_rcv[0]
    r_src_rcv = r_src - x_rcv[1]

    f, g_1_bis_m = g_mat(
        f,
        z_src_list,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        D=D,
        r_rcv_ref=r_src_rcv_ref,
        r=r_src_rcv,
    )

    g_1_cast = np.tile(g_1, (nb_pos_z, 1, 1)).T

    if dist == "D1" or dist == "both":
        d1 = np.sum(np.abs(g_1_cast - g_1_bis_m), axis=0)
        d1 += 1
        d1 = 10 * np.log10(d1)
    if dist == "D2" or dist == "both":
        d2 = np.sum(np.abs(g_1_cast - g_1_bis_m) ** 2, axis=0)
        d2 += 1
        d2 = 10 * np.log10(d2)

    depth_displacement = z_src_list - z_src

    plt.figure()
    if dist == "D1" or dist == "both":
        plt.plot(depth_displacement, d1.flatten(), label="d1")
    if dist == "D2" or dist == "both":
        plt.plot(depth_displacement, d2.flatten(), label="d2")
    plt.xlabel(r"$z - z_s$" + " [m]")
    plt.ylabel(r"$10log_{10}(1+D)$" + " [dB]")
    plt.grid()

    if dist == "both":
        plt.legend()
        dist_lab = "D1D2"
    else:
        dist_lab = dist

    # Save
    root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\ideal_waveguide"
    fpath = os.path.join(
        root,
        f"{dist_lab}_z_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}_dz_{dz}.png",
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

    # Create the list of potential source positions
    r_src_list = np.arange(r_src - covered_range, r_src + covered_range, dr)
    z_src_list = np.arange(zmin, zmax, dz)

    # Ensure that the ref position is included in the list
    r_src_list = np.sort(np.append(r_src_list, r_src))
    z_src_list = np.sort(np.append(z_src_list, z_src))

    nb_pos_r = len(r_src_list)
    nb_pos_z = len(z_src_list)

    # Cast x_rcv as a 2D array
    r_src_rcv_ref = r_src_list - x_rcv[0]
    r_src_rcv = r_src_list - x_rcv[1]

    f, g_1_bis_m = g_mat(
        f,
        z_src_list,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        D=D,
        r_rcv_ref=r_src_rcv_ref,
        r=r_src_rcv,
    )

    # Cast g_1 to the same shape as g_1_bis_m
    g_1_cast = np.tile(g_1, (nb_pos_z, nb_pos_r, 1)).T

    if dist == "D1" or dist == "both":
        d1 = np.sum(np.abs(g_1_cast - g_1_bis_m), axis=0)
    if dist == "D2" or dist == "both":
        d2 = np.sum(np.abs(g_1_cast - g_1_bis_m) ** 2, axis=0)

    range_displacement = r_src_list - r_src
    depth_displacement = z_src_list - z_src

    # plot d1 map
    if dist == "D1" or dist == "both":
        plt.figure()
        plt.pcolormesh(
            range_displacement,
            depth_displacement,
            d1.T,
            shading="auto",
            cmap="jet_r",
            vmin=0,
            vmax=np.median(d1),
        )
        plt.colorbar()
        plt.xlabel(r"$r - r_s$" + " [m]")
        plt.ylabel(r"$z - z_s$" + " [m]")

        # Save
        root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\ideal_waveguide"
        fpath = os.path.join(
            root,
            f"D1_rr_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_dr_{dr}_zz_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}_dr_{dz}.png",
        )
        plt.savefig(fpath)

    # plot d2 map
    if dist == "D2" or dist == "both":
        plt.figure()
        plt.pcolormesh(
            range_displacement,
            depth_displacement,
            d2.T,
            shading="auto",
            cmap="jet_r",
            vmin=0,
            vmax=np.median(d2),
        )
        plt.colorbar()
        plt.xlabel(r"$r - r_s$" + " [m]")
        plt.ylabel(r"$z - z_s$" + " [m]")

        # Save
        root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\ideal_waveguide"
        fpath = os.path.join(
            root,
            f"D2_rr_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_dr_{dr}_zz_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}_dr_{dz}.png",
        )
        plt.savefig(fpath)

    plt.close("all")


# 1) Define receivers and source positions
D = 1000
# Source
z_src = 50
r_src = 1e5
# Receivers
n_rcv = 2
x_rcv = np.linspace(0, 10, n_rcv)
# r_src_rcv = x_rcv - r_src
r_src_rcv = r_src - x_rcv
z_rcv = 100


# 2) Define frequency range
fmin = 1
fmax = 50
nb_freq = 1000
f = np.linspace(fmin, fmax, nb_freq)

# 3) Compute RTF for the reference source position
idx_rcv_ref = 0

f, g_1 = g(
    f, z_src, z_rcv_ref=z_rcv, z_rcv=z_rcv, D=D, r_rcv_ref=r_src_rcv[0], r=r_src_rcv[1]
)

# 4) Derive RTF for another source position
r_src_bis = r_src + 50
r_src_rcv = x_rcv - r_src_bis

f, g_1_bis = g(
    f, z_src, z_rcv_ref=z_rcv, z_rcv=z_rcv, D=D, r_rcv_ref=r_src_rcv[0], r=r_src_rcv[1]
)

plt.figure()
plt.plot(f, np.abs(g_1))
plt.plot(f, np.abs(g_1_bis))
plt.xlabel("Frequency (Hz)")
plt.ylabel(f"$g_1(f)$")
plt.title("RTF")
plt.grid()

# 5) Compute the difference between the two RTFs
diff = np.abs(g_1 - g_1_bis)

plt.figure()
plt.plot(f, diff)
plt.xlabel("Frequency (Hz)")
plt.ylabel(r"$|g_1 - g_1^{bis}|$")
plt.title("Difference between RTFs")
plt.grid()

plt.close("all")

# 6) Compute the relative difference between the two RTFs
d1 = np.sum(diff)
d2 = np.sum(diff**2)
# print(f"d1 = {d1:.2e}")
# print(f"d2 = {d2:.2e}")


# 7) Loop over potential source positions to compute d1 and d2 as a function of source range displacement

PI_var_r(
    D,
    f,
    r_src,
    z_src,
    x_rcv,
    covered_range=1e4,
    dr=10,
    dist="both",
)

PI_var_r(
    D,
    f,
    r_src,
    z_src,
    x_rcv,
    covered_range=0.5 * 1e2,
    dr=0.1,
    dist="both",
)

PI_var_z(
    D,
    f,
    r_src,
    z_src,
    x_rcv,
    zmin=z_src - 30,
    zmax=z_src + 30,
    dz=0.1,
    dist="both",
)

PI_var_z(
    D,
    f,
    r_src,
    z_src,
    x_rcv,
    zmin=1,
    zmax=D - 1,
    dz=0.1,
    dist="both",
)

# 8) Loop over potential source positions to compute d1 and d2 as a function of source range displacement and source depth

Pi_var_rz(
    D,
    f,
    r_src,
    z_src,
    x_rcv,
    covered_range=0.5 * 1e2,
    dr=0.1,
    zmin=z_src - 10,
    zmax=z_src + 10,
    dz=0.1,
    dist="both",
)

Pi_var_rz(
    D,
    f,
    r_src,
    z_src,
    x_rcv,
    covered_range=1e4,
    dr=10,
    zmin=1,
    zmax=D,
    dz=10,
    dist="both",
)
