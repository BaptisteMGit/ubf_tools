#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   ideal_waveguide.py
@Time    :   2024/09/25 10:28:01
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
import scipy.fft as fft
import scipy.signal as sp
import matplotlib.pyplot as plt

from time import time
from publication.PublicationFigure import PubFigure

PubFigure(label_fontsize=22, title_fontsize=24, legend_fontsize=12, ticks_fontsize=20)


# ======================================================================================================================
# Constants
# ======================================================================================================================
c0 = 1500  # m/s
rho_0 = 1000  # kg/m^3
fmin = 0  # Hz
fmax = 50

# ======================================================================================================================
# Functions
# ======================================================================================================================


def nb_propagating_modes(f, c, depth, bottom_bc="pressure_release"):
    """
    Function that calculates the number of propagating modes in a ideal waveguide
    :param f: frequency (Hz)
    :param c: speed of sound in water (m/s)
    :param depth: depth of the waveguide (m)
    :param bottom_bc: bottom boundary condition (pressure_release or perfectly_rigid)
    :return: number of propagating modes
    """
    k = 2 * np.pi * f / c
    if bottom_bc == "perfectly_rigid":
        limit = 1 / 2 * (1 + 4 * depth * f / c)
        n = np.floor(limit).astype(int)  # Perfectly rigid bottom

    elif bottom_bc == "pressure_release":
        limit = k * depth / np.pi
        n = np.floor(k * depth / np.pi).astype(int)  # Pressure release bottom

    # Ensure that the strict condition is met
    if (np.round(limit, 12)).is_integer():
        n -= 1
    # print(f"Number of propagating modes: {n}")

    return n


def cutoff_frequency(c, depth, bottom_bc="pressure_release"):
    """
    Function that calculates the cutoff frequency of an ideal waveguide
    :param c: speed of sound in water (m/s)
    :param depth: depth of the waveguide (m)
    :param bottom_bc: bottom boundary condition (pressure_release or perfectly_rigid)
    :return: cutoff frequency (Hz)
    """

    if bottom_bc == "perfectly_rigid":
        fc = c / (4 * depth)  # Perfectly rigid bottom

    elif bottom_bc == "pressure_release":
        fc = c / (2 * depth)  # Pressure release bottom

    return fc


def kz(m, depth, bottom_bc="pressure_release"):
    """
    Function that calculates the vertical wavenumber
    :param m: mode number
    :param depth: depth of the waveguide (m)
    :param bottom_bc: bottom boundary condition (pressure_release or perfectly_rigid)
    :return: vertical wavenumber
    """
    if bottom_bc == "perfectly_rigid":
        kz_m = (m - 1 / 2) * np.pi / depth  # Perfectly rigid bottom
    elif bottom_bc == "pressure_release":
        kz_m = m * np.pi / depth  # Pressure release bottom

    return kz_m


def kr(m, f, depth, bottom_bc="pressure_release"):
    """
    Function that calculates the radial wavenumber
    :param m: mode number
    :param f: frequency (Hz)
    :param depth: depth of the waveguide (m)
    :param bottom_bc: bottom boundary condition (pressure_release or perfectly_rigid)
    :return: radial wavenumber
    """
    k = 2 * np.pi * f / c0
    kz_m = kz(m, depth, bottom_bc)
    kr_m = np.sqrt(k**2 - kz_m**2)

    return kr_m


def alpha(depth, r):

    alpha_r = np.exp(-1j * np.pi / 4) * 1 / (depth * np.sqrt(2 * np.pi * r))

    return alpha_r


def A_l(r0, rl):
    """
    Function that calculates the RTF A factor defined as A_l = alpha(rl) / alpha(r0)
    :param r0: distance between the source and the ref receiver (m)
    :param rl: distance between the source and the l-th receiver (m)
    :return: RTF A factor
    """

    dr = rl - r0
    A_l = 1 / np.sqrt(1 + dr / r0)

    return A_l


def psi(m, z, depth, bottom_bc="pressure_release"):

    return np.sin(kz(m, depth, bottom_bc) * z)


def psi_normalised(m, z, depth, rho, bottom_bc="pressure_release"):

    return psi(m, z, depth, bottom_bc) * np.sqrt(2 * rho / depth)


def u_m(m, f, z_src, z, depth, bottom_bc="pressure_release"):

    u = (
        psi(m, z_src, depth, bottom_bc)
        * psi(m, z, depth, bottom_bc)
        * 1
        / np.sqrt(kr(m, f, depth, bottom_bc))
    )

    return u


def g(f, z_src, z_rcv_ref, z_rcv, depth, r_rcv_ref, r, bottom_bc="pressure_release"):

    f = f[f > cutoff_frequency(c0, depth, bottom_bc)]

    g = []
    for fi in f:
        n = nb_propagating_modes(fi, c0, depth, bottom_bc)
        m = np.arange(1, n + 1)

        phi_l = u_m(m, fi, z_src, z=z_rcv, depth=depth, bottom_bc=bottom_bc) * np.exp(
            -1j * kr(m, fi, depth, bottom_bc) * r
        )
        phi_0 = u_m(
            m, fi, z_src, z=z_rcv_ref, depth=depth, bottom_bc=bottom_bc
        ) * np.exp(-1j * kr(m, fi, depth, bottom_bc) * r_rcv_ref)
        g_fi = A_l(r_rcv_ref, r) * np.sum(phi_l) / np.sum(phi_0)
        g.append(g_fi)

    g = np.array(g)

    return f, g


def g_mat(
    f,
    z_src,
    z_rcv_ref,
    z_rcv,
    depth,
    r_rcv_ref,
    r,
    bottom_bc="pressure_release",
    smooth_tf=False,
):
    """
    Derive the RTF g(f) for a range of frequencies f, z_src depths and ranges r."""
    # Ensure z_src is an array
    z_src = np.atleast_1d(z_src)
    # Ensure r is a 2D array for compatibility issues
    r = np.atleast_2d(r)

    rflat = r.flatten()  # Flatten the ranges corresponding to several receivers
    zz_src, rr = np.meshgrid(z_src, rflat)
    f = f[f > cutoff_frequency(c0, depth, bottom_bc)]

    # Define the g matrix
    shape = (len(f),) + zz_src.shape
    g_matrix = np.zeros(shape, dtype=np.complex64)

    # phi_l_mat = np.zeros((len(f),) + zz_src.shape, dtype=np.complex64)
    for i, fi in enumerate(f):
        n = nb_propagating_modes(fi, c0, depth, bottom_bc)
        m = np.arange(1, n + 1)

        zz_src, rr, mm = np.meshgrid(z_src, r, m)
        _, rr_ref, _ = np.meshgrid(z_src, r_rcv_ref, m)

        phi_l = u_m(mm, fi, zz_src, z=z_rcv, depth=depth, bottom_bc=bottom_bc) * np.exp(
            -1j * kr(mm, fi, depth, bottom_bc) * rr
        )
        phi_0 = u_m(
            mm, fi, zz_src, z=z_rcv_ref, depth=depth, bottom_bc=bottom_bc
        ) * np.exp(-1j * kr(mm, fi, depth, bottom_bc) * rr_ref)
        g_fi = (
            A_l(rr_ref[..., 0], rr[..., 0])
            * np.sum(phi_l, axis=-1)
            / np.sum(phi_0, axis=-1)
        )
        g_matrix[i, ...] = g_fi

    # Reshape to differentiate receivers
    g_matrix = g_matrix.reshape(
        (len(f),) + r.shape + z_src.shape
    )  # Final shape = (nf, nr, nrcv, nz)

    return f, g_matrix


def h(f, z_src, z, r, depth, bottom_bc="pressure_release"):

    f = f[f > cutoff_frequency(c0, depth, bottom_bc)]

    # t0 = time()
    h = []
    for fi in f:
        n = nb_propagating_modes(fi, c0, depth, bottom_bc)
        m = np.arange(1, n + 1)

        if 0 in kr(m, fi, depth, bottom_bc):
            print("bdegu")
            kr(m, fi, depth, bottom_bc)

        phi_m = u_m(m, fi, z_src, z=z, depth=depth, bottom_bc=bottom_bc) * np.exp(
            -1j * kr(m, fi, depth, bottom_bc) * r
        )

        h_fi = alpha(depth, r) * np.sum(phi_m)
        h.append(h_fi)

    h = np.array(h)
    # print(f"Elapsed time: {time() - t0:.2f} s")

    return f, h


def h_mat(f, z_src, z_rcv, r_rcv, depth, bottom_bc="pressure_release"):
    """Derive the ideal waveguide tranfert function for the given frequencies f, source depth z_src, receiver depths z_rcv and ranges r_rcv"""
    # Ensure z_src is an array
    z_src = np.atleast_1d(z_src)
    # Ensure r_rcv is a 2D array for compatibility issues
    r_rcv = np.atleast_2d(r_rcv)

    f = f[f > cutoff_frequency(c0, depth, bottom_bc)]
    r_rcv_flat = (
        r_rcv.flatten()
    )  # Flatten the ranges corresponding to several receivers
    zz_src, rr = np.meshgrid(z_src, r_rcv_flat)
    # Define the h matrix
    shape = (len(f),) + zz_src.shape
    h_matrix = np.zeros(shape, dtype=np.complex64)

    for i, fi in enumerate(f):
        n = nb_propagating_modes(fi, c0, depth, bottom_bc)
        m = np.arange(1, n + 1)

        zz_src, rr_rcv, mm = np.meshgrid(z_src, r_rcv_flat, m)

        phi_m = u_m(mm, fi, zz_src, z=z_rcv, depth=depth, bottom_bc=bottom_bc) * np.exp(
            -1j * kr(mm, fi, depth, bottom_bc) * rr_rcv
        )
        h_fi = alpha(depth, rr_rcv[..., 0]) * np.sum(phi_m, axis=-1)
        h_matrix[i, ...] = h_fi

    # Reshape to differentiate receivers
    h_matrix = h_matrix.reshape(
        (len(f),) + z_src.shape + r_rcv.shape
    )  # Final shape = (nf, nsrc, nrcv)

    return f, h_matrix


def print_arrivals(z_src, z_rcv, r, depth, n):
    # Number of terms to include in the sum
    m = np.arange(1, n + 1)
    # Image source - receiver distance follwoing definitions from Jensen p.104
    zm1 = 2 * depth * m - z_src + z_rcv
    zm2 = 2 * depth * (m + 1) - z_src - z_rcv
    zm3 = 2 * depth * m + z_src + z_rcv
    zm4 = 2 * depth * (m + 1) + z_src - z_rcv
    Rm1 = np.sqrt(r**2 + zm1.astype(np.float64) ** 2)
    Rm2 = np.sqrt(r**2 + zm2.astype(np.float64) ** 2)
    Rm3 = np.sqrt(r**2 + zm3.astype(np.float64) ** 2)
    Rm4 = np.sqrt(r**2 + zm4.astype(np.float64) ** 2)

    arrivals = np.empty((len(m), 4))
    for i_m in m:
        t1 = Rm1[i_m - 1] / c0
        t2 = Rm2[i_m - 1] / c0
        t3 = Rm3[i_m - 1] / c0
        t4 = Rm4[i_m - 1] / c0

        print(
            f"m = {m[i_m-1]} : \n"
            + f"\t t1 = {t1}s \n"
            + f"\t t2 = {t2}s \n"
            + f"\t t3 = {t3}s \n"
            + f"\t t4 = {t4}s \n"
        )
        arrivals[i_m - 1][0] = t1
        arrivals[i_m - 1][1] = t2
        arrivals[i_m - 1][2] = t3
        arrivals[i_m - 1][3] = t4

    return arrivals


def image_source_ri(z_src, z_rcv, r, depth, n, t=None):

    # Number of terms to include in the sum
    m = np.arange(1, n + 1)
    # Image source - receiver distance following definitions from Jensen p.104
    zm1 = 2 * depth * m - z_src + z_rcv
    zm2 = 2 * depth * (m + 1) - z_src - z_rcv
    zm3 = 2 * depth * m + z_src + z_rcv
    zm4 = 2 * depth * (m + 1) + z_src - z_rcv
    Rm1 = np.sqrt(r**2 + zm1.astype(np.float64) ** 2)
    Rm2 = np.sqrt(r**2 + zm2.astype(np.float64) ** 2)
    Rm3 = np.sqrt(r**2 + zm3.astype(np.float64) ** 2)
    Rm4 = np.sqrt(r**2 + zm4.astype(np.float64) ** 2)

    if t is None:
        Ts = 0.001
        t = np.arange(0, 10, Ts)

    ir = np.empty_like(t)
    for i_m in m:
        idx_1 = np.argmin(np.abs(Rm1[i_m - 1] / c0 - t))
        idx_2 = np.argmin(np.abs(Rm2[i_m - 1] / c0 - t))
        idx_3 = np.argmin(np.abs(Rm3[i_m - 1] / c0 - t))
        idx_4 = np.argmin(np.abs(Rm4[i_m - 1] / c0 - t))

        ir += (
            sp.unit_impulse(t.shape, idx_1) / Rm1[i_m - 1]
            - sp.unit_impulse(t.shape, idx_2) / Rm2[i_m - 1]
            - sp.unit_impulse(t.shape, idx_3) / Rm3[i_m - 1]
            + sp.unit_impulse(t.shape, idx_4) / Rm4[i_m - 1]
        )

    ir *= 1 / (4 * np.pi)

    return t, ir


# def rtf_distance(ranges, z_src, z_rcv, depth, n_rcv, d_rcv):

#     Ts = 0.001
#     fs = 1 / Ts
#     # Closest power of 2
#     nfft = 2**12
#     f = fft.rfftfreq(nfft, 1 / fs)

#     delta_rcv = np.arange(1, n_rcv, d_rcv)
#     for r in ranges:
#         rl = r + delta_rcv
#         f, g_f = g(f=f, z_src=z_src, z_rcv=z_rcv, depth=depth, r_rcv_ref=r, rl=rl)


def field(f, z_src, r, z, depth, bottom_bc="pressure_release"):
    """
    Derive the pressure field for the ideal waveguide at frequencies f, ranges r and depths z
    """

    # Ensure args are arrays
    z = np.atleast_1d(z)
    r = np.atleast_1d(r)
    f = np.atleast_1d(f)
    rr_2d, zz_2d = np.meshgrid(r, z)

    f = f[f > cutoff_frequency(c0, depth, bottom_bc)]

    # Define the field matrix
    p_field = np.zeros((len(f),) + zz_2d.shape, dtype=np.complex64)

    for i, fi in enumerate(f):
        n = nb_propagating_modes(fi, c0, depth, bottom_bc)
        m = np.arange(1, n + 1)

        rr, zz, mm = np.meshgrid(r, z, m)

        phi = u_m(mm, fi, z_src, z=zz, depth=depth, bottom_bc=bottom_bc) * np.exp(
            -1j * kr(mm, fi, depth, bottom_bc) * rr
        )
        p_field_fi = alpha(depth, rr_2d) * np.sum(phi, axis=-1)
        p_field[i, ...] = p_field_fi

    return f, rr, zz, p_field


def plot_tl(f, r, z, p_field, f_plot, z_src=None, r_rcv=None, z_rcv=None, show=False):
    # Slice to get the pressure field at the desired frequency
    f = np.atleast_1d(f)
    idx_freq = np.argmin(np.abs(f - f_plot))
    f_plot = f[idx_freq]
    p_field = p_field[idx_freq, ...]

    # Derive TL from the pressure field
    p_field[p_field == 0] = 1e-20
    tl = -20 * np.log10(np.abs(p_field))
    dr = r[1] - r[0]
    dz = z[1] - z[0]

    tlmax = np.percentile(tl, 95)
    tlmin = np.percentile(tl, 1)
    plt.figure()
    plt.pcolormesh(r * 1e-3, z, tl, cmap="jet_r", vmin=tlmin, vmax=tlmax)
    plt.colorbar(label=r"$\textrm{TL [dB]}$")
    # Add source position
    if z_src is not None:
        plt.scatter(
            0,
            z_src,
            color="fuchsia",
            marker="o",
            s=200,
        )
    # Add receiver positions
    if r_rcv is not None and z_rcv is not None:
        plt.scatter(r_rcv * 1e-3, z_rcv, color="black", marker="x", s=300, linewidths=3)

    plt.gca().invert_yaxis()
    plt.xlabel(r"$r \, \textrm{[km]}$")
    plt.ylabel(r"$z \, \textrm{[m]}$")
    plt.xlim((r[0] - 1) * 1e-3, r[-1] * 1e-3)
    plt.title(
        r"$\textrm{TL} \,"
        f"(f = {f_plot:.2f} "
        + r"\textrm{Hz}, \,"
        + f"\Delta_r = {dr:.1f} \,"
        + r"\textrm{m}, \,"
        + f"\Delta_z = {dz:.1f} \,"
        + r"\textrm{m})$"
    )
    if show:
        plt.show()


def test_g_mat_h_mat(bottom_bc="pressure_release"):
    fs = 50
    dur = 20
    depth = 1000

    n_rcv = 5
    z_rcv = depth - 1
    delta_rcv = 1000
    x_rcv = np.array([i * delta_rcv for i in range(n_rcv)])

    z_src = 5  # Source depth (m)
    r_src = 2.5 * 1e4  # Source range (m)
    r = r_src - x_rcv

    fmin, fmax, fs = 0, 50, 100
    f = np.arange(fmin, fmax, 0.1)

    # Derive RTF using g_mat
    f, g_ref = g_mat(
        f,
        z_src,
        z_rcv_ref=z_rcv,
        z_rcv=z_rcv,
        depth=depth,
        r_rcv_ref=r[0],
        r=r[1:],
        bottom_bc=bottom_bc,
    )
    g_ref = np.squeeze(g_ref, axis=(1, 3))

    # Derive RTF using h_mat
    f, h_ref = h_mat(f, z_src, z_rcv, r[0], depth, bottom_bc)
    f, h_rcv = h_mat(f, z_src, z_rcv, r[1:], depth, bottom_bc)
    g_hmat = h_rcv / h_ref
    g_hmat = np.squeeze(g_hmat, axis=(1, 2))

    print(np.allclose(g_ref, g_hmat, atol=1e-3))
    print(np.abs(g_ref - g_hmat))

    plt.figure()
    plt.plot(f, np.abs(g_ref), label="g_mat")
    plt.plot(f, np.abs(g_hmat), label="g_hmat", linestyle="--")
    plt.legend()
    plt.xlabel(r"$f$")
    plt.ylabel(r"$rtf$")

    plt.show()


def waveguide_params():
    # Define receivers and source positions
    depth = 1000
    # Source
    z_src = 5
    r_src = 30 * 1e3
    # Receivers
    z_rcv = depth - 1

    # Ri duration
    duration = 500

    return depth, r_src, z_src, z_rcv, duration


if __name__ == "__main__":
    # bottom_bc = "perfectly_rigid"
    # bottom_bc = "pressure_release"

    # # test_g_mat_h_mat(bottom_bc=bottom_bc)
    # depth = 1000
    # f = 50
    # k = 2 * np.pi * f / 1500
    # M = nb_propagating_modes(f, 1500, depth, bottom_bc="pressure_release")
    # print(M)
    # kzm = M * np.pi / depth
    # krm = np.sqrt(k**2 - kzm**2)
    # print(f"k_zm : {kzm}")
    # print(f"k_rm : {krm}")

    # zs = 5
    # z0 = 999
    # print(f"sin(kzm zs) : {np.sin(kzm * zs)}")
    # print(f"Approx term : {kzm * zs  - 1/6 * (kzm * zs)**3}")
    # print(f"Approx term : {-1 + 1/6 * (M*np.pi/depth)**3 * (depth-z0)**3}")
    # print_arrivals(z_src=5, z_rcv=999, r=30 * 1e3, depth=1000, n=4)

    #     D = 1e3
    #     f = 5
    #     z_src = 5

    #     dz = 1
    #     dr = 1
    #     R = 20 * 1e3
    #     z = np.arange(0, D, dz)
    #     r = np.arange(1, R, dr)

    #     n_rcv = 5
    #     delta_rcv = 100
    #     r_rcv_0 = R - 5 * 1e3
    #     z_rcv_0 = D - 5
    #     r_rcv = np.array([r_rcv_0 + delta_rcv * i for i in range(n_rcv)])
    #     z_rcv = np.array([z_rcv_0] * n_rcv)

    #     f, rr, zz, p_field = field(f, z_src, r, z, D)
    #     f0 = 10
    #     plot_tl(f, r, z, p_field, f_plot=f0, z_src=z_src, r_rcv=r_rcv, z_rcv=z_rcv)

    #### FIBERSCOPE CONFIG ####
    depth = 10
    z_src = 4
    z_rcv = depth - 0.40
    r = 4.5  # P1
    r = -4.5 + 25  # P4
    n = 100
    t, ir = image_source_ri(z_src, z_rcv, r, depth, n, t=None)

    plt.figure()
    plt.plot(t, ir)
    plt.xlim(0, 0.1)
    plt.xlabel(r"$t \, \textrm{[s]}$")
    plt.ylabel(r"$\textrm{Impulse response}$")
    plt.savefig("impulse_response.png")
    # plt.show()
