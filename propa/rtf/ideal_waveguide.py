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


def nb_propagating_modes(f, c, D):
    """
    Function that calculates the number of propagating modes in a ideal waveguide
    :param f: frequency (Hz)
    :param c: speed of sound in water (m/s)
    :param D: depth of the waveguide (m)
    :return: number of propagating modes
    """

    n = np.floor(1 / 2 * (1 + 4 * D * f / c)).astype(int)

    return n


def cutoff_frequency(c, D):
    """
    Function that calculates the cutoff frequency of an ideal waveguide
    :param c: speed of sound in water (m/s)
    :param D: depth of the waveguide (m)
    :return: cutoff frequency (Hz)
    """

    fc = c / (4 * D)

    return fc


def kz(m, D):
    """
    Function that calculates the vertical wavenumber
    :param m: mode number
    :return: vertical wavenumber
    """

    kz_m = (m - 1 / 2) * np.pi / D

    return kz_m


def kr(m, f, D):
    """
    Function that calculates the radial wavenumber
    :param m: mode number
    :param f: frequency (Hz)
    :return: radial wavenumber
    """

    kr_m = np.sqrt((2 * np.pi * f / c0) ** 2 - kz(m, D) ** 2)

    return kr_m


def alpha(D, r):

    alpha_r = np.exp(1j * np.pi / 4) * 1 / (D * np.sqrt(2 * np.pi * r))

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


def psi(m, z, D):

    return np.sin(kz(m, D) * z)


def psi_normalised(m, z, D, rho):

    return psi(m, z, D) * np.sqrt(2 * rho / D)


def u_m(m, f, z_src, z, D):

    u = psi(m, z_src, D) * psi(m, z, D) * 1 / np.sqrt(kr(m, f, D))

    return u


def g(f, z_src, z_rcv_ref, z_rcv, D, r_rcv_ref, r):

    f = f[f > cutoff_frequency(c0, D)]

    g = []
    for fi in f:
        n = nb_propagating_modes(fi, c0, D)
        m = np.arange(1, n + 1)

        phi_l = u_m(m, fi, z_src, z=z_rcv, D=D) * np.exp(-1j * kr(m, fi, D) * r)
        phi_0 = u_m(m, fi, z_src, z=z_rcv_ref, D=D) * np.exp(
            -1j * kr(m, fi, D) * r_rcv_ref
        )
        g_fi = A_l(r_rcv_ref, r) * np.sum(phi_l) / np.sum(phi_0)
        g.append(g_fi)

    g = np.array(g)

    return f, g


def g_mat(f, z_src, z_rcv_ref, z_rcv, D, r_rcv_ref, r):
    """
    Derive the RTF g(f) for a range of frequencies f, z_src depths and ranges r."""
    # Ensure z_src is an array
    z_src = np.atleast_1d(z_src)
    # Ensure r is an array
    r = np.atleast_1d(r)

    rflat = r.flatten()  # Flatten the ranges corresponding to several receivers
    zz_src, rr = np.meshgrid(z_src, rflat)
    f = f[f > cutoff_frequency(c0, D)]

    # Define the g matrix
    shape = (len(f),) + zz_src.shape
    g_matrix = np.zeros(shape, dtype=np.complex64)

    for i, fi in enumerate(f):
        n = nb_propagating_modes(fi, c0, D)
        m = np.arange(1, n + 1)

        zz_src, rr, mm = np.meshgrid(z_src, r, m)
        _, rr_ref, _ = np.meshgrid(z_src, r_rcv_ref, m)

        phi_l = u_m(mm, fi, zz_src, z=z_rcv, D=D) * np.exp(-1j * kr(mm, fi, D) * rr)
        phi_0 = u_m(mm, fi, zz_src, z=z_rcv_ref, D=D) * np.exp(
            -1j * kr(mm, fi, D) * rr_ref
        )
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


def h(f, z_src, z, r, D):

    f = f[f > cutoff_frequency(c0, D)]

    t0 = time()
    h = []
    for fi in f:
        n = nb_propagating_modes(fi, c0, D) - 1
        m = np.arange(1, n + 1)

        phi_m = u_m(m, fi, z_src, z=z, D=D) * np.exp(-1j * kr(m, fi, D) * r)

        h_fi = alpha(D, r) * np.sum(phi_m)
        h.append(h_fi)

    h = np.array(h)
    # print(f"Elapsed time: {time() - t0:.2f}s")

    # t0 = time()
    # n = nb_propagating_modes(f, c0, D)
    # n_max = np.max(n)
    # m = np.arange(1, n_max + 1)

    # h = 0
    # for i, mi in enumerate(m):
    #     phi_mi_f = u_m(mi, f, z_src, z=z, D=D) * np.exp(1j * kr(mi, f, D) * r)
    #     mask = n >= mi
    #     h += np.where(mask, phi_mi_f, 0)
    #     # h_mat[i, :] = phi_mi_f

    # h *= alpha(D, r)

    # print(f"Elapsed time: {time() - t0:.2f}s")

    # print()
    # for i_n in range(n_max):
    #     h_fi = alpha(D, r) * np.sum(h_mat[i_n, :])
    #     h.append(h_fi)

    # mask =

    return f, h


def image_source_ri(z_src, z, r, D, n, t=None):

    # Number of terms to include in the sum
    m = np.arange(1, n + 1)
    # Image source - receiver distance follwoing definitions from Jensen p.104
    zm1 = 2 * D * m - z_src + z
    zm2 = 2 * D * (m + 1) - z_src - z
    zm3 = 2 * D * m + z_src + z
    zm4 = 2 * D * (m + 1) + z_src - z
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


def rtf_distance(ranges, z_src, z_rcv, D, n_rcv, d_rcv):

    Ts = 0.001
    fs = 1 / Ts
    # Closest power of 2
    nfft = 2**12
    f = fft.rfftfreq(nfft, 1 / fs)

    delta_rcv = np.arange(1, n_rcv, d_rcv)
    for r in ranges:
        rl = r + delta_rcv
        f, g_f = g(f=f, z_src=z_src, z_rcv=z_rcv, D=D, r_rcv_ref=r, rl=rl)


if __name__ == "__main__":
    pass
