#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   normal_modes.py
@Time    :   2025/09/16 13:16:15
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class with usefull tools for normal modes computation
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
from publication.publication_figure import PubFigure, color

PubFigure()


# Fonctions de calcul des nombres d'ondes
def pekeris_eq_2_lhs(kr, omega, c_w, d):
    """Left-hand side of transcendental equation (Eq 2.186, Jensen et al. 2011, p.123)"""
    return np.tan(d * np.sqrt((omega / c_w) ** 2 - kr**2))


def pekeris_eq_2_rhs(kr, omega, c_w, c_s, rho_w, rho_s):
    """Right-hand side of transcendental equation (Eq 2.186, Jensen et al. 2011, p.123)"""
    return -(rho_w / rho_s) * np.sqrt(
        ((omega / c_w) ** 2 - kr**2) / (kr**2 - (omega / c_s) ** 2)
    )


def pekeris_kr_m(f, c_w, c_s, rho_w, rho_s, d):
    """Calculate the horizontal wavenumbers of the propagating modes in a Pekeris waveguide."""

    f = np.atleast_1d(f)
    omega = 2 * np.pi * f

    # Bornes (Cf eq. 2.187) pour avoir des solutions réelles
    kr_min = omega / c_s * (1 + 1e-10)
    kr_max = omega / c_w * (1 - 1e-10)
    kr = np.linspace(kr_min, kr_max, int(1e4))  # (n_search, nf)

    # Find roots (by equating lhs and rhs and then looking for sign changes in diff)
    lhs = pekeris_eq_2_lhs(kr, omega, c_w, d)  # (n_search, nf)
    rhs = pekeris_eq_2_rhs(kr, omega, c_w, c_s, rho_w, rho_s)  # (n_search, nf)
    diff = lhs - rhs
    diff_shift = np.roll(diff, 1, axis=0)
    idx_roots = (diff < 0) & (diff_shift > 0)

    # # Single frequency implementation
    # krm = kr[idx_roots]
    # # Sort roots in descending order
    # krm = np.sort(krm)[::-1]

    # Multi frequency (19/06/2026)
    krm = [
        kr[idx_roots[:, i_f], i_f] for i_f in range(f.size)
    ]  # Get krm for each frequency
    max_nb_modes = np.max(pekeris_n_modes(f, c_w, c_s, d))

    krm_ = []
    for krm_f in krm:
        # Sort in descending order
        sorted_krm_f = np.sort(krm_f)[::-1]
        # Padd to max modes numer to store in a 2D array
        krm_f_padded = np.pad(
            sorted_krm_f, (0, max_nb_modes - sorted_krm_f.size), constant_values=np.nan
        )
        # Add to list
        krm_.append(krm_f_padded)

    krm = np.array(krm_)

    return krm.astype(np.float32)


def cp_m(krm, omega):
    """Calculate the phase speeds of the propagating modes"""
    return (omega / krm).astype(np.float32)


def pekeris_n_modes(f, c_w, c_s, d):
    """Calculate the number of propagating modes in a Pekeris waveguide."""
    # TODO add page ref/eq
    n_max = np.int32(np.floor(2 * d / c_w * np.sqrt(1 - (c_w / c_s) ** 2) * f + 1 / 2))
    return n_max


def pekeris_cutoff_frequency(m, c_w, c_s, d):
    """Calculate the cut-off frequency of mode m in a Pekeris waveguide."""
    # TODO add page ref/eq
    f_c = ((m - 1 / 2) * c_w) / (2 * d * np.sqrt(1 - (c_w / c_s) ** 2))
    return f_c


def pekeris_modes(f, c_w, c_s, rho_w, rho_s, d):
    """Calculate the horizontal wavenumbers and phase speeds of the propagating modes in a Pekeris waveguide."""

    # Handle array of frequencies
    freq = np.atleast_1d(f)
    # Add an extra frequency point to compute last group speed
    # default_df = 0.01
    # df = (freq[-1] - freq[-2]) if freq.size > 1 else default_df
    # freq = np.append(freq, freq[-1] + df)

    # nf = freq.size
    # n_modes = np.max(
    #     pekeris_n_modes(freq, c_w, c_s, d)
    # )  # Max number of propagating modes

    # # Init arrays (Set to nan to avoid division by zero in cp_m)
    # krm_arr = np.empty((nf, n_modes)) * np.nan
    # kzm_arr = np.empty((nf, n_modes)) * np.nan
    # cpm_arr = np.empty((nf, n_modes)) * np.nan
    # cgm_arr = np.empty((nf, n_modes)) * np.nan
    # thetam_arr = np.empty((nf, n_modes)) * np.nan

    # Direct array derivation
    omega = 2 * np.pi * freq
    k = omega / c_w
    # Horizontal wavenumbers
    krm_arr = pekeris_kr_m(freq, c_w, c_s, rho_w, rho_s, d)
    # Vertical wavenumbers in water
    kzm_arr = np.sqrt(k[:, np.newaxis] ** 2 - krm_arr**2).astype(np.float32)
    # Phase speeds
    cpm_arr = cp_m(krm=krm_arr, omega=omega[:, np.newaxis])
    # Group speeds
    cgm_arr = pekeris_cg_m(f, c_w, c_s, rho_w, rho_s, d)
    # Mode angle
    thetam_arr = np.arctan(kzm_arr / krm_arr).astype(np.float32) * 180 / np.pi

    # for i, f in enumerate(freq):
    #     omega_i = 2 * np.pi * f

    #     # Horizontal wavenumbers
    #     krm_i = pekeris_kr_m(f, c_w, c_s, rho_w, rho_s, d)
    #     krm_arr[i, : len(krm_i)] = krm_i  # Fill array with actual values

    #     # Vertical wavenumbers in water
    #     kzm_i = np.sqrt((omega_i / c_w) ** 2 - krm_i**2)
    #     kzm_arr[i, : len(kzm_i)] = kzm_i  # Fill array with actual values

    #     # Phase speeds
    #     cpm_i = cp_m(krm_i, omega_i)
    #     cpm_arr[i, : len(cpm_i)] = cpm_i  # Fill array with actual values

    #     # Mode angle
    #     thetam_i = np.arctan(kzm_i / krm_i) * 180 / np.pi
    #     thetam_arr[i, : len(thetam_i)] = thetam_i

    # # Group speeds (approximation) : forward difference scheme -> u_m = d(omega)/d(kr) = delta_omega / (krm(omega+delta_omega) - krm(omega))
    # d_omega = 2 * np.pi * np.diff(freq)
    # d_omega = d_omega[:, np.newaxis]
    # d_krm = np.diff(krm_arr, axis=0)
    # cgm_arr = d_omega / d_krm

    # # Remove last row (extra frequency point only used for cg estimation)
    # krm_arr = krm_arr[:-1, :]
    # kzm_arr = kzm_arr[:-1, :]
    # cmp_arr = cmp_arr[:-1, :]

    return krm_arr, kzm_arr, cpm_arr, cgm_arr, thetam_arr


def pekeris_ir_duration(f, c_w, c_s, rho_w, rho_s, r, d):
    """Derive signal dispersion in a Pekeris waveguide. This can be used to estimate the duration of the impulse response.

    The duration is estimated as the time difference between the fastest arrival, defined by the maximum sound celerity in the
    waveguide, and the slowest arrival given by the minimum group speed of the fastest and slowest modes at range r.

    """
    # TODO add page ref/eq

    # Derive modal properties
    _, _, cmp, cgm, _ = pekeris_modes(f, c_w, c_s, rho_w, rho_s, d)

    # Minimum group speed
    cg_min = np.nanmin(cgm)

    # # Maximum phase speed
    # cp_max = np.nanmax(cmp)

    # Duration of the impulse response
    cmax = max(c_w, c_s)
    T_ir = r / cg_min - r / cmax

    return T_ir


def pekeris_cg_m(f, c_w, c_s, rho_w, rho_s, d):

    f = np.atleast_1d(f)
    omega = 2 * np.pi * f
    # Get horizontal wavenumbers
    krm = pekeris_kr_m(f, c_w, c_s, rho_w, rho_s, d)

    # Water layer
    k_w = omega / c_w
    kzm_w = np.sqrt(k_w[:, np.newaxis] ** 2 - krm**2)
    # Sediment layer
    k_s = omega / c_s
    kzm_s = np.sqrt(krm**2 - k_s[:, np.newaxis] ** 2)

    rho_ratio = rho_s / rho_s
    a = rho_ratio / (kzm_s**2 + rho_ratio**2 * kzm_w**2)
    cgm = (
        krm
        / omega[:, np.newaxis]
        * c_w**2
        * c_s**2
        * (kzm_s * d + a * (kzm_w**2 + kzm_s**2))
        / (c_s**2 * (kzm_s * d + a * kzm_s**2) + c_w**2 * a * kzm_w**2)
    )

    return cgm.astype(np.float32)


def pekeris_diag_test(f, c_w, c_s, rho_w, rho_s, d):
    """Run test to ensure everything is ok."""
    # Derive all modal properties
    krm, kzm, cpm, cgm, thetam = pekeris_modes(f, c_w, c_s, rho_w, rho_s, d)
    # Number of modes for each frequency
    n_modes = pekeris_n_modes(f, c_w, c_s, d)

    pekeris_dispersion_cg_f(f, cgm, n_modes)
    pekeris_dispersion_cp_f(f, cpm, n_modes)
    pekeris_dispersion_cp_cg_f(f, cpm, cgm, n_modes)


def pekeris_dispersion_cg_f(f, cgm, n_modes):
    plt.figure()

    # Iterate over modes
    modes = np.arange(1, np.max(n_modes) + 1, 1)
    for i_m, m in enumerate(modes):
        plt.plot(f, cgm[:, i_m], label=f"Mode {m}", color=color(i_m))

    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$c_g$ [m s$^{-1}$]")
    plt.legend()
    plt.show()


def pekeris_dispersion_cp_f(f, cpm, n_modes):
    plt.figure()

    # Iterate over modes
    modes = np.arange(1, np.max(n_modes) + 1, 1)
    for i_m, m in enumerate(modes):
        plt.plot(f, cpm[:, i_m], label=f"Mode {m}", color=color(i_m))

    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$c_{\phi}$ [m s$^{-1}$]")
    plt.legend()
    plt.show()


def pekeris_dispersion_cp_cg_f(f, cpm, cgm, n_modes):
    plt.figure()

    # Iterate over modes
    modes = np.arange(1, np.max(n_modes) + 1, 1)
    for i_m, m in enumerate(modes):
        plt.plot(f, cpm[:, i_m], label=f"Mode {m}", linestyle="--", color=color(i_m))
        plt.plot(f, cgm[:, i_m], label=f"Mode {m}", linestyle="-", color=color(i_m))

    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$c_g, c_{\phi}$ [m s$^{-1}$]")
    plt.legend()
    plt.show()


# TODO : A Corriger !!!
# def pekeris_modal_fct(z, f, c_w, c_s, rho_w, rho_s, d=depth):
#     """Calculate modal functions at depth z in a Pekeris waveguide. """

#     # Ensure z is a numpy array
#     z = np.atleast_1d(z)
#     z_w = z[z <= d]  # Depths in water
#     z_s = z[z > d]   # Depths in sediment

#     # Get horizontal wavenumbers
#     krm = pekeris_kr_m(f, c_w, c_s, rho_w, rho_s, d)
#     krm = krm.squeeze()

#     # Water layer
#     kzm_w = np.sqrt((2 * np.pi * f / c_water) ** 2 - krm**2)
#     # Sediment layer
#     kzm_s = np.sqrt(krm**2 - (2 * np.pi * f / c_sediment) ** 2)

#     # Normalization constant
#     Am = np.sqrt(2 / d)
#     phim_arr = np.empty((len(krm), len(z)), dtype=complex)
#     for i, kzm in enumerate(kzm_w):
#         # Modal functions
#         phi_m_w = Am * np.sin(kzm * z_w)
#         phi_m_s = Am * np.sin(kzm * d) * np.exp(1j*kzm_s[i] * (z_s - d))
#         phi_m = np.hstack((phi_m_w, phi_m_s))
#         phim_arr[i, :] = phi_m

#     return phim_arr


if __name__ == "__main__":
    c_w = 1500
    c_s = 2000
    rho_w = 1.0 * 1e3
    rho_s = 2.03 * 1e3
    r = 3000
    d = 35

    # Compute kr
    f = np.linspace(5, 100, 500)
    # pekeris_kr_m(f, c_w, c_s, rho_w, rho_s, d)

    # # Compute all information about the modes
    # krm_arr, kzm_arr, cpm_arr, cgm_arr, thetam_arr = pekeris_modes(
    #     f, c_w, c_s, rho_w, rho_s, d
    # )

    pekeris_diag_test(f, c_w, c_s, rho_w, rho_s, d)

    # Impulse duration
    # f = 800
    # tau = pekeris_ir_duration(f, c_w, c_s, rho_w, rho_s, r, d)
    # print(tau)
