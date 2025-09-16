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


# Fonctions de calcul des nombres d'ondes
def eq_2_lhs(kr, omega, c_w, d):
    """Left-hand side of transcendental equation (Eq 2.186, Jensen et al. 2011, p.123)"""
    return np.tan(d * np.sqrt((omega / c_w) ** 2 - kr**2))


def eq_2_rhs(kr, omega, c_w, c_s, rho_w, rho_s):
    """Right-hand side of transcendental equation (Eq 2.186, Jensen et al. 2011, p.123)"""
    return -(rho_w / rho_s) * np.sqrt(
        ((omega / c_w) ** 2 - kr**2) / (kr**2 - (omega / c_s) ** 2)
    )


def kr_m(f, c_w, c_s, rho_w, rho_s, d):
    """Calculate the horizontal wavenumbers of the propagating modes in a Pekeris waveguide."""

    omega = 2 * np.pi * f

    # Bornes (Cf eq. 2.187) pour avoir des solutions réelles
    kr_min = omega / c_s * 1.00001
    kr_max = omega / c_w * 0.99999
    kr = np.linspace(kr_min, kr_max, int(1e6))

    # Find roots (by equating lhs and rhs looking for sign changes in diff)
    lhs = eq_2_lhs(kr, omega, c_w, d)
    rhs = eq_2_rhs(kr, omega, c_w, c_s, rho_w, rho_s)
    diff = lhs - rhs
    diff_shift = np.roll(diff, 1)
    idx_roots = (diff < 0) & (diff_shift > 0)
    krm = kr[idx_roots]

    # Sort roots in descending order
    krm = np.sort(krm)[::-1]

    return krm


def cp_m(krm, omega):
    """Calculate the phase speeds of the propagating modes"""
    return omega / krm


def pekeris_n_modes(f, c_w, c_s, d):
    """Calculate the number of propagating modes in a Pekeris waveguide."""
    n_max = int(np.floor(2 * d / c_w * np.sqrt(1 - (c_w / c_s) ** 2) * f + 1 / 2))
    return n_max


def pekeris_cutoff_frequency(m, c_w, c_s, d):
    """Calculate the cut-off frequency of mode m in a Pekeris waveguide."""
    f_c = ((m - 1 / 2) * c_w) / (2 * d * np.sqrt(1 - (c_w / c_s) ** 2))
    return f_c


def pekeris_modes(f, c_w, c_s, rho_w, rho_s, d):
    """Calculate the horizontal wavenumbers and phase speeds of the propagating modes in a Pekeris waveguide."""

    # Handle array of frequencies
    freq = np.atleast_1d(f)
    # Add an extra frequency point to compute last group speed
    default_df = 0.01
    df = (freq[-1] - freq[-2]) if freq.size > 1 else default_df
    freq = np.append(freq, freq[-1] + df)

    nf = freq.size
    n_modes = pekeris_n_modes(
        np.max(freq), c_w, c_s, d
    )  # Max number of propagating modes

    # Init arrays
    krm_arr = (
        np.empty((nf, n_modes)) * np.nan
    )  # Set to nan to avoid division by zero in cp_m
    kzm_arr = np.empty((nf, n_modes)) * np.nan
    cmp_arr = np.empty((nf, n_modes)) * np.nan
    cgm_arr = np.empty((nf, n_modes)) * np.nan
    thetam_arr = np.empty((nf, n_modes)) * np.nan

    for i, f in enumerate(freq):
        omega_i = 2 * np.pi * f

        # Horizontal wavenumbers
        krm_i = kr_m(f, c_w, c_s, rho_w, rho_s, d)
        krm_arr[i, : len(krm_i)] = krm_i  # Fill array with actual values

        # Vertical wavenumbers
        kzm_i = np.sqrt((omega_i / c_w) ** 2 - krm_i**2)
        kzm_arr[i, : len(kzm_i)] = kzm_i  # Fill array with actual values

        # Phase speeds
        cpm_i = cp_m(krm_i, omega_i)
        cmp_arr[i, : len(cpm_i)] = cpm_i  # Fill array with actual values

        # Mode angle
        thetam_i = np.arctan(kzm_i / krm_i) * 180 / np.pi
        thetam_arr[i, : len(thetam_i)] = thetam_i

    # Group speeds (approximation) : forward difference scheme -> u_m = d(omega)/d(kr) = delta_omega / (krm(omega+delta_omega) - krm(omega))
    d_omega = 2 * np.pi * np.diff(freq)
    d_omega = d_omega[:, np.newaxis]
    d_krm = np.diff(krm_arr, axis=0)
    cgm_arr = d_omega / d_krm

    # Remove last row (extra frequency point)
    krm_arr = krm_arr[:-1, :]
    kzm_arr = kzm_arr[:-1, :]
    cmp_arr = cmp_arr[:-1, :]

    return krm_arr, kzm_arr, cmp_arr, cgm_arr, thetam_arr


def pekeris_ir_duration(f, c_w, c_s, rho_w, rho_s, r, d):
    """Derive signal dispersion in a Pekeris waveguide. This can be used to estimate the duration of the impulse response.

    The duration is estimated as the time difference between the fastest arrival, defined by the maximum sound celerity in the
    waveguide, and the slowest arrival given by the minimum group speed of the fastest and slowest modes at range r.

    """
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


# TODO : A Corriger !!!
# def pekeris_modal_fct(z, f, c_w, c_s, rho_w, rho_s, d=depth):
#     """Calculate modal functions at depth z in a Pekeris waveguide. """

#     # Ensure z is a numpy array
#     z = np.atleast_1d(z)
#     z_w = z[z <= d]  # Depths in water
#     z_s = z[z > d]   # Depths in sediment

#     # Get horizontal wavenumbers
#     krm = kr_m(f, c_w, c_s, rho_w, rho_s, d)
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
