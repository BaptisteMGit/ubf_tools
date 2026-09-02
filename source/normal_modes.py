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
import time
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

from propa.kraken_toolbox.src.kraken_env import (
    KrakenEnv,
    KrakenMedium,
    KrakenBottomHalfspace,
    KrakenField,
    KrakenFlp,
)
from propa.kraken_toolbox.src.kraken_manager import KrakenManager

# from propa.kraken_toolbox import plot_utils as pu

from scipy.optimize import root_scalar
from publication.publication_figure import PubFigure, color

PubFigure()

# ======================================================================================================================
# Image sources
# ======================================================================================================================


def image_sources_arrivals(z_src, z_rcv, r, depth, n, c0=1500):
    # Number of terms to include in the sum
    m = np.arange(0, n)
    # Image source - receiver distance following definitions from Jensen p.104
    zm1 = 2 * depth * m - z_src + z_rcv
    zm2 = 2 * depth * (m + 1) - z_src - z_rcv
    zm3 = 2 * depth * m + z_src + z_rcv
    zm4 = 2 * depth * (m + 1) + z_src - z_rcv

    # # Correction 27/08/2025
    # zm1 = 2 * depth * m + z_src - z_rcv
    # zm2 = 2 * depth * (m + 1) - z_src - z_rcv
    # zm3 = 2 * depth * m + z_src + z_rcv
    # zm4 = 2 * depth * (m + 1) + z_src - z_rcv

    Rm1 = np.sqrt(r**2 + zm1.astype(np.float64) ** 2)
    Rm2 = np.sqrt(r**2 + zm2.astype(np.float64) ** 2)
    Rm3 = np.sqrt(r**2 + zm3.astype(np.float64) ** 2)
    Rm4 = np.sqrt(r**2 + zm4.astype(np.float64) ** 2)

    arrivals = np.empty((len(m), 4))
    for i_m in m:
        t1 = Rm1[i_m] / c0
        t2 = Rm2[i_m] / c0
        t3 = Rm3[i_m] / c0
        t4 = Rm4[i_m] / c0

        print(
            f"m = {m[i_m]} : \n"
            + f"\t t1 = {t1}s \n"
            + f"\t t2 = {t2}s \n"
            + f"\t t3 = {t3}s \n"
            + f"\t t4 = {t4}s \n"
        )
        arrivals[i_m][0] = t1
        arrivals[i_m][1] = t2
        arrivals[i_m][2] = t3
        arrivals[i_m][3] = t4

    return arrivals


def group_image_source_arrivals(arrivals):
    all_arrivals = arrivals.flatten()
    all_arrivals = np.sort(all_arrivals)

    grouped_arrivals = []
    grp_size = arrivals.shape[1]
    for grp_idx in range(arrivals.shape[0]):
        arrivals_grp_i = all_arrivals[grp_idx * grp_size : (grp_idx + 1) * grp_size]
        grouped_arrivals.append(arrivals_grp_i)
    grouped_arrivals = np.array(grouped_arrivals)
    return grouped_arrivals


# ======================================================================================================================
# Pekeris waveguide
# ======================================================================================================================


# Fonctions de calcul des nombres d'ondes
def pekeris_eq_2_lhs(kr, omega, c1, d):
    """Left-hand side of transcendental equation (Eq 2.186, Jensen et al. 2011, p.123)"""
    # Water layer
    kw = omega / c1  # k in water
    kz1 = np.sqrt(kw**2 - kr**2)  # kz in water

    # Left hand side of transcendental equation (Eq 2.186, Jensen et al. 2011, p.123)
    lhs = np.tan(d * kz1)

    return lhs


def pekeris_eq_2_rhs(kr, omega, c1, c2, rho1, rho2):
    """Right-hand side of transcendental equation (Eq 2.186, Jensen et al. 2011, p.123)"""
    # Water layer
    kw = omega / c1  # k in water
    kzm_1 = np.sqrt(kw**2 - kr**2)  # kz in water

    # Sediment layer
    k2 = omega / c2  # k in sediment
    # kz in sediment, kr > k2 for real solution to transcendental equation (complex solution correspond to leaky modes (Jensen et al. 2011, p.123))
    kzm_2 = 1j * np.sqrt(kr**2 - k2**2)

    # Right hand side of transcendental equation (Eq 2.186, Jensen et al. 2011, p.123)
    rhs = -(1j * rho2 * kzm_1) / (rho1 * kzm_2)
    # rhs is real for kr > k2
    rhs = np.real(rhs)

    return rhs


def pekeris_kr_m(f, c1, c2, rho1, rho2, d, debug=False):
    """Derive horizontal wavenumbers of the propagating modes in a Pekeris waveguide by
    resolution of the transcendental equation (Eq 5.81, Jensen et al. 2011, p.355, Eq. 27, Pekeris 1948).
    """

    f = np.atleast_1d(f)

    # Implementation with root_scalar (25/08/2026), more robust
    t0 = time.time()

    n_search = int(1e4)
    krm = []
    for i_f, ff in enumerate(f):
        omega = 2 * np.pi * ff

        # Bornes (Cf eq. 2.187) pour avoir des solutions réelles
        kr_min = omega / c2 * (1 + 1e-15)  # k in sediment
        kr_max = omega / c1 * (1 - 1e-15)  # k in water

        if omega == 0:  # No mode for f = 0
            krm.append([])
            continue

        kr = np.linspace(kr_min, kr_max, n_search)  # (n_search)

        def func_pek(kr):
            return pekeris_eq_2_lhs(kr, omega, c1, d) - pekeris_eq_2_rhs(
                kr, omega, c1, c2, rho1, rho2
            )

        # # Build research intervals
        # test_brackets = []
        # for i_test in range(kr.size - 1):
        #     fct_i = func_pek(kr[i_test])
        #     fct_ip1 = func_pek(kr[i_test + 1])
        #     # Check if they have different signs (bounds must have different size to ensure it contains a zero)
        #     diff_signs = fct_i / fct_ip1 <= 0
        #     if diff_signs:
        #         bracket = [kr[i_test], kr[i_test + 1]]
        #         test_brackets.append(bracket)
        # print(test_brackets)

        # Build research intervals (faster)
        fval = func_pek(kr)
        idx = np.where(np.signbit(fval[:-1]) != np.signbit(fval[1:]))[0]
        test_brackets = np.column_stack((kr[idx], kr[idx + 1]))

        krm_f = []
        if test_brackets.size > 0:  # Assert at least one interval is suitable
            for i_brack in range(test_brackets.shape[0]):
                # Solve in kr[i], kr[i+1]
                bracket = (test_brackets[i_brack, 0], test_brackets[i_brack, 1])
                sol = root_scalar(func_pek, bracket=bracket, xtol=1e-20)
                krm_f_ = sol.root
                if np.abs(func_pek(krm_f_)) < 1 and np.isreal(krm_f_):
                    krm_f.append(krm_f_)
                    # print(func_pek(krm_f_))

        krm.append(krm_f)

    max_nb_modes = np.max(pekeris_n_modes(f, c1, c2, d))

    krm_ = []
    for krm_f in krm:
        # Sort in descending order
        sorted_krm_f = np.sort(krm_f)[::-1]
        # Padd to max modes number to store in a 2D array
        krm_f_padded = np.pad(
            sorted_krm_f,
            (0, max_nb_modes - sorted_krm_f.size),
            constant_values=np.nan,
        )
        # Add to list
        krm_.append(krm_f_padded)

    krm = np.array(krm_)  # (nf, nmodes)

    if debug:
        krm_root_scalar = krm
        print(krm_root_scalar)
        print(f"Delay with root_scalar {time.time()-t0}")

        # Without root_scalar (this implementation is usually quicker but may require a lot of points to catch all zeros and may leads to memory overflow )
        t0 = time.time()

        omega = 2 * np.pi * f
        # Bornes (Cf eq. 2.187) pour avoir des solutions réelles
        kr_min = omega / c2 * (1 + 1e-10)  # k in sediment
        kr_max = omega / c1 * (1 - 1e-10)  # k in water
        # WARNING : high number of points increases precision but can lead to memory overflow in case we compute kr for many frequencies at the same time
        # n_search = 2 * 1e4 looks like a good compromise
        n_search = int(100 * 1e4)
        kr = np.linspace(kr_min, kr_max, int(2 * 1e4))  # (n_search, nf)

        # Find roots (by equating lhs and rhs and then looking for sign changes in diff)
        lhs = pekeris_eq_2_lhs(kr, omega, c1, d)  # (n_search, nf)
        rhs = pekeris_eq_2_rhs(kr, omega, c1, c2, rho1, rho2)  # (n_search, nf)
        diff = lhs - rhs
        diff_shift = np.roll(diff, 1, axis=0)
        idx_roots = (diff < 0) & (diff_shift > 0)

        # Multi frequency (19/06/2026)
        krm = [
            kr[idx_roots[:, i_f], i_f] for i_f in range(f.size)
        ]  # Get krm for each frequency

        max_nb_modes = np.max(pekeris_n_modes(f, c1, c2, d))

        krm_ = []
        for krm_f in krm:
            # Sort in descending order
            sorted_krm_f = np.sort(krm_f)[::-1]
            # Padd to max modes number to store in a 2D array
            krm_f_padded = np.pad(
                sorted_krm_f,
                (0, max_nb_modes - sorted_krm_f.size),
                constant_values=np.nan,
            )
            # Add to list
            krm_.append(krm_f_padded)

        krm = np.array(krm_)  # (nf, nmodes)

        print(f"Delay without root_scalar {time.time()-t0}")
        print(f"Solutions match : {np.allclose(krm, krm_root_scalar)}")

    return krm.astype(np.float32)


def cp_m(krm, omega):
    """Calculate the phase speeds of the propagating modes"""
    return (omega / krm).astype(np.float32)


def pekeris_n_modes(f, c1, c2, d):
    """Calculate the number of propagating modes in a Pekeris waveguide."""
    # Equivalent to Eq. 2.191, Jensen p. 125 or Eq. 8.89, Jensen p.637
    f = np.atleast_1d(f)
    n_max = np.int32(np.floor(2 * d / c1 * np.sqrt(1 - (c1 / c2) ** 2) * f + 1 / 2))
    if f.size == 1:
        n_max = n_max[0]  # scalar input f -> scalar output n_max
    return n_max


def pekeris_cutoff_frequency(m, c1, c2, d):
    """Calculate the cut-off frequency of mode m in a Pekeris waveguide."""
    # Eq. 2.191, Jensen p. 125 or Eq. 8.89, Jensen p.637
    f_c = ((m - 1 / 2) * c1) / (2 * d * np.sqrt(1 - (c1 / c2) ** 2))
    return f_c


def pekeris_ir_duration(f, c1, c2, rho1, rho2, r, d):
    """Derive signal dispersion in a Pekeris waveguide. This can be used to estimate the duration of the impulse response.

    The duration is estimated as the time difference between the fastest arrival, defined by the maximum sound celerity in the
    waveguide, and the slowest arrival given by the minimum group speed of the fastest and slowest modes at range r.

    """
    # Eq. 8.16, Jensen p. 616

    # Horizontal wavenumbers
    krm = pekeris_kr_m(f, c1, c2, rho1, rho2, d)
    # Group speeds
    cgm = pekeris_cg_m(f, krm, c1, c2, rho1, rho2, d)

    # Minimum group speed
    cg_min = np.nanmin(cgm)

    # # Maximum phase speed
    # # Phase speeds
    # omega = 2 * np.pi * f
    # cpm = cp_m(krm=krm, omega=omega[:, np.newaxis])
    # cp_max = np.nanmax(cmp)

    # Duration of the impulse response
    cmax = max(c1, c2)
    T_ir = r / cg_min - r / cmax

    return T_ir


def pekeris_kzm_12(f, c1, c2, krm):

    f = np.atleast_1d(f)
    omega = 2 * np.pi * f

    # Water layer
    k1 = omega / c1
    k1 = k1[:, np.newaxis]  # (nf, 1)
    kzm_1 = np.sqrt(k1**2 - krm**2)

    # Sediment layer
    k2 = omega / c2
    k2 = k2[:, np.newaxis]  # (nf, 1)

    idx_krm_lt_k2 = krm < k2
    kzm_2 = np.where(
        idx_krm_lt_k2, np.sqrt(k2**2 - krm**2), 1j * np.sqrt(krm**2 - k2**2)
    )

    return kzm_1, kzm_2


def pekeris_cg_m(f, krm, c1, c2, rho1, rho2, d):

    # Add an extra frequency point to compute last group speed
    f = np.atleast_1d(f)
    default_df = 0.01
    df = (f[-1] - f[-2]) if f.size > 1 else default_df
    freq = np.append(f, f[-1] + df)
    fp1 = freq[-1]

    # Compute krm for only one more frequency point to reduce computing time
    krm_fp1 = pekeris_kr_m(fp1, c1, c2, rho1, rho2, d)
    # Get rid of extra modes (modes that do not exist below krm_fp1)
    krm_fp1 = krm_fp1[:, 0 : krm.shape[1]]

    # Add point
    krm = np.concatenate([krm, krm_fp1], axis=0)

    # Group speeds (approximation) : forward difference scheme -> u_m = d(omega)/d(kr) = delta_omega / (krm(omega+delta_omega) - krm(omega))
    d_omega = 2 * np.pi * np.diff(freq)
    d_omega = d_omega[:, np.newaxis]
    d_krm = np.diff(krm, axis=0)
    cgm = d_omega / d_krm

    # TODO check where it comes from ?
    # rho_ratio = rho2 / rho2
    # a = rho_ratio / (kzm_2**2 + rho_ratio**2 * kzm_1**2)
    # cgm = (
    #     krm
    #     / omega[:, np.newaxis]
    #     * c1**2
    #     * c2**2
    #     * (kzm_2 * d + a * (kzm_1**2 + kzm_2**2))
    #     / (c2**2 * (kzm_2 * d + a * kzm_2**2) + c1**2 * a * kzm_1**2)
    # )

    return cgm.astype(np.float32)


def plot_pekeris_cg_f(f, cgm, n_modes=None):

    if n_modes is None:
        n_modes = cgm.shape[1]

    plt.figure()

    # Iterate over modes
    modes = np.arange(1, np.max(n_modes) + 1, 1)
    for i_m, m in enumerate(modes):
        plt.plot(f, cgm[:, i_m], label=f"Mode {m}", color=color(i_m))

    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$c_g$ [m s$^{-1}$]")
    plt.legend()
    # plt.show()

    fig, ax = plt.gcf(), plt.gca()

    return fig, ax


def plot_pekeris_cp_f(f, cpm, n_modes=None):

    if n_modes is None:
        n_modes = cpm.shape[1]

    plt.figure(figsize=(16, 10))

    # Iterate over modes
    modes = np.arange(1, np.max(n_modes) + 1, 1)
    for i_m, m in enumerate(modes):
        plt.plot(f, cpm[:, i_m], label=f"Mode {m}", color=color(i_m))

    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$c_{\phi}$ [m s$^{-1}$]")
    plt.legend()
    # plt.show()

    fig, ax = plt.gcf(), plt.gca()

    return fig, ax


def plot_pekeris_cp_cg_f(f, cpm, cgm, n_modes=None):

    if n_modes is None:
        n_modes = cpm.shape[1]

    # Mode number m
    modes = np.arange(1, np.max(n_modes) + 1, 1)

    fig, ax = plt.subplots(figsize=(16, 12), nrows=1, ncols=1)
    i_ax = 0

    # Iterate over modes
    for i_m, m in enumerate(modes):
        # i_ax = i_m // row_size
        # ax = axs[i_ax]
        ax.plot(
            f,
            cpm[:, i_m],
            label=rf"$c_{{\phi}}$ (m = {{{m}}})",
            linestyle="--",
            color=color(i_m),
        )
        ax.plot(
            f,
            cgm[:, i_m],
            label=rf"$c_g$ (m = {{{m}}})",
            linestyle="-",
            color=color(i_m),
        )
        ax.legend(fontsize=14, ncols=2)

    fig.supxlabel("Fréquence [Hz]")
    fig.supylabel(r"$c_g, c_{\phi}$ [m s$^{-1}$]")

    return fig, ax


def pekeris_green_fct(f, c1, c2, rho1, rho2, attn2, d, z_s, z_r, r):
    """Calculate green's functions at depth z in a Pekeris waveguide."""

    # Horizontal wavenumbers
    krm = pekeris_kr_m(f, c1, c2, rho1, rho2, d)  # (n_f, n_modes)

    # Get vertical wavenumber in water and sediment
    kzm_1, kzm_2 = pekeris_kzm_12(f, c1, c2, krm)  # (n_f, n_modes)

    # Get modal function at z_s and z_r
    phim_zs = pekeris_modal_fct(z_s, f, c1, c2, d, krm)  # (nf, n_modes, 1)
    phim_zr = pekeris_modal_fct(z_r, f, c1, c2, d, krm)  # (nf, n_modes, 1)

    # Remove extra dim
    phim_zs = np.squeeze(phim_zs, axis=-1)
    phim_zr = np.squeeze(phim_zr, axis=-1)

    # Compute modal loss tangent
    # alpha_2_dB_lambda = 10
    alpha_2 = alpha_dB_lambda_to_nepers_m(f, attn2, c2)
    # print(
    #     f"Att coeff in sediment alpha_2 = {alpha_2_dB_lambda} dB / lambda = {alpha_2} nepers / m"
    # )
    alpha_m = pekeris_alpha_attn(f, krm, c1, alpha_2, c2, rho2, d)
    # Apply to krm
    krm = apply_alpha_att_perturbation(krm, alpha_m)

    # Based on Pekeris 1948 formulation
    f = np.atleast_1d(f)
    omega = 2 * np.pi * f
    omega_2d = omega[:, np.newaxis]  # (nf, 1)
    r = np.atleast_1d(r)
    r_2d = r[np.newaxis, :]  # (1, nr)

    # Common factor (does not depend on f)
    # Q = omega * (rho1 * np.pi / d) * np.sqrt(8 / r) * np.exp(+1j * np.pi / 4)
    Q = (
        omega_2d * (rho1 * np.pi / d) * np.sqrt(8 / r_2d) * np.exp(+1j * np.pi / 4)
    )  # (nf, nr)

    # Pekeris 1948, Eq. 25
    def F(x):
        return x / (
            x - np.sin(x) * np.cos(x) - (rho1 / rho2) ** 2 * np.sin(x) ** 2 * np.tan(x)
        )

    # Modal amplitude
    Am = F(kzm_1 * d) * phim_zs * phim_zr  # (nf, nmodes)
    # Modal phase
    # gm = Am * np.exp(-1j * krm * r) / np.sqrt(krm)        # r is scalar
    gm = (
        Am[..., np.newaxis]
        * np.exp(-1j * krm[..., np.newaxis] * r_2d[np.newaxis, ...])
        / np.sqrt(krm[..., np.newaxis])
    )  # r is 1D array

    # Green's function
    gf = Q * np.nansum(gm, axis=1)  # p = Q sum Am e^(-i krm r) / sqrt(krm)

    # # Remove dimension if r is scalar
    # gf = np.squeeze(gf)

    return gf


def pekeris_modal_fct(z, f, c1, c2, d, krm):
    """Calculate modal functions at depth z in a Pekeris waveguide.

    Derivation according to Pekeris 1948 (eq. 38).
    """

    # Get vertical wavenumber in water and sediment
    kzm_1, kzm_2 = pekeris_kzm_12(f, c1, c2, krm)  # (n_f, n_modes)

    # Cast to 3D (nf, nmodes, 1)
    kzm_1_3d = kzm_1[..., np.newaxis]
    kzm_2_3d = kzm_2[..., np.newaxis]

    # Ensure z is a numpy array
    z = np.atleast_1d(z)
    z1 = z[z <= d]  # Depths in water
    z2 = z[z > d]  # Depths in sediment

    # Cast to 2D (1, nz)
    z1_2d = z1[np.newaxis]
    z2_2d = z2[np.newaxis]

    # Define modal fct according to eq. 48
    # z <= D -> mode in water column
    phi_m_1 = np.sin(kzm_1_3d * z1_2d)  # (nf, nmodes, nz)
    # z > D -> mode in sediment
    phi_m_2 = np.sin(kzm_1_3d * d) * np.exp(
        1j * kzm_2_3d * (z2_2d - d)
    )  # (nf, nmodes, nz)

    # phi_m = np.hstack((phi_m_1, phi_m_2))
    phi_m = np.concatenate([phi_m_1, phi_m_2], axis=-1)  # (nf, nmodes, nz)

    return phi_m


def pekeris_alpha_attn(f, krm, c1, alpha_2, c2, rho2, d):
    """
    Derive loss tangent alpha.

    Jensen p.387, Eq. 5.180
    """
    # Get vertical wavenumber in water and sediment
    kzm_1, kzm_2 = pekeris_kzm_12(f, c1, c2, krm)  # (n_f, n_modes)

    f = np.atleast_1d(f)
    omega = 2 * np.pi * f
    omega_2d = omega[:, np.newaxis]

    alpha_2_2d = alpha_2[:, np.newaxis]
    # From Eq. 5.180
    alpha_m = (
        (1j * alpha_2_2d * omega_2d)
        / (krm * c2 * rho2)
        * np.sin(kzm_1 * d) ** 2
        / (2 * kzm_2)
    )
    return alpha_m


def apply_alpha_att_perturbation(krm, alpha_m):
    """Apply perturbation theory.
    Eq. 5.167
    For the convention e^(-iwt)
    krm = krm_0 + 2i * alpham * krm_0
    e^i(kr - wt) = e^i(kr_0 r - wt) x e^(-2 alpha_m kr_0)

    For the convention e^(+iwt) :
    krm = krm_0 - 2i * alpham * krm_0  to obtain attenuation
    e^i(wt - kr) = e^i(wt - kr_0 r) x e^(-2 alpha_m kr_0)

    """

    krm_attn = krm - 2 * 1j * alpha_m * krm

    return krm_attn


def alpha_dB_lambda_to_nepers_m(f, alpha, c):
    """Convert attenuation coefficient from dB / lambda to nepers/m.

    alpha = attenuation coeff in dB / lambda
    f = frequencies in Hz
    c = sound celerity in m/s

    alpha_dB_lambda = -20 log ( exp(-alpha(x + lambda)) / exp(-alpha(x)) ) = alpha * lambda * 20 log(e)
    """
    f = np.atleast_1d(f)
    lambd = c / f

    alpha_nepers_m = alpha / (lambd * 20 * np.log10(np.e))

    return alpha_nepers_m


def impulse_response(gf, f, fs, tau, output_mult_fact=4):

    # Number of freq samples
    df = f[1] - f[0]
    nt = int(fs / df)
    # Number of temporal samples
    nfft_inv = (
        nt * output_mult_fact
    )  # To improve temporal resolution for visual inspection

    # # Correct for propagation delay
    delay = np.exp(1j * 2 * np.pi * tau * f)
    gf *= delay

    # Impulse response in time domain
    gt = np.fft.irfft(gf, n=nfft_inv)

    # Time vector
    T = nt * 1 / fs
    dt = T / nfft_inv
    t = np.arange(0, T, dt)

    return t, gt


def transmission_loss(gf):
    gf[(gf == 0) | np.isnan(gf)] = 1e-20

    p0 = 1e6  # TODO : This does not make sense but leads to coherent TL values, it needs to be clarified
    # TL = - 20 log (p/p0)
    tl = -20 * np.log10(np.abs(gf) / p0)

    return tl


def plot_modal_function_single_freq(modal_fcts, z, depth=None, n_modes=None):

    if n_modes is None:
        n_modes = modal_fcts.shape[0]

    # Mode number m
    modes = np.arange(1, np.max(n_modes) + 1, 1)

    # Plot modal functions
    fig, axs = plt.subplots(1, len(modes), sharey=True)
    axs = np.atleast_1d(axs)
    # Revert y-axis
    axs[0].invert_yaxis()

    max_amp = np.nanmax(np.abs(modal_fcts))
    axs[0].set_ylabel("Profondeur [m]")

    for i, m in enumerate(modes):
        axs[i].plot(np.real(modal_fcts[i]), z, label=f"Real", linestyle="-", color="k")
        axs[i].plot(np.imag(modal_fcts[i]), z, label=f"Im", linestyle="--", color="r")
        axs[i].set_title(f"Mode {m}")
        axs[i].set_xlim([-1.1 * max_amp, 1.1 * max_amp])
        axs[i].grid()

        # Add depth line
        if depth is not None:
            axs[i].axhline(depth, linestyle="--", color="k")

    return fig, axs


def plot_modal_function_multi_freq(modal_fcts, z, f, depth=None, n_modes=None):

    nfreq = modal_fcts.shape[0]
    if not f.size == nfreq:
        raise ValueError(f"Incoherent dim for f and modal_fcts.shape[0]")

    if n_modes is None:
        n_modes = modal_fcts.shape[1]

    # Mode number m
    modes = np.arange(1, np.max(n_modes) + 1, 1)

    # Plot modal functions
    fig, axs = plt.subplots(1, len(modes), sharey=True)
    axs = np.atleast_1d(axs)
    # Revert y-axis
    axs[0].invert_yaxis()

    max_amp = np.nanmax(np.abs(modal_fcts))
    axs[0].set_ylabel("Profondeur [m]")

    for i, m in enumerate(modes):
        for j in range(nfreq):
            axs[i].plot(
                np.real(modal_fcts[j, i, :]),
                z,
                # label=f"Real",
                linestyle="-",
                color=color(j),
                label=f"{f[j]} Hz",
            )
            axs[i].plot(
                np.imag(modal_fcts[j, i, :]),
                z,
                # label=f"Im",
                linestyle="--",
                color=color(j),
                # label=f"{f[j]} Hz",
            )

        axs[i].set_title(f"Mode {m}")
        axs[i].set_xlim([-1.1 * max_amp, 1.1 * max_amp])
        axs[i].grid()
        axs[i].legend(loc="upper left")

        # Add depth line
        if depth is not None:
            axs[i].axhline(depth, linestyle="--", color="k")

    return fig, axs


def plot_modal_function(modal_fcts, z, f=None, depth=None, n_modes=None):

    # Check if multiple freq provided
    f = np.atleast_1d(f)
    if f.size > 1:
        fig, axs = plot_modal_function_multi_freq(
            modal_fcts=modal_fcts, z=z, f=f, depth=depth, n_modes=n_modes
        )

    else:
        fig, axs = plot_modal_function_single_freq(
            modal_fcts=modal_fcts, z=z, depth=depth, n_modes=n_modes
        )

    return fig, axs


def plot_green_fct(f, gf, z_s, z_r, r):

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Amplitude
    axs[0].plot(f, np.abs(gf), color="k")
    axs[0].set_ylabel(r"$\lvert H(f) \rvert$")

    # Phase
    axs[1].plot(f, np.angle(gf), color="k")
    axs[1].set_ylabel("arg(H(f)) [rad]")

    fig.supxlabel("Fréquence [Hz]")
    fig.suptitle(rf"z_s = {z_s:.1f} m, z_r = {z_r:.1f} m, r = {r:.1f} m")

    return fig, axs


def plot_impulse_response(t, gt, z_s, z_r, r):

    plt.figure()
    plt.plot(t, gt, color="k")
    # plt.xlabel("Temps [s]")
    plt.xlabel(r"Temps réduit $t - r / c_{max}$ [s]")
    plt.ylabel("h(t) [Pa]")
    plt.title(rf"z_s = {z_s:.1f} m, z_r = {z_r:.1f} m, r = {r:.1f} m")

    fig, ax = plt.gcf(), plt.gca()
    return fig, ax


def plot_impulse_response_and_source_image_arrivals(t, gt, z_s, z_r, r, arrivals, tau):

    # Plot impulse response
    fig, ax = plot_impulse_response(t, gt, z_s, z_r, r)

    # Add arrivals
    # for arr in arrivals.flatten():
    #     corrected_arr = arr - tau
    #     ax.axvline(corrected_arr, linestyle="--", color="b")

    for i in range(arrivals.shape[0]):
        for j in range(arrivals.shape[1]):
            corrected_arr = arrivals[i, j] - tau
            if j == 0:
                ax.axvline(
                    corrected_arr, linestyle="--", color=color(i), label=f"n = {i}"
                )
            else:
                ax.axvline(corrected_arr, linestyle="--", color=color(i))

    return fig, ax


def plot_impulse_response_and_stft(
    t,
    gt,
    z_s,
    z_r,
    r,
    fmax,
    nfft=1024,
    nperseg=51,
    noverlap=50,
    stft_vmin_percentile=90,
    stft_vmax_percentile=99.99,
):

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Signal
    axs[0].plot(t, gt, color="k")
    axs[0].set_ylabel("h(t) [Pa]")
    axs[0].set_xlim(t.min(), t.max())

    # STFT
    fs = 1 / (t[1] - t[0])
    f, t, Sxx = sp.stft(
        gt,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling="psd",
    )

    Sxx_db = 20 * np.log10(abs(Sxx))

    im = axs[1].pcolormesh(
        t,
        f,
        Sxx_db,
        cmap="jet",
        vmin=np.percentile(Sxx_db, stft_vmin_percentile),
        vmax=np.percentile(Sxx_db, stft_vmax_percentile),
    )
    # plt.colorbar(im, ax=axs[1])
    axs[1].set_ylim(0, fmax)

    axs[1].set_ylabel("Fréquence [Hz]")

    # fig.supxlabel("Temps [s]")
    fig.supxlabel(r"Temps réduit $t - r / c_{max}$ [s]")
    fig.suptitle(rf"z_s = {z_s:.1f} m, z_r = {z_r:.1f} m, r = {r:.1f} m")

    return fig, axs


def plot_impulse_response_stft_dispersion_curve(
    t,
    gt,
    z_s,
    z_r,
    r,
    fmax,
    f,
    cgm,
    tau,
    nfft=1024,
    nperseg=51,
    noverlap=50,
    stft_vmin_percentile=90,
    stft_vmax_percentile=99.99,
):

    # Theoretical dispersion curves
    tm_theo = r / cgm - tau
    # No correction
    # tm_theo = r / cgm

    plt.figure()

    # STFT
    fs = 1 / (t[1] - t[0])
    ff, tt, Sxx = sp.stft(
        gt,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling="psd",
    )

    Sxx_db = 20 * np.log10(abs(Sxx))

    im = plt.pcolormesh(
        tt,
        ff,
        Sxx_db,
        cmap="jet",
        vmin=np.percentile(Sxx_db, stft_vmin_percentile),
        vmax=np.percentile(Sxx_db, stft_vmax_percentile),
    )
    # plt.colorbar(im, ax=plt.gca())

    # Add dispersion curves
    for i_m in range(tm_theo.shape[1]):
        plt.plot(tm_theo[:, i_m], f, color="k")

    plt.ylim(0, fmax)
    plt.ylabel("Fréquence [Hz]")
    # plt.xlabel("Temps [s]")
    plt.xlabel(r"Temps réduit $t - r / c_{max}$ [s]")
    plt.title(rf"z_s = {z_s:.1f} m, z_r = {z_r:.1f} m, r = {r:.1f} m")

    fig, ax = plt.gcf(), plt.gca()
    return fig, ax


def plot_tl_fr(
    g_fr,
    r,
    f,
    tl_vmin_percentile=1,
    tl_vmax_percentile=95,
):

    tl = transmission_loss(g_fr)
    tlmax = np.percentile(tl, tl_vmax_percentile)
    tlmin = np.percentile(tl, tl_vmin_percentile)

    plt.figure()
    im = plt.pcolormesh(r, f, tl, vmin=tlmin, vmax=tlmax, cmap="jet")
    plt.xlabel("Distance r [m]")
    plt.ylabel("Fréquence [Hz]")
    plt.colorbar(im, label="TL [dB]")

    fig, ax = plt.gcf(), plt.gca()
    return fig, ax


def plot_tl_r(
    g_r,
    r,
    spherical_loss=False,
    cylindrical_loss=False,
):

    tl = transmission_loss(g_r)

    plt.figure()
    plt.plot(r, tl, color="k")
    plt.xlabel("Distance r [m]")
    plt.ylabel("TL [dB]")
    plt.gca().invert_yaxis()

    icol = 0
    if spherical_loss:
        tl_spherical = 20 * np.log10(r)
        plt.plot(r, tl_spherical, color=color(icol), label=r"$20 \log_{10} (r)$")
        icol += 1

    if cylindrical_loss:
        tl_cylindrical = 10 * np.log10(r)
        plt.plot(r, tl_cylindrical, color=color(icol), label=r"$10 \log_{10} (r)$")
        icol += 1

    if spherical_loss or cylindrical_loss:
        plt.legend()

    fig, ax = plt.gcf(), plt.gca()
    return fig, ax


# ======================================================================================================================
# Tests
# ======================================================================================================================
def test_pekeris_env():
    # Waveguide parameters
    rho1 = 1.0 * 1e3  # density in water (kg/m^3)
    c1 = 1500  # sound celerity in water (m/s)
    c2 = 1600  # sound celerity in fluid sediment (m/s)
    rho2 = 1.5 * 1e3  # density in fluid sediment (kg/m^3)
    d = 100  # waveguide depth (m)

    return c1, c2, rho1, rho2, d


def test_pekeris_sig():
    fmax = 250  # Max frequency (Hz)
    T = 5  # Signal duration to generate (s)
    fs = 2 * fmax  # Sampling frequency (Hz) = Nyquist
    ts = 1 / fs  # sampling interval (s)
    nt = T * fs  # Number of samples
    f = np.fft.rfftfreq(n=nt, d=ts)  # Frequency vector

    return fmax, T, fs, nt, f


def test_kr_pekeris():
    """
    Test function to derive horizontal wavenumber kr by resolution of the transcendental equation.
    Two implementations are compared and one can check than the solution matches the one given by the code of J.Bonnel (tuto).

    """
    f = np.linspace(10, 250, 20)
    # Waveguide parameters
    c1, c2, rho1, rho2, d = test_pekeris_env()

    for f_i in f:
        krm = pekeris_kr_m(f_i, c1, c2, rho1, rho2, d, debug=True)
        print(f"f = {f_i} Hz\n krm = {krm.flatten()}")


def test_cp_pekeris():
    """Compute and plot phase speed cp."""

    c1, c2, rho1, rho2, d = test_pekeris_env()
    _, _, _, _, freq = test_pekeris_sig()

    # Direct array derivation
    omega = 2 * np.pi * freq
    # Horizontal wavenumbers
    krm = pekeris_kr_m(freq, c1, c2, rho1, rho2, d)
    # Phase speeds
    cpm = cp_m(krm=krm, omega=omega[:, np.newaxis])

    plot_pekeris_cp_f(f=freq, cpm=cpm)
    plt.show()


def test_cg_pekeris():
    """Compute and plot group speed cg."""

    c1, c2, rho1, rho2, d = test_pekeris_env()
    _, _, _, _, freq = test_pekeris_sig()

    # Horizontal wavenumbers
    krm = pekeris_kr_m(freq, c1, c2, rho1, rho2, d)
    # Group speeds
    cgm = pekeris_cg_m(freq, krm, c1, c2, rho1, rho2, d)

    plot_pekeris_cg_f(f=freq, cgm=cgm)
    plt.show()


def test_cp_cg_pekeris():
    """
    Compute and plot phase speed cp and group speed cg.
    """

    c1, c2, rho1, rho2, d = test_pekeris_env()
    _, _, _, _, freq = test_pekeris_sig()

    # Angular freq
    omega = 2 * np.pi * freq
    # Horizontal wavenumbers
    krm = pekeris_kr_m(freq, c1, c2, rho1, rho2, d)
    # Phase speeds
    cpm = cp_m(krm=krm, omega=omega[:, np.newaxis])
    # Group speeds
    cgm = pekeris_cg_m(freq, krm, c1, c2, rho1, rho2, d)

    plot_pekeris_cp_cg_f(f=freq, cpm=cpm, cgm=cgm)
    plt.show()


def test_modal_fct_pekeris():

    c1, c2, rho1, rho2, d = test_pekeris_env()
    _, _, _, _, freq = test_pekeris_sig()

    # Horizontal wavenumbers
    krm = pekeris_kr_m(freq, c1, c2, rho1, rho2, d)

    dz = 0.01
    z = np.arange(dz, d + 30, dz)
    phim_arr = pekeris_modal_fct(z, freq, c1, c2, d, krm)

    # Single freq
    f0 = 100
    idx_f0 = np.argmin(np.abs(freq - f0))
    phim_f0 = phim_arr[idx_f0, ...]
    M_f0 = np.max(pekeris_n_modes(f0, c1, c2, d))
    plot_modal_function(modal_fcts=phim_f0, z=z, depth=d, n_modes=M_f0)

    # Multi freq
    fplot = [25, 50, 75]
    idx_fplot = [np.argmin(np.abs(freq - f)) for f in fplot]
    phim_fplot = phim_arr[idx_fplot, :]
    M_fplot = np.max(pekeris_n_modes(fplot, c1, c2, d))
    plot_modal_function(modal_fcts=phim_fplot, z=z, f=fplot, depth=d, n_modes=M_fplot)

    plt.show()


def run_full_diag(sig_param, env_param, src_rcv_param):

    # Unpack params
    # Signal
    freq = sig_param.get("freq", None)
    fs = sig_param.get("fs", None)
    fmax = sig_param.get("fmax", None)
    # Environment
    c1 = env_param.get("c1", None)
    c2 = env_param.get("c2", None)
    rho1 = env_param.get("rho1", None)
    rho2 = env_param.get("rho2", None)
    attn2 = env_param.get("attn2", None)
    d = env_param.get("depth", None)

    # Source / receiver
    z_s = src_rcv_param.get("z_s", 5)
    z = src_rcv_param.get("z_rcv", 20)
    r_grid = src_rcv_param.get("r_rcv", np.array([1e4]))
    r0 = src_rcv_param.get("r0", 1e4)

    r_grid = np.atleast_1d(r_grid)

    # Frequency to plot
    f0 = fmax
    # Number of modes to reprensent
    M_f0 = np.max(pekeris_n_modes(f0, c1, c2, d))
    M_f0 = min(M_f0, 10)
    if M_f0 > 0:
        print(f"Maximum number of modes = {M_f0}")
    else:
        print("No propagative modes found.")
        return

    ###### Modal group and phase speed ######
    # Angular freq
    omega = 2 * np.pi * freq
    # Horizontal wavenumbers
    krm = pekeris_kr_m(freq, c1, c2, rho1, rho2, d)
    # Phase speeds
    cpm = cp_m(krm=krm, omega=omega[:, np.newaxis])
    # Group speeds
    cgm = pekeris_cg_m(freq, krm, c1, c2, rho1, rho2, d)
    # Plot
    plot_pekeris_cp_cg_f(f=freq, cpm=cpm, cgm=cgm, n_modes=M_f0)

    ###### Modal function ######
    # dz = 0.1
    # zphi = np.arange(dz, d + 30, dz)
    nz = 5000
    zphi = np.linspace(0, d + max(50, 0.1 * d), nz)
    phim_arr = pekeris_modal_fct(zphi, freq, c1, c2, d, krm)

    idx_f0 = np.argmin(np.abs(freq - f0))
    phim_f0 = phim_arr[idx_f0, ...]
    plot_modal_function(modal_fcts=phim_f0, z=zphi, depth=d, n_modes=M_f0)

    # # Multi freq
    # fplot = [25, 50, 75]
    # idx_fplot = [np.argmin(np.abs(freq - f)) for f in fplot]
    # phim_fplot = phim_arr[idx_fplot, :]
    # M_fplot = np.max(pekeris_n_modes(fplot, c1, c2, d))
    # plot_modal_function(modal_fcts=phim_fplot, z=z, f=fplot, depth=d, n_modes=M_fplot)

    ###### Green's function at z, r ######
    # z_s = 5
    # z, r = 20, 10000

    g_fr = pekeris_green_fct(freq, c1, c2, rho1, rho2, attn2, d, z_s, z, r_grid)
    idx_r0 = np.argmin(np.abs(r_grid - r0))
    gf = g_fr[:, idx_r0]

    # gf = pekeris_green_fct(freq, c1, c2, rho1, rho2, d, z_s, z, r)
    plot_green_fct(freq, gf, z_s=z_s, z_r=z, r=r0)

    # Impulse response
    output_mult_fact = 4
    tau = r0 / max(c2, c1)  # Minimum arrival time
    t, gt = impulse_response(gf, freq, fs, tau=tau, output_mult_fact=output_mult_fact)
    plot_impulse_response(t, gt, z_s, z, r0)

    # Comparison to source image method
    n_img = 10
    arrivals = image_sources_arrivals(z_src=z_s, z_rcv=z, r=r0, depth=d, n=n_img, c0=c1)
    # grouped_arrivals = group_image_source_arrivals(arrivals)
    plot_impulse_response_and_source_image_arrivals(
        t, gt, z_s, z, r0, arrivals, tau=tau
    )

    # stft
    # Réglage identique au code du tuto de J.Bonnel
    nfft = 1024
    nperseg = 51 * output_mult_fact
    noverlap = nperseg - 1
    plot_impulse_response_and_stft(
        t, gt, z_s, z, r0, fmax, nfft=nfft, nperseg=nperseg, noverlap=noverlap
    )

    # Stft and theoretical dispersion curves
    plot_impulse_response_stft_dispersion_curve(
        t,
        gt,
        z_s,
        z,
        r0,
        fmax,
        freq,
        cgm,
        tau=tau,
        nfft=nfft,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    if r_grid.size > 1:
        # Plot TL(f, r)
        idx_f_tl = np.logical_and(freq > 50, freq < 60)
        f_tl = freq[idx_f_tl]
        g_fr_ftl = g_fr[idx_f_tl, ...]
        plot_tl_fr(
            g_fr_ftl,
            r_grid,
            f_tl,
            tl_vmin_percentile=1,
            tl_vmax_percentile=95,
        )

        # Plot TL(f=f0, r)
        f0 = 80
        idx_f0 = np.argmin(np.abs(freq - f0))
        g_r_ftl = g_fr[idx_f0, ...]
        plot_tl_r(
            g_r_ftl,
            r_grid,
            spherical_loss=True,
            cylindrical_loss=True,
        )

    plt.show()


def test_pekeris_JBonnel():
    "Test Pekeris functions by comparison with results from the Matlab tuto code by JBonnel."

    # Waveguide parameters
    rho1 = 1.0 * 1e3  # density in water (kg/m^3)
    c1 = 1500  # sound celerity in water (m/s)
    c2 = 1600  # sound celerity in fluid sediment (m/s)
    rho2 = 1.5 * 1e3  # density in fluid sediment (kg/m^3)
    d = 100  # waveguide depth (m)

    # Signal properties
    fmax = 250  # Max frequency (Hz)
    T = 10  # Signal duration to generate (s)
    fs = 2 * fmax  # Sampling frequency (Hz) = Nyquist
    ts = 1 / fs  # sampling interval (s)
    T = T - ts
    nt = int(T * fs)  # Number of samples
    freq = np.fft.rfftfreq(n=nt, d=ts)  # Frequency vector

    # Source / receiver properties
    z_s = d
    z_rcv = d
    r_rcv = 10 * 1e3  # receiver range (m)

    # Run test
    src_rcv_param = {"z_s": z_s, "z_rcv": z_rcv, "r_rcv": r_rcv}
    sig_param = {"freq": freq, "fs": fs, "fmax": fmax}
    env_param = {"c1": c1, "c2": c2, "rho1": rho1, "rho2": rho2, "depth": d}
    run_full_diag(sig_param=sig_param, env_param=env_param, src_rcv_param=src_rcv_param)
    # run_full_diag(freq, c1, c2, rho1, rho2, d, fs, fmax)


def test_pekeris_Jensen_p119():
    "Test Pekeris functions by comparison with results in Jensen. Pekeris definition p.119."

    # Waveguide parameters
    rho1 = 1.0 * 1e3  # density in water (kg/m^3)
    c1 = 1500  # sound celerity in water (m/s)
    rho2 = 1.8 * 1e3  # density in fluid sediment (kg/m^3)
    c2 = 1800  # sound celerity in fluid sediment (m/s)
    d = 100  # waveguide depth (m)

    # Signal properties
    fmax = 50  # Max frequency (Hz)
    T = 10  # Signal duration to generate (s)
    fs = 2 * fmax  # Sampling frequency (Hz) = Nyquist
    ts = 1 / fs  # sampling interval (s)
    nt = T * fs  # Number of samples
    freq = np.fft.rfftfreq(n=nt, d=ts)  # Frequency vector

    # Source / receiver properties
    z_s = 36
    z_rcv = 36
    r_rcv = 30 * 1e3

    # Run test
    src_rcv_param = {"z_s": z_s, "z_rcv": z_rcv, "r_rcv": r_rcv}
    sig_param = {"freq": freq, "fs": fs, "fmax": fmax}
    env_param = {"c1": c1, "c2": c2, "rho1": rho1, "rho2": rho2, "depth": d}
    run_full_diag(sig_param=sig_param, env_param=env_param, src_rcv_param=src_rcv_param)
    # run_full_diag(freq, c1, c2, rho1, rho2, d, fs, fmax)


def test_pekeris_Jensen_p350():
    "Test Pekeris functions by comparison with results in Jensen. Pekeris definition p.350."

    # Waveguide parameters
    rho1 = 1.0 * 1e3  # density in water (kg/m^3)
    c1 = 1500  # sound celerity in water (m/s)
    rho2 = 1.0 * 1e3  # density in fluid sediment (kg/m^3)
    c2 = 2000  # sound celerity in fluid sediment (m/s)
    d = 100  # waveguide depth (m)

    # Signal properties
    fmax = 75  # Max frequency (Hz)
    T = 10  # Signal duration to generate (s)
    fs = 2 * fmax  # Sampling frequency (Hz) = Nyquist
    ts = 1 / fs  # sampling interval (s)
    nt = T * fs  # Number of samples
    freq = np.fft.rfftfreq(n=nt, d=ts)  # Frequency vector

    # Source / receiver properties
    z_s = 25
    z_rcv = 50
    r_rcv = 30 * 1e3

    # Run test
    src_rcv_param = {"z_s": z_s, "z_rcv": z_rcv, "r_rcv": r_rcv}
    sig_param = {"freq": freq, "fs": fs, "fmax": fmax}
    env_param = {"c1": c1, "c2": c2, "rho1": rho1, "rho2": rho2, "depth": d}
    run_full_diag(sig_param=sig_param, env_param=env_param, src_rcv_param=src_rcv_param)


def test_pekeris_Jensen_p636():
    "Test Pekeris functions by comparison with results in Jensen. Pekeris definition p.636."

    # Waveguide parameters
    rho1 = 1.0 * 1e3  # density in water (kg/m^3)
    c1 = 1500  # sound celerity in water (m/s)
    rho2 = 1.5 * 1e3  # density in fluid sediment (kg/m^3)
    c2 = 1600  # sound celerity in fluid sediment (m/s)
    d = 100  # waveguide depth (m)

    # Signal properties
    fmax = 75  # Max frequency (Hz)
    T = 15  # Signal duration to generate (s)
    fs = 2 * fmax  # Sampling frequency (Hz) = Nyquist
    ts = 1 / fs  # sampling interval (s)
    nt = T * fs  # Number of samples
    freq = np.fft.rfftfreq(n=nt, d=ts)  # Frequency vector

    # Source / receiver properties
    z_s = 25
    z_rcv = 20
    r_rcv = 30 * 1e3

    # Run test
    src_rcv_param = {"z_s": z_s, "z_rcv": z_rcv, "r_rcv": r_rcv}
    sig_param = {"freq": freq, "fs": fs, "fmax": fmax}
    env_param = {"c1": c1, "c2": c2, "rho1": rho1, "rho2": rho2, "depth": d}
    run_full_diag(sig_param=sig_param, env_param=env_param, src_rcv_param=src_rcv_param)


def test_pekeris_short_ir():
    "Test Pekeris functions for a waveguide with short IR."

    # Waveguide parameters
    rho1 = 1.0 * 1e3  # density in water (kg/m^3)
    c1 = 1500  # sound celerity in water (m/s)
    rho2 = 1.5 * 1e3  # density in fluid sediment (kg/m^3)
    c2 = 1600  # sound celerity in fluid sediment (m/s)
    d = 100  # waveguide depth (m)

    # Signal properties
    fmax = 100  # Max frequency (Hz)
    T = 5  # Signal duration to generate (s)
    fs = 2 * fmax  # Sampling frequency (Hz) = Nyquist
    ts = 1 / fs  # sampling interval (s)
    nt = int(T * fs)  # Number of samples
    freq = np.fft.rfftfreq(n=nt, d=ts)  # Frequency vector

    # Source / receiver properties
    z_s = 5
    z_rcv = d - 0.5
    r_rcv = np.linspace(10 * 1e3, 50 * 1e3, int(1e4))
    r0 = 30 * 1e3

    # Run test
    src_rcv_param = {"z_s": z_s, "z_rcv": z_rcv, "r_rcv": r_rcv, "r0": r0}
    sig_param = {"freq": freq, "fs": fs, "fmax": fmax}
    env_param = {"c1": c1, "c2": c2, "rho1": rho1, "rho2": rho2, "depth": d}
    run_full_diag(sig_param=sig_param, env_param=env_param, src_rcv_param=src_rcv_param)
    # run_full_diag(freq, c1, c2, rho1, rho2, d, fs, fmax)


######
# Validation avec kraken
####


# def compare_cp_cg_f(f, cpm, cgm, cpm_kraken, cgm_kraken, n_modes=None):

#     if n_modes is None:
#         n_modes = cpm.shape[1]

#     # Mode number m
#     modes = np.arange(1, np.max(n_modes) + 1, 1)

#     fig, ax = plt.subplots(figsize=(16, 12), nrows=1, ncols=1)
#     i_ax = 0

#     # Iterate over modes
#     for i_m, m in enumerate(modes):
#         # i_ax = i_m // row_size
#         # ax = axs[i_ax]
#         ax.plot(
#             f,
#             cpm[:, i_m],
#             label=rf"$c_{{\phi}}$ (m = {{{m}}})",
#             linestyle="--",
#             color=color(i_m),
#         )
#         ax.plot(
#             f,
#             cgm[:, i_m],
#             label=rf"$c_g$ (m = {{{m}}})",
#             linestyle="-",
#             color=color(i_m),
#         )

#         # ax.plot(
#         #     f,
#         #     cpm[:, i_m],
#         #     label=rf"$c_{{\phi}}$ (m = {{{m}}})",
#         #     linestyle="--",
#         #     color=color(i_m),
#         # )
#         # ax.plot(
#         #     f,
#         #     cgm[:, i_m],
#         #     label=rf"$c_g$ (m = {{{m}}})",
#         #     linestyle="-",
#         #     color=color(i_m),
#         # )

#         ax.legend(fontsize=14, ncols=2)

#     fig.supxlabel("Fréquence [Hz]")
#     fig.supylabel(r"$c_g, c_{\phi}$ [m s$^{-1}$]")

#     return fig, ax


if __name__ == "__main__":

    # test_kr_pekeris()     # OK
    # test_cp_pekeris()     # OK
    # test_cg_pekeris()     # Ok
    # test_cp_cg_pekeris()  # OK
    # test_modal_fct_pekeris()  # OK

    # test_pekeris_JBonnel()  # OK
    # test_pekeris_Jensen_p119()  # OK
    # test_pekeris_Jensen_p636()  # OK
    # test_pekeris_Jensen_p350()

    test_pekeris_short_ir()  # OK

# ======================================================================================================================
# Left overs
# ======================================================================================================================

# def pekeris_modes(f, c1, c2, rho1, rho2, d):
#     """Calculate the horizontal wavenumbers and phase speeds of the propagating modes in a Pekeris waveguide."""

#     # Handle array of frequencies
#     freq = np.atleast_1d(f)
#     # Add an extra frequency point to compute last group speed
#     # default_df = 0.01
#     # df = (freq[-1] - freq[-2]) if freq.size > 1 else default_df
#     # freq = np.append(freq, freq[-1] + df)

#     # nf = freq.size
#     # n_modes = np.max(
#     #     pekeris_n_modes(freq, c1, c2, d)
#     # )  # Max number of propagating modes

#     # # Init arrays (Set to nan to avoid division by zero in cp_m)
#     # krm_arr = np.empty((nf, n_modes)) * np.nan
#     # kzm_arr = np.empty((nf, n_modes)) * np.nan
#     # cpm_arr = np.empty((nf, n_modes)) * np.nan
#     # cgm_arr = np.empty((nf, n_modes)) * np.nan
#     # thetam_arr = np.empty((nf, n_modes)) * np.nan

#     # Direct array derivation
#     omega = 2 * np.pi * freq
#     k = omega / c1
#     # Horizontal wavenumbers
#     krm_arr = pekeris_kr_m(freq, c1, c2, rho1, rho2, d)
#     # Vertical wavenumbers in water
#     kzm_arr = np.sqrt(k[:, np.newaxis] ** 2 - krm_arr**2).astype(np.float32)
#     # Phase speeds
#     cpm_arr = cp_m(krm=krm_arr, omega=omega[:, np.newaxis])
#     # Group speeds
#     cgm_arr = pekeris_cg_m(f, c1, c2, rho1, rho2, d)
#     # Mode angle
#     thetam_arr = np.arctan(kzm_arr / krm_arr).astype(np.float32) * 180 / np.pi

#     # for i, f in enumerate(freq):
#     #     omega_i = 2 * np.pi * f

#     #     # Horizontal wavenumbers
#     #     krm_i = pekeris_kr_m(f, c1, c2, rho1, rho2, d)
#     #     krm_arr[i, : len(krm_i)] = krm_i  # Fill array with actual values

#     #     # Vertical wavenumbers in water
#     #     kzm_i = np.sqrt((omega_i / c1) ** 2 - krm_i**2)
#     #     kzm_arr[i, : len(kzm_i)] = kzm_i  # Fill array with actual values

#     #     # Phase speeds
#     #     cpm_i = cp_m(krm_i, omega_i)
#     #     cpm_arr[i, : len(cpm_i)] = cpm_i  # Fill array with actual values

#     #     # Mode angle
#     #     thetam_i = np.arctan(kzm_i / krm_i) * 180 / np.pi
#     #     thetam_arr[i, : len(thetam_i)] = thetam_i

#     # # Group speeds (approximation) : forward difference scheme -> u_m = d(omega)/d(kr) = delta_omega / (krm(omega+delta_omega) - krm(omega))
#     # d_omega = 2 * np.pi * np.diff(freq)
#     # d_omega = d_omega[:, np.newaxis]
#     # d_krm = np.diff(krm_arr, axis=0)
#     # cgm_arr = d_omega / d_krm

#     # # Remove last row (extra frequency point only used for cg estimation)
#     # krm_arr = krm_arr[:-1, :]
#     # kzm_arr = kzm_arr[:-1, :]
#     # cmp_arr = cmp_arr[:-1, :]

#     return krm_arr, kzm_arr, cpm_arr, cgm_arr, thetam_arr


# def pekeris_cg_m(f, c1, c2, rho1, rho2, d):

#     # Add an extra frequency point to compute last group speed
#     f = np.atleast_1d(f)
#     default_df = 0.01
#     df = (f[-1] - f[-2]) if f.size > 1 else default_df
#     freq = np.append(f, f[-1] + df)

#     # Get horizontal wavenumbers
#     krm = pekeris_kr_m(freq, c1, c2, rho1, rho2, d)

#     # Group speeds (approximation) : forward difference scheme -> u_m = d(omega)/d(kr) = delta_omega / (krm(omega+delta_omega) - krm(omega))
#     d_omega = 2 * np.pi * np.diff(freq)
#     d_omega = d_omega[:, np.newaxis]
#     d_krm = np.diff(krm, axis=0)
#     cgm = d_omega / d_krm

#     # TODO check where it comes from ?
#     # rho_ratio = rho2 / rho2
#     # a = rho_ratio / (kzm_2**2 + rho_ratio**2 * kzm_1**2)
#     # cgm = (
#     #     krm
#     #     / omega[:, np.newaxis]
#     #     * c1**2
#     #     * c2**2
#     #     * (kzm_2 * d + a * (kzm_1**2 + kzm_2**2))
#     #     / (c2**2 * (kzm_2 * d + a * kzm_2**2) + c1**2 * a * kzm_1**2)
#     # )

#     return cgm.astype(np.float32)


# def pekeris_diag_test(f, c1, c2, rho1, rho2, d):
#     """Run test to ensure everything is ok."""
#     # Derive all modal properties
#     krm, kzm, cpm, cgm, thetam = pekeris_modes(f, c1, c2, rho1, rho2, d)
#     # Number of modes for each frequency
#     n_modes = pekeris_n_modes(f, c1, c2, d)

#     # plot_pekeris_cg_f(f, cgm, n_modes)
#     # plot_pekeris_cp_f(f, cpm, n_modes)
#     plot_pekeris_cp_cg_f(f, cpm, cgm, n_modes)

#     # plt.show()


# def pekeris_green_fct(f, c1, c2, rho1, rho2, d, z_s, z_r, r):
#     """Calculate green's functions at depth z in a Pekeris waveguide."""

#     # Derive all modal properties
#     krm, kzm, cpm, cgm, thetam = pekeris_modes(f, c1, c2, rho1, rho2, d)

#     # Get vertical wavenumber in water and sediment
#     kzm_1, kzm_2 = pekeris_kzm_12(f, c1, c2, krm)  # (n_f, n_modes)

#     # Based on Pekeris 1948 formulation
#     omega = 2 * np.pi * f
#     omega_2d = omega[np.newaxis]
#     # r_2d = r[np.newaxis]

#     # # Common factor
#     # Q = omega * (rho1 * np.pi / d) * np.sqrt(8 / r) * np.exp(+1j * np.pi / 4)

#     # def F(x):
#     #     """Equation 25 and A73"""
#     #     return x / (
#     #         x
#     #         - np.sin(x) * np.cos(x)
#     #         - (rho1 / rho2) ** 2 * np.sin(x) ** 2 * np.tan(x)
#     #     )

#     # # # Solution in water layer
#     # # if z_r < d:
#     # #     A_m = (
#     # #         F(kzm_1 * d) * np.sin(kzm_1 * z_s) * np.sin(kzm_1 * z_r)
#     # #     )  # Modal amplitude
#     # # # Solutution in sediment
#     # # else:
#     # #     A_m = (
#     # #         F(kzm_1 * d)
#     # #         * np.sin(kzm_1 * z_s)
#     # #         * np.sin(kzm_1 * d)
#     # #         * np.exp(1j * kzm_2 * (z_r - d))
#     # #     )  # Modal amplitude

#     # # A_m *= np.exp(-1j * krm * r) / np.sqrt(krm)  # Propagative term

#     # # gfm = Q * np.sum(A_m)

#     # Adaptation from Bonnel et al 2020 (MATLAB provided tool box) # TODO : update and verify according to Jensen2011
#     A_2 = (
#         2
#         * rho1
#         * kzm_1
#         * kzm_2
#         / (
#             kzm_1 * kzm_2 * d
#             - 1 / 2 * kzm_2 * np.sin(2 * kzm_1 * d)
#             + rho1 / rho2 * kzm_1 * np.sin(kzm_1 * d) ** 2
#         )
#     )
#     # Global multiplicative factor
#     A_2 = 1 / 4 * A_2 * 1j * np.exp(1j * np.pi / 4) / rho1

#     gfm = (
#         A_2
#         * np.sin(kzm_1 * z_s)
#         * np.exp(-1j * krm * r)
#         / np.sqrt(krm * r)
#         * np.sin(kzm_1 * z_r)
#     )

#     # # Replace np.nan by 0
#     gfm[np.isnan(gfm)] = 0
#     g_f = np.sum(gfm, axis=1)

#     return g_f
