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

from scipy.optimize import root_scalar
from publication.publication_figure import PubFigure, color

PubFigure()


# ======================================================================================================================
# Pekeris waveguide
# ======================================================================================================================


# Fonctions de calcul des nombres d'ondes
def pekeris_eq_2_lhs(kr, omega, c_w, d):
    """Left-hand side of transcendental equation (Eq 2.186, Jensen et al. 2011, p.123)"""
    # Water layer
    kw = omega / c_w  # k in water
    kz_w = np.sqrt(kw**2 - kr**2)  # kz in water

    # Left hand side of transcendental equation (Eq 2.186, Jensen et al. 2011, p.123)
    lhs = np.tan(d * kz_w)

    return lhs


def pekeris_eq_2_rhs(kr, omega, c_w, c_s, rho_w, rho_s):
    """Right-hand side of transcendental equation (Eq 2.186, Jensen et al. 2011, p.123)"""
    # Water layer
    kw = omega / c_w  # k in water
    kzm_w = np.sqrt(kw**2 - kr**2)  # kz in water

    # Sediment layer
    k_s = omega / c_s  # k in sediment
    # kz in sediment, kr > k_s for real solution to transcendental equation (complex solution correspond to leaky modes (Jensen et al. 2011, p.123))
    kzm_s = 1j * np.sqrt(kr**2 - k_s**2)

    # Right hand side of transcendental equation (Eq 2.186, Jensen et al. 2011, p.123)
    rhs = -(1j * rho_s * kzm_w) / (rho_w * kzm_s)
    # rhs is real for kr > k_s
    rhs = np.real(rhs)

    return rhs


def pekeris_kr_m(f, c_w, c_s, rho_w, rho_s, d, debug=False):
    """Derive horizontal wavenumbers of the propagating modes in a Pekeris waveguide by
    resolution of the transcendental equation (Eq 5.81, Jensen et al. 2011, p.355, Eq. 27, Pekeris 1948).
    """

    f = np.atleast_1d(f)

    # Implementation with root_scalar (25/08/2026), more robust
    t0 = time.time()

    n_search = int(1e5)
    krm = []
    for i_f, ff in enumerate(f):
        omega = 2 * np.pi * ff

        # Bornes (Cf eq. 2.187) pour avoir des solutions réelles
        kr_min = omega / c_s * (1 + 1e-15)  # k in sediment
        kr_max = omega / c_w * (1 - 1e-15)  # k in water

        if omega == 0:  # No mode for f = 0
            krm.append([])
            continue

        kr = np.linspace(kr_min, kr_max, n_search)  # (n_search)

        def func_pek(kr):
            return pekeris_eq_2_lhs(kr, omega, c_w, d) - pekeris_eq_2_rhs(
                kr, omega, c_w, c_s, rho_w, rho_s
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

    max_nb_modes = np.max(pekeris_n_modes(f, c_w, c_s, d))

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
        kr_min = omega / c_s * (1 + 1e-10)  # k in sediment
        kr_max = omega / c_w * (1 - 1e-10)  # k in water
        # WARNING : high number of points increases precision but can lead to memory overflow in case we compute kr for many frequencies at the same time
        # n_search = 2 * 1e4 looks like a good compromise
        n_search = int(100 * 1e4)
        kr = np.linspace(kr_min, kr_max, int(2 * 1e4))  # (n_search, nf)

        # Find roots (by equating lhs and rhs and then looking for sign changes in diff)
        lhs = pekeris_eq_2_lhs(kr, omega, c_w, d)  # (n_search, nf)
        rhs = pekeris_eq_2_rhs(kr, omega, c_w, c_s, rho_w, rho_s)  # (n_search, nf)
        diff = lhs - rhs
        diff_shift = np.roll(diff, 1, axis=0)
        idx_roots = (diff < 0) & (diff_shift > 0)

        # Multi frequency (19/06/2026)
        krm = [
            kr[idx_roots[:, i_f], i_f] for i_f in range(f.size)
        ]  # Get krm for each frequency

        max_nb_modes = np.max(pekeris_n_modes(f, c_w, c_s, d))

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


def pekeris_n_modes(f, c_w, c_s, d):
    """Calculate the number of propagating modes in a Pekeris waveguide."""
    # Equivalent to Eq. 2.191, Jensen p. 125 or Eq. 8.89, Jensen p.637
    n_max = np.int32(np.floor(2 * d / c_w * np.sqrt(1 - (c_w / c_s) ** 2) * f + 1 / 2))
    return n_max


def pekeris_cutoff_frequency(m, c_w, c_s, d):
    """Calculate the cut-off frequency of mode m in a Pekeris waveguide."""
    # Eq. 2.191, Jensen p. 125 or Eq. 8.89, Jensen p.637
    f_c = ((m - 1 / 2) * c_w) / (2 * d * np.sqrt(1 - (c_w / c_s) ** 2))
    return f_c


def pekeris_ir_duration(f, c_w, c_s, rho_w, rho_s, r, d):
    """Derive signal dispersion in a Pekeris waveguide. This can be used to estimate the duration of the impulse response.

    The duration is estimated as the time difference between the fastest arrival, defined by the maximum sound celerity in the
    waveguide, and the slowest arrival given by the minimum group speed of the fastest and slowest modes at range r.

    """
    # Eq. 8.16, Jensen p. 616

    # Horizontal wavenumbers
    krm = pekeris_kr_m(f, c_w, c_s, rho_w, rho_s, d)
    # Group speeds
    cgm = pekeris_cg_m(f, krm, c_w, c_s, rho_w, rho_s, d)

    # Minimum group speed
    cg_min = np.nanmin(cgm)

    # # Maximum phase speed
    # # Phase speeds
    # omega = 2 * np.pi * f
    # cpm = cp_m(krm=krm, omega=omega[:, np.newaxis])
    # cp_max = np.nanmax(cmp)

    # Duration of the impulse response
    cmax = max(c_w, c_s)
    T_ir = r / cg_min - r / cmax

    return T_ir


def pekeris_kzm_ws(f, c_w, c_s, krm):

    f = np.atleast_1d(f)
    omega = 2 * np.pi * f

    # Water layer
    k_w = omega / c_w
    k_w = k_w[:, np.newaxis]  # (nf, 1)
    kzm_w = np.sqrt(k_w**2 - krm**2)

    # Sediment layer
    k_s = omega / c_s
    k_s = k_s[:, np.newaxis]  # (nf, 1)

    idx_krm_lt_k_s = krm < k_s
    kzm_s = np.where(
        idx_krm_lt_k_s, np.sqrt(k_s**2 - krm**2), 1j * np.sqrt(krm**2 - k_s**2)
    )

    return kzm_w, kzm_s


def pekeris_cg_m(f, krm, c_w, c_s, rho_w, rho_s, d):

    # Add an extra frequency point to compute last group speed
    f = np.atleast_1d(f)
    default_df = 0.01
    df = (f[-1] - f[-2]) if f.size > 1 else default_df
    freq = np.append(f, f[-1] + df)
    fp1 = freq[-1]

    # Compute krm for only one more frequency point to reduce computing time
    krm_fp1 = pekeris_kr_m(fp1, c_w, c_s, rho_w, rho_s, d)

    # Add point
    krm = np.concatenate([krm, krm_fp1], axis=0)

    # Group speeds (approximation) : forward difference scheme -> u_m = d(omega)/d(kr) = delta_omega / (krm(omega+delta_omega) - krm(omega))
    d_omega = 2 * np.pi * np.diff(freq)
    d_omega = d_omega[:, np.newaxis]
    d_krm = np.diff(krm, axis=0)
    cgm = d_omega / d_krm

    # TODO check where it comes from ?
    # rho_ratio = rho_s / rho_s
    # a = rho_ratio / (kzm_s**2 + rho_ratio**2 * kzm_w**2)
    # cgm = (
    #     krm
    #     / omega[:, np.newaxis]
    #     * c_w**2
    #     * c_s**2
    #     * (kzm_s * d + a * (kzm_w**2 + kzm_s**2))
    #     / (c_s**2 * (kzm_s * d + a * kzm_s**2) + c_w**2 * a * kzm_w**2)
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


def pekeris_green_fct(f, c_w, c_s, rho_w, rho_s, d, z_s, z_r, r):
    """Calculate green's functions at depth z in a Pekeris waveguide."""

    # Horizontal wavenumbers
    krm = pekeris_kr_m(f, c_w, c_s, rho_w, rho_s, d)  # (n_f, n_modes)

    # Get vertical wavenumber in water and sediment
    kzm_w, kzm_s = pekeris_kzm_ws(f, c_w, c_s, krm)  # (n_f, n_modes)

    # Get modal function at z_s and z_r
    phim_zs = pekeris_modal_fct(z_s, f, c_w, c_s, d, krm)  # (nf, n_modes, 1)
    phim_zr = pekeris_modal_fct(z_r, f, c_w, c_s, d, krm)  # (nf, n_modes, 1)

    # Remove extra dim
    phim_zs = np.squeeze(phim_zs)
    phim_zr = np.squeeze(phim_zr)

    # Based on Pekeris 1948 formulation
    omega = 2 * np.pi * f
    omega_2d = omega[np.newaxis]
    # r_2d = r[np.newaxis]

    # Common factor (does not depend on f)
    Q = omega * (rho_w * np.pi / d) * np.sqrt(8 / r) * np.exp(+1j * np.pi / 4)

    # Pekeris 1948, Eq. 25
    def F(x):
        return x / (
            x
            - np.sin(x) * np.cos(x)
            - (rho_w / rho_s) ** 2 * np.sin(x) ** 2 * np.tan(x)
        )

    # Modal amplitude
    Am = F(kzm_w * d) * phim_zs * phim_zr
    # Modal phase
    gm = Am * np.exp(-1j * krm * r) / np.sqrt(krm)
    # Green's function
    gf = Q * np.nansum(gm, axis=1)  # p = Q sum Am e^(-i krm r) / sqrt(krm)

    return gf


def pekeris_modal_fct(z, f, c_w, c_s, d, krm):
    """Calculate modal functions at depth z in a Pekeris waveguide.

    Derivation according to Pekeris 1948 (eq. 38).
    """

    # Get vertical wavenumber in water and sediment
    kzm_w, kzm_s = pekeris_kzm_ws(f, c_w, c_s, krm)  # (n_f, n_modes)

    # Cast to 3D (nf, nmodes, 1)
    kzm_w_3d = kzm_w[..., np.newaxis]
    kzm_s_3d = kzm_s[..., np.newaxis]

    # Ensure z is a numpy array
    z = np.atleast_1d(z)
    z_w = z[z <= d]  # Depths in water
    z_s = z[z > d]  # Depths in sediment

    # Cast to 2D (1, nz)
    z_w_2d = z_w[np.newaxis]
    z_s_2d = z_s[np.newaxis]

    # Define modal fct according to eq. 48
    # z <= D -> mode in water column
    phi_m_w = np.sin(kzm_w_3d * z_w_2d)  # (nf, nmodes, nz)
    # z > D -> mode in sediment
    phi_m_s = np.sin(kzm_w_3d * d) * np.exp(
        1j * kzm_s_3d * (z_s_2d - d)
    )  # (nf, nmodes, nz)

    # phi_m = np.hstack((phi_m_w, phi_m_s))
    phi_m = np.concatenate([phi_m_w, phi_m_s], axis=-1)  # (nf, nmodes, nz)

    return phi_m


def impulse_response(gf, f, fs, output_mult_fact=4):

    # Number of freq samples
    df = f[1] - f[0]
    nt = int(fs / df)
    # Number of temporal samples
    nfft_inv = (
        nt * output_mult_fact
    )  # To improve temporal resolution for visual inspection

    # # Correct for propagation delay
    # tau_rcv = 30 * 1e3 / 1500
    # delay_rcv = np.exp(1j * 2 * np.pi * tau_rcv * f)
    # gf *= delay_rcv

    # Impulse response in time domain
    gt = np.fft.irfft(gf, n=nfft_inv)

    # Time vector
    T = nt * 1 / fs
    dt = T / nfft_inv
    t = np.arange(0, T, dt)

    return t, gt


def plot_modal_function(modal_fcts, z, depth=None, n_modes=None):

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
    axs[0].set_ylabel("Depth [m]")

    for i, m in enumerate(modes):
        axs[i].plot(np.real(modal_fcts[i]), z, label=f"Real", linestyle="-", color="k")
        axs[i].plot(np.imag(modal_fcts[i]), z, label=f"Im", linestyle="--", color="r")
        axs[i].set_xlabel(f"Mode {m}")
        axs[i].set_xlim([-1.1 * max_amp, 1.1 * max_amp])
        axs[i].grid()

        # Add depth line
        if depth is not None:
            axs[i].axhline(depth, linestyle="--", color="k")


def plot_green_fct(f, gf, z_s, z_r, r):

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Amplitude
    axs[0].plot(f, np.abs(gf))
    axs[0].set_ylabel(r"$\lvert H(f) \rvert$")

    # Phase
    axs[1].plot(f, np.angle(gf))
    axs[1].set_ylabel("arg(H(f)) [rad]")

    fig.supxlabel("Fréquence [Hz]")
    fig.suptitle(rf"z_s = {z_s:.1f} m, z_r = {z_r:.1f} m, r = {r:.1f} m")


def plot_impulse_response(t, gt, z_s, z_r, r):

    plt.figure()
    plt.plot(t, gt)
    plt.xlabel("Temps [s]")
    plt.ylabel("h(t)")
    plt.title(rf"z_s = {z_s:.1f} m, z_r = {z_r:.1f} m, r = {r:.1f} m")


def plot_impulse_response_and_stft(
    t, gt, z_s, z_r, r, fmax, nfft=1024, nperseg=51, noverlap=50
):

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Signal
    axs[0].plot(t, gt)
    axs[0].set_ylabel("h(t)")
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
        vmin=np.percentile(Sxx_db, 75),
        vmax=np.percentile(Sxx_db, 99.5),
    )
    # plt.colorbar(im, ax=axs[1])
    axs[1].set_ylim(0, fmax)

    axs[1].set_ylabel("Fréquence [Hz]")

    fig.supxlabel("Temps [s]")
    fig.suptitle(rf"z_s = {z_s:.1f} m, z_r = {z_r:.1f} m, r = {r:.1f} m")


def plot_impulse_response_stft_dispersion_curve(
    t, gt, z_s, z_r, r, fmax, f, cgm, cw, nfft=1024, nperseg=51, noverlap=50
):

    # Theoretical dispersion curves
    # tm_theo = (
    #     r / cgm - r / cw
    # )  # range over group_speed minus correction for time origin

    tm_theo = r / cgm

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
        vmin=np.percentile(Sxx_db, 75),
        vmax=np.percentile(Sxx_db, 99.5),
    )
    # plt.colorbar(im, ax=plt.gca())

    # Add dispersion curves
    for i_m in range(tm_theo.shape[1]):
        plt.plot(tm_theo[:, i_m], f, color="k")

    plt.ylim(0, fmax)
    plt.ylabel("Fréquence [Hz]")
    plt.xlabel("Temps [s]")
    plt.title(rf"z_s = {z_s:.1f} m, z_r = {z_r:.1f} m, r = {r:.1f} m")


# ======================================================================================================================
# Tests
# ======================================================================================================================
def test_pekeris_env():
    # Waveguide parameters
    rho_w = 1.0 * 1e3  # density in water (kg/m^3)
    c_w = 1500  # sound celerity in water (m/s)
    c_s = 1600  # sound celerity in fluid sediment (m/s)
    rho_s = 1.5 * 1e3  # density in fluid sediment (kg/m^3)
    d = 100  # waveguide depth (m)

    return c_w, c_s, rho_w, rho_s, d


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
    c_w, c_s, rho_w, rho_s, d = test_pekeris_env()

    for f_i in f:
        krm = pekeris_kr_m(f_i, c_w, c_s, rho_w, rho_s, d, debug=True)
        print(f"f = {f_i} Hz\n krm = {krm.flatten()}")


def test_cp_pekeris():
    """Compute and plot phase speed cp."""

    c_w, c_s, rho_w, rho_s, d = test_pekeris_env()
    _, _, _, _, freq = test_pekeris_sig()

    # Direct array derivation
    omega = 2 * np.pi * freq
    # Horizontal wavenumbers
    krm = pekeris_kr_m(freq, c_w, c_s, rho_w, rho_s, d)
    # Phase speeds
    cpm = cp_m(krm=krm, omega=omega[:, np.newaxis])

    plot_pekeris_cp_f(f=freq, cpm=cpm)
    plt.show()


def test_cg_pekeris():
    """Compute and plot group speed cg."""

    c_w, c_s, rho_w, rho_s, d = test_pekeris_env()
    _, _, _, _, freq = test_pekeris_sig()

    # Horizontal wavenumbers
    krm = pekeris_kr_m(freq, c_w, c_s, rho_w, rho_s, d)
    # Group speeds
    cgm = pekeris_cg_m(freq, krm, c_w, c_s, rho_w, rho_s, d)

    plot_pekeris_cg_f(f=freq, cgm=cgm)
    plt.show()


def test_cp_cg_pekeris():
    """
    Compute and plot phase speed cp and group speed cg.
    """

    c_w, c_s, rho_w, rho_s, d = test_pekeris_env()
    _, _, _, _, freq = test_pekeris_sig()

    # Angular freq
    omega = 2 * np.pi * freq
    # Horizontal wavenumbers
    krm = pekeris_kr_m(freq, c_w, c_s, rho_w, rho_s, d)
    # Phase speeds
    cpm = cp_m(krm=krm, omega=omega[:, np.newaxis])
    # Group speeds
    cgm = pekeris_cg_m(freq, krm, c_w, c_s, rho_w, rho_s, d)

    plot_pekeris_cp_cg_f(f=freq, cpm=cpm, cgm=cgm)
    plt.show()


def test_modal_fct_pekeris():

    c_w, c_s, rho_w, rho_s, d = test_pekeris_env()
    _, _, _, _, freq = test_pekeris_sig()

    # Horizontal wavenumbers
    krm = pekeris_kr_m(freq, c_w, c_s, rho_w, rho_s, d)

    dz = 0.01
    z = np.arange(dz, d + 30, dz)
    phim_arr = pekeris_modal_fct(z, freq, c_w, c_s, d, krm)

    f0 = 100
    idx_f0 = np.argmin(np.abs(freq - f0))
    phim_f0 = phim_arr[idx_f0, ...]
    M_f0 = np.max(pekeris_n_modes(f0, c_w, c_s, d))

    plot_modal_function(modal_fcts=phim_f0, z=z, depth=d, n_modes=M_f0)
    plt.show()


def run_full_diag(sig_param, env_param, src_rcv_param):

    # Unpack params
    # Signal
    freq = sig_param.get("freq", None)
    fs = sig_param.get("fs", None)
    fmax = sig_param.get("fmax", None)
    # Environment
    c_w = env_param.get("c_w", None)
    c_s = env_param.get("c_s", None)
    rho_w = env_param.get("rho_w", None)
    rho_s = env_param.get("rho_s", None)
    d = env_param.get("depth", None)

    # Source / receiver
    z_s = src_rcv_param.get("z_s", 5)
    z = src_rcv_param.get("z_rcv", 20)
    r = src_rcv_param.get("r_rcv", 1e4)

    ###### Modal group and phase speed ######
    # Angular freq
    omega = 2 * np.pi * freq
    # Horizontal wavenumbers
    krm = pekeris_kr_m(freq, c_w, c_s, rho_w, rho_s, d)
    # Phase speeds
    cpm = cp_m(krm=krm, omega=omega[:, np.newaxis])
    # Group speeds
    cgm = pekeris_cg_m(freq, krm, c_w, c_s, rho_w, rho_s, d)
    # Plot
    plot_pekeris_cp_cg_f(f=freq, cpm=cpm, cgm=cgm)

    ###### Modal function ######
    dz = 0.1
    zphi = np.arange(dz, d + 30, dz)
    phim_arr = pekeris_modal_fct(zphi, freq, c_w, c_s, d, krm)

    # f0 = fmax / 2
    f0 = fmax
    idx_f0 = np.argmin(np.abs(freq - f0))
    phim_f0 = phim_arr[idx_f0, ...]

    M_f0 = np.max(pekeris_n_modes(f0, c_w, c_s, d))
    plot_modal_function(modal_fcts=phim_f0, z=zphi, depth=d, n_modes=M_f0)

    ###### Green's function at z, r ######
    # z_s = 5
    # z, r = 20, 10000
    gf = pekeris_green_fct(freq, c_w, c_s, rho_w, rho_s, d, z_s, z, r)
    plot_green_fct(freq, gf, z_s=z_s, z_r=z, r=r)

    # Impulse response
    output_mult_fact = 4
    t, gt = impulse_response(gf, freq, fs, output_mult_fact=output_mult_fact)
    plot_impulse_response(t, gt, z_s, z, r)

    # stft
    # Réglage identique au code du tuto de J.Bonnel
    nfft = 1024
    nperseg = 51 * output_mult_fact
    noverlap = nperseg - 1
    plot_impulse_response_and_stft(
        t, gt, z_s, z, r, fmax, nfft=nfft, nperseg=nperseg, noverlap=noverlap
    )

    # Stft and theoretical dispersion curves
    plot_impulse_response_stft_dispersion_curve(
        t,
        gt,
        z_s,
        z,
        r,
        fmax,
        freq,
        cgm,
        c_w,
        nfft=nfft,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    plt.show()


def test_pekeris_JBonnel():
    "Test Pekeris functions by comparison with results from the Matlab tuto code by JBonnel."

    # Waveguide parameters
    rho_w = 1.0 * 1e3  # density in water (kg/m^3)
    c_w = 1500  # sound celerity in water (m/s)
    c_s = 1600  # sound celerity in fluid sediment (m/s)
    rho_s = 1.5 * 1e3  # density in fluid sediment (kg/m^3)
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
    env_param = {"c_w": c_w, "c_s": c_s, "rho_w": rho_w, "rho_s": rho_s, "depth": d}
    run_full_diag(sig_param=sig_param, env_param=env_param, src_rcv_param=src_rcv_param)
    # run_full_diag(freq, c_w, c_s, rho_w, rho_s, d, fs, fmax)


def test_pekeris_Jensen_p119():
    "Test Pekeris functions by comparison with results in Jensen. Pekeris definition p.119."

    # Waveguide parameters
    rho_w = 1.0 * 1e3  # density in water (kg/m^3)
    c_w = 1500  # sound celerity in water (m/s)
    rho_s = 1.8 * 1e3  # density in fluid sediment (kg/m^3)
    c_s = 1800  # sound celerity in fluid sediment (m/s)
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
    env_param = {"c_w": c_w, "c_s": c_s, "rho_w": rho_w, "rho_s": rho_s, "depth": d}
    run_full_diag(sig_param=sig_param, env_param=env_param, src_rcv_param=src_rcv_param)
    # run_full_diag(freq, c_w, c_s, rho_w, rho_s, d, fs, fmax)


def test_pekeris_Jensen_p350():
    "Test Pekeris functions by comparison with results in Jensen. Pekeris definition p.350."

    # Waveguide parameters
    rho_w = 1.0 * 1e3  # density in water (kg/m^3)
    c_w = 1500  # sound celerity in water (m/s)
    rho_s = 1.0 * 1e3  # density in fluid sediment (kg/m^3)
    c_s = 1550  # sound celerity in fluid sediment (m/s)
    d = 100  # waveguide depth (m)

    # Signal properties
    fmax = 75  # Max frequency (Hz)
    T = 22  # Signal duration to generate (s)
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
    env_param = {"c_w": c_w, "c_s": c_s, "rho_w": rho_w, "rho_s": rho_s, "depth": d}
    run_full_diag(sig_param=sig_param, env_param=env_param, src_rcv_param=src_rcv_param)


def test_pekeris_Jensen_p636():
    "Test Pekeris functions by comparison with results in Jensen. Pekeris definition p.636."

    # Waveguide parameters
    rho_w = 1.0 * 1e3  # density in water (kg/m^3)
    c_w = 1500  # sound celerity in water (m/s)
    rho_s = 1.5 * 1e3  # density in fluid sediment (kg/m^3)
    c_s = 1600  # sound celerity in fluid sediment (m/s)
    d = 100  # waveguide depth (m)

    # Signal properties
    fmax = 75  # Max frequency (Hz)
    T = 22  # Signal duration to generate (s)
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
    env_param = {"c_w": c_w, "c_s": c_s, "rho_w": rho_w, "rho_s": rho_s, "depth": d}
    run_full_diag(sig_param=sig_param, env_param=env_param, src_rcv_param=src_rcv_param)


def test_pekeris_short_ir():
    "Test Pekeris functions for a waveguide with short IR."

    # Waveguide parameters
    rho_w = 1.0 * 1e3  # density in water (kg/m^3)
    c_w = 1500  # sound celerity in water (m/s)
    rho_s = 1.5 * 1e3  # density in fluid sediment (kg/m^3)
    c_s = 1600  # sound celerity in fluid sediment (m/s)
    d = 100  # waveguide depth (m)

    # Signal properties
    fmax = 100  # Max frequency (Hz)
    T = 15  # Signal duration to generate (s)
    fs = 2 * fmax  # Sampling frequency (Hz) = Nyquist
    ts = 1 / fs  # sampling interval (s)
    nt = T * fs  # Number of samples
    freq = np.fft.rfftfreq(n=nt, d=ts)  # Frequency vector

    # Source / receiver properties
    z_s = 5
    z_rcv = d - 0.5
    r_rcv = 30 * 1e3

    # Run test
    src_rcv_param = {"z_s": z_s, "z_rcv": z_rcv, "r_rcv": r_rcv}
    sig_param = {"freq": freq, "fs": fs, "fmax": fmax}
    env_param = {"c_w": c_w, "c_s": c_s, "rho_w": rho_w, "rho_s": rho_s, "depth": d}
    run_full_diag(sig_param=sig_param, env_param=env_param, src_rcv_param=src_rcv_param)
    # run_full_diag(freq, c_w, c_s, rho_w, rho_s, d, fs, fmax)


if __name__ == "__main__":

    # test_kr_pekeris()     # OK
    # test_cp_pekeris()     # OK
    # test_cg_pekeris()     # Ok
    # test_cp_cg_pekeris()  # OK
    # test_modal_fct_pekeris() # OK

    # test_pekeris_JBonnel()  # OK
    # test_pekeris_Jensen_p119()  # OK
    # test_pekeris_Jensen_p636()  # OK
    test_pekeris_Jensen_p350()

    # test_pekeris_short_ir()  # OK

# ======================================================================================================================
# Left overs
# ======================================================================================================================

# def pekeris_modes(f, c_w, c_s, rho_w, rho_s, d):
#     """Calculate the horizontal wavenumbers and phase speeds of the propagating modes in a Pekeris waveguide."""

#     # Handle array of frequencies
#     freq = np.atleast_1d(f)
#     # Add an extra frequency point to compute last group speed
#     # default_df = 0.01
#     # df = (freq[-1] - freq[-2]) if freq.size > 1 else default_df
#     # freq = np.append(freq, freq[-1] + df)

#     # nf = freq.size
#     # n_modes = np.max(
#     #     pekeris_n_modes(freq, c_w, c_s, d)
#     # )  # Max number of propagating modes

#     # # Init arrays (Set to nan to avoid division by zero in cp_m)
#     # krm_arr = np.empty((nf, n_modes)) * np.nan
#     # kzm_arr = np.empty((nf, n_modes)) * np.nan
#     # cpm_arr = np.empty((nf, n_modes)) * np.nan
#     # cgm_arr = np.empty((nf, n_modes)) * np.nan
#     # thetam_arr = np.empty((nf, n_modes)) * np.nan

#     # Direct array derivation
#     omega = 2 * np.pi * freq
#     k = omega / c_w
#     # Horizontal wavenumbers
#     krm_arr = pekeris_kr_m(freq, c_w, c_s, rho_w, rho_s, d)
#     # Vertical wavenumbers in water
#     kzm_arr = np.sqrt(k[:, np.newaxis] ** 2 - krm_arr**2).astype(np.float32)
#     # Phase speeds
#     cpm_arr = cp_m(krm=krm_arr, omega=omega[:, np.newaxis])
#     # Group speeds
#     cgm_arr = pekeris_cg_m(f, c_w, c_s, rho_w, rho_s, d)
#     # Mode angle
#     thetam_arr = np.arctan(kzm_arr / krm_arr).astype(np.float32) * 180 / np.pi

#     # for i, f in enumerate(freq):
#     #     omega_i = 2 * np.pi * f

#     #     # Horizontal wavenumbers
#     #     krm_i = pekeris_kr_m(f, c_w, c_s, rho_w, rho_s, d)
#     #     krm_arr[i, : len(krm_i)] = krm_i  # Fill array with actual values

#     #     # Vertical wavenumbers in water
#     #     kzm_i = np.sqrt((omega_i / c_w) ** 2 - krm_i**2)
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


# def pekeris_cg_m(f, c_w, c_s, rho_w, rho_s, d):

#     # Add an extra frequency point to compute last group speed
#     f = np.atleast_1d(f)
#     default_df = 0.01
#     df = (f[-1] - f[-2]) if f.size > 1 else default_df
#     freq = np.append(f, f[-1] + df)

#     # Get horizontal wavenumbers
#     krm = pekeris_kr_m(freq, c_w, c_s, rho_w, rho_s, d)

#     # Group speeds (approximation) : forward difference scheme -> u_m = d(omega)/d(kr) = delta_omega / (krm(omega+delta_omega) - krm(omega))
#     d_omega = 2 * np.pi * np.diff(freq)
#     d_omega = d_omega[:, np.newaxis]
#     d_krm = np.diff(krm, axis=0)
#     cgm = d_omega / d_krm

#     # TODO check where it comes from ?
#     # rho_ratio = rho_s / rho_s
#     # a = rho_ratio / (kzm_s**2 + rho_ratio**2 * kzm_w**2)
#     # cgm = (
#     #     krm
#     #     / omega[:, np.newaxis]
#     #     * c_w**2
#     #     * c_s**2
#     #     * (kzm_s * d + a * (kzm_w**2 + kzm_s**2))
#     #     / (c_s**2 * (kzm_s * d + a * kzm_s**2) + c_w**2 * a * kzm_w**2)
#     # )

#     return cgm.astype(np.float32)


# def pekeris_diag_test(f, c_w, c_s, rho_w, rho_s, d):
#     """Run test to ensure everything is ok."""
#     # Derive all modal properties
#     krm, kzm, cpm, cgm, thetam = pekeris_modes(f, c_w, c_s, rho_w, rho_s, d)
#     # Number of modes for each frequency
#     n_modes = pekeris_n_modes(f, c_w, c_s, d)

#     # plot_pekeris_cg_f(f, cgm, n_modes)
#     # plot_pekeris_cp_f(f, cpm, n_modes)
#     plot_pekeris_cp_cg_f(f, cpm, cgm, n_modes)

#     # plt.show()


# def pekeris_green_fct(f, c_w, c_s, rho_w, rho_s, d, z_s, z_r, r):
#     """Calculate green's functions at depth z in a Pekeris waveguide."""

#     # Derive all modal properties
#     krm, kzm, cpm, cgm, thetam = pekeris_modes(f, c_w, c_s, rho_w, rho_s, d)

#     # Get vertical wavenumber in water and sediment
#     kzm_w, kzm_s = pekeris_kzm_ws(f, c_w, c_s, krm)  # (n_f, n_modes)

#     # Based on Pekeris 1948 formulation
#     omega = 2 * np.pi * f
#     omega_2d = omega[np.newaxis]
#     # r_2d = r[np.newaxis]

#     # # Common factor
#     # Q = omega * (rho_w * np.pi / d) * np.sqrt(8 / r) * np.exp(+1j * np.pi / 4)

#     # def F(x):
#     #     """Equation 25 and A73"""
#     #     return x / (
#     #         x
#     #         - np.sin(x) * np.cos(x)
#     #         - (rho_w / rho_s) ** 2 * np.sin(x) ** 2 * np.tan(x)
#     #     )

#     # # # Solution in water layer
#     # # if z_r < d:
#     # #     A_m = (
#     # #         F(kzm_w * d) * np.sin(kzm_w * z_s) * np.sin(kzm_w * z_r)
#     # #     )  # Modal amplitude
#     # # # Solutution in sediment
#     # # else:
#     # #     A_m = (
#     # #         F(kzm_w * d)
#     # #         * np.sin(kzm_w * z_s)
#     # #         * np.sin(kzm_w * d)
#     # #         * np.exp(1j * kzm_s * (z_r - d))
#     # #     )  # Modal amplitude

#     # # A_m *= np.exp(-1j * krm * r) / np.sqrt(krm)  # Propagative term

#     # # gfm = Q * np.sum(A_m)

#     # Adaptation from Bonnel et al 2020 (MATLAB provided tool box) # TODO : update and verify according to Jensen2011
#     A_2 = (
#         2
#         * rho_w
#         * kzm_w
#         * kzm_s
#         / (
#             kzm_w * kzm_s * d
#             - 1 / 2 * kzm_s * np.sin(2 * kzm_w * d)
#             + rho_w / rho_s * kzm_w * np.sin(kzm_w * d) ** 2
#         )
#     )
#     # Global multiplicative factor
#     A_2 = 1 / 4 * A_2 * 1j * np.exp(1j * np.pi / 4) / rho_w

#     gfm = (
#         A_2
#         * np.sin(kzm_w * z_s)
#         * np.exp(-1j * krm * r)
#         / np.sqrt(krm * r)
#         * np.sin(kzm_w * z_r)
#     )

#     # # Replace np.nan by 0
#     gfm[np.isnan(gfm)] = 0
#     g_f = np.sum(gfm, axis=1)

#     return g_f
