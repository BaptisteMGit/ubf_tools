#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_pekeris_analytic.py
@Time    :   2026/09/01 12:46:13
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
from propa.kraken_toolbox.read_modes import readmodes

from source.normal_modes import *

from scipy.optimize import root_scalar
from publication.publication_figure import PubFigure, color

PubFigure()


HERE = os.path.dirname(os.path.abspath(__file__))
FILENAME = "pekeris_normal_modes_validation_against_kraken"


def compute_kraken_solution(sig_param, env_param, src_rcv_param):
    # Unpack params
    # Signal
    freq = sig_param.get("freq", None)
    fs = sig_param.get("fs", None)
    fmax = sig_param.get("fmax", None)
    # Environment
    d = env_param.get("depth", None)
    c1 = env_param.get("c1", None)
    c2 = env_param.get("c2", None)
    rho1 = env_param.get("rho1", None)
    rho2 = env_param.get("rho2", None)
    attn2 = env_param.get("attn2", None)

    # Source / receiver
    z_s = src_rcv_param.get("z_s", 5)
    z = src_rcv_param.get("z_rcv", 20)
    r_grid = src_rcv_param.get("r_rcv", np.array([1e4]))
    r0 = src_rcv_param.get("r0", 1e4)

    # Convert r_grid to km
    r_grid = r_grid * 1e-3

    # Keeps only frequencies above mode 1 cut-off
    kraken_freq = freq[freq > pekeris_cutoff_frequency(m=1, c1=c1, c2=c2, d=d)]

    # Use only propative modes
    nb_modes = pekeris_n_modes(f=fmax, c1=c1, c2=c2, d=d)

    # ----------------------------------------------------------------------
    # 1. Environment: Pekeris waveguide
    # ----------------------------------------------------------------------
    medium = KrakenMedium(
        ssp_interpolation_method="C_linear",
        z_ssp=[0.0, d],
        c_p=[c1, c1],  # isovelocity water column
        rho=rho1,
    )

    bottom_hs = KrakenBottomHalfspace(
        halfspace_properties={
            "z": d,
            "c_p": c2,
            "c_s": 0.0,  # fluid sediment: no shear waves
            "rho": rho2,
            "a_p": attn2,  # dB/wavelength
            "a_s": 0.0,  # fluid sediment: no shear waves
        },
        add_sediment_buffer_layer=False,  # direct half-space -> classic Pekeris model
        # fmin=kraken_freq.min(),
        # alpha_wavelength=10,
        # add_sediment_buffer_layer=True,
    )

    field = KrakenField(
        phase_speed_limits=[0, c2 + 0.1],
        src_depth=z_s,
        n_rcv_z=201,
        rcv_z_min=0.0,
        rcv_z_max=d + max(50, 0.1 * d),
        rcv_r_max=r_grid.max(),
    )

    env = KrakenEnv(
        title="Case 1 - Pekeris waveguide (range-independent, single frequency)",
        env_root=HERE,
        env_filename=FILENAME,
        freq=kraken_freq,
        kraken_medium=medium,
        kraken_bottom_hs=bottom_hs,
        kraken_field=field,
        nmedia=None,  # derived automatically -> 1 (no buffer layer)
        # nmedia=2,
    )
    # assert env.nmedia == 1
    env.write_env()
    print(f"Wrote {env.env_fpath} (nmedia={env.nmedia})")

    flp = KrakenFlp(
        env=env,
        src_type="point_source",
        mode_theory="adiabatic",  # irrelevant for a range-independent run, kept simple
        mode_addition="coherent",
        nb_modes=nb_modes,
        src_depth=z_s,
        n_rcv_z=1,
        rcv_z_min=z,
        rcv_z_max=z,
        n_rcv_r=r_grid.size,
        rcv_r_min=r_grid.min(),
        rcv_r_max=r_grid.max(),
    )
    flp.write_flp()
    print(f"Wrote {flp.flp_fpath}")

    # ----------------------------------------------------------------------
    # 2. Run KRAKEN + FIELD (requires real binaries -- see KrakenManager /
    #    propa.kraken_toolbox.params.KRAKEN_BIN_DIRECTORY).
    # ----------------------------------------------------------------------
    manager = KrakenManager(verbose=True)
    pressure_field, field_pos = manager.runkraken(
        env=env, flp=flp, frequencies=env.freq
    )
    print("KRAKEN/FIELD run completed.")

    # Squeeze
    pressure_field = pressure_field.squeeze()

    # Get green's function
    c0 = 1500
    k0 = 2 * np.pi * kraken_freq / c0
    norm_factor = np.exp(1j * k0) / (4 * np.pi)
    # norm_factor[:] = 1
    gf = norm_factor[:, np.newaxis] * pressure_field

    return kraken_freq, gf, field_pos


def plot_mode_shapes(mod_fpath, freq, n_modes=4):
    """Plot the real part of the first 'n_modes' mode shapes as a
    function of depth."""
    Modes = readmodes(mod_fpath, freq=freq)
    n_modes = min(n_modes, Modes["M"])

    fig, ax = plt.subplots(figsize=(6, 8))
    for i in range(n_modes):
        ax.plot(Modes["phi"][:, i].real, Modes["z"], label=f"Mode {i + 1}")
        # ax.plot(
        #     Modes["phi"][:, i].imag, Modes["z"], label=f"Mode {i + 1}", linestyle="--"
        # )

    ax.invert_yaxis()
    ax.set_xlabel("Mode amplitude (real part)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"Mode shapes at {freq} Hz  (Number of modes = {Modes['M']})")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


def compare_modal_fct(sig_param, env_param, fcomp=None):
    # Signal
    freq = sig_param.get("freq", None)
    fmax = sig_param.get("fmax", None)

    # Environment
    c1 = env_param.get("c1", None)
    c2 = env_param.get("c2", None)
    rho1 = env_param.get("rho1", None)
    rho2 = env_param.get("rho2", None)
    d = env_param.get("depth", None)

    mod_fpath = os.path.join(HERE, FILENAME + ".mod")

    # Horizontal wavenumbers
    krm = pekeris_kr_m(freq, c1, c2, rho1, rho2, d)

    ###### Modal function ######
    nz = 5000
    zphi = np.linspace(0, d + max(50, 0.1 * d), nz)
    phim_arr = pekeris_modal_fct(zphi, freq, c1, c2, d, krm)

    if fcomp is None:
        fcomp = [fmax / 2]

    for f0 in fcomp:

        # Number of modes to reprensent
        M_f0 = np.max(pekeris_n_modes(f0, c1, c2, d))
        M_f0 = min(M_f0, 10)
        # Get modal function at f0
        idx_f0 = np.argmin(np.abs(freq - f0))
        phim_f0 = phim_arr[idx_f0, ...]
        # Plot
        fig, axs = plot_modal_function(
            modal_fcts=phim_f0, z=zphi, depth=d, n_modes=M_f0
        )

        # plot_mode_shapes(mod_fpath=mod_fpath, freq=f0, n_modes=M_f0)

        # Add kraken modes
        Modes = readmodes(mod_fpath, freq=f0)
        # n_modes = min(n_modes, Modes["M"])

        for i in range(M_f0):
            axs[i].plot(
                Modes["phi"][:, i].real / np.max(Modes["phi"][:, i].real),
                # Modes["phi"][:, i].real,
                Modes["z"],
                color="b",
                linestyle="--",
                label="Kraken",
            )
            axs[i].legend()

    fig.savefig(os.path.join(HERE, "mode_shapes.png"))
    plt.close(fig)


def compare_green_fct(sig_param, env_param, src_rcv_param, kraken_gfr, kraken_f):
    # Signal
    freq = sig_param.get("freq", None)
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

    # Analytic green's function
    g_fr = pekeris_green_fct(freq, c1, c2, rho1, rho2, attn2, d, z_s, z, r_grid)
    idx_r0 = np.argmin(np.abs(r_grid - r0))
    gf = g_fr[:, idx_r0]

    # Normalize
    gf /= np.max(np.abs(gf))
    fig, axs = plot_green_fct(freq, gf, z_s=z_s, z_r=z, r=r0)

    kraken_gf = kraken_gfr[:, idx_r0]
    kraken_gf /= np.max(np.abs(kraken_gf))
    axs[0].plot(kraken_f, np.abs(kraken_gf), color="b", linestyle="--", label="Kraken")
    axs[1].plot(
        kraken_f, np.angle(kraken_gf), color="b", linestyle="--", label="Kraken"
    )
    axs[0].legend()

    fig.savefig(os.path.join(HERE, "green_fct.png"))
    plt.close(fig)


def compare_tl_fr(r_grid, freq, g_fr, kraken_f, kraken_gfr, fmin=25, fmax=75):
    # Plot TL(f, r)
    idx_f_tl = np.logical_and(freq > fmin, freq < fmax)
    f_tl = freq[idx_f_tl]
    g_fr_ftl = g_fr[idx_f_tl, ...]
    fig, ax = plot_tl_fr(
        g_fr_ftl,
        r_grid,
        f_tl,
        tl_vmin_percentile=1,
        tl_vmax_percentile=95,
    )

    fig.savefig(os.path.join(HERE, "tl_fr.png"))
    plt.close(fig)

    # kraken
    idx_f_tl = np.logical_and(kraken_f > fmin, kraken_f < fmax)
    kraken_f_tl = kraken_f[idx_f_tl]
    kraken_g_fr_ftl = kraken_gfr[idx_f_tl, ...]

    tl = -20 * np.log10(np.abs(kraken_g_fr_ftl))
    tlmax = np.percentile(tl, 1)
    tlmin = np.percentile(tl, 95)

    plt.figure()
    im = plt.pcolormesh(r_grid, kraken_f_tl, tl, vmin=tlmin, vmax=tlmax, cmap="jet")
    plt.xlabel("Distance r [m]")
    plt.ylabel("Fréquence [Hz]")
    plt.colorbar(im, label="TL [dB]")

    # fig, ax = plot_tl_fr(
    #     kraken_g_fr_ftl,
    #     r_grid,
    #     kraken_f_tl,
    #     tl_vmin_percentile=1,
    #     tl_vmax_percentile=95,
    # )
    fig = plt.gcf()
    fig.savefig(os.path.join(HERE, "tl_fr_kraken.png"))
    plt.close(fig)

    g_fr_ftl /= np.max(np.abs(g_fr_ftl))
    kraken_g_fr_ftl /= np.max(np.abs(kraken_g_fr_ftl))
    ratio = np.abs(g_fr_ftl / kraken_g_fr_ftl)
    diff_dB = 10 * np.log10(ratio)
    plt.figure()
    im = plt.pcolormesh(r_grid, kraken_f_tl, diff_dB, vmin=-5, vmax=5, cmap="bwr")
    plt.xlabel("Distance r [m]")
    plt.ylabel("Fréquence [Hz]")
    plt.colorbar(im, label=r"$\Delta$ Kraken - Analytic [dB]")

    fig = plt.gcf()
    fig.savefig(os.path.join(HERE, "tl_fr_kraken_diff.png"))
    plt.close(fig)


def compare_tl_r(r_grid, freq, g_fr, kraken_f, kraken_gfr, f0=100):
    # Plot TL(f=f0, r)
    idx_f0 = np.argmin(np.abs(freq - f0))
    g_r_ftl = g_fr[idx_f0, ...]
    fig, ax = plot_tl_r(
        g_r_ftl,
        r_grid,
        spherical_loss=True,
        cylindrical_loss=True,
    )
    fig.savefig(os.path.join(HERE, "tl_r.png"))
    plt.close(fig)

    # kraken
    idx_f0 = np.argmin(np.abs(kraken_f - f0))
    kraken_g_r_ftl = kraken_gfr[idx_f0, ...]
    tl = -20 * np.log10(np.abs(kraken_g_r_ftl))
    plt.figure()
    plt.plot(r_grid, tl, color="k")
    plt.xlabel("Distance r [m]")
    plt.ylabel("TL [dB]")
    plt.gca().invert_yaxis()

    icol = 0
    tl_spherical = 20 * np.log10(r_grid)
    plt.plot(r_grid, tl_spherical, color=color(icol), label=r"$20 \log_{10} (r)$")
    icol += 1

    tl_cylindrical = 10 * np.log10(r_grid)
    plt.plot(r_grid, tl_cylindrical, color=color(icol), label=r"$10 \log_{10} (r)$")
    icol += 1
    plt.legend()
    fig, ax = plt.gcf(), plt.gca()

    # fig, ax = plot_tl_r(
    #     kraken_g_r_ftl,
    #     r_grid,
    #     spherical_loss=True,
    #     cylindrical_loss=True,
    # )
    fig.savefig(os.path.join(HERE, "tl_r_kraken.png"))
    plt.close(fig)


def compare_krm(sig_param, env_param, fcomp):

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

    krm = pekeris_kr_m(freq, c1, c2, rho1, rho2, d)  # (n_f, n_modes)

    mod_fpath = os.path.join(HERE, FILENAME + ".mod")
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))

    for f0 in fcomp:

        idx_f0 = np.argmin(np.abs(freq - f0))
        krm_f0 = krm[idx_f0, :]
        axs[0].scatter(
            np.arange(1, krm_f0.size + 1),
            np.real(krm_f0),
            label=f"{f0:.1f}Hz",
            marker="o",
            s=150,
        )
        axs[1].scatter(
            np.arange(1, krm_f0.size + 1),
            np.imag(krm_f0),
            label=f"{f0:.1f}Hz",
            marker="o",
            s=150,
        )

        # Add kraken modes
        Modes = readmodes(mod_fpath, freq=f0)
        krm_f0_kraken = Modes["k"]
        axs[0].scatter(
            np.arange(1, krm_f0_kraken.size + 1),
            np.real(krm_f0_kraken),
            label=f"Kraken {f0:.1f}Hz",
            marker="x",
            s=150,
        )
        axs[1].scatter(
            np.arange(1, krm_f0_kraken.size + 1),
            np.imag(krm_f0_kraken),
            label=f"Kraken {f0:.1f}Hz",
            marker="x",
            s=150,
        )
    axs[0].legend()
    axs[1].legend()
    axs[0].set_title("Re(kr)")
    axs[1].set_title("Im(kr)")

    fig.supxlabel("Modes")
    fig.savefig(os.path.join(HERE, "krm.png"))
    plt.close(fig)


def compare_kzm(sig_param, env_param, fcomp):

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

    # Get vertical wavenumber in water and sediment
    krm = pekeris_kr_m(freq, c1, c2, rho1, rho2, d)  # (n_f, n_modes)
    kzm_1, kzm_2 = pekeris_kzm_12(freq, c1, c2, krm)  # (n_f, n_modes)

    mod_fpath = os.path.join(HERE, FILENAME + ".mod")
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))

    for f0 in fcomp:

        idx_f0 = np.argmin(np.abs(freq - f0))
        kzm_1_f0 = krm[idx_f0, :]
        axs[0].scatter(
            np.arange(1, kzm_1_f0.size + 1),
            np.real(kzm_1_f0),
            label=f"{f0:.1f}Hz",
            marker="o",
            s=150,
        )
        axs[1].scatter(
            np.arange(1, kzm_1_f0.size + 1),
            np.imag(kzm_1_f0),
            label=f"{f0:.1f}Hz",
            marker="o",
            s=150,
        )

        # # Add kraken modes
        # Modes = readmodes(mod_fpath, freq=f0)
        # krm_f0_kraken = Modes["k"]
        # axs[0].scatter(
        #     np.arange(1, krm_f0_kraken.size + 1),
        #     np.real(krm_f0_kraken),
        #     label=f"Kraken {f0:.1f}Hz",
        #     marker="x",
        #     s=150,
        # )
        # axs[1].scatter(
        #     np.arange(1, krm_f0_kraken.size + 1),
        #     np.imag(krm_f0_kraken),
        #     label=f"Kraken {f0:.1f}Hz",
        #     marker="x",
        #     s=150,
        # )
    axs[0].legend()
    axs[1].legend()
    axs[0].set_title("Re(kz1)")
    axs[1].set_title("Im(kz1)")

    fig.supxlabel("Modes")
    fig.savefig(os.path.join(HERE, "kz1m.png"))
    plt.close(fig)


def kraken_validation(sig_param, env_param, src_rcv_param):
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

    # Run kraken
    kraken_f, kraken_gfr, field_pos = compute_kraken_solution(
        sig_param=sig_param, env_param=env_param, src_rcv_param=src_rcv_param
    )

    # Compare modes
    fcomp = [fmax]
    compare_modal_fct(sig_param=sig_param, env_param=env_param, fcomp=fcomp)

    # # env_param["attn2"] *= 35
    # # attn2 = env_param["attn2"]
    g_fr = pekeris_green_fct(
        freq, c1, c2, rho1, rho2, attn2, d, z_s, z, r_grid
    )  # Analytic green's function

    # # Compare tl_fr
    # compare_tl_fr(r_grid, freq, g_fr, kraken_f, kraken_gfr, fmin=25, fmax=75)

    # # Compare tl_r
    # compare_tl_r(r_grid, freq, g_fr, kraken_f, kraken_gfr, f0=100)

    # # Compare frequency response
    # compare_green_fct(
    #     sig_param=sig_param,
    #     env_param=env_param,
    #     src_rcv_param=src_rcv_param,
    #     kraken_f=kraken_f,
    #     kraken_gfr=kraken_gfr,
    # )

    # Compare krm
    fcomp = [25, 50, 75, 100]
    compare_krm(sig_param=sig_param, env_param=env_param, fcomp=fcomp)

    # # Compare kzm
    # fcomp = [25, 50, 75, 100]
    # compare_kzm(sig_param=sig_param, env_param=env_param, fcomp=fcomp)

    # plt.show()


def test_kraken_validation():
    # # Waveguide parameters
    # rho1 = 1.0 * 1e3  # density in water (kg/m^3)
    # c1 = 1500  # sound celerity in water (m/s)
    # rho2 = 1.5 * 1e3  # density in fluid sediment (kg/m^3)
    # c2 = 1600  # sound celerity in fluid sediment (m/s)
    # attn2 = 0.2  # compressional wave attenuation in fluid sediment in dB / wavelength
    # d = 100  # waveguide depth (m)

    # # Signal properties
    # fmax = 75  # Max frequency (Hz)
    # T = 15  # Signal duration to generate (s)
    # fs = 2 * fmax  # Sampling frequency (Hz) = Nyquist
    # ts = 1 / fs  # sampling interval (s)
    # nt = T * fs  # Number of samples
    # freq = np.fft.rfftfreq(n=nt, d=ts)  # Frequency vector

    # # Source / receiver properties
    # z_s = 25
    # z_rcv = 20
    # r_rcv = np.atleast_1d(30 * 1e3)

    # Waveguide parameters
    rho1 = 1.0 * 1e3  # density in water (kg/m^3)
    c1 = 1500  # sound celerity in water (m/s)
    rho2 = 1.5 * 1e3  # density in fluid sediment (kg/m^3)
    c2 = 1600  # sound celerity in fluid sediment (m/s)
    attn2 = 0.0  # compressional wave attenuation in fluid sediment in dB / wavelength
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
    r_rcv = np.linspace(10 * 1e3, 100 * 1e3, int(1e4))
    r0 = 50 * 1e3

    # Run test
    src_rcv_param = {"z_s": z_s, "z_rcv": z_rcv, "r_rcv": r_rcv, "r0": r0}
    sig_param = {"freq": freq, "fs": fs, "fmax": fmax}
    env_param = {
        "c1": c1,
        "c2": c2,
        "rho1": rho1,
        "rho2": rho2,
        "attn2": attn2,
        "depth": d,
    }
    kraken_validation(
        sig_param=sig_param, env_param=env_param, src_rcv_param=src_rcv_param
    )


if __name__ == "__main__":
    test_kraken_validation()
