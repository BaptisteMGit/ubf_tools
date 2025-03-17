#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   check_ideal_waveguide_properties.py
@Time    :   2024/09/27 15:25:23
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import copy
import scipy.signal as signal
from propa.rtf.ideal_waveguide import *
from signals.signals import ricker_pulse
from propa.kraken_toolbox.run_kraken import runkraken
from localisation.verlinden.testcases.testcase_envs import TestCase1_0

ROOT_DATA = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\data"


def nb_propagating_modes_freq(fmin, fmax, depth):
    # Number of propagating modes
    f = np.linspace(fmin, fmax, 1000)
    n = nb_propagating_modes(f, c0, depth)
    fc = cutoff_frequency(c0, depth)

    plt.figure()
    plt.plot(f, n)
    plt.axvline(fc, color="red", linestyle="--", label=f"Cutoff frequency: {fc:.2f} Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Number of propagating modes")
    plt.title("Number of propagating modes in an ideal waveguide")
    plt.legend()
    plt.grid()
    plt.show()


def plot_modes(f, depth):

    # Modes
    n = nb_propagating_modes(f, c0, depth)
    z = np.linspace(0, depth, 1000)

    m = np.arange(1, n + 1)
    psi_m = [psi_normalised(l, z, depth, rho_0) for l in m]
    psi_m_nonorm = [psi(l, z, depth) for l in m]

    # Derive intergral of psi_m**2 / rho_0
    integral = [np.trapz(psi_m[i] ** 2 / rho_0, z) for i in range(n)]
    print(integral)

    nb_mode_to_plot = 4
    nb_mode_to_plot = min(nb_mode_to_plot, n)
    fig, ax = plt.subplots(1, nb_mode_to_plot, figsize=(16, 10), sharey=True)
    ax[0].invert_yaxis()

    for i in range(nb_mode_to_plot):
        ax[i].plot(psi_m[i], z)
        # ax[i].plot(psi_m_nonorm[i], z, linestyle="--")
        ax[i].set_title(f"Mode {i + 1}")
        max_amp = np.max(np.abs(psi_m[i])) * 1.1
        ax[i].set_xlim([-max_amp, max_amp])
        ax[i].axvline(0, color="k", linestyle="--")

    plt.show()


def plot_tf(mode=None, src_image=None, kraken=None):

    if mode is None or src_image is None or kraken is None:
        mode, src_image, kraken = get_irs()

    # Transfert function of the window of width T centered on T/2
    T = mode["t"][-1]
    T = 0.1
    Xf = T * np.sinc(mode["f"] * T) * np.exp(-1j * np.pi * mode["f"] * T)
    plt.figure()
    plt.plot(mode["f"], np.abs(Xf), label="Window")
    plt.xlabel(r"$f \, \textrm[Hz]$")
    plt.ylabel(r"$|H(f)|$")

    plt.figure()
    plt.plot(mode["f"], np.abs(mode["h_f"]), label="Modes")
    plt.plot(src_image["f"], np.abs(src_image["h_f"]), label="Image source")
    plt.plot(kraken["f"], np.abs(kraken["h_f"]), label="Kraken")
    # plt.xlim([0, 50])
    # plt.ylim([0, np.max(np.abs(src_image["h_f_bw"])) * 1.1])
    plt.xlabel(r"$f \, \textrm[Hz]$")
    plt.ylabel(r"$|H(f)|$")
    plt.title(r"$\textrm{Transfer function}$")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(mode["f"], np.angle(mode["h_f_bw"]), label="Modes")
    plt.plot(src_image["f"], np.angle(src_image["h_f_bw"]), label="Image source")
    plt.plot(kraken["f"], np.angle(np.conj(kraken["h_f"])), label="Kraken")
    plt.xlim([0, 50])
    plt.xlabel(r"$f \, \textrm[Hz]$")
    plt.ylabel(r"$\Phi(f)$")
    plt.title(r"$\textrm{Transfer function}$")
    plt.legend()
    plt.grid()

    mod_ratio = {}
    mod_ratio["src_image_mode"] = np.abs(src_image["h_f"]) / np.abs(mode["h_f"])
    mod_ratio["kraken_mode"] = np.abs(kraken["h_f"] / mode["h_f"])
    # Derive mean deviation
    f = mode["f"]
    f_bw = f[f <= 50]
    ratio_bw = mod_ratio["kraken_mode"][f <= 50]
    ratio_no_nan = np.empty_like(ratio_bw)
    ratio_no_nan = ratio_bw
    not_nan_idx = ~(np.logical_or(np.isnan(ratio_no_nan), np.isinf(ratio_no_nan)))
    ratio_no_nan = ratio_no_nan[not_nan_idx]
    f_no_nan = f_bw[not_nan_idx]
    p = np.polyfit(f_no_nan, ratio_no_nan, deg=1)
    fit_bias = p[0] * f_no_nan + p[1]

    plt.figure()
    plt.plot(src_image["f"], mod_ratio["src_image_mode"], label="Image source / modes")
    plt.plot(kraken["f"], mod_ratio["kraken_mode"], label="Kraken / modes ")
    plt.plot(f_no_nan, fit_bias, color="r", linestyle="--", label=f"{p[0]}f + {p[1]}")
    plt.legend()

    plt.show()


def get_irs():

    # Modes
    fmin, fmax = 0, 50

    # Load tf source image
    fpath = os.path.join(ROOT_DATA, "src_image_tf.csv")
    f_img, h_img_real, h_img_imag = np.loadtxt(fpath, delimiter=",", unpack=True)
    h_img = h_img_real + 1j * h_img_imag

    # Load ir source image
    fpath = os.path.join(ROOT_DATA, "src_image_ir.csv")
    t_img, ir_img = np.loadtxt(fpath, delimiter=",", unpack=True)

    # Filter in the 0 - 50 Hz band
    h_img_bw = np.zeros_like(h_img)
    h_img_bw[f_img <= fmax] = h_img[f_img <= fmax]
    h_img_bw[f_img < fmin] = 0
    # Filtered impulse response
    ir_img_bw = fft.irfft(h_img_bw)

    src_image = {
        "t": t_img,
        "ir": ir_img,
        "f": f_img,
        "h_f": h_img,
        "h_f_bw": h_img_bw,
        "ir_bw": ir_img_bw,
    }

    # Load tf mode
    fpath = os.path.join(ROOT_DATA, "mode_tf.csv")
    f_mode, h_f_mode_real, h_f_mode_imag = np.loadtxt(fpath, delimiter=",", unpack=True)
    h_f_mode = h_f_mode_real + 1j * h_f_mode_imag

    # Load ir mode
    fpath = os.path.join(ROOT_DATA, "mode_ir.csv")
    t_mode, ir_mode = np.loadtxt(fpath, delimiter=",", unpack=True)

    # Filter in the 0 - 50 Hz band
    h_f_mode_bw = np.zeros_like(h_f_mode)
    h_f_mode_bw[f_mode <= fmax] = h_f_mode[f_mode <= fmax]
    h_f_mode_bw[f_mode < fmin] = 0

    # Filtered impulse response
    ir_mode_bw = fft.irfft(h_f_mode_bw)

    mode = {
        "t": t_mode,
        "ir": ir_mode,
        "f": f_mode,
        "h_f": h_f_mode,
        "h_f_bw": h_f_mode_bw,
        "ir_bw": ir_mode_bw,
    }

    # Load tf
    fpath = os.path.join(ROOT_DATA, "kraken_tf.csv")
    f_kraken, h_kraken_real, h_kraken_imag = np.loadtxt(
        fpath, delimiter=",", unpack=True
    )
    h_kraken = h_kraken_real + 1j * h_kraken_imag

    # k0 = 2 * np.pi * f_kraken / c0
    # norm_factor = np.exp(1j * k0) / (4)
    # norm_factor = 0.0951102 * f_kraken + 0.02735
    # h_kraken = h_kraken * 1 / norm_factor

    # Load ir kraken
    fpath = os.path.join(ROOT_DATA, "kraken_ir.csv")
    t_kraken, ir_kraken = np.loadtxt(fpath, delimiter=",", unpack=True)

    kraken = {"t": t_kraken, "ir": ir_kraken, "f": f_kraken, "h_f": h_kraken}

    return mode, src_image, kraken


def plot_ir(mode=None, src_image=None, kraken=None):
    if mode is None or src_image is None or kraken is None:
        # Impulse response
        mode, src_image, kraken = get_irs()

    plt.figure()
    plt.plot(mode["t"], mode["ir"], label="Modes")
    plt.plot(src_image["t"], src_image["ir"], label="Image source")
    plt.plot(kraken["t"], kraken["ir"], label="Kraken")
    # plt.plot(src_image["t"], src_image["filtered_ir"], label="Filtered image source")
    # plt.xlim([20, 20.2])
    plt.xlabel("Time (s)")
    plt.ylabel("h(t)")
    plt.title("Impulse response")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(mode["t"], mode["ir_bw"], label="Modes")
    plt.plot(src_image["t"], src_image["ir_bw"], label="Image source")
    plt.plot(kraken["t"], kraken["ir"], label="Kraken")
    # plt.plot(src_image["t"], src_image["filtered_ir"], label="Filtered image source")
    # plt.xlim([20, 20.2])
    plt.xlabel("Time (s)")
    plt.ylabel("h(t)")
    plt.title("Impulse response in the 0 - 50 Hz band")
    plt.legend()
    plt.grid()

    plt.show()


def compare_tf():
    mode, src_image, kraken = get_irs()

    # Plot transfert function
    plot_tf(mode=mode, src_image=src_image, kraken=kraken)

    # Plot impulse response
    plot_ir(mode=mode, src_image=src_image, kraken=kraken)

    # Create ricker pulse
    duration = src_image["t"][-1]
    ts = src_image["t"][1] - src_image["t"][0]
    fs = 1 / ts
    fc = 20
    s_pulse, t_pulse = ricker_pulse(fc, fs, duration + ts, t0=0, center=True)
    s_pulse_spectrum = fft.rfft(s_pulse)
    # plt.figure()
    # plt.plot(t_pulse, s_pulse)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.title("Ricker pulse")
    # plt.grid()

    # Convolution
    s_conv_modes = fft.irfft(s_pulse_spectrum * mode["h_f_bw"])
    s_conv_src_image = fft.irfft(s_pulse_spectrum * src_image["h_f_bw"])
    s_conv_kraken = fft.irfft(s_pulse_spectrum * kraken["h_f"])

    plt.figure()
    plt.plot(t_pulse, s_conv_modes, label="Modes")
    plt.plot(t_pulse, s_conv_src_image, label="Image source")
    plt.plot(t_pulse, s_conv_kraken, label="Kraken")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Convolution")
    plt.grid()
    plt.show()


def derive_kraken_tf():

    # Load params
    depth, r_src, z_src, z_rcv, duration = waveguide_params()

    # Create the frequency vector
    ts = 1e-3
    nt = int(duration / ts)
    f = fft.rfftfreq(nt, ts)

    # Init env
    tc_varin = {
        "freq": f,
        "src_depth": z_src,
        "max_range_m": r_src,
        "mode_theory": "adiabatic",
        "flp_n_rcv_z": 1,
        "flp_rcv_z_min": z_rcv,
        "flp_rcv_z_max": z_rcv,
        "min_depth": depth,
        "max_depth": depth,
        "dr_flp": r_src,
        "nb_modes": 100,
        "bottom_boundary_condition": "vacuum",
        "nmedia": 1,
        "phase_speed_limits": [200, 20000],
    }
    tc = TestCase1_0(mode="prod", testcase_varin=tc_varin)
    title = "Ideal waveguide"
    tc.title = title
    tc.env_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\tmp"
    tc.update(tc_varin)

    # For too long frequencies vector field fails to compute -> we will iterate over frequency subband to compute the transfert function
    fmax = 50
    fmin = cutoff_frequency(c0, depth, bottom_bc="pressure_release")
    n_subband = 500
    i_subband = 1
    f0 = fmin
    f1 = f[n_subband]
    h_kraken = np.zeros_like(f, dtype=complex)

    while f0 < fmax:
        # Frequency subband
        f_kraken = f[(f < f1) & (f >= f0)]
        # print(i_subband, f0, f1, len(f_kraken))
        pad_before = np.sum(f < f0)
        pad_after = np.sum(f >= f1)

        # Update env
        varin_update = {"freq": f_kraken}
        tc.update(varin_update)

        pressure_field, field_pos = runkraken(
            env=tc.env,
            flp=tc.flp,
            frequencies=tc.env.freq,
            parallel=True,
            verbose=True,
        )

        h_kraken_subband = np.squeeze(pressure_field, axis=(1, 2, 3))[:, 1]
        # print(pad_before, pad_after)
        # Zero padding of the transfert function to match the length of the global transfert function
        h_kraken += np.pad(h_kraken_subband, (pad_before, pad_after))

        # Update frequency subband
        i_subband += 1
        f0 = f1
        f1 = f[n_subband * i_subband]

    # Some nan values can appear in the transfert function
    h_kraken = np.nan_to_num(h_kraken)

    # Save transfert function as a csv
    fpath = os.path.join(ROOT_DATA, "kraken_tf.csv")
    np.savetxt(fpath, np.array([f, h_kraken.real, h_kraken.imag]).T, delimiter=",")

    ir_kraken = fft.irfft(h_kraken)
    t_kraken = np.arange(0, len(ir_kraken)) * ts

    # Save kraken ir
    fpath = os.path.join(ROOT_DATA, "kraken_ir.csv")
    np.savetxt(fpath, np.array([t_kraken, ir_kraken]).T, delimiter=",")


def derive_src_image_tf():
    # Load params
    depth, r_src, z_src, z_rcv, duration = waveguide_params()

    # Image source method
    ts = 1e-3
    fs = 1 / ts
    t = np.arange(0, duration, ts)
    t_img, ir_img = image_source_ri(z_src, z, r, depth, n=int(1e3), t=t)
    f_img = fft.rfftfreq(len(ir_img), 1 / fs)
    h_img = fft.rfft(ir_img)

    # Save transfert function as a csv
    fpath = os.path.join(ROOT_DATA, "src_image_tf.csv")
    np.savetxt(fpath, np.array([f_img, h_img.real, h_img.imag]).T, delimiter=",")

    # Save impulse response as a csv
    fpath = os.path.join(ROOT_DATA, "src_image_ir.csv")
    np.savetxt(fpath, np.array([t_img, ir_img]).T, delimiter=",")


def derive_mode_tf():

    # Load params
    depth, r_src, z_src, z_rcv, duration = waveguide_params()

    # Create the frequency vector
    ts = 1e-3
    nt = int(duration / ts)
    f = fft.rfftfreq(nt, ts)

    # Mode analytical solution
    f_mode, h_f_mode = h(f, z_src, z, r, depth)
    # Zero padding of the transfert function to match the length of the other transfert functions
    h_f_mode = np.pad(h_f_mode, (len(f) - len(h_f_mode), 0))
    f_mode = f

    ir_mode = fft.irfft(h_f_mode)
    t_mode = np.arange(0, len(ir_mode)) * ts

    # Save transfert function as a csv
    fpath = os.path.join(ROOT_DATA, "mode_tf.csv")
    np.savetxt(fpath, np.array([f_mode, h_f_mode.real, h_f_mode.imag]).T, delimiter=",")

    # Save impulse response as a csv
    fpath = os.path.join(ROOT_DATA, "mode_ir.csv")
    np.savetxt(fpath, np.array([t_mode, ir_mode]).T, delimiter=",")


# def plot_rtf(D):

#     # # RTF
#     D = 100
#     z_src = 25
#     z_rcv_ref = 90
#     z_rcv = z_rcv_ref

#     r_rcv_ref = 1e5
#     rl = r_rcv_ref + 10

#     f = np.arange(fmin, fmax, 0.1)
#     t0 = time()
#     f_h, h_l = h(f, z_src, z_rcv, rl, D)
#     f_h, h_ref = h(f, z_src, z_rcv, r_rcv_ref, D)
#     rtf_mano = h_l / h_ref
#     print(f"Elapsed time: {time() - t0:.2f}s")
#     t0 = time()
#     f, g_l = g(f, z_src, z_rcv_ref, z_rcv, D, r_rcv_ref, rl)
#     print(f"Elapsed time: {time() - t0:.2f}s")

#     # print(np.abs(g_l / rtf_mano))
#     plt.figure()
#     plt.plot(f, np.abs(g_l), label="g")
#     plt.plot(f_h, np.abs(rtf_mano), label="mano")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("RTF")
#     plt.title("RTF")
#     plt.grid()
#     plt.legend()
#     plt.show()


if __name__ == "__main__":
    # D = 1000
    fmin = 0.1
    fmax = 50
    # z_src = 5
    # z = 999
    # r = 30 * 1e3

    depth = 100
    z_src = 10
    z = 50
    r = 30 * 1e3

    f0 = 30
    # plot_modes(f0, D)
    # nb_propagating_modes_freq(0, 50, D)
    # plot_tf(fmin, fmax, z_src, z, r, depth)
    # plot_ir(D)
    # derive_kraken_tf()
    # derive_mode_tf()
    # derive_src_image_tf()

    compare_tf()

    # # RTF
    # D = 100
    # z_src = 25
    # z_rcv_ref = 90
    # z_rcv = z_rcv_ref

    # r_rcv_ref = 1e5
    # rl = r_rcv_ref + 10

    # f = np.arange(fmin, fmax, 0.1)
    # t0 = time()
    # f, g_l = g(f, z_src, z_rcv_ref, z_rcv, D, r_rcv_ref, rl)
